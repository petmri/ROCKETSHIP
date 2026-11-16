function run_testretest_matlab(jobs_json, bids_root, out_root, baseline, varargin)
% RUN_TESTRETEST_MATLAB  MATLAB arm of the test-retest validation.
%
%   run_testretest_matlab(jobs_json, bids_root, out_root, baseline)
%   run_testretest_matlab(..., 'subject', 'sub-X', 'skip_existing', true, 'keep_inputs', false)
%
% Fits every session of a job list written by run_testretest_export_matlab_jobs.py, with
% the inputs and the settings the Python fit of that session used. It calls
% A_make_R1maps_func, B_AIF_fitting_func and D_fit_voxels_func directly, in the order and
% with the argument layout of run_dce_cli, because run_dce_cli cannot be matched: it
% overwrites relaxivity, has no per-session baseline, and writes into the input folders
% (matlab_arm.md).
%
%   jobs_json  job list, e.g. <DCE_BIDS>/derivatives/rocketship-testretest/matlab_jobs_AC.json
%   bids_root  the DCE BIDS root on this machine; job paths are relative to it
%   out_root   where fits go, one folder per session, e.g.
%              <DCE_BIDS>/derivatives/rocketship-testretest/matlab_AC
%   baseline   what to do where the Python fit auto-detected the baseline end:
%                'python'  pass the frame Python chose (isolates the fit from the detector)
%                'auto'    let MATLAB detect it (find_end_ss_tv)
%              A frame the rater marked is passed either way. No default: this choice is
%              part of what "matched" means and belongs in the pre-registration.
%
% Settings that are not study settings (noise, SNR filter, drift, smoothing, CPU count)
% are read from the script_preferences.txt of the ROCKETSHIP checkout on the MATLAB path,
% as run_dce_cli does, and recorded in matlab_provenance.json. Voxel-fit bounds and
% starting values come from dce/dce_preferences.txt, as always.
%
% Outputs per session: <rootname>_<model>_fit_<parameter>.nii (the names the Python arm
% writes, so the collector reads both), matlab_fit.log, and in out_root
% matlab_fit_manifest.tsv and matlab_provenance.json.
%
% Status 2026-11-16: written against the MATLAB sources of the validation branch but NOT
% YET RUN. MATLAB was not available where it was written. Run one session first and
% compare its maps with the Python fit before starting the cohort.

    options = struct('subject', '', 'session', '', 'skip_existing', true, 'keep_inputs', false);
    for k = 1:2:numel(varargin)
        if ~isfield(options, varargin{k})
            error('run_testretest_matlab:option', 'Unknown option %s.', varargin{k});
        end
        options.(varargin{k}) = varargin{k + 1};
    end
    if nargin < 4 || ~any(strcmp(baseline, {'python', 'auto'}))
        error('run_testretest_matlab:baseline', 'baseline must be ''python'' or ''auto''.');
    end

    repo = fileparts(which('run_dce_cli'));
    if isempty(repo)
        error('run_testretest_matlab:path', 'Put the ROCKETSHIP checkout on the MATLAB path first.');
    end
    addpath(fullfile(repo, 'dce'));
    addpath(fullfile(repo, 'external_programs'));
    addpath(fullfile(repo, 'external_programs', 'niftitools'));
    addpath(fullfile(repo, 'parametric_scripts'));

    data = jsondecode(fileread(jobs_json));
    jobs = data.jobs;
    if ~exist(out_root, 'dir')
        mkdir(out_root);
    end

    % Non-study settings, from the checkout's script_preferences.txt. Three reads with the
    % key lists run_dce_cli uses for parts A, B and D: the parser matches keys by prefix
    % (start_t would otherwise fill start_time), so the lists are kept as they are there.
    preference_file = fullfile(repo, 'script_preferences.txt');
    prefs = struct();
    prefs.a = parse_preference_file(preference_file, 0, ...
        {'noise_pathpick' 'noise_pixsize' 'dynamic_files' ...
        'aif_files' 'roi_files' 't1map_files' 'noise_files' 'drift_files' ...
        'rootname' 'fileorder' 'quant' 'roimaskroi' 'aifmaskroi' 'aif_rr_type' ...
        'tr' 'fa' 'hematocrit' 'snr_filter' 'relaxivity' 'injection_time' ...
        'injection_duration' 'drift_global' 'blood_t1', 'start_t', 'end_t' ...
        'time_resolution' 'force_use_default_relaxivity'});
    prefs.b = parse_preference_file(preference_file, 0, ...
        {'start_time', 'end_time', 'auto_find_injection', 'start_injection', ...
        'end_injection', 'fit_aif', 'time_resolution', 'aif_type', ...
        'import_aif_path', 'timevectyn', 'timevectpath', 'rootname', ...
        'fileorder', 'aif_rr_type'});
    prefs.d = parse_preference_file(preference_file, 0, ...
        {'tofts', 'ex_tofts', 'fxr', 'auc', 'nested', 'patlak', ...
        'tissue_uptake', 'two_cxm', 'FXL_rr', 'time_smoothing', ...
        'time_smoothing_window', 'xy_smooth_size', 'number_cpus', 'roi_list', ...
        'fit_voxels', 'outputft'});

    provenance = struct();
    provenance.written = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
    provenance.matlab_version = version;
    provenance.rocketship_path = repo;
    [~, revision] = system(sprintf('git -C "%s" rev-parse HEAD', repo));
    [~, dirty] = system(sprintf('git -C "%s" status --porcelain', repo));
    provenance.rocketship_revision = strtrim(revision);
    provenance.rocketship_dirty = ~isempty(strtrim(dirty));
    provenance.jobs_json = jobs_json;
    provenance.rater = data.rater;
    provenance.baseline_mode = baseline;
    provenance.aif_rr_type = 'aif_roi_static';
    provenance.script_preferences = prefs;
    provenance.gpufit_available = exist('GpufitCudaAvailableMex', 'file') == 3;
    write_text(fullfile(out_root, 'matlab_provenance.json'), jsonencode(provenance));

    manifest_path = fullfile(out_root, 'matlab_fit_manifest.tsv');
    is_new = ~exist(manifest_path, 'file');
    manifest = fopen(manifest_path, 'a');
    if is_new
        fprintf(manifest, 'subject\tsession\tstatus\tsteady_state_argument\tduration_sec\tdetail\n');
    end
    cleanup = onCleanup(@() fclose(manifest));

    for i = 1:numel(jobs)
        if iscell(jobs)
            job = jobs{i};
        else
            job = jobs(i);
        end
        if ~isempty(options.subject) && ~strcmp(job.subject, options.subject)
            continue;
        end
        if ~isempty(options.session) && ~strcmp(job.session, options.session)
            continue;
        end
        session_dir = fullfile(out_root, job.subject, job.session);
        models = fieldnames(job.models);
        first_map = fullfile(session_dir, [job.rootname '_' models{1} '_fit_Ktrans.nii']);
        if options.skip_existing && exist(first_map, 'file')
            fprintf('%s %s: already fitted, skipped\n', job.subject, job.session);
            continue;
        end
        if ~exist(session_dir, 'dir')
            mkdir(session_dir);
        end

        % Where Python auto-detected the baseline end, 'auto' hands the detection to MATLAB
        marked_by_rater = strcmp(job.steady_state_source, 'aif_sidecar');
        if marked_by_rater || strcmp(baseline, 'python')
            steady_state_time = job.steady_state_end;
        else
            steady_state_time = -2;
        end

        started = tic;
        status = 'ok';
        detail = '';
        diary(fullfile(session_dir, 'matlab_fit.log'));
        try
            fit_session(job, bids_root, session_dir, steady_state_time, prefs, options.keep_inputs);
        catch failure
            status = 'failed';
            detail = strrep(strrep(failure.message, sprintf('\n'), ' '), sprintf('\t'), ' ');
            disp(getReport(failure));
        end
        diary off;
        close all;
        fprintf(manifest, '%s\t%s\t%s\t%g\t%.1f\t%s\n', job.subject, job.session, status, ...
            steady_state_time, toc(started), detail);
        fprintf('[%d/%d] %s %s: %s %s\n', i, numel(jobs), job.subject, job.session, status, detail);
    end
end


function fit_session(job, bids_root, session_dir, steady_state_time, prefs, keep_inputs)
    % The dynamic series is staged in the session folder itself: stage A writes next to
    % it, and that is what keeps the outputs out of the rater run.
    inputs_dir = fullfile(session_dir, 'inputs');
    dynamic = stage_input(fullfile(bids_root, job.dynamic), session_dir);
    aif = stage_input(fullfile(bids_root, job.aif), inputs_dir);
    roi = stage_input(fullfile(bids_root, job.roi), inputs_dir);
    t1map = stage_input(fullfile(bids_root, job.t1map), inputs_dir);

    % An empty start_t / end_t is the whole series (the job list has null there)
    start_t = job.start_t;
    end_t = job.end_t;

    filevolume = 1;  % 4D files
    lut = 1;
    noise_files = '';
    drift_files = '';
    % Study settings come from the job; the rest from the preference file, as in run_dce_cli
    [A_results, A_vars, errormsg] = A_make_R1maps_func(filevolume, ...
        str2double(prefs.a.noise_pathpick), str2double(prefs.a.noise_pixsize), lut, ...
        {dynamic}, {aif}, {roi}, {t1map}, noise_files, drift_files, ...
        job.rootname, prefs.a.fileorder, str2double(prefs.a.quant), ...
        str2double(prefs.a.roimaskroi), str2double(prefs.a.aifmaskroi), 'aif_roi_static', ...
        job.tr_ms, job.fa_deg, job.hematocrit, str2double(prefs.a.snr_filter), ...
        job.relaxivity, steady_state_time, str2double(prefs.a.drift_global), ...
        job.blood_t1_sec, str2double(prefs.a.injection_duration), start_t, end_t, false);
    if ~isempty(errormsg)
        error('run_testretest_matlab:stageA', '%s', errormsg);
    end

    % start/end of injection are found automatically (-1), as run_dce_cli does with
    % auto_find_injection = 1; no manual time vector
    time_resolution_min = job.time_resolution_sec / 60;
    [B_results, B_vars] = B_AIF_fitting_func(A_results, ...
        str2double(prefs.b.start_time), str2double(prefs.b.end_time), -1, -1, ...
        job.aif_type, '', time_resolution_min, '', A_vars, false);

    dce_model = struct('tofts', 0, 'ex_tofts', 0, 'fxr', 0, 'auc', 0, 'nested', 0, ...
        'patlak', 0, 'tissue_uptake', 0, 'two_cxm', 0, 'FXL_rr', 0, 'fractal', 0);
    requested = fieldnames(job.models);
    for k = 1:numel(requested)
        if ~isfield(dce_model, requested{k})
            error('run_testretest_matlab:model', 'Unknown model %s in the job list.', requested{k});
        end
        dce_model.(requested{k}) = job.models.(requested{k});
    end
    roi_list = '';
    neuroecon = 0;
    D_fit_voxels_func(B_results, dce_model, prefs.d.time_smoothing, ...
        str2double(prefs.d.time_smoothing_window), str2double(prefs.d.xy_smooth_size), ...
        str2double(prefs.d.number_cpus), roi_list, str2double(prefs.d.fit_voxels), ...
        neuroecon, str2double(prefs.d.outputft), B_vars, false);

    if ~keep_inputs
        delete(dynamic);
        rmdir(inputs_dir, 's');
    end
end


function staged = stage_input(source, folder)
    % Copy one input next to the fit, unzipped: the rater runs stay read-only
    if ~exist(source, 'file')
        error('run_testretest_matlab:input', 'Input not found: %s', source);
    end
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    [~, name, extension] = fileparts(source);
    if strcmp(extension, '.gz')
        gunzip(source, folder);
        staged = fullfile(folder, name);
    else
        copyfile(source, folder);
        staged = fullfile(folder, [name extension]);
    end
end


function write_text(path, text)
    handle = fopen(path, 'w');
    fprintf(handle, '%s\n', text);
    fclose(handle);
end
