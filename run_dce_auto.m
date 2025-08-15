function run_dce_auto(subject_tp_path)
    % Automated DCE processing using script_preferences.txt configuration
    % Subject timepoint path should be the full path to the subject directory
    
    % Find and add subpaths 
    mfilepath=fileparts(which('run_dce_auto'));
    addpath(fullfile(mfilepath,'dce'));
    addpath(fullfile(mfilepath,'external_programs'));
    addpath(fullfile(mfilepath,'external_programs/niftitools'));
    addpath(fullfile(mfilepath,'parametric_scripts'));
    
    %% RUNA
    % load A prefs
    script_prefs = parse_preference_file('script_preferences.txt', 0, ...
        {'filevolume', 'noise_pathpick', 'noise_pixsize', 'LUT', 'dynamic_files', ...
        'aif_files', 'roi_files', 't1map_files', 'noise_files', 'drift_files', ...
        'rootname', 'fileorder', 'quant', 'roimaskroi', 'aifmaskroi', 'aif_rr_type', ...
        'tr', 'fa', 'hematocrit', 'snr_filter', 'relaxivity', 'injection_time', ...
        'drift_global', 'blood_t1', 'injection_duration', 'start_t', 'end_t'});

    % type casts
    filevolume = str2num(script_prefs.filevolume);
    noise_pathpick = str2num(script_prefs.noise_pathpick);
    noise_pixsize = str2num(script_prefs.noise_pixsize);
    LUT = str2num(script_prefs.LUT);
    
    % gather filenames
    tmp = dir(strcat(subject_tp_path, script_prefs.dynamic_files, '*'));
    dynamic_files = cellstr(strcat(tmp.folder, '/', tmp.name));

    tmp = dir(strcat(subject_tp_path, script_prefs.aif_files, '*'));
    aif_files = cellstr(strcat(tmp.folder, '/', tmp.name));

    tmp = dir(strcat(subject_tp_path, script_prefs.roi_files, '*'));
    roi_files = cellstr(strcat(tmp.folder, '/', tmp.name));

    tmp = dir(strcat(subject_tp_path, script_prefs.t1map_files, '*'));
    t1map_files = cellstr(strcat(tmp.folder, '/', tmp.name));

    if ~strcmp(script_prefs.noise_files,'')
        tmp = dir(strcat(subject_tp_path, script_prefs.dynamic_files, '*'));
        noise_files = cellstr(strcat(tmp.folder, '/', tmp.name));
    else
        noise_files = '';
    end

    if ~strcmp(script_prefs.drift_files,'')
        tmp = dir(strcat(subject_tp_path, script_prefs.dynamic_files, '*'));
        drift_files = cellstr(strcat(tmp.folder, '/', tmp.name));
    else
        drift_files = '';
    end

    quant = str2num(script_prefs.quant);
    roimaskroi = str2num(script_prefs.roimaskroi);
    aifmaskroi = str2num(script_prefs.aifmaskroi);

    dce_json = strcat(subject_tp_path, 'DCE.json');
    if exist(dce_json, 'file')
        fid = fopen(dce_json);
        raw = fread(fid,inf);
        str = char(raw');
        fclose(fid);
        json = jsondecode(str);
        tr = json.RepetitionTime;
        fa = json.FlipAngle;
    else
        tr = str2num(script_prefs.tr);
        fa = str2num(script_prefs.fa);
    end

    hematocrit = str2num(script_prefs.hematocrit);
    snr_filter = str2num(script_prefs.snr_filter);
    relaxivity = str2num(script_prefs.relaxivity);
    injection_time = str2num(script_prefs.injection_time);
    drift_global = str2num(script_prefs.drift_global);
    blood_t1 = str2num(script_prefs.blood_t1);
    injection_duration = str2num(script_prefs.injection_duration);
    
    if isempty(script_prefs.start_t)
        start_t = [];
    else
        start_t = str2num(script_prefs.start_t);
    end
    
    if isempty(script_prefs.end_t)
        end_t = [];
    else
        end_t = str2num(script_prefs.end_t);
    end

    % main function call
    try
        [A_results, A_vars, errormsg] = A_make_R1maps_func(filevolume, noise_pathpick, ...
            noise_pixsize, LUT, dynamic_files,aif_files, ...
            roi_files, t1map_files, ...
            noise_files, drift_files, ...
            script_prefs.rootname, script_prefs.fileorder, quant, roimaskroi, ...
            aifmaskroi, script_prefs.aif_rr_type, tr, fa, hematocrit, snr_filter, ...
            relaxivity, injection_time, drift_global, blood_t1, injection_duration, ...
            start_t, end_t);
    catch L
        disp(L.message)
        return;
    end

    if ~isempty(errormsg)
        error(errormsg);
        return;
    end

    %% RUNB
    % load B prefs
    script_prefs = parse_preference_file('script_preferences.txt', 0, ...
        {'start_time', 'end_time', 'auto_find_injection', 'start_injection', ...
        'end_injection', 'fit_aif', 'time_resolution', 'aif_type', ...
        'import_aif_path', 'timevectyn', 'timevectpath'});

    % type casts
    start_time = str2num(script_prefs.start_time);
    end_time = str2num(script_prefs.end_time);
    auto_find_injection = str2num(script_prefs.auto_find_injection);
    start_injection = str2num(script_prefs.start_injection);
    end_injection = str2num(script_prefs.end_injection);
    fit_aif = str2num(script_prefs.fit_aif);
    time_resolution = str2num(script_prefs.time_resolution);
    aif_type = str2num(script_prefs.aif_type);
    import_aif_path = script_prefs.import_aif_path;
    timevectyn = str2num(script_prefs.timevectyn);
    timevectpath = script_prefs.timevectpath;

    % main function call
    fail = 0;
    while (fail < 3)
        try
            [B_results, B_vars] = B_AIF_fitting_func(A_results, A_vars, ...
                start_time, end_time, start_injection, ...
                end_injection, fit_aif, import_aif_path, ...
                time_resolution, timevectpath);
            break;
        catch L
            disp("RUNB failed. Repeating in case of bad read...")
            disp(L.message)
        end
        fail = fail + 1;
    end
    if fail >= 3
        warning("RUNB failed and could not recover.");
        return;
    end
    %% RUND
    script_prefs = parse_preference_file('script_preferences.txt', 0, ...
        {'tofts', 'ex_tofts', 'fxr', 'auc', 'nested', 'patlak', ...
        'tissue_uptake', 'two_cxm', 'FXL_rr', 'time_smoothing', ...
        'time_smoothing_window', 'xy_smooth_size', 'number_cpus', 'roi_list', ...
        'fit_voxels', 'outputft'});

    % type casts
    dce_model.tofts = str2num(script_prefs.tofts);
    dce_model.ex_tofts = str2num(script_prefs.ex_tofts);
    dce_model.fxr = str2num(script_prefs.fxr);
    dce_model.auc = str2num(script_prefs.auc);
    dce_model.nested = str2num(script_prefs.nested);
    dce_model.patlak = str2num(script_prefs.patlak);
    dce_model.tissue_uptake = str2num(script_prefs.tissue_uptake);
    dce_model.two_cxm = str2num(script_prefs.two_cxm);
    dce_model.FXL_rr = str2num(script_prefs.FXL_rr);
    dce_model.fractal = 0;

    time_smoothing_window = str2num(script_prefs.time_smoothing_window);
    xy_smooth_size = str2num(script_prefs.xy_smooth_size);
    number_cpus = str2num(script_prefs.number_cpus);
    fit_voxels = str2num(script_prefs.fit_voxels);
    outputft = str2num(script_prefs.outputft);

    if (~isempty(script_prefs.roi_list))
        roi_list = split(script_prefs.roi_list);
    else
        roi_list = '';
    end

    neuroecon = 0;

    % main function call
    try
        D_results = D_fit_voxels_func(B_results, B_vars, dce_model, ...
            script_prefs.time_smoothing,time_smoothing_window, ...
            xy_smooth_size,number_cpus,roi_list,fit_voxels,neuroecon, ...
            outputft);
    catch L
        disp("RUND failed:")
        disp(L.message)
    end
end