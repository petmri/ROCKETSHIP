function output = generate_dce_tofts_parity_map(varargin)
% generate_dce_tofts_parity_map Build MATLAB DCE parity baselines.
%
% This runs parts A->B->D non-interactively on a dataset and writes
% `<rootname>_<model>_fit_*.nii` maps into `outputRoot`.
%
% Example:
%   output = generate_dce_tofts_parity_map();
%   output = generate_dce_tofts_parity_map( ...
%       'subjectRoot', '/path/to/tests/data/BBB data p19', ...
%       'outputRoot', '/path/to/tests/data/BBB data p19/processed/results_matlab', ...
%       'models', {'tofts','ex_tofts','patlak','tissue_uptake','2cxm'});

thisFile = mfilename('fullpath');
testsMatlabDir = fileparts(thisFile);
helpersDir = fullfile(testsMatlabDir, 'helpers');
if exist(helpersDir, 'dir')
    addpath(helpersDir);
end

if exist('add_rocketship_paths', 'file')
    repoRoot = add_rocketship_paths();
else
    repoRoot = fileparts(fileparts(testsMatlabDir));
    addpath(repoRoot);
    addpath(fullfile(repoRoot, 'dce'));
    addpath(fullfile(repoRoot, 'dsc'));
    addpath(fullfile(repoRoot, 'external_programs'));
    addpath(fullfile(repoRoot, 'external_programs', 'niftitools'));
    addpath(fullfile(repoRoot, 'parametric_scripts'));
    addpath(fullfile(repoRoot, 'parametric_scripts', 'custom_scripts'));
end

% Force DCE helpers to win over same-name DSC helpers.
addpath(fullfile(repoRoot, 'dce'), '-begin');
selectedAifFitHelper = which('AIFbiexpfithelp');
if isempty(selectedAifFitHelper) || isempty(strfind(selectedAifFitHelper, [filesep 'dce' filesep])) %#ok<STREMP>
    error('Unexpected AIFbiexpfithelp on path: %s', selectedAifFitHelper);
end

p = inputParser;
addParameter(p, 'subjectRoot', fullfile(repoRoot, 'tests/data', 'BBB data p19'), @is_text_scalar);
addParameter(p, 'outputRoot', '', @is_text_scalar);
addParameter(p, 'rootname', 'Dyn-1', @is_text_scalar);
addParameter(p, 'trMs', 8.29, @isscalar);
addParameter(p, 'faDeg', 15.0, @isscalar);
addParameter(p, 'timeResolutionSec', 15.84, @isscalar);
% -1 = auto: B_AIF_fitting_func derives the injection window from Stage-A's
% detected steady state, the same way the Python parity fixture does. Pinning
% these to fixed minutes silently decouples the two pipelines' Stage-B inputs
% (the AIF onset lands on a different frame) while every other setting matches,
% which is exactly what the parity suite is meant to be comparing.
addParameter(p, 'startInjectionMin', -1, @isscalar);
addParameter(p, 'endInjectionMin', -1, @isscalar);
% -2 = auto-detect the steady-state end (find_end_ss_tv), matching the Python
% fixture's steady_state_auto_method = 'tv'.
addParameter(p, 'steadyStateTime', -2, @isscalar);
addParameter(p, 'hematocrit', 0.42, @isscalar);
addParameter(p, 'snrFilter', 5.0, @isscalar);
addParameter(p, 'relaxivity', 3.6, @isscalar);
addParameter(p, 'models', {'tofts'}, @is_model_list);
addParameter(p, 'roiList', {}, @is_text_list_or_scalar);
% Optional explicit input paths. When set, they override the flat subjectRoot/processed layout,
% which lets this generator target the BIDS_test sub-10bbbdownsample fixture (or any BIDS subject).
addParameter(p, 'dynamicPath', '', @is_text_scalar);
addParameter(p, 'aifRoiPath', '', @is_text_scalar);
addParameter(p, 'brainRoiPath', '', @is_text_scalar);
addParameter(p, 't1MapPath', '', @is_text_scalar);
addParameter(p, 'noiseRoiPath', '', @is_text_scalar);
% Stop after writing the Stage-B AIF contract JSON. Regenerating that payload otherwise
% costs a full voxelwise Stage-D run it does not depend on.
addParameter(p, 'stageBOnly', false, @(v) islogical(v) || isnumeric(v));
parse(p, varargin{:});

subjectRoot = char(p.Results.subjectRoot);
outputRoot = char(p.Results.outputRoot);
rootname = char(p.Results.rootname);
modelList = normalize_models(p.Results.models);
roiList = normalize_text_list(p.Results.roiList);
if isempty(outputRoot)
    outputRoot = fullfile(subjectRoot, 'processed', 'results_matlab');
end

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

processedRoot = fullfile(subjectRoot, 'processed');
dynamicPath = pick_path(p.Results.dynamicPath, fullfile(subjectRoot, 'Dynamic_t1w.nii'));
t1AifPath = pick_path(p.Results.aifRoiPath, fullfile(processedRoot, 'T1_AIF_roi.nii'));
t1RoiPath = pick_path(p.Results.brainRoiPath, fullfile(processedRoot, 'T1_brain_roi.nii'));
t1MapPath = pick_path(p.Results.t1MapPath, fullfile(processedRoot, 'T1_map_t1_fa_fit_fa10.nii'));
noisePath = pick_path(p.Results.noiseRoiPath, fullfile(processedRoot, 'T1_noise_roi.nii'));

required = {dynamicPath, t1AifPath, t1RoiPath, t1MapPath, noisePath};
for i = 1:numel(required)
    if ~exist(required{i}, 'file')
        error('Missing required input file: %s', required{i});
    end
end

% Baselines must come from the CPU fit()/confint() path: gpufit zero-pads the CI columns and
% diverges from MATLAB CPU, which shipped all-zero CI maps undetected for months (issue #3,
% project-management/projects/archived/batch-parity/batch_parity.md).
% Mirrors the fitter's own USE_GPU test (FXLfit_generic.m) -- checking force_cpu alone fires
% on CUDA-less machines already on the CPU path. Checked, not set: dce_preferences.txt is
% tracked CRLF, so a generator rewriting it is its own hazard.
try
    gpu_available = GpufitCudaAvailableMex;
catch
    gpu_available = 0;
end
cpu_prefs = parse_preference_file('dce_preferences.txt', 0, {'force_cpu'}, {'0'});
force_cpu_raw = cpu_prefs.force_cpu;
if ischar(force_cpu_raw) || isstring(force_cpu_raw)
    force_cpu_val = str2double(force_cpu_raw);
else
    force_cpu_val = double(force_cpu_raw);
end
force_cpu = ~isnan(force_cpu_val) && force_cpu_val ~= 0;
if gpu_available && ~force_cpu
    error(['generate_dce_tofts_parity_map:ForceCpuRequired\n' ...
        'CUDA gpufit is available and dce/dce_preferences.txt has force_cpu = %s, so this\n' ...
        'run would fit on the GPU. Parity baselines must come from the CPU path.\n' ...
        'Set "force_cpu = 1", re-run, then revert it to 0.\n' ...
        'Full recipe: tests/README.md.'], strtrim(num2str(force_cpu_val)));
end

filevolume = 1;
noise_pathpick = true;
noise_pixsize = 16;
LUT = 1;
filelist = {dynamicPath};
t1aiffiles = {t1AifPath};
t1roifiles = {t1RoiPath};
t1mapfiles = {t1MapPath};
noisefiles = {noisePath};
driftfiles = {};
fileorder = 'xyzt';
quant = true;
mask_roi = true;
mask_aif = true;
aif_rr_type = 'aif_roi';
tr = p.Results.trMs;
fa = p.Results.faDeg;
hematocrit = p.Results.hematocrit;
snr_filter = p.Results.snrFilter;
relaxivity = p.Results.relaxivity;
steady_state_time = p.Results.steadyStateTime;
if steady_state_time >= 1
    steady_state_time = round(steady_state_time);
end
drift_global = false;
blood_t1 = 0;
injection_duration = 1;
start_t = 0;
end_t = 0;

[~, A_vars, errormsg] = A_make_R1maps_func(filevolume, noise_pathpick, ...
    noise_pixsize, LUT, filelist, t1aiffiles, t1roifiles, t1mapfiles, ...
    noisefiles, driftfiles, rootname, fileorder, quant, mask_roi, ...
    mask_aif, aif_rr_type, tr, fa, hematocrit, snr_filter, relaxivity, ...
    steady_state_time, drift_global, blood_t1, injection_duration, ...
    start_t, end_t, false);
if ~isempty(errormsg)
    error('A_make_R1maps_func failed: %s', errormsg);
end

resultsAPath = fullfile(outputRoot, ['A_' rootname 'R1info.mat']);
startTime = 0;
endTime = 0;
fitAif = 1;
importAifPath = '';
timeResolutionMin = p.Results.timeResolutionSec / 60.0;
timevectPath = '';
[~, B_vars] = B_AIF_fitting_func(resultsAPath, startTime, endTime, ...
    p.Results.startInjectionMin, p.Results.endInjectionMin, fitAif, ...
    importAifPath, timeResolutionMin, timevectPath, A_vars, false);

resultsBPath = fullfile(outputRoot, ['B_' rootname 'fitted_R1info.mat']);

% Stage-B AIF contract payload. The parity suite compares only final maps, so a
% structurally different AIF fit stays invisible behind passing map checks -- that is
% exactly how issue #2 survived for months. See tests/python/test_stage_b_aif_parity.py.
stageBContractPath = fullfile(outputRoot, [rootname '_stage_b_aif.json']);
write_stage_b_aif_contract(stageBContractPath, B_vars, rootname);
output = struct();
output.subjectRoot = subjectRoot;
output.outputRoot = outputRoot;
output.models = modelList;
output.resultsAPath = resultsAPath;
output.resultsBPath = resultsBPath;
output.stageBContractPath = stageBContractPath;
if p.Results.stageBOnly
    output.ktransPaths = {};
    fprintf('Stage-B AIF contract written: %s\n', stageBContractPath);
    return;
end

dce_model = struct( ...
    'tofts', 0, ...
    'ex_tofts', 0, ...
    'fxr', 0, ...
    'fractal', 0, ...
    'auc', 0, ...
    'nested', 0, ...
    'patlak', 0, ...
    'tissue_uptake', 0, ...
    'two_cxm', 0, ...
    'FXL_rr', 0);
for i = 1:numel(modelList)
    modelName = modelList{i};
    if strcmp(modelName, '2cxm')
        dce_model.two_cxm = 1;
    elseif isfield(dce_model, modelName)
        dce_model.(modelName) = 1;
    else
        error('Unsupported model requested: %s', modelName);
    end
end

time_smoothing = 'none';
time_smoothing_window = 0;
xy_smooth_size = 0;
number_cpus = 1;
roi_list = roiList;
fit_voxels = 1;
neuroecon = 0;
outputft = 1;

D_fit_voxels_func(resultsBPath, dce_model, time_smoothing, ...
    time_smoothing_window, xy_smooth_size, number_cpus, roi_list, ...
    fit_voxels, neuroecon, outputft, B_vars, false);

missing = {};
ktransPaths = cell(numel(modelList), 1);
for i = 1:numel(modelList)
    modelName = modelList{i};
    ktransPaths{i} = fullfile(outputRoot, [rootname '_' modelName '_fit_Ktrans.nii']);
    if ~exist(ktransPaths{i}, 'file')
        missing{end + 1} = ktransPaths{i}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    errText = sprintf('%s\n', missing{:});
    error('Expected Ktrans map(s) not found:\n%s', errText);
end

output.ktransPaths = ktransPaths;

fprintf('MATLAB DCE baseline written for models: %s\n', strjoin(modelList, ', '));
end

function ok = is_text_scalar(value)
ok = ischar(value) || (isstring(value) && isscalar(value));
end

function ok = is_model_list(value)
ok = is_text_scalar(value) || iscellstr(value) || ...
    (iscell(value) && all(cellfun(@is_text_scalar, value)));
end

function ok = is_text_list_or_scalar(value)
ok = is_text_scalar(value) || isempty(value) || iscellstr(value) || ...
    (iscell(value) && all(cellfun(@is_text_scalar, value))) || isstring(value);
end

function out = normalize_models(value)
if is_text_scalar(value)
    raw = {char(value)};
elseif isstring(value)
    raw = cellstr(value(:));
elseif iscell(value)
    raw = cell(size(value));
    for i = 1:numel(value)
        raw{i} = char(value{i});
    end
else
    error('Unsupported model list input');
end

out = {};
for i = 1:numel(raw)
    modelName = lower(strtrim(raw{i}));
    if isempty(modelName)
        continue;
    end
    if any(strcmp(modelName, out))
        continue;
    end
    out{end + 1} = modelName; %#ok<AGROW>
end

if isempty(out)
    out = {'tofts'};
end
end

function out = normalize_text_list(value)
if isempty(value)
    out = {};
    return;
end

if is_text_scalar(value)
    out = {char(value)};
    return;
end

if isstring(value)
    value = cellstr(value(:));
end

if ~iscell(value)
    error('Unsupported text-list input');
end

out = {};
for i = 1:numel(value)
    text = strtrim(char(value{i}));
    if isempty(text)
        continue;
    end
    out{end + 1} = text; %#ok<AGROW>
end
end

function out = pick_path(override, fallback)
% Use the explicit override path when provided, otherwise the flat-layout fallback.
override = char(override);
if isempty(strtrim(override))
    out = fallback;
else
    out = override;
end
end
