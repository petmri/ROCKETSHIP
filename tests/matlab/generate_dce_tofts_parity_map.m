function output = generate_dce_tofts_parity_map(varargin)
% generate_dce_tofts_parity_map Build a MATLAB Tofts Ktrans parity baseline.
%
% This runs parts A->B->D non-interactively on a dataset and writes
% `<rootname>_tofts_fit_Ktrans.nii` into `outputRoot`.
%
% Example:
%   output = generate_dce_tofts_parity_map();
%   output = generate_dce_tofts_parity_map( ...
%       'subjectRoot', '/path/to/test_data/BBB data p19', ...
%       'outputRoot', '/path/to/test_data/BBB data p19/processed/results_matlab');

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
addParameter(p, 'subjectRoot', fullfile(repoRoot, 'test_data', 'BBB data p19'), @is_text_scalar);
addParameter(p, 'outputRoot', '', @is_text_scalar);
addParameter(p, 'rootname', 'Dyn-1', @is_text_scalar);
addParameter(p, 'trMs', 8.29, @isscalar);
addParameter(p, 'faDeg', 15.0, @isscalar);
addParameter(p, 'timeResolutionSec', 15.84, @isscalar);
addParameter(p, 'startInjectionMin', 0.5, @isscalar);
addParameter(p, 'endInjectionMin', 0.7, @isscalar);
addParameter(p, 'steadyStateTime', -2, @isscalar);
addParameter(p, 'hematocrit', 0.42, @isscalar);
addParameter(p, 'snrFilter', 5.0, @isscalar);
addParameter(p, 'relaxivity', 3.6, @isscalar);
parse(p, varargin{:});

subjectRoot = char(p.Results.subjectRoot);
outputRoot = char(p.Results.outputRoot);
rootname = char(p.Results.rootname);
if isempty(outputRoot)
    outputRoot = fullfile(subjectRoot, 'processed', 'results_matlab');
end

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

dynamicPath = fullfile(subjectRoot, 'Dynamic_t1w.nii');
processedRoot = fullfile(subjectRoot, 'processed');
t1AifPath = fullfile(processedRoot, 'T1_AIF_roi.nii');
t1RoiPath = fullfile(processedRoot, 'T1_brain_roi.nii');
t1MapPath = fullfile(processedRoot, 'T1_map_t1_fa_fit_fa10.nii');
noisePath = fullfile(processedRoot, 'T1_noise_roi.nii');

required = {dynamicPath, t1AifPath, t1RoiPath, t1MapPath, noisePath};
for i = 1:numel(required)
    if ~exist(required{i}, 'file')
        error('Missing required input file: %s', required{i});
    end
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
[~, B_vars] = B_AIF_fitting_func(resultsAPath, A_vars, startTime, ...
    endTime, p.Results.startInjectionMin, p.Results.endInjectionMin, ...
    fitAif, importAifPath, timeResolutionMin, timevectPath, false);

resultsBPath = fullfile(outputRoot, ['B_' rootname 'fitted_R1info.mat']);
dce_model = struct( ...
    'tofts', 1, ...
    'ex_tofts', 0, ...
    'fxr', 0, ...
    'fractal', 0, ...
    'auc', 0, ...
    'nested', 0, ...
    'patlak', 0, ...
    'tissue_uptake', 0, ...
    'two_cxm', 0, ...
    'FXL_rr', 0);

time_smoothing = 'none';
time_smoothing_window = 0;
xy_smooth_size = 0;
number_cpus = 1;
roi_list = {};
fit_voxels = 1;
neuroecon = 0;
outputft = 1;

D_fit_voxels_func(resultsBPath, B_vars, dce_model, time_smoothing, ...
    time_smoothing_window, xy_smooth_size, number_cpus, roi_list, ...
    fit_voxels, neuroecon, outputft, false);

ktransPath = fullfile(outputRoot, [rootname '_tofts_fit_Ktrans.nii']);
if ~exist(ktransPath, 'file')
    error('Expected Ktrans map not found: %s', ktransPath);
end

output = struct();
output.subjectRoot = subjectRoot;
output.outputRoot = outputRoot;
output.ktransPath = ktransPath;
output.resultsAPath = resultsAPath;
output.resultsBPath = resultsBPath;

fprintf('MATLAB Tofts baseline written: %s\n', ktransPath);
end

function ok = is_text_scalar(value)
ok = ischar(value) || (isstring(value) && isscalar(value));
end
