function output = generate_t1_parity_map(varargin)
% generate_t1_parity_map Build a MATLAB T1 map baseline for end-to-end parity.
%
% Loops the single-voxel fitParameter (already validated against the Python port
% at the function level) over a small VFA fixture and writes a T1 map NIfTI, so
% the Python parametric T1 pipeline can be compared map-to-map against MATLAB.
% This exercises the full MATLAB T1 fit + map-assembly path headlessly, without
% the GUI batch/JOB_struct machinery.
%
% Example:
%   generate_t1_parity_map( ...
%       'vfaFiles', {'flip-02deg_VFA.nii.gz','flip-05deg_VFA.nii.gz','flip-10deg_VFA.nii.gz'}, ...
%       'flipAngles', [2 5 10], 'trMs', 8.012, ...
%       'outputPath', '.../results_matlab/T1_map_t1_fa_fit.nii');

thisFile = mfilename('fullpath');
testsMatlabDir = fileparts(thisFile);
helpersDir = fullfile(testsMatlabDir, 'helpers');
if exist(helpersDir, 'dir')
    addpath(helpersDir);
end
if exist('add_rocketship_paths', 'file')
    add_rocketship_paths();
else
    repoRoot = fileparts(fileparts(testsMatlabDir));
    addpath(repoRoot);
    addpath(fullfile(repoRoot, 'external_programs'));
    addpath(fullfile(repoRoot, 'external_programs', 'niftitools'));
    addpath(fullfile(repoRoot, 'parametric_scripts'));
    addpath(fullfile(repoRoot, 'parametric_scripts', 'custom_scripts'));
end

p = inputParser;
addParameter(p, 'vfaFiles', {}, @iscell);
addParameter(p, 'flipAngles', [], @isnumeric);
addParameter(p, 'trMs', 8.012, @isscalar);
addParameter(p, 'fitType', 't1_fa_fit', @(v) ischar(v) || (isstring(v) && isscalar(v)));
addParameter(p, 'outputPath', '', @(v) ischar(v) || (isstring(v) && isscalar(v)));
addParameter(p, 'rsquaredThreshold', 0, @isscalar);
parse(p, varargin{:});

vfaFiles = p.Results.vfaFiles;
flipAngles = p.Results.flipAngles(:);
trMs = p.Results.trMs;
fitType = char(p.Results.fitType);
outputPath = char(p.Results.outputPath);
rsquaredThreshold = p.Results.rsquaredThreshold;

if numel(vfaFiles) < 2
    error('generate_t1_parity_map: need at least two VFA files.');
end
if numel(flipAngles) ~= numel(vfaFiles)
    error('generate_t1_parity_map: flipAngles count (%d) must match vfaFiles count (%d).', ...
        numel(flipAngles), numel(vfaFiles));
end
if isempty(outputPath)
    error('generate_t1_parity_map: outputPath is required.');
end

% Load + stack VFA volumes into [X, Y, Z, N].
nFlips = numel(vfaFiles);
stack = [];
for i = 1:nFlips
    if ~exist(vfaFiles{i}, 'file')
        error('generate_t1_parity_map: missing VFA file: %s', vfaFiles{i});
    end
    nii = load_untouch_nii(vfaFiles{i});
    img = double(nii.img);
    if isempty(stack)
        [nx, ny, nz] = size(img);
        stack = zeros(nx, ny, nz, nFlips);
    end
    stack(:, :, :, i) = img;
end

% Loop the single-voxel fitParameter over the map.
T1 = nan(nx, ny, nz);
for x = 1:nx
    for y = 1:ny
        for z = 1:nz
            si = squeeze(stack(x, y, z, :));
            if ~all(isfinite(si)) || all(si == 0)
                continue;
            end
            try
                fitOut = fitParameter(flipAngles, fitType, si, trMs, '', 0, '', 0, rsquaredThreshold);
                T1(x, y, z) = fitOut(1);
            catch
                T1(x, y, z) = NaN;
            end
        end
    end
end

outDir = fileparts(outputPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
niiOut = make_nii(T1);
save_nii(niiOut, outputPath);

output = struct();
output.outputPath = outputPath;
output.fitType = fitType;
output.size = [nx, ny, nz];
output.finiteVoxels = sum(isfinite(T1(:)));
fprintf('MATLAB T1 parity map written: %s (%d finite voxels)\n', outputPath, output.finiteVoxels);
end
