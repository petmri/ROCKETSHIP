function manifest = generate_synthetic_datasets(varargin)
% generate_synthetic_datasets Build deterministic synthetic fixtures.
%
% This function creates BIDS-like synthetic datasets by modifying the
% existing test fixture at tests/data/BIDS_example:
% 1) noisy_low      - low Gaussian noise on DCE + VFA images
% 2) noisy_high     - higher Gaussian noise on DCE + VFA images
% 3) downsample_x2  - 2x spatial downsample of DCE/anat derivatives + VFA
% 4) bolus_delay    - delayed/attenuated DCE bolus dynamics
%
% Example:
%   manifest = generate_synthetic_datasets();
%   manifest = generate_synthetic_datasets('outputRoot', fullfile(tempdir, 'rocketship_synth'));

repoRoot = add_rocketship_paths();

p = inputParser;
addParameter(p, 'sourceRoot', fullfile(repoRoot, 'tests/data', 'BIDS_example'), @ischar);
addParameter(p, 'outputRoot', fullfile(repoRoot, 'tests/data', 'synthetic', 'generated'), @ischar);
addParameter(p, 'seed', 1729, @isscalar);
addParameter(p, 'clean', true, @islogical);
parse(p, varargin{:});

sourceRoot = p.Results.sourceRoot;
outputRoot = p.Results.outputRoot;
seed = p.Results.seed;
clean = p.Results.clean;

if ~exist(sourceRoot, 'dir')
    error('Source root does not exist: %s', sourceRoot);
end

if clean && exist(outputRoot, 'dir')
    rmdir(outputRoot, 's');
end
if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

rng(seed, 'twister');

relativeFiles = {
    fullfile('rawdata', 'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_DCE.json')
    fullfile('rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-01_VFA.nii.gz')
    fullfile('rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-01_VFA.json')
    fullfile('rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-02_VFA.nii.gz')
    fullfile('rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-02_VFA.json')
    fullfile('rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-03_VFA.nii.gz')
    fullfile('rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-03_VFA.json')
    fullfile('derivatives', 'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii.gz')
    fullfile('derivatives', 'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-AIF_T1map.nii.gz')
    fullfile('derivatives', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_space-DCEref_T1map.nii')
    fullfile('derivatives', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_space-DCEref_desc-bfczunified_VFA.nii')
    fullfile('derivatives', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_space-DCEref_desc-brain_mask.nii.gz')
};

variants = {
    struct('name', 'noisy_low', 'type', 'noise', 'noiseFrac', 0.015)
    struct('name', 'noisy_high', 'type', 'noise', 'noiseFrac', 0.050)
    struct('name', 'downsample_x2', 'type', 'downsample', 'factor', 2)
    struct('name', 'bolus_delay', 'type', 'bolus', 'delayFrames', 2, 'attenuation', 0.90)
};

manifest = struct();
manifest.meta = struct();
manifest.meta.sourceRoot = sourceRoot;
manifest.meta.outputRoot = outputRoot;
manifest.meta.seed = seed;
manifest.meta.generated_utc = datestr(now, 30);
manifest.variants = cell(numel(variants), 1);

for i = 1:numel(variants)
    variant = variants{i};
    variantRoot = fullfile(outputRoot, variant.name);
    copy_variant_skeleton(sourceRoot, variantRoot, relativeFiles);

    switch variant.type
        case 'noise'
            apply_noise_variant(variantRoot, variant.noiseFrac);
        case 'downsample'
            apply_downsample_variant(variantRoot, variant.factor);
        case 'bolus'
            apply_bolus_variant(variantRoot, variant.delayFrames, variant.attenuation);
        otherwise
            error('Unknown variant type: %s', variant.type);
    end

    write_variant_note(variantRoot, variant);

    manifest.variants{i} = struct( ...
        'name', variant.name, ...
        'root', variantRoot, ...
        'type', variant.type ...
    );
end

manifestPath = fullfile(outputRoot, 'manifest.json');
fid = fopen(manifestPath, 'w');
if fid == -1
    error('Unable to write synthetic manifest: %s', manifestPath);
end
fwrite(fid, jsonencode(manifest), 'char');
fclose(fid);

fprintf('Synthetic datasets written to:\n%s\n', outputRoot);
for i = 1:numel(variants)
    fprintf('  - %s\n', fullfile(outputRoot, variants{i}.name));
end
end

function copy_variant_skeleton(sourceRoot, variantRoot, relativeFiles)
for i = 1:numel(relativeFiles)
    src = fullfile(sourceRoot, relativeFiles{i});
    dst = fullfile(variantRoot, relativeFiles{i});
    dstDir = fileparts(dst);
    if ~exist(dstDir, 'dir')
        mkdir(dstDir);
    end
    if ~exist(src, 'file')
        error('Missing required source fixture file: %s', src);
    end
    copyfile(src, dst);
end
end

function apply_noise_variant(variantRoot, noiseFrac)
dcePath = fullfile(variantRoot, 'derivatives', 'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii.gz');
vfaPaths = {
    fullfile(variantRoot, 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-01_VFA.nii.gz')
    fullfile(variantRoot, 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-02_VFA.nii.gz')
    fullfile(variantRoot, 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-03_VFA.nii.gz')
};

apply_noise_to_nifti(dcePath, noiseFrac);
for i = 1:numel(vfaPaths)
    apply_noise_to_nifti(vfaPaths{i}, noiseFrac);
end
end

function apply_downsample_variant(variantRoot, factor)
imageItems = {
    struct('path', fullfile(variantRoot, 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-01_VFA.nii.gz'), 'hasTime', false)
    struct('path', fullfile(variantRoot, 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-02_VFA.nii.gz'), 'hasTime', false)
    struct('path', fullfile(variantRoot, 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-03_VFA.nii.gz'), 'hasTime', false)
    struct('path', fullfile(variantRoot, 'derivatives', 'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii.gz'), 'hasTime', true)
    struct('path', fullfile(variantRoot, 'derivatives', 'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-AIF_T1map.nii.gz'), 'hasTime', false)
    struct('path', fullfile(variantRoot, 'derivatives', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_space-DCEref_T1map.nii'), 'hasTime', false)
    struct('path', fullfile(variantRoot, 'derivatives', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_space-DCEref_desc-bfczunified_VFA.nii'), 'hasTime', false)
};

maskPath = fullfile(variantRoot, 'derivatives', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_space-DCEref_desc-brain_mask.nii.gz');

for i = 1:numel(imageItems)
    downsample_nifti(imageItems{i}.path, factor, false, imageItems{i}.hasTime);
end
downsample_nifti(maskPath, factor, true, false);
end

function apply_bolus_variant(variantRoot, delayFrames, attenuation)
dcePath = fullfile(variantRoot, 'derivatives', 'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii.gz');

nii = load_untouch_nii(dcePath);
img = double(nii.img);
if ndims(img) ~= 4
    error('Expected 4D DCE image for bolus delay variant: %s', dcePath);
end

shifted = img;
if delayFrames > 0
    shifted(:, :, :, (delayFrames+1):end) = img(:, :, :, 1:(end-delayFrames));
    shifted(:, :, :, 1:delayFrames) = repmat(img(:, :, :, 1), 1, 1, 1, delayFrames);
end

% Mild attenuation to emulate lower dose / reduced enhancement.
shifted = shifted * attenuation;

overwrite_nifti(dcePath, nii, shifted, false);
end

function apply_noise_to_nifti(pathIn, noiseFrac)
nii = load_untouch_nii(pathIn);
img = double(nii.img);

sigma = noiseFrac * max(std(img(:)), 1);
noisy = img + sigma * randn(size(img));
noisy = max(noisy, 0);

overwrite_nifti(pathIn, nii, noisy, false);
end

function downsample_nifti(pathIn, factor, isMask, hasTime)
if nargin < 4
    hasTime = false;
end

nii = load_untouch_nii(pathIn);
img = double(nii.img);

down = downsample_spatial(img, factor, hasTime);
if isMask
    down = down > 0;
end

overwrite_nifti(pathIn, nii, down, isMask);
end

function out = downsample_spatial(img, factor, hasTime)
d = ndims(img);
if d == 4
    if hasTime
        out = img(1:factor:end, 1:factor:end, 1:factor:end, :);
    else
        out = img(1:factor:end, 1:factor:end, 1:factor:end, 1:factor:end);
    end
elseif d == 3
    if hasTime
        out = img(1:factor:end, 1:factor:end, :);
    else
        out = img(1:factor:end, 1:factor:end, 1:factor:end);
    end
elseif d == 2
    out = img(1:factor:end, 1:factor:end);
else
    error('Unsupported dimensionality for downsample: %d', d);
end
end

function overwrite_nifti(originalPath, niiTemplate, newData, isMask)
if nargin < 4
    isMask = false;
end

savePath = originalPath;
if endsWith(originalPath, '.nii.gz')
    savePath = originalPath(1:end-3); % write as .nii
end

if isMask
    newData = uint8(newData > 0);
else
    newData = cast_like(newData, niiTemplate.img);
end

niiTemplate = update_header_dimensions(niiTemplate, newData);
niiTemplate.img = newData;
save_untouch_nii(niiTemplate, savePath);

if ~strcmp(savePath, originalPath) && exist(originalPath, 'file')
    delete(originalPath);
end
end

function y = cast_like(x, template)
targetClass = class(template);
switch targetClass
    case {'uint8', 'uint16', 'uint32'}
        maxVal = double(intmax(targetClass));
        y = cast(min(max(x, 0), maxVal), targetClass);
    case {'int8', 'int16', 'int32'}
        minVal = double(intmin(targetClass));
        maxVal = double(intmax(targetClass));
        y = cast(min(max(x, minVal), maxVal), targetClass);
    case {'single', 'double'}
        y = cast(x, targetClass);
    otherwise
        y = cast(x, 'single');
end
end

function nii = update_header_dimensions(nii, img)
imgSize = size(img);
numDims = ndims(img);

dim = ones(1, 8);
dim(1) = numDims;
dim(2:(1+numDims)) = imgSize;
nii.hdr.dime.dim = dim;

if isfield(nii.hdr.dime, 'glmax')
    nii.hdr.dime.glmax = max(double(img(:)));
end
if isfield(nii.hdr.dime, 'glmin')
    nii.hdr.dime.glmin = min(double(img(:)));
end
end

function write_variant_note(variantRoot, variant)
notePath = fullfile(variantRoot, 'SYNTHETIC_NOTES.json');
fid = fopen(notePath, 'w');
if fid == -1
    error('Unable to write variant note: %s', notePath);
end
fwrite(fid, jsonencode(variant), 'char');
fclose(fid);
end
