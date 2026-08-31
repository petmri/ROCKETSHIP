function repoRoot = add_rocketship_paths(repoRoot)
% add_rocketship_paths Add project paths required by algorithm tests.

if nargin < 1 || isempty(repoRoot)
    helperPath = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(fileparts(helperPath))));
end

pathsToAdd = {
    repoRoot
    fullfile(repoRoot, 'dce')
    fullfile(repoRoot, 'dsc')
    fullfile(repoRoot, 'external_programs')
    fullfile(repoRoot, 'external_programs', 'niftitools')
    fullfile(repoRoot, 'parametric_scripts')
    fullfile(repoRoot, 'parametric_scripts', 'custom_scripts')
    fullfile(repoRoot, 'tests', 'matlab')
    fullfile(repoRoot, 'tests', 'matlab', 'helpers')
};

for i = 1:numel(pathsToAdd)
    if exist(pathsToAdd{i}, 'dir')
        addpath(pathsToAdd{i});
    end
end
end
