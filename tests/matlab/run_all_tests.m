function results = run_all_tests(varargin)
% run_all_tests Run ROCKETSHIP algorithm test suites.
%
% Parameters:
%   'suite'              - 'unit' | 'integration' | 'all' (default: 'all')
%   'includeIntegration' - include integration suite when suite='all' (default: false)

p = inputParser;
addParameter(p, 'suite', 'all', @(x) ischar(x) || isstring(x));
addParameter(p, 'includeIntegration', false, @islogical);
parse(p, varargin{:});

suiteName = lower(char(p.Results.suite));

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
add_rocketship_paths(repoRoot);

import matlab.unittest.TestSuite

unitDir = fullfile(repoRoot, 'tests', 'matlab', 'unit');
integrationDir = fullfile(repoRoot, 'tests', 'matlab', 'integration');

switch suiteName
    case 'unit'
        suite = TestSuite.fromFolder(unitDir, 'IncludingSubfolders', true);
    case 'integration'
        suite = TestSuite.fromFolder(integrationDir, 'IncludingSubfolders', true);
    case 'all'
        suite = TestSuite.fromFolder(unitDir, 'IncludingSubfolders', true);
        if p.Results.includeIntegration
            suite = [suite, TestSuite.fromFolder(integrationDir, 'IncludingSubfolders', true)]; %#ok<AGROW>
        end
    otherwise
        error('Unknown suite "%s". Use unit, integration, or all.', suiteName);
end

results = run(suite);

if nargout == 0 && any([results.Failed])
    error('One or more tests failed.');
end
end
