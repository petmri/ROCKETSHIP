classdef TestPreferenceParsing < matlab.unittest.TestCase
    methods (Test)
        function parsesDelimiterAndBlankValues(testCase)
            tmpFile = [tempname, '.txt'];
            cleaner = onCleanup(@() delete_if_exists(tmpFile));

            fid = fopen(tmpFile, 'w');
            testCase.assertGreaterThan(fid, 0);
            cleanupFid = onCleanup(@() fclose(fid));
            fprintf(fid, 'alpha = 1\n');
            fprintf(fid, 'beta =\n');
            fprintf(fid, '%% comment line\n');
            fprintf(fid, 'gamma = value with spaces\n');
            clear cleanupFid

            prefs = parse_preference_file(tmpFile, 0, {'alpha', 'beta', 'gamma'}, {'', '', ''});

            testCase.verifyEqual(prefs.alpha, '1');
            testCase.verifyEqual(prefs.beta, '');
            testCase.verifyEqual(prefs.gamma, 'value with spaces');
            clear cleaner
            delete_if_exists(tmpFile);
        end
    end
end

function delete_if_exists(path)
if exist(path, 'file') == 2
    delete(path);
end
end
