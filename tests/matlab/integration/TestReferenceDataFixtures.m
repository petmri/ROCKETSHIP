classdef TestReferenceDataFixtures < matlab.unittest.TestCase

    properties
        RepoRoot
    end

    methods (TestClassSetup)
        function setupPaths(testCase)
            testCase.RepoRoot = add_rocketship_paths();
        end
    end

    methods (Test)
        function testBbbFixtureFilesExist(testCase)
            requiredFiles = {
                fullfile(testCase.RepoRoot, 'tests/data', 'BBB data p19', 'Dynamic_t1w.nii')
                fullfile(testCase.RepoRoot, 'tests/data', 'BBB data p19', 'fa2.nii')
                fullfile(testCase.RepoRoot, 'tests/data', 'BBB data p19', 'fa5.nii')
                fullfile(testCase.RepoRoot, 'tests/data', 'BBB data p19', 'fa10.nii')
                fullfile(testCase.RepoRoot, 'tests/data', 'BBB data p19', 'processed', 'T1_map_t1_fa_fit_fa10.nii')
                fullfile(testCase.RepoRoot, 'tests/data', 'BBB data p19', 'processed', 'results', 'Dyn-1_patlak_fit_Ktrans.nii')
            };

            for i = 1:numel(requiredFiles)
                testCase.verifyTrue(isfile(requiredFiles{i}), sprintf('Missing fixture file: %s', requiredFiles{i}));
            end
        end

        function testBidsFixtureFilesExist(testCase)
            requiredFiles = {
                fullfile(testCase.RepoRoot, 'tests/data', 'BIDS_example', 'rawdata', 'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_DCE.json')
                fullfile(testCase.RepoRoot, 'tests/data', 'BIDS_example', 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-01_VFA.nii.gz')
                fullfile(testCase.RepoRoot, 'tests/data', 'BIDS_example', 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-02_VFA.nii.gz')
                fullfile(testCase.RepoRoot, 'tests/data', 'BIDS_example', 'rawdata', 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_flip-03_VFA.nii.gz')
            };

            for i = 1:numel(requiredFiles)
                testCase.verifyTrue(isfile(requiredFiles{i}), sprintf('Missing fixture file: %s', requiredFiles{i}));
            end
        end

        function testReferenceMapMatchesDynamicSpatialSize(testCase)
            dynamicPath = fullfile(testCase.RepoRoot, 'tests/data', 'BBB data p19', 'Dynamic_t1w.nii');
            ktransPath = fullfile(testCase.RepoRoot, 'tests/data', 'BBB data p19', 'processed', 'results', 'Dyn-1_patlak_fit_Ktrans.nii');

            dynamicNii = load_untouch_nii(dynamicPath);
            ktransNii = load_untouch_nii(ktransPath);

            dynamicSize = size(dynamicNii.img);
            ktransSize = size(ktransNii.img);

            testCase.verifyEqual(ktransSize(1), dynamicSize(1));
            testCase.verifyEqual(ktransSize(2), dynamicSize(2));

            if numel(dynamicSize) >= 3 && numel(ktransSize) >= 3
                testCase.verifyEqual(ktransSize(3), dynamicSize(3));
            end
        end
    end
end
