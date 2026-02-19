classdef TestSyntheticDatasetGeneration < matlab.unittest.TestCase

    properties
        RepoRoot
    end

    methods (TestClassSetup)
        function setupPaths(testCase)
            testCase.RepoRoot = add_rocketship_paths();
        end
    end

    methods (Test)
        function testGeneratorCreatesExpectedVariants(testCase)
            outRoot = fullfile(tempdir, 'rocketship_synth_tests');
            manifest = generate_synthetic_datasets('outputRoot', outRoot, 'clean', true, 'seed', 1001);

            testCase.verifyTrue(isfolder(outRoot));
            testCase.verifyEqual(numel(manifest.variants), 4);
            testCase.verifyTrue(isfile(fullfile(outRoot, 'manifest.json')));

            variantNames = cellfun(@(v) v.name, manifest.variants, 'UniformOutput', false);
            expectedNames = {'noisy_low', 'noisy_high', 'downsample_x2', 'bolus_delay'};
            testCase.verifyEqual(sort(variantNames(:)), sort(expectedNames(:)));
        end

        function testNoisyVariantPreservesDimensionsAndChangesValues(testCase)
            sourceDce = fullfile(testCase.RepoRoot, 'tests/data', 'BIDS_test', 'derivatives', ...
                'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii.gz');
            sourceNii = load_untouch_nii(sourceDce);

            outRoot = fullfile(tempdir, 'rocketship_synth_noise_test');
            generate_synthetic_datasets('outputRoot', outRoot, 'clean', true, 'seed', 1002);

            noisyDce = fullfile(outRoot, 'noisy_low', 'derivatives', 'sub-01', 'ses-01', ...
                'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii');
            noisyNii = load_untouch_nii(noisyDce);

            testCase.verifyEqual(size(noisyNii.img), size(sourceNii.img));
            diffCount = nnz(abs(double(noisyNii.img) - double(sourceNii.img)) > 0);
            testCase.verifyGreaterThan(diffCount, 0);
        end

        function testDownsampleVariantReducesSpatialDimensions(testCase)
            sourceDce = fullfile(testCase.RepoRoot, 'tests/data', 'BIDS_test', 'derivatives', ...
                'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii.gz');
            sourceNii = load_untouch_nii(sourceDce);
            sourceSize = size(sourceNii.img);

            outRoot = fullfile(tempdir, 'rocketship_synth_downsample_test');
            generate_synthetic_datasets('outputRoot', outRoot, 'clean', true, 'seed', 1003);

            downDce = fullfile(outRoot, 'downsample_x2', 'derivatives', 'sub-01', 'ses-01', ...
                'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii');
            downNii = load_untouch_nii(downDce);
            downSize = size(downNii.img);

            testCase.verifyEqual(downSize(4), sourceSize(4));
            testCase.verifyLessThan(downSize(1), sourceSize(1));
            testCase.verifyLessThan(downSize(2), sourceSize(2));
            testCase.verifyLessThanOrEqual(downSize(3), sourceSize(3));
        end

        function testBolusDelayVariantAltersDynamics(testCase)
            sourceDce = fullfile(testCase.RepoRoot, 'tests/data', 'BIDS_test', 'derivatives', ...
                'sub-01', 'ses-01', 'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii.gz');
            sourceNii = load_untouch_nii(sourceDce);
            sourceImg = double(sourceNii.img);

            outRoot = fullfile(tempdir, 'rocketship_synth_bolus_test');
            generate_synthetic_datasets('outputRoot', outRoot, 'clean', true, 'seed', 1004);

            bolusDce = fullfile(outRoot, 'bolus_delay', 'derivatives', 'sub-01', 'ses-01', ...
                'dce', 'sub-01_ses-01_desc-bfcz_DCE.nii');
            bolusNii = load_untouch_nii(bolusDce);
            bolusImg = double(bolusNii.img);

            testCase.verifyEqual(size(bolusImg), size(sourceImg));

            sourceTrace = squeeze(mean(mean(mean(sourceImg, 1), 2), 3));
            bolusTrace = squeeze(mean(mean(mean(bolusImg, 1), 2), 3));

            % Delay + attenuation should modify the global timecourse.
            testCase.verifyGreaterThan(norm(sourceTrace - bolusTrace), 0);
            testCase.verifyLessThan(max(bolusTrace), max(sourceTrace));
        end
    end
end
