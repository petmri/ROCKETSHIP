classdef TestDscCoreAlgorithms < matlab.unittest.TestCase

    methods (TestClassSetup)
        function setupPaths(~)
            add_rocketship_paths();
        end
    end

    methods (Test)
        function testImportAifTruncatesToShortestLength(testCase)
            meanAIF = linspace(0.1, 1.2, 12)';
            bolusTime = 3;
            timeVect = (0:0.1:1.6)';
            concentration = reshape(linspace(0.01, 0.5, 2 * 3 * numel(timeVect)), [2, 3, numel(timeVect)]);

            [adjustedAIF, adjustedTime, adjustedConc, meanSignal] = import_AIF(meanAIF, bolusTime, timeVect, concentration, 3.4, 0.03);

            expectedLength = numel(meanAIF) - bolusTime + 1;
            testCase.verifyEqual(numel(adjustedAIF), expectedLength);
            testCase.verifyEqual(numel(adjustedTime), expectedLength);
            testCase.verifyEqual(size(adjustedConc, 3), expectedLength);
            testCase.verifyEqual(numel(meanSignal), expectedLength);
            assert_all_finite(testCase, adjustedAIF, 'adjustedAIF');
            assert_all_finite(testCase, meanSignal, 'meanSignal');
        end

        function testPreviousAifTruncatesTimeAndConcentration(testCase)
            meanAIF = linspace(0.1, 1.0, 10)';
            timeVect = (0:0.1:1.4)';
            concentration = reshape(linspace(0.02, 0.8, 2 * 2 * numel(timeVect)), [2, 2, numel(timeVect)]);

            [adjustedAIF, adjustedTime, adjustedConc] = previous_AIF(meanAIF, meanAIF, 0, timeVect, concentration);

            testCase.verifyEqual(numel(adjustedAIF), numel(meanAIF));
            testCase.verifyEqual(numel(adjustedTime), numel(meanAIF));
            testCase.verifyEqual(size(adjustedConc, 3), numel(meanAIF));
            assert_all_finite(testCase, adjustedAIF, 'adjustedAIF');
        end

        function testSignalToConcentrationOnSyntheticScan(testCase)
            dimx = 8;
            dimy = 8;
            dimt = 20;

            imageArray = 120 * ones(dimx, dimy, dimt);
            imageArray(end-4:end, end-4:end, :) = 10;
            for t = 8:dimt
                imageArray(1:end-5, 1:end-5, t) = 60;
            end

            outDir = tempname;
            mkdir(outDir);

            [concentrationArray, baseConcentrationArray, timeVect, baseTimeVect, wholeTimeVect, bolusIndex, ...
                baseSignalArray, ~, backgroundSignal] = DSC_signal2concentration(imageArray, 0.03, 1.0, 3.4, ...
                [outDir filesep], 0, [], 0);

            testCase.verifyGreaterThan(bolusIndex, 1);
            testCase.verifyEqual(size(concentrationArray, 3), numel(timeVect));
            testCase.verifyEqual(size(baseConcentrationArray, 3), numel(baseTimeVect));
            testCase.verifyEqual(numel(wholeTimeVect), dimt);
            testCase.verifyGreaterThan(backgroundSignal, 0);
            assert_all_finite(testCase, concentrationArray, 'concentrationArray');
            assert_all_finite(testCase, baseSignalArray, 'baseSignalArray');
            testCase.verifyTrue(isfile(fullfile(outDir, 'conc_full_scan.nii')));
        end

        function testSSvdDeconvolutionProducesFiniteMaps(testCase)
            conc = zeros(2, 2, 10);
            timeIndex = (0:9);
            for i = 1:2
                for j = 1:2
                    conc(i, j, :) = exp(-((timeIndex - (2 + i + j/2)).^2) / 6);
                end
            end
            AIF = exp(-((timeIndex - 2).^2) / 4)';

            outDir = tempname;
            mkdir(outDir);

            [CBF, CBV, MTT] = DSC_convolution_sSVD(conc, AIF, 0.1, 0.73, 1.04, 20, 1, [outDir filesep]);

            testCase.verifyEqual(size(CBF), [2 2]);
            testCase.verifyEqual(size(CBV), [2 2]);
            testCase.verifyEqual(size(MTT), [2 2]);
            assert_all_finite(testCase, CBF, 'CBF_sSVD');
            assert_all_finite(testCase, CBV, 'CBV_sSVD');
            assert_all_finite(testCase, MTT, 'MTT_sSVD');
        end

        function testOSvdDeconvolutionProducesFiniteMaps(testCase)
            conc = zeros(2, 2, 10);
            timeIndex = (0:9);
            for i = 1:2
                for j = 1:2
                    conc(i, j, :) = exp(-((timeIndex - (1 + i + j/2)).^2) / 6);
                end
            end
            AIF = exp(-((timeIndex - 2).^2) / 4)';

            outDir = tempname;
            mkdir(outDir);

            [CBF, CBV, MTT] = DSC_convolution_oSVD(conc, AIF, 0.1, 0.73, 1.04, 0.8, 1, [outDir filesep]);

            testCase.verifyEqual(size(CBF), [2 2]);
            testCase.verifyEqual(size(CBV), [2 2]);
            testCase.verifyEqual(size(MTT), [2 2]);
            assert_all_finite(testCase, CBF, 'CBF_oSVD');
            assert_all_finite(testCase, CBV, 'CBV_oSVD');
            assert_all_finite(testCase, MTT, 'MTT_oSVD');
        end
    end
end
