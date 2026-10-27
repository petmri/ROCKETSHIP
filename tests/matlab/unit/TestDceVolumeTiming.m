classdef TestDceVolumeTiming < matlab.unittest.TestCase
    % BIDS VolumeTiming -> part-B frame times (dce_volume_timer) and the start_t/end_t frame
    % window it shares with loadIMGVOL (dce_frame_window).

    methods (TestClassSetup)
        function setupPaths(~)
            add_rocketship_paths();
        end
    end

    methods (Test)
        function frameWindowKeepsLegacyLoadImgVolCropping(testCase)
            % The branches loadIMGVOL used before it called the helper, including its quirks.
            testCase.verifyEqual(dce_frame_window(10, [], []), 1:10);
            testCase.verifyEqual(dce_frame_window(10, 3, []), 3:10);
            testCase.verifyEqual(dce_frame_window(10, 1, []), 1:10);
            testCase.verifyEqual(dce_frame_window(10, 3, 8), 3:8);
            testCase.verifyEqual(dce_frame_window(10, [], 6), 1:6);
            testCase.verifyEqual(dce_frame_window(10, 0, 6), 1:6);
            testCase.verifyEqual(dce_frame_window(10, 7, 5), 1:5);
            testCase.verifyEqual(dce_frame_window(10, NaN, []), 1:10);
        end

        function volumeTimerCropsAndRebasesToMinutes(testCase)
            % Same case as test_dynamic_metadata_volume_timing_outranks_scalar_spacing in
            % tests/python/test_dce_pipeline.py: frames 2..4 of five uneven frames.
            [timer, step] = dce_volume_timer([0 30 90 120 180], 5, 2, 4);
            testCase.verifyEqual(timer, [0 1 1.5], 'AbsTol', 1e-12);
            testCase.verifyEqual(step * 60, 45, 'AbsTol', 1e-12);
        end

        function volumeTimerAcceptsColumnInputAndFullWindow(testCase)
            [timer, step] = dce_volume_timer([5; 15; 25; 45], 4, [], []);
            testCase.verifyEqual(size(timer), [1 4]);
            testCase.verifyEqual(timer * 60, [0 10 20 40], 'AbsTol', 1e-12);
            testCase.verifyEqual(step * 60, 10, 'AbsTol', 1e-12);
        end

        function singleFrameWindowReportsWholeSeriesStep(testCase)
            [timer, step] = dce_volume_timer([0 10 20 40], 4, [], 1);
            testCase.verifyEqual(timer, 0);
            testCase.verifyEqual(step * 60, 10, 'AbsTol', 1e-12);
        end

        function rejectsMalformedVolumeTiming(testCase)
            testCase.verifyError(@() dce_volume_timer([0 10 20], 4, [], []), ...
                'dce_volume_timer:LengthMismatch');
            testCase.verifyError(@() dce_volume_timer([0 10 10 20], 4, [], []), ...
                'dce_volume_timer:NotIncreasing');
            testCase.verifyError(@() dce_volume_timer([0 10 NaN 20], 4, [], []), ...
                'dce_volume_timer:NotIncreasing');
            testCase.verifyError(@() dce_volume_timer([0 10 20 30], 4, 6, []), ...
                'dce_volume_timer:WindowOutOfRange');
        end
    end
end
