classdef TestDceCoreModels < matlab.unittest.TestCase

    methods (TestClassSetup)
        function setupPaths(~)
            add_rocketship_paths();
        end
    end

    methods (Test)
        function testToftsForwardZeroInput(testCase)
            timer = (0:0.2:4)';
            Cp = zeros(size(timer));

            Ct = model_tofts_cfit(0.02, 0.3, Cp, timer);
            testCase.verifyEqual(Ct, zeros(size(timer)), 'AbsTol', 1e-12);
        end

        function testPatlakLinearRecoversSyntheticParameters(testCase)
            fixture = make_synthetic_dce_fixture();
            result = model_patlak_linear(fixture.Ct_patlak, fixture.Cp', fixture.timer);

            testCase.verifyLessThan(abs(result(1) - fixture.ktrans), 1e-3);
            testCase.verifyLessThan(abs(result(2) - fixture.vp), 1e-3);
            testCase.verifyGreaterThanOrEqual(result(3), 0);
        end

        function testToftsFitRecoversSyntheticParameters(testCase)
            fixture = make_synthetic_dce_fixture();
            prefs = default_dce_fit_prefs();

            [fitResult, residuals] = model_tofts(fixture.Ct_tofts, fixture.Cp', fixture.timer, prefs);

            testCase.verifyLessThan(abs(fitResult(1) - fixture.ktrans), 5e-3);
            testCase.verifyLessThan(abs(fitResult(2) - fixture.ve), 5e-2);
            testCase.verifyGreaterThanOrEqual(fitResult(3), 0);
            assert_all_finite(testCase, residuals, 'Tofts residuals');
        end

        function testExtendedToftsFitRecoversSyntheticParameters(testCase)
            fixture = make_synthetic_dce_fixture();
            prefs = default_dce_fit_prefs();

            [fitResult, residuals] = model_extended_tofts(fixture.Ct_extended_tofts, fixture.Cp', fixture.timer, prefs);

            testCase.verifyLessThan(abs(fitResult(1) - fixture.ktrans), 7e-3);
            testCase.verifyLessThan(abs(fitResult(2) - fixture.ve), 8e-2);
            testCase.verifyLessThan(abs(fitResult(3) - fixture.vp), 1e-2);
            testCase.verifyGreaterThanOrEqual(fitResult(4), 0);
            assert_all_finite(testCase, residuals, 'Extended Tofts residuals');
        end

        function test2CxmForwardFiniteWhenKtransGteFp(testCase)
            timer = (0:0.1:5)';
            Cp = exp(-timer / 1.5) + 0.01;

            Ct = model_2cxm_cfit(0.2, 0.3, 0.05, 0.1, Cp, timer);
            assert_all_finite(testCase, Ct, '2CXM Ct');
            testCase.verifyEqual(size(Ct), size(timer));
        end

        function testFxrForwardIsFinite(testCase)
            fixture = make_synthetic_dce_fixture();

            R1t = model_fxr_cfit(fixture.ktrans, fixture.ve, fixture.tau, fixture.Cp, fixture.timer, ...
                fixture.R1o, fixture.R1i, fixture.r1, fixture.fw);

            assert_all_finite(testCase, R1t, 'FXR R1t');
            testCase.verifyEqual(size(R1t), size(fixture.timer));
        end
    end
end
