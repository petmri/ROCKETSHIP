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

        function testVpForwardMatchesScaledCp(testCase)
            fixture = make_synthetic_dce_fixture();
            Ct = model_vp_cfit(fixture.vp, fixture.Cp, fixture.timer);

            testCase.verifyEqual(Ct, fixture.vp .* fixture.Cp, 'AbsTol', 1e-12);
            assert_all_finite(testCase, Ct, 'VP forward Ct');
        end

        function testTissueUptakeForwardFinite(testCase)
            fixture = make_synthetic_dce_fixture();
            Ct = model_tissue_uptake_cfit(fixture.ktrans, fixture.fp, fixture.tp, fixture.Cp, fixture.timer);

            testCase.verifyEqual(size(Ct), size(fixture.timer));
            testCase.verifyEqual(Ct(1), 0, 'AbsTol', 1e-12);
            assert_all_finite(testCase, Ct, 'Tissue uptake Ct');
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

        function testVpFitRecoversSyntheticParameters(testCase)
            fixture = make_synthetic_dce_fixture();
            prefs = default_dce_fit_prefs();

            [fitResult, residuals] = model_vp(fixture.Ct_vp, fixture.Cp', fixture.timer, prefs);

            testCase.verifyLessThan(abs(fitResult(1) - fixture.vp), 1e-3);
            testCase.verifyGreaterThanOrEqual(fitResult(2), 0);
            assert_all_finite(testCase, residuals, 'VP residuals');
        end

        function testTissueUptakeFitRecoversSyntheticParameters(testCase)
            fixture = make_synthetic_dce_fixture();
            prefs = default_dce_fit_prefs();

            [fitResult, residuals] = model_tissue_uptake(fixture.Ct_tissue_uptake, fixture.Cp', fixture.timer, prefs);

            testCase.verifyLessThan(abs(fitResult(1) - fixture.ktrans), 1e-2);
            testCase.verifyLessThan(abs(fitResult(2) - fixture.fp), 5e-2);
            testCase.verifyLessThan(abs(fitResult(3) - fixture.vp), 5e-2);
            testCase.verifyGreaterThanOrEqual(fitResult(4), 0);
            assert_all_finite(testCase, residuals, 'Tissue uptake residuals');
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

        function test2CxmFitRecoversSyntheticParameters(testCase)
            fixture = make_synthetic_dce_fixture();
            prefs = default_dce_fit_prefs();

            [fitResult, residuals] = model_2cxm(fixture.Ct_2cxm, fixture.Cp', fixture.timer, prefs);

            testCase.verifyLessThan(abs(fitResult(1) - fixture.ktrans), 2e-2);
            testCase.verifyLessThan(abs(fitResult(2) - fixture.ve), 1e-1);
            testCase.verifyLessThan(abs(fitResult(3) - fixture.vp), 2e-2);
            testCase.verifyLessThan(abs(fitResult(4) - fixture.fp), 2e-1);
            testCase.verifyGreaterThanOrEqual(fitResult(5), 0);
            assert_all_finite(testCase, residuals, '2CXM residuals');
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

        function testFxrFitRecoversSyntheticParameters(testCase)
            fixture = make_synthetic_dce_fixture();
            prefs = default_dce_fit_prefs();

            [fitResult, residuals] = model_fxr(fixture.R1t_fxr, fixture.Cp', fixture.timer, ...
                fixture.R1o, fixture.R1i, fixture.r1, fixture.fw, prefs);

            testCase.verifyLessThan(abs(fitResult(1) - fixture.ktrans), 2e-2);
            testCase.verifyLessThan(abs(fitResult(2) - fixture.ve), 2e-1);
            testCase.verifyLessThan(abs(fitResult(3) - fixture.tau), 2e-2);
            testCase.verifyGreaterThanOrEqual(fitResult(4), 0);
            assert_all_finite(testCase, residuals, 'FXR residuals');
        end
    end
end
