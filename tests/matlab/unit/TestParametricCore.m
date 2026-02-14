classdef TestParametricCore < matlab.unittest.TestCase

    methods (TestClassSetup)
        function setupPaths(~)
            add_rocketship_paths();
        end
    end

    methods (Test)
        function testT2LinearFastRecoversKnownT2(testCase)
            te = [10; 20; 40; 60];
            trueT2 = 80;
            rho = 1000;
            si = rho .* exp(-te ./ trueT2);

            fitOutput = fitParameter(te, 't2_linear_fast', si, 0, '', 0, '', 0, 0);

            testCase.verifyLessThan(abs(fitOutput(1) - trueT2), 1e-6);
            testCase.verifyGreaterThan(fitOutput(3), 0.999);
            testCase.verifyGreaterThanOrEqual(fitOutput(6), 0);
        end

        function testT1FaLinearFitRecoversKnownT1(testCase)
            fa = [2; 5; 10; 15];
            tr = 8;
            trueT1 = 1200;
            m0 = 1200;
            theta = fa .* (pi / 180);
            si = m0 .* ((1 - exp(-tr ./ trueT1)) .* sin(theta)) ./ (1 - exp(-tr ./ trueT1) .* cos(theta));

            fitOutput = fitParameter(fa, 't1_fa_linear_fit', si, tr, '', 0, '', 0, 0);

            testCase.verifyLessThan(abs(fitOutput(1) - trueT1), 100);
            testCase.verifyGreaterThan(fitOutput(3), 0.99);
            testCase.verifyGreaterThanOrEqual(fitOutput(6), 0);
        end

        function testFitParameterAppliesRSquaredGate(testCase)
            te = [10; 20; 40; 60];
            si = ones(size(te));

            fitOutput = fitParameter(te, 't2_linear_simple', si, 0, '', 0, '', 0, 0.95);

            testCase.verifyEqual(fitOutput(1), -2);
            testCase.verifyEqual(fitOutput(2), -2);
            testCase.verifyEqual(fitOutput(6), -2);
        end
    end
end
