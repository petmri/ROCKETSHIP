function assert_all_finite(testCase, values, label)
% assert_all_finite Assert no NaN/Inf in numeric outputs.

if nargin < 3
    label = 'values';
end

testCase.verifyTrue(all(isfinite(values(:))), sprintf('%s contains non-finite values.', label));
end
