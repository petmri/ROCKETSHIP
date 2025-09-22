%% AIFbiexpconvolve Cp

%% This models the AIF as a biexponential function convolved with a
%% rectangular function to model the injection

% function Cp = AIFbiexpcon(x, xdata)
function Cp = AIFbiexpcon(A, B, c, d, t_base_end, t0_exp, T1, fittingAU, baseline)
% A = x(1);
% B = x(2);
% c = x(3);
% d = x(4);
% T1 = xdata{1}.timer;
% step=xdata{1}.step;
% maxer=xdata{1}.maxer;

T1 = T1(:);
% OUT = find(step > 0);

%Set as model A, MacGrath, MRM 2009
%B = maxer-A;

% if exist('xdata', 'var') && isfield(xdata{1}, 'raw') && xdata{1}.raw == true && isfield(xdata{1}, 'baseline')
if ~fittingAU
%     baseline = baseline;
% else
    baseline = 0;
end

Cp = zeros(size(T1));
for j = 1:numel(T1)
    % Baseline
    if T1(j) < t_base_end
        Cp(j) = baseline;
    % Linear upslope from t_base_end to t0_exp
    elseif T1(j) < t0_exp
        Cp(j) = baseline + (A-baseline) * ((T1(j)-t_base_end)/(t0_exp-t_base_end)) + (B-baseline) * ((T1(j)-t_base_end)/(t0_exp-t_base_end));
    % Bi-Exponential Decay    
    else
        Cp(j) = A * exp(-c * (T1(j) - t0_exp)) + B * exp(-d * (T1(j) - t0_exp));
    end
end
% Ensure output is the same shape and size as T1
Cp = reshape(Cp, size(T1));
