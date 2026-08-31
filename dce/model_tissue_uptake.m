function [x, residuals] = model_tissue_uptake(Ct,Cp,timer,prefs)

% Use Curvefitting tool box instead of optimization toolbox (lsqcurvefit)
% as curvefitting will easily return confidence intervals on the fit
% performance of the two appears to be the same
options = fitoptions('Method', 'NonlinearLeastSquares',...
    'Algorithm', 'Trust-Region',...
    'MaxIter', prefs.MaxIter,...
    'MaxFunEvals', prefs.MaxFunEvals,...
    'TolFun', prefs.TolFun,...
    'TolX', prefs.TolX,...
    'Display', 'off',...
    'Lower',[prefs.lower_limit_ktrans prefs.lower_limit_fp prefs.lower_limit_tp],...
    'Upper', [prefs.upper_limit_ktrans prefs.upper_limit_fp prefs.upper_limit_tp],...
    'StartPoint', [prefs.initial_value_ktrans prefs.initial_value_fp prefs.initial_value_tp],...
    'Robust', prefs.Robust);
ft = fittype('model_tissue_uptake_cfit( Ktrans, Fp, Tp, Cp, T1)',...
    'independent', {'T1', 'Cp'},...
    'coefficients',{'Ktrans', 'Fp', 'Tp'});

lower = [prefs.lower_limit_ktrans prefs.lower_limit_fp prefs.lower_limit_tp];
upper = [prefs.upper_limit_ktrans prefs.upper_limit_fp prefs.upper_limit_tp];
base_start = [prefs.initial_value_ktrans prefs.initial_value_fp prefs.initial_value_tp];

start_candidates = [
    base_start;
    [prefs.initial_value_ktrans*10 prefs.initial_value_fp prefs.initial_value_tp];
    [prefs.initial_value_ktrans*100 prefs.initial_value_fp prefs.initial_value_tp];
    [prefs.initial_value_ktrans max(prefs.initial_value_fp*2, prefs.initial_value_ktrans*1.5) max(prefs.initial_value_tp*0.5, prefs.lower_limit_tp)];
];

best_sse = inf;
best_f = [];
best_gof = [];
best_output = [];

for i = 1:size(start_candidates,1)
    sp = start_candidates(i,:);
    sp = max(sp, lower);
    sp = min(sp, upper);
    % keep away from singular fp ~= ktrans region when possible
    if (sp(2) - sp(1)) < 1e-4
        sp(2) = min(upper(2), max(lower(2), sp(1) + 1e-4));
    end
    try
        cur_options = fitoptions(options, 'StartPoint', sp);
        [cand_f, cand_gof, cand_output] = fit([timer, Cp'],Ct,ft, cur_options);
        if cand_gof.sse < best_sse
            best_sse = cand_gof.sse;
            best_f = cand_f;
            best_gof = cand_gof;
            best_output = cand_output;
        end
    catch
        % Ignore failed starts and continue.
    end
end

if isempty(best_f)
    [best_f, best_gof, best_output] = fit([timer, Cp'],Ct,ft, options);
end

f = best_f;
gof = best_gof;
output = best_output;
try
    confidence_interval = confint(f,0.95);
catch
    confidence_interval = repmat([f.Ktrans f.Fp f.Tp], [2 1]);
end

% Calulated wanted parameters
if abs(f.Fp - f.Ktrans) < 1e-12
    PS = 1e8;
else
    PS = f.Ktrans/(1-f.Ktrans/f.Fp);
end
vp = (f.Fp+PS)*f.Tp;
vp_low_ci = (f.Fp+PS)*confidence_interval(1,3);
vp_high_ci = (f.Fp+PS)*confidence_interval(2,3);

%Calculate the R2 fit
x(1) = f.Ktrans;			% ktrans
x(2) = f.Fp;				% Fp
x(3) = vp;  				% vp
x(4) = gof.sse;				% SSE
x(5) = confidence_interval(1,1);% (95 lower CI of ktrans)
x(6) = confidence_interval(2,1);% (95 upper CI of ktrans)
x(7) = confidence_interval(1,2);% (95 lower CI of Fp)
x(8) = confidence_interval(2,2);% (95 upper CI of Fp)
x(9) = vp_low_ci;% (95 lower CI of vp)
x(10) = vp_high_ci;% (95 upper CI of vp)

residuals = output.residuals;