% AIFbixexpfithelper is a wrapper function to fit the AIF to a biexponential model

%{

AIFbiexpcon.m is the file defining the function. You can of course alter
this as you wish

out is the fitted timecurve over the desired time interval
x is the parameters of the fit
xdata stores the input parameters of the fit

%}

function [out, x, xdata, rsquare] = AIFbiexpfithelp(xdata, verbose)

warning off

if ~iscell(xdata)
    foo{1} = xdata;
    xdata = foo;
end

Cp = xdata{1}.Cp;
Cp = Cp(:);

t  = xdata{1}.timer;
oldt= t;
oldt = oldt(:);

xdata{1}.timer = t;
t  = t(:);

% Get preferences
prefs = parse_preference_file('dce_preferences.txt',0,...
    {'aif_lower_limits' 'aif_upper_limits' 'aif_initial_values' ...
    'aif_TolFun' 'aif_TolX' 'aif_MaxIter' 'aif_MaxFunEvals' 'aif_Robust' ...
    'aif_peak_weight_exponent'});
lower_limits = str2num(prefs.aif_lower_limits);
upper_limits = str2num(prefs.aif_upper_limits);
initial_values = str2num(prefs.aif_initial_values);
TolFun = str2num(prefs.aif_TolFun);
TolX = str2num(prefs.aif_TolX);
MaxIter = str2num(prefs.aif_MaxIter);
MaxFunEvals = str2num(prefs.aif_MaxFunEvals);
Robust = prefs.aif_Robust;
aif_peak_weight_exponent = str2num(prefs.aif_peak_weight_exponent); %#ok<ST2NM>
if isempty(aif_peak_weight_exponent)
    aif_peak_weight_exponent = 2;
end
aif_peak_weight_floor = 1e-3;

if verbose>0
    fprintf('lower_limits = %s\n',num2str(lower_limits));
    fprintf('upper_limits = %s\n',num2str(upper_limits));
    fprintf('initial_values = %s\n',num2str(initial_values));
    fprintf('TolFun = %s\n',num2str(TolFun));
    fprintf('TolX = %s\n',num2str(TolX));
    fprintf('MaxIter = %s\n',num2str(MaxIter));
    fprintf('MaxFunEvals = %s\n',num2str(MaxFunEvals));
    fprintf('Robust = %s\n\n',Robust);
end

%configure the optimset for use with lsqcurvefit
options = optimset('lsqcurvefit');

%increase the number of function evaluations for more accuracy
% options.MaxFunEvals = MaxFunEvals;
% options.MaxIter     = MaxIter;
% options.TolFun      = TolFun;
% options.TolX        = TolX;
% options.Diagnostics = 'off';
% options.Display     = 'off';
% options.Algorithm   = 'levenberg-marquardt';
% options.Robust      = Robust;
options = fitoptions('Method', 'NonlinearLeastSquares',...
    'Algorithm', 'Levenberg-Marquardt',...
    'MaxIter', MaxIter,...
    'MaxFunEvals', MaxFunEvals,...
    'TolFun', TolFun,...
    'TolX', TolX,...
    'Display', 'off',...
    'Lower',lower_limits,...
    'Upper', upper_limits,...
    'StartPoint', initial_values,...
    'Robust', Robust);

% Choose upper and lower bounds only for trust-region methods.
% lb = [0 0 0 0];
% ub = [5 5 5 5];
% initial_values = [1 1 1 1];

%% Split the fitting between the biexponential phase and the linear phase
t = oldt;
if verbose>0
    figure, plot(t, Cp./max(Cp), 'b.');
    title('Weighting, Injection Period, AIF Curve'), xlabel('time (min)');
end


%[x y] = ginput(1);
% x = 1;
% temp = abs(x-t);
% ind  = find(temp == min(temp));

timer = t;
start = xdata{1}.step;
ended = start(2);
start = start(1);

start_index = find(abs(timer - start) == min(abs(timer - start)), 1);
end_index   = find(abs(timer - ended) == min(abs(timer - ended)), 1);
if end_index < start_index
    end_index = start_index;
end

% Optional behaviour flags. Defaults reproduce the production Stage-B fit.
%   fit_t_base_end : false -> t_base_end is held at timer(start_index) (an input, owned by the
%                    baseline-end precedence upstream); true -> it is fitted, which is what the
%                    Stage-A timing pass (find_end_ss_biexp) needs.
% See docs/project-management/projects/batch-parity/aif_fitting_parity.md.
if isfield(xdata{1}, 'fit_t_base_end')
    fit_t_base_end = logical(xdata{1}.fit_t_base_end);
else
    fit_t_base_end = false;
end

step = zeros(size(timer));
step(start_index:end_index) = 1;
xdata{1}.step = step;


[~, max_index] = max(Cp.*step);
% WW= sort(Cp.*step, 'descend');
% ind(1) = find(Cp == WW(1));
% ind(2) = find(Cp == WW(2));
% ind(3) = find(Cp == WW(3));

% Every sample carries equal weight except the peak, which gets a prior de-weighting derived
% from how far it stands above the rest of the curve.
%
% This has to be data-based rather than residual-based. The peak has leverage 1 in this model --
% it is the only sample in [t_base_end, t0_exp) and the model's maximum A+B can sit exactly on
% it -- so a noise-inflated peak is interpolated rather than flagged and ends up with a *small*
% residual, which the robust estimator cannot see (the classic masking problem). Left at full
% weight it drags the fit: one exponential is spent reaching that single sample and is no longer
% available to describe the washout, so every other frame fits worse.
%
% The weight is the peak's excess over the median relative to the next largest sample's excess,
% which makes it dimensionless -- a raw 1/(peak-median) would change with the concentration
% units and can exceed 1. It never reaches 0: at exactly zero the sample stops constraining the
% upslope duration at all and the fit jumps to a different solution.
% See docs/project-management/projects/batch-parity/aif_fitting_parity.md.
% Only the production fit de-weights the peak. What is unreliable about the peak is its
% *height*, not its *position* -- and position is exactly what the Stage-A timing pass
% (find_end_ss_biexp, which sets fit_t_base_end) is estimating, with the peak as its primary
% evidence. De-weighting it there instead drags t_base_end earlier: measured on
% sub-10bbbdownsample it moved the fitted baseline end from 1.80 frames to 1.16, i.e.
% end_ss 3 -> 2, which is simply wrong for that series.
W = ones(size(Cp));
peak_weight = 1;
if ~fit_t_base_end && numel(Cp) >= 3 && all(isfinite(Cp))
    median_level = median(Cp);
    excess_peak = Cp(max_index) - median_level;
    others = Cp;
    others(max_index) = [];
    excess_ref = max(others) - median_level;
    if isfinite(excess_peak) && excess_peak > 0
        if isfinite(excess_ref) && excess_ref > 0
            peak_weight = (excess_ref / excess_peak) ^ aif_peak_weight_exponent;
        else
            peak_weight = aif_peak_weight_floor;
        end
    end
end
if ~isfinite(peak_weight)
    peak_weight = aif_peak_weight_floor;
end
peak_weight = min(max(peak_weight, aif_peak_weight_floor), 1);
W(max_index) = peak_weight;
% MATLAB multiplies the robust IRLS weights into options.Weights, so this prior weight and the
% robust weight compose the same way they do on the Python side.
options.Weights = W;
if verbose>0
    fprintf('AIF peak prior weight = %g (exponent %g)\n', peak_weight, aif_peak_weight_exponent);
end

step(max_index+1:end) = 0;
xdata{1}.step = step;

if isempty(find(step==1,1))
    % Something has gone wrong, reset to default
    step = zeros(size(timer));
    step(start_index:end_index) = 1;
    xdata{1}.step = step;
end

if verbose>0
    hold on,
    plot(t(max_index), Cp(max_index)/max(Cp), 'kx', 'MarkerSize', 30, ...
        'DisplayName', 'Measured peak');
    plot_aif_transition_lines(t(start_index), t(end_index));
end

% Alter the weightings here.
% W(max_index) =1;
% W(max_index+1)= 1;
% W(max_index-1)= 1;

if verbose>0
    plot(t, W, 'gx');
end

maxer = Cp(max_index);
if ~isfinite(maxer) || maxer <= 0
    % Degenerate injection window; fall back to the global maximum.
    [maxer, max_index] = max(Cp);
end
xdata{1}.maxer = maxer;
% Cp = Cp.*W;

end_baseline = find(xdata{1}.step > 0);
baseline = mean(Cp(1:end_baseline(1)));
xdata{1}.baseline = baseline;
upper_limits(1) = maxer*2;
upper_limits(2) = maxer*2;
initial_values(1) = maxer*0.5;
initial_values(2) = maxer*0.5;
options.Upper = upper_limits;
options.StartPoint = initial_values;
if verbose>0
    disp('Fitting AIF values, limits and initial values adjusted');
    fprintf('lower_limits = %s\n',num2str(lower_limits));
    fprintf('upper_limits = %s\n',num2str(upper_limits));
    fprintf('initial_values = %s\n',num2str(initial_values));
end
% Currently, we use AIF
% [x,resnorm,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(@AIFbiexpcon, ...
%     initial_values, xdata, ...
%     Cp',lower_limits,upper_limits,options);



%% Transition times: t0_exp is parameterised as t_base_end + delta
% Fitting t0_exp directly needs a lower bound to keep it after t_base_end, and the constant
% floor this used to carry (timer(start_index+1) + eps) is not that bound: it is independent of
% the fitted t_base_end, and on a coarse series it sits a whole frame late, which forced the
% upslope to run past the measured peak. delta >= one frame is self-consistent for any
% t_base_end, and one frame is the true floor -- t_base_end sits on the last baseline sample, so
% the earliest the peak can sit is the next one. A curve that jumps straight from baseline to
% max in a single frame is exactly delta == dt, with no interior samples on the ramp.
% See docs/project-management/projects/batch-parity/aif_fitting_parity.md.

timer_diffs = diff(timer);
timer_diffs = timer_diffs(timer_diffs > 0);
if isempty(timer_diffs)
    dt = 1;
else
    dt = median(timer_diffs);
end

% Index guards: end_index-1 and end_index+round(0.2*n) can both fall outside timer on a short
% series or a late-detected injection.
t0_exp_upper_index = min(numel(timer), max(1, end_index + max(1, round(0.2*numel(timer)))));
t0_exp_upper = timer(t0_exp_upper_index);

t_base_end_fixed = timer(start_index);
if fit_t_base_end
    % Timing pass: the baseline end is what we are solving for, so it ranges over the whole
    % pre-peak span rather than the injection window, which can collapse to a single point.
    t_base_end_lower = timer(1);
    t_base_end_upper = timer(max(1, max_index-1));
    if t_base_end_upper <= t_base_end_lower
        t_base_end_upper = min(timer(end), t_base_end_lower + dt);
    end
else
    t_base_end_lower = t_base_end_fixed;
    t_base_end_upper = t_base_end_fixed;
end
t_base_end_init = min(max(t_base_end_fixed, t_base_end_lower), t_base_end_upper);

delta_lower = dt;
delta_upper = max(delta_lower, t0_exp_upper - t_base_end_lower);
% Seed from the *unsnapped* injection window: Stage A reports a fractional end of injection
% (the timing pass's fitted t0_exp), and snapping it to timer(end_index) would discard it.
delta_init = min(max(ended - start, delta_lower), delta_upper);

if fit_t_base_end
    options.Lower      = [lower_limits   t_base_end_lower delta_lower];
    options.Upper      = [upper_limits   t_base_end_upper delta_upper];
    options.StartPoint = [initial_values t_base_end_init  delta_init];
    % Clamp the start point inside the box; MATLAB otherwise silently accepts an infeasible one.
    options.StartPoint = min(max(options.StartPoint, options.Lower), options.Upper);

    ft = fittype('AIFbiexpcon(A, B, c, d, t_base_end, t_base_end + delta, T1, fittingAU, baseline)', ...
        'independent', {'T1'}, ...
        'coefficients', {'A', 'B', 'c', 'd', 't_base_end', 'delta'}, ...
        'problem', {'fittingAU', 'baseline'});

    [f, gof, output] = fit(timer, Cp, ft, options, 'problem', {xdata{1}.fittingAU, xdata{1}.baseline});

    t_base_end_fit = f.t_base_end;
    delta_fit = f.delta;
else
    options.Lower      = [lower_limits   delta_lower];
    options.Upper      = [upper_limits   delta_upper];
    options.StartPoint = [initial_values delta_init];
    options.StartPoint = min(max(options.StartPoint, options.Lower), options.Upper);

    ft = fittype('AIFbiexpcon(A, B, c, d, t_base_end, t_base_end + delta, T1, fittingAU, baseline)', ...
        'independent', {'T1'}, ...
        'coefficients', {'A', 'B', 'c', 'd', 'delta'}, ...
        'problem', {'t_base_end', 'fittingAU', 'baseline'});

    [f, gof, output] = fit(timer, Cp, ft, options, 'problem', ...
        {t_base_end_fixed, xdata{1}.fittingAU, xdata{1}.baseline});

    t_base_end_fit = t_base_end_fixed;
    delta_fit = f.delta;
end
t0_exp_fit = t_base_end_fit + delta_fit;

xdata{1}.timer = oldt;
if verbose>0
    disp(['Adjusted R^2 of AIF fit = ' num2str(gof.adjrsquare)]);
    fprintf('t_base_end = %g, delta = %g, t0_exp = %g\n', t_base_end_fit, delta_fit, t0_exp_fit);
end

out = AIFbiexpcon(f.A, f.B, f.c, f.d, t_base_end_fit, t0_exp_fit, timer, xdata{1}.fittingAU, xdata{1}.baseline)';
% Reported coefficients keep the historical [A B c d t_base_end t0_exp] shape.
x = [f.A, f.B, f.c, f.d, t_base_end_fit, t0_exp_fit];
rsquare = gof.adjrsquare;
