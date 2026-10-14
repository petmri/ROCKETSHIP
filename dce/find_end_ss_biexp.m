function [end_ss, end_injection] = find_end_ss_biexp(signal_intensities)
    % Baseline end from a 6-parameter biexponential fit to the mean AIF signal curve.
    %
    % Port of Python's _biexp_fit_baseline_end (python/dce_pipeline.py), the default
    % steady_state_auto_method there. signal_intensities is a 2D matrix (time points x voxels,
    % same convention as find_end_ss.m's DYNAMLV input).
    %
    % Why the signal curve and not Cp: the baseline end is what *defines* the window that
    % converts signal to R1, so it cannot be read off a concentration curve -- Cp does not exist
    % until this has answered. Fitting the signal breaks that cycle without restructuring Part A.
    % Only the two transition times are kept, and both are invariant to the affine rescaling
    % below, so nothing that survives the fit is distorted by working in signal units. A, B, c
    % and d are discarded, because signal saturation does distort amplitudes and decay rates.
    %
    % The curve is baseline-subtracted and max-normalised before fitting -- the same preparation
    % dce_auto_aif.m already applies -- so the fit runs with fittingAU = false and the amplitude
    % bounds derived from maxer stay on a familiar scale.
    %
    % Weighting is uniform; a noise-inflated peak is handled by the robust estimator
    % (aif_Robust in dce_preferences.txt) rather than by discarding frames.
    %
    % See project-management/projects/archived/batch-parity/aif_fitting_parity.md.

    DYNAMLV = signal_intensities;
    if isvector(DYNAMLV)
        DYNAMLV = DYNAMLV(:);
    end

    % find_end_ss_tv supplies the provisional baseline mean and injection window this fit is
    % seeded from, and is the fallback whenever the fit cannot run or does not converge.
    [end_ss, end_injection] = find_end_ss_tv(DYNAMLV);

    n = size(DYNAMLV, 1);
    if n < 4
        return;
    end

    seed_end_ss = max(1, min(end_ss, n));
    global_curve = mean(DYNAMLV, 2);

    baseline_mean = mean(global_curve(1:seed_end_ss));
    normalized = global_curve - baseline_mean;
    scale = max(normalized);
    if ~isfinite(scale) || scale <= 0
        return;
    end
    normalized = normalized ./ scale;

    % Frame units, matching dce_auto_aif.m's timer = 0:time_points-1. t_base_end therefore comes
    % back as a 0-based frame position, and end_ss (1-based) is one more than it.
    timer = (0:n-1)';
    [~, peak_index] = max(normalized);

    xdata = cell(1);
    xdata{1}.Cp = normalized;
    xdata{1}.timer = timer;
    xdata{1}.step = [seed_end_ss - 1, max(peak_index - 1, seed_end_ss - 1)];
    xdata{1}.fittingAU = false;
    xdata{1}.fit_t_base_end = true;

    try
        [~, x, ~, ~] = AIFbiexpfithelp(xdata, 0);
    catch fit_error
        warning('find_end_ss_biexp:fitFailed', ...
            'Biexponential baseline-end fit failed (%s); falling back to find_end_ss_tv.', ...
            fit_error.message);
        return;
    end

    t_base_end = x(5);
    t0_exp = x(6);
    if ~isfinite(t_base_end) || ~isfinite(t0_exp)
        return;
    end

    end_ss = max(1, min(round(t_base_end) + 1, n));
    % end_injection stays fractional on purpose: it becomes the Part B fit's start point for the
    % upslope duration, and snapping it to a frame would throw that away.
    end_injection = min(max(t0_exp + 1, end_ss), n);

    fprintf('Found end steady state at image number: %d (biexponential fit)\n', end_ss);
end
