function [end_ss, end_injection, details] = find_end_ss_tv(signal_intensities)
    % Total-variation/fused-lasso style denoise + first-significant-upward-jump
    % detector for the end of the pre-contrast steady-state baseline.
    %
    % Port of Python's _tv_baseline_end (python/dce_pipeline.py), used as the
    % default steady_state_auto_method there. signal_intensities is a 2D matrix
    % (time points x voxels, same convention as find_end_ss.m's DYNAMLV input).
    %
    % end_injection follows the same "mean(argmax(DYNAMLV, time)) across voxels"
    % convention Python computes for every auto-detect method (dce_pipeline.py's
    % Stage A, commented "MATLAB auto-find-injection parity"), rather than a
    % method-specific local-max search, so it stays aligned with what Python
    % already treats as the canonical definition.

    % How many robust sigmas above the median a jump must clear to count as the contrast
    % onset. Must equal TV_JUMP_THRESHOLD_SIGMA in python/dce_pipeline.py.
    TV_JUMP_THRESHOLD_SIGMA = 5.0;

    % `details` mirrors the dict Python's _tv_baseline_end returns, so a caller can tell a real
    % detection from a give-up. See the `mode` note at the bottom of this function.
    details = struct('mode', 'tv_jump', 'strength', 0.0, 'valid_jump_count', 0, ...
        'jump_threshold', 0.0, 'detected_jump', 0.0);

    DYNAMLV = signal_intensities;
    if isvector(DYNAMLV)
        DYNAMLV = DYNAMLV(:);
    end

    if size(DYNAMLV, 1) >= 1
        [~, peak_idx] = max(DYNAMLV, [], 1);
        end_injection = mean(peak_idx);
    else
        end_injection = 1;
    end

    if size(DYNAMLV, 1) < 2
        end_ss = 1;
        details.mode = 'fallback_short_signal';
        return;
    end

    x_raw = mean(DYNAMLV, 2);
    n = length(x_raw);
    if n < 3
        end_ss = 1;
        details.mode = 'fallback_short_signal';
        return;
    end

    initial_diff = diff(x_raw);
    mad_val = median(abs(initial_diff - median(initial_diff)));
    lambda_tv = 2.0 * mad_val;
    if mad_val < 1e-6
        lambda_tv = 0.1;
    end

    x = x_raw;
    for iteration = 1:50
        x_old = x;
        d = diff(x);
        d_thresh = sign(d) .* max(abs(d) - lambda_tv / n, 0.0);
        x(1) = x_raw(1);
        for i = 2:n
            x(i) = 0.5 * (x(i - 1) + d_thresh(i - 1)) + 0.5 * x_raw(i);
        end
        if max(abs(x - x_old)) < 1e-6
            break;
        end
    end

    jumps = diff(x);
    % The threshold has to be calibrated on baseline noise, so it must not be measured over a
    % stretch that contains contrast. A leading window (min(n, max(5, 0.2*n)), previously used
    % here) is not that stretch: it spans 12 frames of a 64-frame series, and the bolus usually
    % peaks well inside it. Calibrating there measures the bolus instead of the noise -- on
    % sub-1102140_ses-01 it returned a MAD of 76.0 against a true baseline scatter of 2.5,
    % inflating the threshold ~40x, which pushed the real onset jump (231.5) below it and left
    % the detector with no jump to report at all. The MAD over *all* jumps is robust for the
    % same reason lambda_tv above uses it: contrast frames are a small minority of a DCE series,
    % so the median absolute deviation still reflects the flat part of the curve.
    % Mirrored in python/dce_pipeline.py:_tv_baseline_end -- keep the two in step.
    if ~isempty(jumps)
        jump_median = median(jumps);
        jump_mad = median(abs(jumps - jump_median));
    else
        jump_median = 0.0;
        jump_mad = 0.0;
    end
    if jump_mad < 1e-6
        % Population std (normalize by N, not N-1) to match numpy's default ddof=0.
        jump_mad = 0.01 * std(x_raw, 1);
        if jump_mad < 1e-6
            jump_mad = 0.01;
        end
    end

    jump_threshold = jump_median + TV_JUMP_THRESHOLD_SIGMA * jump_mad;
    significant_jumps = find(jumps > jump_threshold);

    valid_jumps = [];
    for idx = 1:numel(significant_jumps)
        k = significant_jumps(idx);
        if k < numel(jumps)
            next_jump = jumps(k + 1);
            if next_jump > -jump_mad || jumps(k) > 2.0 * jump_threshold
                valid_jumps(end + 1) = k; %#ok<AGROW>
            end
        else
            if jumps(k) > 1.5 * jump_threshold
                valid_jumps(end + 1) = k; %#ok<AGROW>
            end
        end
    end

    % Finding no jump at all is a *failure to detect*, not a detection of frame 1. Returning a
    % bare end_ss = 1 made the two indistinguishable to every caller: a detector that had given
    % up looked exactly like one that had confidently found a bolus arriving on the second
    % frame, and end_ss = 1 is plausible enough that nothing downstream would question it.
    % The warning and the `details.mode` flag make the give-up visible; end_ss itself is
    % unchanged, since 1 is still the safest guess when nothing was found. Mirrors the
    % `fallback_no_jump_detected` mode in python/dce_pipeline.py:_tv_baseline_end.
    details.jump_threshold = jump_threshold;
    details.valid_jump_count = numel(valid_jumps);
    if isempty(valid_jumps)
        end_ss = 1;
        details.mode = 'fallback_no_jump_detected';
        warning('find_end_ss_tv:NoJumpDetected', ...
            ['No contrast-onset jump cleared the threshold (%.4g); falling back to ' ...
             'end_ss = 1. This is a detection failure, not a baseline of one frame -- ' ...
             'check the AIF curve before trusting downstream maps.'], jump_threshold);
    else
        end_ss = valid_jumps(1);
        details.detected_jump = jumps(end_ss);
        baseline_noise = std(x_raw(1:min(10, max(1, floor(n / 4)))), 1);
        if baseline_noise < 1e-6
            baseline_noise = 1e-6;
        end
        details.strength = min(max(1.0 - exp(-(details.detected_jump / baseline_noise) / 2.0), 0.0), 1.0);
    end
    end_ss = max(1, min(end_ss, n));
end
