function [end_ss, end_injection] = find_end_ss_tv(signal_intensities)
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
        return;
    end

    x_raw = mean(DYNAMLV, 2);
    n = length(x_raw);
    if n < 3
        end_ss = 1;
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
    baseline_len = min(n, max(5, floor(0.2 * n)));
    baseline_segment = x(1:baseline_len);
    baseline_jumps = diff(baseline_segment);
    if ~isempty(baseline_jumps)
        baseline_jump_mad = median(abs(baseline_jumps - median(baseline_jumps)));
        baseline_jump_median = median(baseline_jumps);
    else
        baseline_jump_mad = 0.0;
        baseline_jump_median = 0.0;
    end
    if baseline_jump_mad < 1e-6
        % Population std (normalize by N, not N-1) to match numpy's default ddof=0.
        baseline_jump_mad = 0.01 * std(x_raw(1:baseline_len), 1);
        if baseline_jump_mad < 1e-6
            baseline_jump_mad = 0.01;
        end
    end

    jump_threshold = baseline_jump_median + 3.5 * baseline_jump_mad;
    significant_jumps = find(jumps > jump_threshold);

    valid_jumps = [];
    for idx = 1:numel(significant_jumps)
        k = significant_jumps(idx);
        if k < numel(jumps)
            next_jump = jumps(k + 1);
            if next_jump > -baseline_jump_mad || jumps(k) > 2.0 * jump_threshold
                valid_jumps(end + 1) = k; %#ok<AGROW>
            end
        else
            if jumps(k) > 1.5 * jump_threshold
                valid_jumps(end + 1) = k; %#ok<AGROW>
            end
        end
    end

    if isempty(valid_jumps)
        end_ss = 1;
    else
        end_ss = valid_jumps(1);
    end
    end_ss = max(1, min(end_ss, n));
end
