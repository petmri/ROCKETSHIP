function [local_min_index, local_max_index] = find_end_ss(signal_intensities)
    % % Load the data
    % load(filename);
    DYNAMLV = signal_intensities; % Assuming signal_intensities is a 2D matrix with time points in rows
    x = mean(DYNAMLV, 2);
    mse_values = zeros(size(x)-1);
    x = imgaussfilt(x, 1);

    for t = 2:length(x)
        err = 0;
        before_t = mean(x(1:t-1), 1);
        after_t = mean(x(t+1:end), 1);
        for tau = 1:length(x)
            if tau < t
                err = err + (x(tau) - before_t)^2;
            else
                err = err + (x(tau) - after_t)^2;
            end
        end
        err = err / length(x);
        mse_values(t-1) = err;
    end

    [~, min_index] = min(mse_values);
    disp(['Optimal t: ', num2str(min_index)]);

    % Step outwards from min_index to find nearest local min and local max
    local_min_index = min_index;
    while local_min_index > 1 && x(local_min_index) >= x(local_min_index - 1)
        local_min_index = local_min_index - 1;
    end

    local_max_index = min_index;
    while local_max_index < length(x) && x(local_max_index) <= x(local_max_index + 1)
        local_max_index = local_max_index + 1;
    end

    % Display the results
    disp(['Local minimum index: ', num2str(local_min_index)]);
    disp(['Local maximum index: ', num2str(local_max_index)]);

    % Create the plot
    h = plot(x);
    hold on; % Hold the current plot
    plot(min_index, x(min_index), 'bx', 'MarkerSize', 10, 'DisplayName', 'Optimal t'); % Plot optimal t
    plot(local_min_index, x(local_min_index), 'ro', 'MarkerSize', 10, 'DisplayName', 'Local Min'); % Plot local minimum
    plot(local_max_index, x(local_max_index), 'go', 'MarkerSize', 10, 'DisplayName', 'Local Max'); % Plot local maximum
    xlabel('Index');
    ylabel('Value');
    title('Mean Values Plot');
    legend show; % Show legend

    % Enable data cursor mode
    datacursormode on;
    hold off; % Release the hold
end