function h = plot_aif_transition_lines(t_base_end, t0_exp)
%plot_aif_transition_lines Draw the AIF transition times as vertical lines.
%
%   Marks the end of the baseline (t_base_end / end_ss) and the end of the injection
%   (t0_exp / end_injection) on the current axes. Both are drawn spanning the full y range so
%   they read as event markers, rather than as a data series that is zero everywhere with a
%   spike at each location -- which is what this replaced, and which rescaled the y axis and
%   looked like a signal.
%
%   Works in whatever x units the current axes use (minutes or frame numbers); pass the values
%   already in those units. Returns the line handles so callers can add them to a legend.

hold_state = ishold;
hold on;
y = ylim;
h = gobjects(1, 2);
h(1) = plot([t_base_end t_base_end], y, '--', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5, ...
    'DisplayName', 'End of baseline (t\_base\_end)');
h(2) = plot([t0_exp t0_exp], y, ':', 'Color', [0.49 0.18 0.56], 'LineWidth', 1.5, ...
    'DisplayName', 'End of injection (t0\_exp)');
ylim(y);  % keep the data's own limits; the markers should not rescale the axes
if ~hold_state
    hold off;
end
end
