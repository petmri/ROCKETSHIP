function [timer_min, step_min] = dce_volume_timer(volume_timing_sec, n_frames_raw, start_t, end_t)
% dce_volume_timer Frame times (minutes, from 0) of the analysed window, from BIDS VolumeTiming.
%
%  volume_timing_sec - sidecar VolumeTiming: start of every acquired frame, in seconds
%  n_frames_raw      - frame count of the dynamic image before start_t/end_t cropping
%  start_t, end_t    - the frame-window preferences, as passed to A_make_R1maps_func
%
%  timer_min - row vector, one entry per kept frame, first entry 0
%  step_min  - median frame step in minutes (of the window, or of the whole series when the
%              window holds a single frame), reported as the time resolution
%
% Mirrors the VolumeTiming branch of python/dce_pipeline.py::_resolve_dynamic_metadata.

volume_timing_sec = double(volume_timing_sec(:)');
if numel(volume_timing_sec) ~= n_frames_raw
    error('dce_volume_timer:LengthMismatch', ...
        'VolumeTiming has %d entries but the dynamic image has %d frames', ...
        numel(volume_timing_sec), n_frames_raw);
end
if any(~isfinite(volume_timing_sec)) || any(diff(volume_timing_sec) <= 0)
    error('dce_volume_timer:NotIncreasing', 'VolumeTiming must be finite and strictly increasing');
end

idx = dce_frame_window(n_frames_raw, start_t, end_t);
if isempty(idx) || idx(end) > n_frames_raw
    error('dce_volume_timer:WindowOutOfRange', ...
        'Frame window start_t/end_t selects no frames of the %d in the dynamic image', n_frames_raw);
end
window_sec = volume_timing_sec(idx);
timer_min = (window_sec - window_sec(1)) / 60;

if numel(window_sec) > 1
    step_min = median(diff(window_sec)) / 60;
else
    step_min = median(diff(volume_timing_sec)) / 60;
end
end
