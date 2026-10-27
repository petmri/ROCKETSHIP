function idx = dce_frame_window(n_frames, start_t, end_t)
% dce_frame_window 1-based frames of a dynamic series kept by the start_t/end_t preferences.
%
% Shared by loadIMGVOL (which crops the image) and dce_volume_timer (which crops the
% sidecar VolumeTiming), so the frame times can never drift from the frames they describe.
% The branch order reproduces the original loadIMGVOL cropping exactly.

if ~isempty(start_t) && ~isempty(end_t) && start_t > 0 && end_t > start_t
    idx = start_t:end_t;
elseif ~isempty(start_t) && isempty(end_t) && start_t > 1
    idx = start_t:n_frames;
elseif ~isempty(end_t) && end_t > 0
    idx = 1:end_t;
else
    idx = 1:n_frames;
end
end
