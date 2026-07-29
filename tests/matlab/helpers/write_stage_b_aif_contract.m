function payload = write_stage_b_aif_contract(jsonPath, Bdata, rootname)
%write_stage_b_aif_contract Serialize MATLAB Stage-B AIF outputs as a parity contract.
%
% Freezes the Stage-B quantities the DCE parity suite otherwise never compares --
% the fitted AIF `Cp_use`, the injection `step` window, the baseline/upslope timing
% terms, and the AIF peak index -- so a structurally different Stage-B fit fails on
% its own terms instead of hiding behind final maps that still correlate.
%
%   payload = write_stage_b_aif_contract(jsonPath, Bdata, rootname)
%
% Consumed by tests/python/test_stage_b_aif_parity.py and diffed for MATLAB-side
% drift by tests/contracts/check_matlabref_map_drift.py.

if nargin < 3
    rootname = '';
end

xdata1 = Bdata.xdata{1};

cp_use = row_vector(Bdata.Cp_use);
if isempty(cp_use)
    error('write_stage_b_aif_contract:EmptyCpUse', 'Bdata.Cp_use is empty.');
end
[~, max_index] = max(cp_use);

payload = struct();
payload.meta = struct( ...
    'version', 'stage_b_aif_v1', ...
    'generated_utc', datestr(now, 'yyyy-mm-ddTHH:MM:SSZ'), ...
    'source', 'B_AIF_fitting_func', ...
    'rootname', char(rootname), ...
    'aif_name', char(Bdata.aif_name));

% Curves. CpROI/Cp_use are concentration; Stlv_* are the signal-domain pair.
payload.curves = struct( ...
    'timer', row_vector(Bdata.timer), ...
    'CpROI', row_vector(Bdata.CpROI), ...
    'Cp_use', cp_use, ...
    'Stlv_use', row_vector(xdata1.Stlv));

% Scalars. start_time/end_time are 1-based inclusive indices into the full series.
payload.window = struct( ...
    'step', row_vector(xdata1.step), ...
    'start_injection', double(Bdata.start_injection), ...
    'end_injection', double(Bdata.end_injection), ...
    'start_time', double(Bdata.start_time), ...
    'end_time', double(Bdata.end_time), ...
    'time_resolution', double(Bdata.time_resolution), ...
    'max_index', double(max_index), ...
    'numvoxels', double(Bdata.numvoxels));

% [A B c d t_base_end t0_exp]; empty outside the fitted branch.
payload.fit = struct( ...
    'params_cp', row_vector(get_or_empty(Bdata, 'aif_fit_params_cp')), ...
    'params_stlv', row_vector(get_or_empty(Bdata, 'aif_fit_params_stlv')), ...
    'rsquare_cp', scalar_or_nan(get_or_empty(Bdata, 'aif_fit_rsquare_cp')), ...
    'rsquare_stlv', scalar_or_nan(get_or_empty(Bdata, 'aif_fit_rsquare_stlv')));

jsonText = jsonencode(payload);
fid = fopen(jsonPath, 'w');
if fid < 0
    error('write_stage_b_aif_contract:OpenFailed', 'Could not open %s for writing.', jsonPath);
end
fwrite(fid, jsonText, 'char');
fclose(fid);
end


function v = row_vector(x)
if isempty(x)
    v = [];
    return;
end
v = double(reshape(x, 1, []));
end


function v = scalar_or_nan(x)
if isempty(x)
    v = NaN;
else
    v = double(x(1));
end
end


function v = get_or_empty(s, name)
if isfield(s, name)
    v = s.(name);
else
    v = [];
end
end
