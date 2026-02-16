# DCE Options Reference (CLI + GUI)

This file is the shared options reference for:
- Python CLI: `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_cli.py`
- Python GUI: `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_gui.py`

Default config template:
- `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_default.json`

## Precedence
For options in `stage_overrides`:
1. Explicit value in runtime config / CLI `--set KEY=VALUE`
2. Value from `dce_preferences.txt` (if enabled)
3. Built-in Python fallback

## Top-level config keys
- `subject_source_path`: source BIDS path (rawdata side)
- `subject_tp_path`: processed/derivatives path for this timepoint
- `output_dir`: output folder for maps, logs, figures, summary
- `checkpoint_dir`: optional stage checkpoint folder
- `backend`: `auto|cpu|gpufit`
  - `auto`: probe in order `gpufit_cuda -> cpufit_cpu -> gpufit_cpu_fallback -> pure_cpu`
  - `cpu`: force pure CPU fitting path (no acceleration backend)
  - `gpufit`: require `pygpufit` import; CUDA is used when available, otherwise fallback path
- `write_xls`: write ROI spreadsheet output
- `aif_mode`: `auto|fitted|raw|imported`
- `imported_aif_path`: used when imported AIF mode is selected
- `dynamic_files`: dynamic DCE NIfTI list
- `aif_files`: AIF ROI/mask files
- `roi_files`: tissue ROI/mask files
- `t1map_files`: T1 map files
- `noise_files`: optional noise mask files
- `drift_files`: optional drift files (reserved)
- `model_flags`: map of model enable flags
- `stage_overrides`: advanced settings and fit controls

## `model_flags`
- `tofts`, `ex_tofts`, `patlak`, `tissue_uptake`, `two_cxm`, `fxr`, `auc`, `nested`, `FXL_rr`
- Value convention: `1` enabled, `0` disabled

## `stage_overrides` groups

### Runtime / staging
- `stage_a_mode`: `real|scaffold`
- `stage_b_mode`: `real|scaffold|auto`
- `stage_d_mode`: `real|scaffold|auto`
- `rootname`: output name prefix
- `write_param_maps`: bool for map writing

### Preferences bridging
- `use_dce_preferences`: bool to enable `dce_preferences.txt` defaults
- `dce_preferences_path`: explicit path override
- `force_cpu`: when backend is `auto`, force CPU path if non-zero

### Acquisition / timing
- `dce_metadata_path`: explicit metadata JSON path
- `tr_sec`, `tr_ms`, `fa_deg`
- `time_resolution_sec`, `time_resolution_min`
- `time_vector_path`, `timevectpath`, `timer_path`
- `steady_state_start`, `steady_state_end`
- `start_time`, `end_time`, `start_time_min`, `end_time_min`
- `start_injection_min`, `end_injection_min`
- `injection_duration`

### Stage A concentration conversion
- `relaxivity`
- `hematocrit`
- `blood_t1_ms`, `blood_t1_sec`
- `noise_pixsize`
- `snr_filter`

### Stage B AIF fit
- `aif_curve_mode`: `fitted|raw|imported|auto`
- `aif_lower_limits`: 4 values `[A,B,c,d]`
- `aif_upper_limits`: 4 values `[A,B,c,d]`
- `aif_initial_values`: 4 values `[A,B,c,d]`
- `aif_TolFun`, `aif_TolX`, `aif_MaxIter`, `aif_MaxFunEvals`, `aif_Robust`

### Stage D fit controls
- `time_smoothing`, `time_smoothing_window`
- `fxr_fw`

### Voxel fit bounds / initial values
- `voxel_lower_limit_ktrans`, `voxel_upper_limit_ktrans`, `voxel_initial_value_ktrans`
- `voxel_lower_limit_ve`, `voxel_upper_limit_ve`, `voxel_initial_value_ve`
- `voxel_lower_limit_vp`, `voxel_upper_limit_vp`, `voxel_initial_value_vp`
- `voxel_lower_limit_fp`, `voxel_upper_limit_fp`, `voxel_initial_value_fp`
- `voxel_lower_limit_tp`, `voxel_upper_limit_tp`, `voxel_initial_value_tp`
- `voxel_lower_limit_tau`, `voxel_upper_limit_tau`, `voxel_initial_value_tau`
- `voxel_lower_limit_ktrans_RR`, `voxel_upper_limit_ktrans_RR`, `voxel_initial_value_ktrans_RR`
- `voxel_value_ve_RR`
- `voxel_TolFun`, `voxel_TolX`, `voxel_MaxIter`, `voxel_MaxFunEvals`, `voxel_Robust`

### Acceleration tuning
- `gpu_tolerance`
- `gpu_max_n_iterations`
- `gpu_initial_value_ktrans`
- `gpu_initial_value_ve`
- `gpu_initial_value_vp`
- `gpu_initial_value_fp`

Notes:
- Stage-D acceleration currently applies to `tofts` and `patlak`.
- Stage summary for part D includes:
  - `selected_backend`
  - `acceleration_backend`
  - `backend_reason`
  - `backend_used`

## Notes
- MATLAB-style numeric expressions in preferences (for example `10^-7`) are supported when loaded from `dce_preferences.txt`.
- GUI v1 provides `Browse...` dialogs for all path/file input widgets currently shown in the form.
- `imported_aif_path` exists at the config level, but current GUI form does not expose a dedicated field yet; set it via JSON config when using imported AIF mode.
- Not all MATLAB-era options are fully consumed by current Python runtime yet; see active backlog:
  - `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/PORTING_STATUS.md`
  - `/Users/samuelbarnes/code/ROCKETSHIP/TODO.txt`
