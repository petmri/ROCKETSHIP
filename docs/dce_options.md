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
- `write_postfit_arrays`: bool for optional Part E array export (`*_postfit_arrays.npz`)

### Preferences bridging
- `use_dce_preferences`: bool to enable `dce_preferences.txt` defaults (default: `false` — Python workflows use JSON defaults/config only; enable this to opt into the legacy MATLAB-style preference bridge). There is no CLI flag for the preference file; set `use_dce_preferences`/`dce_preferences_path` via `stage_overrides` (or `--set`).
- `dce_preferences_path`: explicit path override
- `force_cpu`: when backend is `auto`, force CPU path if non-zero

### Acquisition / timing
- `dce_metadata_path`: explicit metadata JSON path
- `tr_sec`, `tr_ms`, `fa_deg`
- MATLAB script aliases: `tr` (ms), `fa` (deg)
- `time_resolution_sec`, `time_resolution_min`
- MATLAB script alias: `time_resolution` (sec)
- Strict resolution behavior (real Stage A):
  - Preferred: resolve from DCE metadata JSON sidecar (or explicit `dce_metadata_path` JSON).
  - If no metadata JSON is available, you must set all three manually: TR (`tr_ms`/`tr_sec`), FA (`fa_deg`/`fa`), and time resolution (`time_resolution_sec`/`time_resolution`).
  - Partial manual override with metadata JSON present is rejected (set all three or none).
- `time_vector_path`, `timevectpath`, `timer_path`
- MATLAB script toggle: `timevectyn` controls whether legacy `timevectpath` is used
- `steady_state_start`, `steady_state_end`: manual pin, highest priority. Prefer the AIF
  sidecar mechanism below for fixed/predictable runs instead of setting these directly;
  this remains available as a low-level escape hatch.
- AIF JSON sidecar `SteadyStateEndTimeIndex`: a `<file>.json` sidecar next to
  `aif_files[0]` (same discovery convention as the DCE metadata sidecar: swap
  `.nii`/`.nii.gz` for `.json`) may set a 1-based `SteadyStateEndTimeIndex` field to pin
  a fixed, predictable baseline end (e.g. `{"SteadyStateEndTimeIndex": 3}`). This is the
  documented way to get a fixed/reproducible run without disabling auto-detection for
  everyone else; used when `steady_state_end` is not set, and takes precedence over
  `steady_state_auto_method`.
- `steady_state_auto_method`: automatic baseline-end detector, used only when neither
  `steady_state_end` nor the AIF sidecar's `SteadyStateEndTimeIndex` is set
  - `legacy_sobel`: MATLAB `dce_auto_aif`-style global-signal Sobel/line-fit heuristic
  - `piecewise_constant`: MATLAB `find_end_ss`-style two-constant brute-force split with local-min backtrack
  - `glr`: GLR-like one-sided change-in-mean detector (ported from `synthetic_dce` `ismrm_submit/end_baseline_detect.py`)
  - `tv`: total-variation/fused-lasso style denoise + first significant upward jump detector (same source)
  - `biexp_fit`: 6-parameter biexponential fit to the mean AIF *signal* curve, seeded by
    `tv`. Unlike the shape heuristics this also reports where the upslope ends, which
    becomes `end_injection` and the Stage-B fit's start point for the upslope duration.
    Falls back to its `tv` seed if the fit cannot run or does not converge.
  - Aliases accepted: `legacy`, `dce_auto_aif`, `sobel`, `piecewise`, `find_end_ss`, `edge`, `find_end_ss_edge`, `tv`, `find_end_ss_tv`, `biexp`, `find_end_ss_biexp`
  - Precedence overall: `steady_state_end` > AIF sidecar `SteadyStateEndTimeIndex` > `steady_state_auto_method`
  - If none of the above is set, Python defaults to `biexp_fit`
- `start_time`, `end_time`, `start_time_min`, `end_time_min`
- `end_injection_min` (MATLAB script alias: `end_injection`, min). There is no
  `start_injection_min`: the injection start is *defined* as the resolved baseline end, so
  move it with `steady_state_end` / the AIF sidecar / `steady_state_auto_method`. Passing
  `start_injection_min` or `start_injection` is rejected with an error pointing at those.
- `injection_duration`
- `aif_Robust`: robust estimator for the Stage-B AIF fit. `Bisquare` (default) runs a Tukey
  biweight IRLS with a per-iteration MAD scale and leverage correction, matched between Python
  and MATLAB; `LAR` maps to scipy `soft_l1`; `off` disables it.
- `aif_peak_weight_exponent` (default 2): prior de-weighting of the AIF peak sample. The weight
  is the peak's excess over the median relative to the next largest sample's excess, raised to
  this exponent; 0 disables it (weight 1). The peak has leverage 1 in the biexponential model,
  so a residual-based robust scheme cannot see a noise-inflated peak — this weight comes from
  the curve's shape instead. Applied only to the production fit, never to the Stage-A timing
  pass, whose whole job is to locate the peak.
- `save_aif_figure` (default true): write the Stage-B AIF fit figure (`dceAIF_fitting.png`),
  showing measured vs fitted curves with `t_base_end` and `t0_exp` marked as vertical lines.

### Stage A concentration conversion
- `relaxivity`
- `hematocrit`
- `blood_t1_ms`, `blood_t1_sec`
- MATLAB script alias: `blood_t1`
- `noise_pixsize`
- `snr_filter`

### Stage B AIF fit
- `aif_curve_mode`: `fitted|raw|imported|auto`
- MATLAB script alias: `aif_type` (`1=fitted`, `2=raw`, `3=imported`)
- Imported AIF path fallback alias: `import_aif_path`
- `aif_lower_limits`: 4 values `[A,B,c,d]`
- `aif_upper_limits`: 4 values `[A,B,c,d]`
- `aif_initial_values`: 4 values `[A,B,c,d]`
- `aif_TolFun`, `aif_TolX`, `aif_MaxIter`, `aif_MaxFunEvals`, `aif_Robust`

### Stage D fit controls
- `time_smoothing`, `time_smoothing_window`
- `fxr_fw`
- `write_param_maps`: bool (default `true`) — write per-voxel parameter map NIfTIs.
- `fit_voxels`: bool (default `true`). Set `false` for **ROI-only mode**: skip the per-voxel fit and
  fit only each ROI's averaged concentration curve (average-then-fit, matching MATLAB). Much faster,
  and for nonlinear models the pre-fit averaging reduces noise. Requires `roi_files`; parameter maps
  are not written. Each ROI is averaged over its intersection with the primary fit region
  (`roi_files[0]`), so make `roi_files[0]` the encompassing ROI (e.g. the whole-brain mask).
- `time_unit` / `timer_unit` (optional direct-fit hint): `minutes|seconds`
  - No implicit or runtime-selectable algorithm switching.
  - `model_2cxm_fit` uses the OSIPI LEK-style resampled fit path.
  - `model_tissue_uptake_fit` uses the standard least-squares fit path.
  - For `model_tissue_uptake_fit` and `model_2cxm_fit`, internal fitting is always done in minutes with rate constants in 1/min.
  - Input preferences for rate limits/initial values are interpreted in the same units as the input timer, then converted internally.
  - Returned rate parameters (`ktrans`, `fp`) are converted back to match the input timer unit.

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
- Stage-D acceleration currently applies to `tofts`, `ex_tofts`, `patlak`, `tissue_uptake`, and `2cxm`.
- `gpu_tolerance` is a shared accelerated solver tolerance (CPUfit/GPUfit path) for all accelerated Stage-D models; current default is `1e-6` (the previous `1e-12` was too strict for some `ex_tofts` qualification workloads).
- Stage summary for part D includes:
  - `selected_backend`
  - `acceleration_backend`
  - `backend_reason`
  - `backend_used`

## Notes
- MATLAB-style numeric expressions in preferences (for example `10^-7`) are supported when loaded from `dce_preferences.txt`.
- GUI v1 provides `Browse...` dialogs for all path/file input widgets currently shown in the form.
- `imported_aif_path` exists at the config level, but current GUI form does not expose a dedicated field yet; set it via JSON config when using imported AIF mode.
- Script-level option audit (all keys in `script_preferences.txt` with support status):
  - `~/code/ROCKETSHIP/docs/project-management/projects/script-preferences-audit/script_preferences_option_audit.md`
  - `~/code/ROCKETSHIP/docs/project-management/projects/script-preferences-audit/script_preferences_option_audit.json`
- Not all MATLAB-era options are fully consumed by current Python runtime yet; see active backlog:
  - `~/code/ROCKETSHIP/docs/project-management/PORTING_STATUS.md`
  - `~/code/ROCKETSHIP/docs/project-management/TODO.md`
