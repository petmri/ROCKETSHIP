# Python Porting Status

This file is the handoff point for resuming work later.

## Current checkpoint
- Branch: `codex/algorithm-test-suite`
- Latest milestone commit: `d59c492`
- Python test baseline at this checkpoint:
  - `.venv/bin/python -m unittest discover -s tests/python -p 'test_*.py'`
  - `57` run, `0` failed, `2` skipped (parity-gated)

## Current scope covered
Python implementations currently exist in `/Users/samuelbarnes/code/ROCKETSHIP/python/`:

- `dce_models.py`
  - `model_tofts_cfit`
  - `model_patlak_cfit`
  - `model_extended_tofts_cfit`
  - `model_vp_cfit`
  - `model_tissue_uptake_cfit`
  - `model_2cxm_cfit`
  - `model_fxr_cfit`
  - `model_patlak_linear`
  - `model_tofts_fit` (SciPy least-squares)
  - `model_vp_fit`
  - `model_tissue_uptake_fit`
  - `model_2cxm_fit`
  - `model_fxr_fit`
- `dsc_helpers.py`
  - `import_aif`
  - `previous_aif`
- `dsc_models.py`
  - `dsc_convolution_ssvd`
- `parametric_models.py`
  - `t2_linear_fast`
  - `t1_fa_linear_fit`

## DCE CLI scope (agreed)
Target deliverable is an end-to-end Python CLI path for DCE parts `A`, `B`, and `D`.

Confirmed out-of-scope/deprecated for the Python port:
- Any `neuroecon` server execution path.
- Legacy GUI batch queue/prep flow for part D.
- Email notification completion flow.
- Manual click-based AIF tools.
- ImageJ ROI input support (`.roi`).
- MATLAB-specific batch helper scripts.

Confirmed in-scope requirements to retain:
- ROI spreadsheet outputs (`.xls`).

## Execution architecture decisions
- Primary runtime path is a single-process end-to-end pipeline (`A -> B -> D`) with in-memory data flow.
- Intermediate `.mat` handoff files are not part of normal execution.
- Optional stage checkpoint export is allowed for parity/debug only.
- Fitting backend approach:
  - CPU path is required and treated as the parity baseline.
  - GPUfit is optional (`backend=auto|cpu|gpufit`) and can be enabled once installed.

## DCE CLI pipeline status
- In-memory pipeline implemented in:
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_pipeline.py`
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_cli.py`
  - `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_cli.py`
  - `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_gui.py` (GUI wrapper)
- Stage A status:
  - real implementation path is wired (NIfTI input, ROI/AIF extraction, R1/Cp/Ct generation, QC figure saving)
  - stage mode control: `stage_overrides.stage_a_mode = real|scaffold`
- Stage B status:
  - real non-GUI implementation path is wired (time restriction, non-interactive AIF selection/fitting, pass-through arrays, QC figure saving)
  - supported AIF modes: `fitted`, `raw`, `imported`
  - stage mode control: `stage_overrides.stage_b_mode = auto|real|scaffold`
- Stage D status:
  - real in-memory implementation path is wired (voxel/ROI model fitting, parameter-map output, `.xls` ROI tables)
  - supported non-GUI models: `tofts`, `ex_tofts`, `patlak`, `tissue_uptake`, `2cxm`, `fxr`, `auc`
  - stage mode control: `stage_overrides.stage_d_mode = auto|real|scaffold`
- MATLAB preference bridge status:
  - Python now loads MATLAB-style `dce_preferences.txt` defaults (default path: `/Users/samuelbarnes/code/ROCKETSHIP/dce/dce_preferences.txt`)
  - override path is supported via `stage_overrides.dce_preferences_path` or CLI `--dce-preferences`
  - explicit `stage_overrides` values always win over file defaults
  - preference parsing supports MATLAB numeric expressions (for example `10^-7`, `10^(-12)`)
  - Stage D auto-backend now honors `force_cpu` when backend is `auto`
  - Stage D acceleration runtime fallback order is now:
    - selected acceleration backend
    - `cpufit_cpu` (when selected backend is gpufit and pycpufit import is available)
    - pure CPU fitting path
- CLI observability status:
  - emits progress events to stdout (`ROCKETSHIP_EVENT <json>`)
  - writes per-run event log JSONL (`<output_dir>/dce_pipeline_events.jsonl`)
  - default config path is now `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_default.json`
  - default config is prewired to tiny fixture data for one-click local smoke runs
- GUI status (v1):
  - PySide6 GUI implemented in `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_gui.py`
  - GUI edits common top-level config fields and all `stage_overrides` keys
  - GUI runs CLI via subprocess, streams progress, supports hard-stop terminate, and previews PNG QC figures
  - GUI form includes `Browse...` dialogs for path/file inputs
  - `imported_aif_path` currently remains JSON-config-only (not yet a dedicated GUI field)
- Recent parity fixes (MATLAB alignment):
  - Stage A now uses MATLAB-style column-major voxel indexing for `lvind/tumind/noiseind`.
  - Stage D map writeback now uses MATLAB-style linear index mapping.
  - Stage A TR handling now converts `tr_ms -> tr_sec` before R1 calculations.
  - Stage A now auto-converts T1 maps from ms to seconds when magnitudes indicate ms units.
  - Stage A cleanup behavior (`cleanAB`, `cleanR1t`) now mirrors MATLAB logic more closely.
- Dataset parity tests:
  - `tests/python/test_dce_pipeline_parity_metrics.py` adds full-pipeline Tofts map checks (`Ktrans`, `ve`) against MATLAB maps using correlation and MSE tolerances.
  - Stage-D acceleration wiring now includes `tofts`, `ex_tofts`, `patlak`, `tissue_uptake`, and `2cxm` in `python/dce_pipeline.py`.
  - `tests/python/test_dce_pipeline.py` includes unit coverage for accelerated model dispatch, parameter bounds/initialization, and output-column mapping for those five models.
  - opt-in multi-model map parity test:
    - `test_downsample_bbb_p19_models_cpu_and_auto`
    - compares MATLAB vs Python CPU-only and MATLAB vs Python `auto` backend for `tofts`, `ex_tofts`, `patlak`, `tissue_uptake`, `2cxm`
    - also compares Python `auto` vs Python CPU-only maps
    - enabled with `ROCKETSHIP_RUN_PIPELINE_PARITY=1` and `ROCKETSHIP_RUN_MULTI_MODEL_BACKEND_PARITY=1`
    - uses sparse ROI sampling (`ROCKETSHIP_PARITY_MULTI_MODEL_ROI_STRIDE`, default `12`) to keep runtime manageable
  - fast path uses committed fixture `test_data/ci_fixtures/dce/bbb_p19_downsample_x3y3` (with fallback to synthetic-generated path if needed).
  - full-volume `BBB data p19` path is available behind env gating and intended for occasional thorough checks due runtime.
  - both tests require MATLAB baseline map `processed/results_matlab/Dyn-1_tofts_fit_Ktrans.nii` (not legacy `processed/results`).
  - MATLAB baseline generator: `tests/matlab/generate_dce_tofts_parity_map.m`
    - now accepts `'models', {...}` to emit non-Tofts baselines in one run.
  - latest measured parity (MATLAB `results_matlab` baseline):
    - downsample `x3y3` `Ktrans`: `corr=0.99816`, `mse=0.0001815`, `mae=0.004171` (`n=2834`)
    - downsample `x3y3` `ve`: `corr=0.98584`, `mse=0.001059`, `mae=0.001531` (`n=2834`)
    - full-volume `BBB p19` `Ktrans`: `corr=0.99499`, `mse=0.0004671`, `mae=0.005261` (`n=25512`)
    - full-volume `BBB p19` `ve`: `corr=0.98822`, `mse=0.0008778`, `mae=0.001473` (`n=25512`)
- Tiny settings-matrix tests:
  - fixture generator: `tests/python/generate_tiny_dce_settings_fixture.py`
  - fixture path: `test_data/ci_fixtures/dce/tiny_settings_case`
  - test module: `tests/python/test_dce_pipeline_settings_matrix.py`
  - current covered settings/features:
    - Tofts fit bound enforcement (`voxel_lower/upper_limit_*`)
    - Tofts initial guess robustness (`voxel_initial_value_*`)
    - Stage-A static blood T1 override (`stage_overrides.blood_t1_ms|blood_t1_sec`)
    - Stage-A blood T1 override guardrail (rejects non-positive values)
- Preference bridge tests:
  - `tests/python/test_dce_preferences_bridge.py`
  - validates file loading, expression parsing, override precedence, Stage-B AIF fit option wiring, and `force_cpu` behavior
  - `tests/python/test_dce_cli.py`
  - validates `--dce-preferences` and repeatable `--set KEY=VALUE` merging into `stage_overrides`
- Scope guards enforced by config validation:
  - rejects ImageJ `.roi` input
  - accepts backend `auto|cpu|gpufit`
  - accepts AIF mode `auto|fitted|raw|imported`
- Installer/acceleration setup status:
  - installer script: `/Users/samuelbarnes/code/ROCKETSHIP/install_python_acceleration.py`
  - installs pyCpufit/pyGpufit with `--no-deps` to avoid NumPy/SciPy resolver churn
  - prefers latest stable (non-prerelease) release by default when no explicit release tag is provided
  - auto-detects local CUDA version (`CUDA_VERSION`, `nvcc`, `nvidia-smi`) and selects the closest compatible host asset
  - if local CUDA is older than all available CUDA assets and CPU asset exists, installer prefers CPU asset first
- Checkpoint support:
  - writes `a_out.json`, `b_out.json`, `d_out.json` when `checkpoint_dir` is configured
- Example config:
  - `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/dce_cli_config.example.json`
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_default.json`
- Shared options docs:
  - `/Users/samuelbarnes/code/ROCKETSHIP/docs/dce_options.md`

## Parity status against MATLAB baseline
From `.venv`:

```bash
.venv/bin/python tests/python/generate_python_results.py --output /tmp/python_results.json
.venv/bin/python tests/python/compare_with_matlab_baseline.py --python-results /tmp/python_results.json --require-all
```

Expected summary right now:
- `18 pass`
- `0 fail`
- `0 missing`
- `0 skipped`

## Test commands to rerun quickly

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m unittest discover -s tests/python -p 'test_*.py'
.venv/bin/python tests/python/generate_python_results.py --output /tmp/python_results.json
.venv/bin/python tests/python/compare_with_matlab_baseline.py --python-results /tmp/python_results.json --require-all
```

## Environment notes
- Use project-local `venv`: `/Users/samuelbarnes/code/ROCKETSHIP/.venv`
- Install deps:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## CI behavior
Workflow: `/Users/samuelbarnes/code/ROCKETSHIP/.github/workflows/run_DCE.yml`

- `python_checks` job (push + PR): Python unit tests, contract parity checks, and downsample DCE pipeline parity.
- `python_portability` job (push + PR): Python unit/integration suite on `windows-latest` and `macos-latest` (non-parity mode).
- PR MATLAB job: unit/integration checks + DCE smoke run using committed fixture `test_data/ci_fixtures/dce/downsample_x2_bids`.
- Push to `dev/master`: heavier MATLAB matrix (full validation path).

## Next recommended steps
1. Complete MATLAB script-level option mapping audit:
   - map `script_preferences.txt` keys to Python config/stage overrides
   - classify each as `supported`, `dropped-by-design`, or `pending`
   - add targeted tests for newly-wired options
2. Expand tiny fixture variants for edge-case sweeps:
   - low-SNR case
   - non-uniform timer case (`stage_overrides.time_vector_path`)
   - harsh bounds / low-iteration fit case
3. Expand dataset-backed DCE regression beyond current Tofts map checks (`Ktrans`, `ve`) into ROI table values and additional model maps.
4. Decide whether to port `nested` and `FXL_rr` DCE model flows in the Python CLI or keep them explicitly unsupported.
5. Expand DSC parity work (`DSC_convolution_oSVD`) once DCE dataset regression checks are stable.
6. Performance pass (post-stability/parity lock):
   - Investigate DCE pipeline runtime gap where Python runs are currently about `2x-4x` slower than MATLAB on full runs.
   - Profile Python Stage D hot spots and optimize numerical paths/data layout.
   - Evaluate using GPUfit CPU backend options (in addition to GPU mode) as a fast fitting path.
