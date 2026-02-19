# Python Parity Runner

This directory contains Python parity tooling and first model ports.
For a resume-later snapshot, see:
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/PORTING_STATUS.md`

Current focus:
- End-to-end DCE CLI port for parts `A`, `B`, and `D`.
- Default execution: single-process with in-memory stage handoff.
- Deprecated/not being ported: `neuroecon`, GUI batch queue flow, manual click-based AIF tools, ImageJ `.roi` input, email notifications.
- Retained output compatibility: ROI spreadsheet exports (`.xls`).
- GPUfit/Cpufit acceleration is supported as an optional backend; CPU remains the parity baseline.

## Environment setup (macOS)
This project is now configured to use a local `venv` at
`/Users/samuelbarnes/code/ROCKETSHIP/.venv`.

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

## Acceleration installer (Python + MATLAB MEX)

Install/update acceleration packages from release bundles:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python install_python_acceleration.py
```

Current installer contract:
- Installs Python `pyCpufit` + `pyGpufit` from release assets.
- Installs MATLAB files from bundle `matlab/` into `external_programs/`.
- Probes MATLAB for required GPUfit symbols:
  - `GpufitCudaAvailableMex`
  - `gpufit_constrained`
  - `gpufit_constraints`
- Fails by default if those MATLAB GPUfit symbols are missing in the release bundle.

Temporary bypass (until release bundles are updated):

```bash
.venv/bin/python install_python_acceleration.py --allow-missing-matlab-gpufit
```

## Script
- `compare_with_matlab_baseline.py`
- `generate_python_results.py`
- `benchmark_dce_pipeline.py` (full-pipeline runtime benchmark across MATLAB/Python backends)
- `run_dce_python_cli.py` (repo root wrapper for in-memory A->B->D pipeline)
- `run_dce_python_gui.py` (repo root wrapper for PySide6 GUI)

## Full-pipeline benchmark (fast fixture)
Benchmark as many backend configurations as are available on the machine:
- `matlab_cpu`, `matlab_gpufit`, `python_cpu`, `python_cpufit`, `python_gpufit`
- unavailable configs are reported as `SKIP`

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python tests/python/benchmark_dce_pipeline.py
```

Useful options:

```bash
.venv/bin/python tests/python/benchmark_dce_pipeline.py \
  --repeats 3 \
  --models patlak \
  --configs matlab_cpu,python_cpu,python_cpufit,python_gpufit \
  --output-json /tmp/dce_bench_fast.json
```

## First model port
- Source: `python/dce_models.py`
- Function: `model_tofts_cfit`
- Function: `model_patlak_cfit`
- Function: `model_extended_tofts_cfit`
- Function: `model_vp_cfit`
- Function: `model_tissue_uptake_cfit`
- Function: `model_2cxm_cfit`
- Function: `model_fxr_cfit`
- Function: `model_patlak_linear`
- Function: `model_tofts_fit`
- Function: `model_vp_fit`
- Function: `model_tissue_uptake_fit`
- Function: `model_2cxm_fit`
- Function: `model_fxr_fit`
- Source: `python/dsc_helpers.py`
- Function: `import_aif`
- Function: `previous_aif`
- Source: `python/dsc_models.py`
- Function: `dsc_convolution_ssvd`
- Source: `python/parametric_models.py`
- Function: `t2_linear_fast`
- Function: `t1_fa_linear_fit`

Run Python reliability suite (no MATLAB required; parity-gated tests remain skipped unless explicitly enabled):

```bash
.venv/bin/python -m pytest tests/python -q
```

Targeted Phase-1 reliability tests:

```bash
.venv/bin/python -m pytest \
  tests/python/test_install_python_acceleration.py \
  tests/python/test_dce_pipeline_contracts.py -v
```

## What it does
- Loads MATLAB baseline outputs from `tests/contracts/baselines/matlab_reference_v1.json`
- Loads contracts/tolerances from `tests/contracts/`
- Compares Python outputs to MATLAB baseline by contract ID
- Prints pass/fail summary per contract

## Generate Python outputs from current ports

```bash
.venv/bin/python tests/python/generate_python_results.py \
  --output /tmp/python_results_tofts.json
```

## Generate template Python results JSON
Use this before implementing Python models:

```bash
python3 tests/python/compare_with_matlab_baseline.py \
  --write-template tests/python/python_results_template.json
```

## Compare Python results against MATLAB baseline

```bash
.venv/bin/python tests/python/compare_with_matlab_baseline.py \
  --python-results /tmp/python_results_tofts.json
```

Add `--require-all` to fail if any mapped contract output is missing.

## DCE CLI pipeline (single-process, in-memory)
Example config:
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/dce_cli_config.example.json`

Run:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py --config tests/python/dce_cli_config.example.json
```

Run with default config template:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py
```

Optional CLI overrides (repeatable `--set KEY=VALUE`):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py \
  --config tests/python/dce_cli_config.example.json \
  --dce-preferences /Users/samuelbarnes/code/ROCKETSHIP/dce/dce_preferences.txt \
  --set voxel_MaxFunEvals=100 \
  --set blood_t1_ms=1600
```

Current pipeline behavior:
- Runs A->B->D stage order in-memory with no `.mat` handoff files.
- Stage A has a real implementation path (NIfTI + mask-driven Cp/Ct extraction).
- Stage B has a real non-GUI implementation path (timer restriction, AIF `fitted|raw|imported`, and AIF QC figure output).
- Stage D has a real non-GUI implementation path (model fitting, parameter maps, and `.xls` ROI table output).
- Loads MATLAB-style `dce_preferences.txt` defaults (explicit `stage_overrides` values still take precedence).
- Writes run summary to `<output_dir>/dce_pipeline_run.json`.
- Writes event log to `<output_dir>/dce_pipeline_events.jsonl`.
- Optionally writes per-stage checkpoints (`a_out.json`, `b_out.json`, `d_out.json`) if `checkpoint_dir` is set.
- Enforces scope decisions (for example, rejects ImageJ `.roi` inputs).
- Saves QC figures during Stage A and Stage B real runs (for example `dce_timecurves.png`, `dce_roi_overview.png`, `dce_aif_fitting.png`).
- Writes per-model DCE maps (NIfTI when possible; fallback `.npy`) for supported model parameters.

Shared options doc (CLI + GUI):
- `/Users/samuelbarnes/code/ROCKETSHIP/docs/dce_options.md`

## GUI v1 (PySide6)
Install:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pip install -r requirements_gui.txt
```

Run:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_gui.py
```

GUI behavior highlights:
- Path/file inputs in the form have `Browse...` actions.
- One-click smoke run works with default config (`python/dce_default.json`) against tiny fixture data.
- Current GUI form does not expose `imported_aif_path`; set it in JSON when using `aif_mode=imported`.

## Dataset-backed DCE parity checks
Default downsample parity fixture is committed for CI/local use:
- `test_data/ci_fixtures/dce/bbb_p19_downsample_x3y3`

Optional: regenerate a fast real-data fixture (nearest-neighbor downsample of `BBB data p19`):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python tests/python/generate_bbb_p19_downsample.py --clean --factor-x 3 --factor-y 3
```

Generate MATLAB parity baselines.
Tofts-only (default):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
matlab -batch "cd('/Users/samuelbarnes/code/ROCKETSHIP'); addpath('tests/matlab'); generate_dce_tofts_parity_map('subjectRoot','/Users/samuelbarnes/code/ROCKETSHIP/test_data/synthetic/generated/bbb_p19_downsample_x3y3')"
matlab -batch "cd('/Users/samuelbarnes/code/ROCKETSHIP'); addpath('tests/matlab'); generate_dce_tofts_parity_map('subjectRoot','/Users/samuelbarnes/code/ROCKETSHIP/test_data/BBB data p19')"
```

Multi-model baselines (for `tofts`, `ex_tofts`, `patlak`, `tissue_uptake`, `2cxm`):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
matlab -batch "cd('/Users/samuelbarnes/code/ROCKETSHIP'); addpath('tests/matlab'); generate_dce_tofts_parity_map('subjectRoot','/Users/samuelbarnes/code/ROCKETSHIP/test_data/ci_fixtures/dce/bbb_p19_downsample_x3y3','models',{'tofts','ex_tofts','patlak','tissue_uptake','2cxm'})"
```

Run downsampled full-pipeline Tofts parity test (Python vs MATLAB maps for `Ktrans` and `ve`, with corr/MSE tolerances):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pytest \
  tests/python/test_dce_pipeline_parity_metrics.py::test_downsample_bbb_p19_tofts_ktrans \
  --run-parity
```

Optional full-volume parity test (slower; reserve for occasional thorough checks):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pytest \
  tests/python/test_dce_pipeline_parity_metrics.py::test_full_bbb_p19_tofts_ktrans \
  --run-parity --run-full-parity
```

Optional multi-model backend parity (CPU-only vs auto backend vs MATLAB maps).
This is opt-in because it is heavier than the Tofts-only check:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pytest \
  tests/python/test_dce_pipeline_parity_metrics.py::test_downsample_bbb_p19_models_cpu_and_auto \
  --run-parity --run-multi-model-backend-parity
```

Default multi-model behavior:
- Required (gating) checks: `tofts`, `ex_tofts`, `patlak` (with `patlak` CPU-vs-MATLAB and AUTO-vs-CPU kept diagnostic-only by default).
- Diagnostic-only checks: `tissue_uptake`, `2cxm`, and other non-required model checks. These are reported in logs but do not fail the test unless explicitly promoted.

Cross-platform helper runner (avoids manual env var setup):

```bash
cd /path/to/ROCKETSHIP
python tests/python/run_dce_parity.py --suite multi-model
```

Examples:

```bash
python tests/python/run_dce_parity.py --suite tofts-downsample
python tests/python/run_dce_parity.py --suite tofts-full --full-root "/path/to/ROCKETSHIP/test_data/BBB data p19"
python tests/python/run_dce_parity.py --suite multi-model --dataset-root "/path/to/ROCKETSHIP/test_data/ci_fixtures/dce/bbb_p19_downsample_x3y3" --roi-stride 12
python tests/python/run_dce_parity.py --suite multi-model --show-warnings
```

The runner suppresses deprecation warnings by default to keep parity output readable.

Synthetic stability sweeps for unstable models (no MATLAB required):

```bash
cd /path/to/ROCKETSHIP
python tests/python/tune_dce_unstable_models.py --model both --curves 120 --noise-std "0,2e-4,5e-4"
```

Model-specific parity tuning env vars (applied only to the named model):
- `ROCKETSHIP_PARITY_TISSUE_*` for `tissue_uptake`
- `ROCKETSHIP_PARITY_2CXM_*` for `2cxm`

Examples:
- bounds/init: `..._LOWER_LIMIT_*`, `..._UPPER_LIMIT_*`, `..._INITIAL_VALUE_*`
- optimizer: `ROCKETSHIP_PARITY_TISSUE_MAXFUNEVALS`, `ROCKETSHIP_PARITY_TISSUE_MAXITER`, `ROCKETSHIP_PARITY_TISSUE_ROBUST`
- optimizer: `ROCKETSHIP_PARITY_2CXM_MAXFUNEVALS`, `ROCKETSHIP_PARITY_2CXM_MAXITER`, `ROCKETSHIP_PARITY_2CXM_ROBUST`

This check is intended to expose remaining model parity gaps; tune thresholds with env vars below as needed during bring-up.

Tune parity thresholds via environment variables:

```bash
ROCKETSHIP_PARITY_DOWNSAMPLED_KTRANS_CORR_MIN=0.99
ROCKETSHIP_PARITY_DOWNSAMPLED_KTRANS_MSE_MAX=0.001
ROCKETSHIP_PARITY_DOWNSAMPLED_VE_CORR_MIN=0.97
ROCKETSHIP_PARITY_DOWNSAMPLED_VE_MSE_MAX=0.002
ROCKETSHIP_PARITY_FULL_KTRANS_CORR_MIN=0.99
ROCKETSHIP_PARITY_FULL_KTRANS_MSE_MAX=0.001
ROCKETSHIP_PARITY_FULL_VE_CORR_MIN=0.97
ROCKETSHIP_PARITY_FULL_VE_MSE_MAX=0.002
ROCKETSHIP_PARITY_MODEL_KTRANS_CORR_MIN=0.95
ROCKETSHIP_PARITY_MODEL_KTRANS_MSE_MAX=0.01
ROCKETSHIP_PARITY_MODEL_PARAM_CORR_MIN=0.90
ROCKETSHIP_PARITY_MODEL_PARAM_MSE_MAX=0.02
ROCKETSHIP_PARITY_CPU_AUTO_KTRANS_CORR_MIN=0.98
ROCKETSHIP_PARITY_CPU_AUTO_KTRANS_MSE_MAX=0.002
ROCKETSHIP_PARITY_CPU_AUTO_PARAM_CORR_MIN=0.95
ROCKETSHIP_PARITY_CPU_AUTO_PARAM_MSE_MAX=0.01
ROCKETSHIP_PARITY_EX_TOFTS_KTRANS_CORR_MIN=0.85
ROCKETSHIP_PARITY_KTRANS_UPPER_EXCLUDE=1.9
ROCKETSHIP_PARITY_REQUIRED_MODELS=tofts,ex_tofts,patlak
ROCKETSHIP_PARITY_CPU_OPTIONAL_MODELS=patlak
ROCKETSHIP_PARITY_REQUIRE_ALL_MODELS=0
ROCKETSHIP_PARITY_MULTI_MODEL_ROI_STRIDE=12
```

## Tiny settings-matrix tests (very fast)
Generate or refresh tiny fixture:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python tests/python/generate_tiny_dce_settings_fixture.py --clean
```

Run settings/feature coverage tests (constraints, initial guesses, blood T1 override):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pytest tests/python/test_dce_pipeline_settings_matrix.py -v
```
