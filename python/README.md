# ROCKETSHIP Python Usage Guide

This document covers how to run and validate the Python implementation
currently in this repository.

## Current status

- Primary Python entrypoint for DCE pipeline:
  - `/path/to/ROCKETSHIP/run_dce_python_cli.py`
- Primary Python entrypoint for parametric T1 pipeline:
  - `/path/to/ROCKETSHIP/run_parametric_python_cli.py`
- Optional GUI entrypoints:
  - `/path/to/ROCKETSHIP/run_dce_python_gui.py`
  - `/path/to/ROCKETSHIP/run_parametric_python_gui.py`
- BIDS dataset/session discovery utility:
  - `/path/to/ROCKETSHIP/run_bids_discovery.py`
- Current implemented workflow focus:
  - DCE parts `A -> B -> D` (single-process, in-memory handoff; CLI-first with optional GUI wrapper)
  - Parametric VFA T1 mapping (linear, nonlinear, and two-point fit types; CLI + GUI v1)
- Current parity focus:
  - MATLAB-vs-Python parity for model contracts and dataset-backed Tofts maps (`Ktrans`, `ve`)
  - MATLAB-vs-Python parity tests for parametric core fitters (`t2_linear_fast`, `t1_fa_linear_fit`, `t1_fa_fit`) via unit parity tests and contract-runner checks
  - OSIPI-labeled reliability tests for DCE, T1, and SI-to-concentration conversion

For detailed port status and todo items, see:

- `/path/to/ROCKETSHIP/python/ROADMAP.md`
- `/path/to/ROCKETSHIP/python/PORTING_STATUS.md`
- `/path/to/ROCKETSHIP/TODO.md`

## Environment setup

Recommended setup (default):

```bash
cd /path/to/ROCKETSHIP
python3 install_python_acceleration.py
```

What this script does:

- creates/reuses `.venv` (use `--recreate-venv` to rebuild)
- installs Python requirements (including GUI by default)
- downloads latest stable release package from `ironictoo/Gpufit`
- auto-detects host platform/arch and picks matching release asset
- detects local CUDA version (when available) and prefers the closest matching CUDA asset for your host
- falls back to CPU asset IDs when CUDA builds are not a good local match
- installs both `pyCpufit` and `pyGpufit` into the venv
- verifies imports and reports CUDA availability

Common installer options:

- `--release-tag <tag>`: pin to a specific Gpufit release
- `--asset-id <id>`: force specific asset id
- `--venv-path <path>`: custom venv path
- `--no-gui`: skip GUI dependency install

### Manual setup

Use this path only if you do not want to use the automated installer.

```bash
cd /path/to/ROCKETSHIP
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements_gui.txt
```

Manual acceleration package install (from local wheel/source paths):

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pip install /path/to/pyCpufit-*.whl
.venv/bin/python -m pip install /path/to/pyGpufit-*.whl
```

## Run the Python DCE CLI

Use the example config:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py --config tests/python/dce_cli_config.example.json
```

Run with built-in default config template:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py
```

Default template location:

- `/path/to/ROCKETSHIP/python/dce_default.json`
- This default is prewired to the tiny fixture:
  - `/path/to/ROCKETSHIP/tests/data/ci_fixtures/dce/tiny_settings_case`
  - outputs to `/path/to/ROCKETSHIP/out/dce_gui_tiny`

Optional runtime overrides:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py \
  --config tests/python/dce_cli_config.example.json \
  --dce-preferences /path/to/ROCKETSHIP/dce/dce_preferences.txt \
  --set voxel_MaxFunEvals=100 \
  --set blood_t1_ms=1600
```

Typical outputs:

- Stage summary JSON: `<output_dir>/dce_pipeline_run.json`
- Event log JSONL: `<output_dir>/dce_pipeline_events.jsonl`
- Stage checkpoints (optional): `<checkpoint_dir>/a_out.json`, `b_out.json`, `d_out.json`
- DCE model maps (NIfTI when possible; fallback `.npy`)
- ROI spreadsheet output (`.xls`) for ROI-enabled runs
- QC figures for Stage A/B real runs

## Run the Python Parametric T1 CLI

Run with built-in default config template:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_parametric_python_cli.py
```

Default template location:

- `/path/to/ROCKETSHIP/python/parametric_default.json`

Typical outputs:

- Run summary JSON: `<output_dir>/parametric_t1_run.json`
- Event log JSONL: `<output_dir>/parametric_t1_events.jsonl`
- T1 map NIfTI: `<output_dir>/T1_map_<fit_type>_<label>.nii.gz`
- R-squared map NIfTI: `<output_dir>/Rsquared_<fit_type>_<label>.nii.gz`

Parametric input notes:

- `fit_type` supports `t1_fa_linear_fit`, `t1_fa_fit`, and `t1_fa_two_point_fit`.
- `b1_map_file` is optional; when provided, per-voxel effective flip angles are `flip_angles_deg * b1_scale`.
- If `b1_map_file` is omitted, the pipeline auto-detects `B1_scaled_FAreg.nii` or `B1_scaled_FAreg.nii.gz` in the VFA directory.
- `tr_ms` is optional only when VFA sidecars contain `RepetitionTime`; if sidecar TR is missing, `tr_ms` must be explicitly provided.
- `odd_echoes=true` keeps only odd-positioned samples from the VFA stack (indices `0,2,4,...`) before fitting, matching MATLAB workflow behavior.
- `xy_smooth_sigma` (alias `xy_smooth_size`) applies optional per-frame XY Gaussian smoothing before fitting.

## Discover BIDS datasets/sessions

Generate a manifest of all discoverable sessions under a BIDS root:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_bids_discovery.py \
  --bids-root tests/data/BIDS_test \
  --output-json out/bids_manifest.json \
  --print-json
```

This utility reads `rawdata/` and `derivatives/` and emits subject/session pairs that
exist in both trees, so other tools can run over the same discovered set.

## Batch processing DCE across BIDS datasets

Process all (or filtered) sessions in a BIDS dataset:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_bids_batch.py \
  --bids-root /path/to/bids_root \
  --pipeline-folder dceprep \
  --backend gpufit
```

Batch processor features:

- **BIDS-native derivatives**: Use `--pipeline-folder` to read/write in `derivatives/{pipeline}/sub-X/ses-Y/` structure
- **Automatic file discovery**: Scans `derivatives/{pipeline}/sub-*/[ses-*]/` for dceprep-standard files
- **Flexible configuration**: Load base config template and override specific values
- **Session filtering**: Run specific subjects/sessions with `--subject sub-01 --session ses-baseline`
- **Batch aggregation**: Writes summary to `derivatives/reports/{pipeline}/batch_summary_YYYYMMDD.json` with per-session status
- **Error resilience**: Continues on single-session failures, reports aggregate statistics
- **Provenance tracking**: Each session outputs pipeline run report to `sub-X/ses-Y/reports/` with BIDS-compliant naming

Output organization:

- `--pipeline-folder dceprep`: Reads from and writes to `bids_root/derivatives/dceprep/sub-X/ses-Y/` BIDS structure
- `--pipeline-folder dceprep --output-pipeline dce-postproc`: Reads from dceprep, writes to dce-postproc
- `--output-root /path`: Uses custom flat directory `{output-root}/sub-X_ses-Y/` structure
- Neither specified: Defaults to flat `bids_root/derivatives/dce_batch_output/sub-X_ses-Y/`

Configuration precedence (lowest to highest priority):

1. Hardcoded defaults (sensible baseline values)
2. Config template JSON (`--config-template`)
3. Auto-discovered BIDS inputs (override file paths)
4. CLI `--set` overrides (stage_overrides keys)
5. CLI `--backend` and `--dce-models` (highest priority)

Example: BIDS-native run with config template:

```bash
.venv/bin/python run_dce_bids_batch.py \
  --bids-root /path/to/data \
  --pipeline-folder dceprep \
  --config-template python/dce_default.json \
  --set blood_t1_ms=1600 \
  --set aif_curve_mode=fitted \
  --backend auto
```

Example: Process only session 1 with specific models:

```bash
.venv/bin/python run_dce_bids_batch.py \
  --bids-root /path/to/data \
  --pipeline-folder dceprep \
  --session ses-01 \
  --dce-models tofts,ex_tofts,patlak
```

Typical batch output structure (with --pipeline-folder):

```
bids_root/derivatives/
├── reports/
│   └── dceprep/
│       └── batch_summary_20260227.json  # Overall batch status and per-session results
└── dceprep/
    ├── sub-01/
    │   └── ses-baseline/
    │       ├── reports/
    │       │   └── sub-01_ses-baseline_desc-provenance.json
    │       ├── dce/
    │       │   ├── sub-01_ses-baseline_patlak_fit_Ktrans.nii.gz
    │       │   ├── sub-01_ses-baseline_patlak_fit_rois.xls
    │       │   └── [other model outputs...]
    │       └── dce_pipeline_events.jsonl
    ├── sub-02/
    │   └── ses-baseline/
    │       └── [same structure...]
    └── ...
```

Or with flat `--output-root` structure:

```
output_root/
├── batch_summary.json
├── sub-01_ses-baseline/
│   ├── dce_pipeline_run.json
│   └── [outputs...]
└── ...
```

## Input expectations (DCE)

Config fields are parsed by:

- `/path/to/ROCKETSHIP/python/dce_cli.py`
- `/path/to/ROCKETSHIP/python/dce_pipeline.py`

Key expectations:

- **Configuration priority**: Python workflows use `dce_default.json` as the primary config. Optional MATLAB-style `dce_preferences.txt` can override defaults. The precedence chain is:
  - `dce_default.json` (base defaults)
  - `dce_preferences.txt` (optional legacy override from `dce/dce_preferences.txt` or `./dce_preferences.txt`)
  - `stage_overrides` in JSON config (explicit overrides)
  - CLI arguments (final overrides)
- Dynamic image + ROI/AIF/T1/noise masks are provided (NIfTI path lists)
- TR/FA/time-resolution for real Stage A are resolved strictly:
  - preferred from DCE metadata JSON sidecar (or explicit `stage_overrides.dce_metadata_path`)
  - if no metadata JSON is available, you must provide all three manually in `stage_overrides`
    (`tr_ms`/`tr_sec`, `fa_deg`/`fa`, `time_resolution_sec`/`time_resolution`)
  - partial manual override when metadata JSON is present is rejected (set all three or none)
  - there is no fallback to repository preference files for scan parameters
- Supported backend values: `auto`, `cpu`, `gpufit`
- Backend selection behavior:
  - `auto`: tries `pygpufit` with CUDA first, then `pycpufit` CPU backend, then pure CPU path
  - `cpu`: forces pure Python/Scipy CPU fitting path (no gpufit/cpufit acceleration)
  - `gpufit`: requires `pygpufit` import and then uses CUDA when available, otherwise fallback to `pycpufit`
- Current acceleration coverage in Stage D:
  - accelerated: `tofts`, `patlak`, `extended tofts`, `2cxm`, `tissue uptake`
  - non-accelerated (currently pure CPU path): other models
- Stage D logs backend selection on each run:
  - `[DCE] Stage-D backend selection: requested=... selected=... acceleration=... reason=...`
- Supported AIF curve modes: `auto`, `fitted`, `raw`, `imported`
- Static blood-T1 override for Stage A is available via `stage_overrides.blood_t1_ms` (or `blood_t1_sec`)

Part E work-in-progress:

- Python post-fit statistical core is available in `/path/to/ROCKETSHIP/python/dce_postfit_analysis.py`.
- Current coverage includes f-test and AIC/relative-likelihood helpers plus ROI CSV and voxel-map reconstruction utilities.
- Reproducible output helpers are available via `run_ftest_analysis(...)` and `run_aic_analysis(...)` (JSON/CSV/NPY artifacts).
- Part E outputs now include statistical summary fields and optional plot PNGs for troubleshooting (`p` histograms, model-count histograms, best-vs-second likelihood histograms).
- Stage D can optionally export Part E-ready fit arrays (`*_postfit_arrays.npz`) using `stage_overrides.write_postfit_arrays=true`.
- NPZ loader is available via `load_dce_fit_stats_from_npz(...)` with runner script `/path/to/ROCKETSHIP/tests/python/run_dce_postfit_analysis.py`.

Preference precedence (highest to lowest):

1. CLI arguments (e.g., `--roi-mask-path`)
2. `stage_overrides` in JSON config
3. MATLAB-style `dce_preferences.txt` (optional legacy file)
4. `dce_default.json` base values
5. Python built-in fallback defaults

Shared options documentation for CLI + GUI:

- `/Users/samuelbarnes/code/ROCKETSHIP/docs/dce_options.md`

## Pipeline outputs and provenance

Each DCE run generates `dce_pipeline_run.json` in the output directory with the following structure:

```json
{
  "meta": {
    "pipeline": "dce_cli_in_memory",
    "status": "ok",
    "single_process": true,
    "duration_sec": 42.3,
    "dce_preferences_path": null,
    "summary_path": "/path/to/dce_pipeline_run.json"
  },
  "provenance": {
    "execution_timestamp": "2026-02-26T15:30:00+00:00",
    "duration_sec": 42.3,
    "inputs": {
      "dynamic": ["/path/to/dynamic_image.nii.gz"],
      "aif_mask": ["/path/to/aif_mask.nii.gz"],
      "roi_mask": ["/path/to/roi_mask.nii.gz"],
      "t1_map": ["/path/to/t1_map.nii.gz"],
      "noise_mask": null
    },
    "backend_requested": "auto",
    "backend_used": "cpufit_cpu"
  },
  "config": { /* full resolved configuration */ },
  "stages": {
    "A": { /* Stage A output metadata */ },
    "B": { /* Stage B output metadata */ },
    "D": { /* Stage D output metadata */ }
  }
}
```

The provenance section provides:

- **execution_timestamp**: ISO 8601 timestamp for reproducibility and audit trails
- **duration_sec**: Wall-clock execution time (useful for performance analysis)
- **inputs**: Full resolved paths to all input files (supports tracing data lineage)
- **backend_requested** + **backend_used**: Documents actual backend selection for each run
  - Helpful for debugging when `--backend auto` chains multiple fallbacks

## What is intentionally not supported in Python port scope

- Manual click-based AIF tools
- ImageJ ROI input (`.roi`)
- Legacy MATLAB GUI batch queue/prep flow for part D
- `neuroecon` execution path
- legacy email-completion flow

## Run Python tests

### Unit + integration-style Python tests

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest tests/python -q
```

Coverage summary + XML report:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest tests/python -q \
  --cov=python \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=60
```

This includes Phase-1 reliability coverage such as installer asset-selection logic and
pipeline output/event contract checks (`tests/python/test_install_python_acceleration.py`,
`tests/python/test_dce_pipeline_contracts.py`).

### Contract parity against MATLAB baseline JSON

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python tests/contracts/generate_python_results.py --output /tmp/python_results.json
.venv/bin/python tests/contracts/compare_with_matlab_baseline.py --python-results /tmp/python_results.json --require-all
```

### Dataset-backed DCE pipeline parity (Tofts `Ktrans` + `ve`)

Fast downsample parity:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest \
  tests/python/test_dce_pipeline_parity_metrics.py::test_downsample_bbb_p19_tofts_ktrans \
  --parity
```

Optional VE parity mask threshold (measurable-Ktrans voxels only):

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest \
  tests/python/test_dce_pipeline_parity_metrics.py::test_downsample_bbb_p19_tofts_ktrans \
  --parity \
  --parity-ve-ktrans-min 1e-6
```

Notes:

- `--parity-ve-ktrans-min` default is `1e-6`
- VE parity is evaluated only where both MATLAB and Python Ktrans exceed that threshold

Optional full-volume parity (slower; reserve for occasional thorough checks):

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest \
  tests/python/test_dce_pipeline_parity_metrics.py::test_full_bbb_p19_tofts_ktrans \
  --parity --full-parity
```

Optional CPU model-map + ROI table parity (`tofts`, `ex_tofts`, `patlak`, `tissue_uptake`):

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest \
  tests/python/test_dce_pipeline_parity_metrics.py::test_downsample_bbb_p19_model_maps_and_roi_xls_cpu \
  --parity
```

Default downsample fixture used for parity:

- `/path/to/ROCKETSHIP/tests/data/ci_fixtures/dce/bbb_p19_downsample_x3y3`

### Tiny settings matrix (fast)

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python tests/data/scripts/generate_tiny_dce_settings_fixture.py --clean
.venv/bin/python -m pytest tests/python/test_dce_pipeline_settings_matrix.py -v
```

### OSIPI T1 + SI-to-concentration checks

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest \
  tests/python/test_osipi_t1_reliability.py \
  tests/python/test_osipi_si_to_conc_reliability.py \
  -v
```

### OSIPI primary backend consistency checks

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest tests/python/test_osipi_backend_consistency.py -v
```

### OSIPI primary merge-gate reliability summary

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python tests/python/run_osipi_reliability.py \
  --suite all \
  --summary-json /tmp/osipi_primary_reliability_summary.json
```

## CI behavior (high level)

Workflow:

- `/path/to/ROCKETSHIP/.github/workflows/run_DCE.yml`

CI currently runs:

- Python unit tests
- Python contract parity checks
- Python downsample dataset parity check
- Python cross-platform portability checks on Windows and macOS (non-parity mode)
- MATLAB unit/integration checks
- MATLAB DCE smoke/full jobs (event-dependent)

## GUI (PySide6) v1

Install GUI dependency:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pip install -r requirements_gui.txt
```

Launch GUI:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_dce_python_gui.py
```

Launch parametric T1 GUI:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_parametric_python_gui.py
```

One-click test run:

- Launch the GUI and click `Run DCE` without changing fields.
- It uses `/path/to/ROCKETSHIP/python/dce_default.json` and the tiny fixture by default.

GUI v1 behavior:

- Edits common top-level config fields + all `stage_overrides` keys.
- Runs CLI in a subprocess and streams progress from stdout events.
- Uses hard-stop process termination.
- Displays QC PNG figures when generated.
- Provides `Browse...` dialogs for all path/file input fields in the form.
- If `aif_mode=imported` is needed, set `imported_aif_path` in JSON config (current GUI form does not expose it yet).

Parametric GUI v1 behavior:

- Edits the parametric T1 config (`vfa_files`, flip angles, TR, thresholds, B1 map, script-preferences path, output controls).
- Runs `run_parametric_python_cli.py` in a subprocess and streams event progress.
- Shows summary metrics from `parametric_t1_run.json` and lists output artifact paths.
