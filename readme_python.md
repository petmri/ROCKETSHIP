# ROCKETSHIP Python Usage Guide

This document covers how to run and validate the Python implementation
currently in this repository.

## Current status

- Primary Python entrypoint for DCE pipeline:
  - `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_cli.py`
- Optional GUI entrypoint:
  - `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_gui.py`
- Current implemented workflow focus:
  - DCE parts `A -> B -> D` (single-process, in-memory handoff; CLI-first with optional GUI wrapper)
- Current parity focus:
  - MATLAB-vs-Python parity for model contracts and dataset-backed Tofts maps (`Ktrans`, `ve`)

For detailed port status and todo items, see:
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/PORTING_STATUS.md`

## Environment setup

Recommended setup (default):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
python3 install_python_acceleration.py
```

What this script does:
- creates/reuses `.venv` (use `--recreate-venv` to rebuild)
- installs Python requirements (including GUI by default)
- downloads latest prerelease package from `ironictoo/Gpufit`
- auto-detects host platform/arch and picks matching release asset
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
cd /Users/samuelbarnes/code/ROCKETSHIP
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements_gui.txt
```

Manual acceleration package install (from local wheel/source paths):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pip install /path/to/pyCpufit-*.whl
.venv/bin/python -m pip install /path/to/pyGpufit-*.whl
```

## Run the Python DCE CLI

Use the example config:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py --config tests/python/dce_cli_config.example.json
```

Run with built-in default config template:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py
```

Default template location:
- `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_default.json`
- This default is prewired to the tiny fixture:
  - `/Users/samuelbarnes/code/ROCKETSHIP/test_data/ci_fixtures/dce/tiny_settings_case`
  - outputs to `/Users/samuelbarnes/code/ROCKETSHIP/out/dce_gui_tiny`

Optional runtime overrides:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py \
  --config tests/python/dce_cli_config.example.json \
  --dce-preferences /Users/samuelbarnes/code/ROCKETSHIP/dce/dce_preferences.txt \
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

## Input expectations (DCE)

Config fields are parsed by:
- `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_cli.py`
- `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_pipeline.py`

Key expectations:
- Dynamic image + ROI/AIF/T1/noise masks are provided (NIfTI path lists)
- `stage_overrides` should provide TR/FA/timing parameters for parity-safe runs
- MATLAB-style `dce_preferences.txt` defaults are loaded automatically from `/Users/samuelbarnes/code/ROCKETSHIP/dce/dce_preferences.txt` when present
- Supported backend values: `auto`, `cpu`, `gpufit`
- Backend selection behavior:
  - `auto`: tries `pygpufit` with CUDA first, then `pycpufit` CPU backend, then `pygpufit` CPU fallback, then pure CPU path
  - `cpu`: forces pure Python/Scipy CPU fitting path (no gpufit/cpufit acceleration)
  - `gpufit`: requires `pygpufit` import and then uses CUDA when available, otherwise fallback path
- Current acceleration coverage in Stage D:
  - accelerated: `tofts`, `patlak`
  - non-accelerated (currently pure CPU path): other models
- Stage D logs backend selection on each run:
  - `[DCE] Stage-D backend selection: requested=... selected=... acceleration=... reason=...`
- Supported AIF curve modes: `auto`, `fitted`, `raw`, `imported`
- Static blood-T1 override for Stage A is available via `stage_overrides.blood_t1_ms` (or `blood_t1_sec`)

Preference precedence:
- explicit `stage_overrides` value
- `dce_preferences.txt` value
- Python built-in fallback default

Shared options documentation for CLI + GUI:
- `/Users/samuelbarnes/code/ROCKETSHIP/docs/dce_options.md`

## What is intentionally not supported in Python port scope

- Manual click-based AIF tools
- ImageJ ROI input (`.roi`)
- Legacy MATLAB GUI batch queue/prep flow for part D
- `neuroecon` execution path
- legacy email-completion flow

## Run Python tests

### Unit + integration-style Python tests

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m unittest discover -s tests/python -p 'test_*.py'
```

### Contract parity against MATLAB baseline JSON

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python tests/python/generate_python_results.py --output /tmp/python_results.json
.venv/bin/python tests/python/compare_with_matlab_baseline.py --python-results /tmp/python_results.json --require-all
```

### Dataset-backed DCE pipeline parity (Tofts `Ktrans` + `ve`)

Fast downsample parity:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
ROCKETSHIP_RUN_PIPELINE_PARITY=1 .venv/bin/python -m unittest \
  tests.python.test_dce_pipeline_parity_metrics.TestDcePipelineParityMetrics.test_downsample_bbb_p19_tofts_ktrans
```

Optional VE parity mask threshold (measurable-Ktrans voxels only):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
ROCKETSHIP_RUN_PIPELINE_PARITY=1 ROCKETSHIP_PARITY_VE_KTRANS_MIN=1e-6 .venv/bin/python -m unittest \
  tests.python.test_dce_pipeline_parity_metrics.TestDcePipelineParityMetrics.test_downsample_bbb_p19_tofts_ktrans
```

Notes:
- `ROCKETSHIP_PARITY_VE_KTRANS_MIN` default is `1e-6`
- VE parity is evaluated only where both MATLAB and Python Ktrans exceed that threshold

Optional full-volume parity (slower; reserve for occasional thorough checks):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
ROCKETSHIP_RUN_PIPELINE_PARITY=1 ROCKETSHIP_RUN_FULL_VOLUME_PARITY=1 .venv/bin/python -m unittest \
  tests.python.test_dce_pipeline_parity_metrics.TestDcePipelineParityMetrics.test_full_bbb_p19_tofts_ktrans
```

Default downsample fixture used for parity:
- `/Users/samuelbarnes/code/ROCKETSHIP/test_data/ci_fixtures/dce/bbb_p19_downsample_x3y3`

### Tiny settings matrix (fast)

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python tests/python/generate_tiny_dce_settings_fixture.py --clean
.venv/bin/python -m unittest tests.python.test_dce_pipeline_settings_matrix -v
```

## CI behavior (high level)

Workflow:
- `/Users/samuelbarnes/code/ROCKETSHIP/.github/workflows/run_DCE.yml`

CI currently runs:
- Python unit tests
- Python contract parity checks
- Python downsample dataset parity check
- MATLAB unit/integration checks
- MATLAB DCE smoke/full jobs (event-dependent)

## GUI (PySide6) v1

Install GUI dependency:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pip install -r requirements_gui.txt
```

Launch GUI:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_gui.py
```

One-click test run:
- Launch the GUI and click `Run DCE` without changing fields.
- It uses `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_default.json` and the tiny fixture by default.

GUI v1 behavior:
- Edits common top-level config fields + all `stage_overrides` keys.
- Runs CLI in a subprocess and streams progress from stdout events.
- Uses hard-stop process termination.
- Displays QC PNG figures when generated.
- Provides `Browse...` dialogs for all path/file input fields in the form.
- If `aif_mode=imported` is needed, set `imported_aif_path` in JSON config (current GUI form does not expose it yet).
