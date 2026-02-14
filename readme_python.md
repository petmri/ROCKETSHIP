# ROCKETSHIP Python Usage Guide

This document covers how to run and validate the Python implementation
currently in this repository.

## Current status

- Primary Python entrypoint for DCE pipeline:
  - `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_cli.py`
- Current implemented workflow focus:
  - DCE parts `A -> B -> D` (non-GUI, single-process, in-memory handoff)
- Current parity focus:
  - MATLAB-vs-Python parity for model contracts and dataset-backed Tofts maps (`Ktrans`, `ve`)

For detailed port status and todo items, see:
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/PORTING_STATUS.md`

## Environment setup

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

## Run the Python DCE CLI

Use the example config:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py --config tests/python/dce_cli_config.example.json
```

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
- Stage checkpoints (optional): `<checkpoint_dir>/a_out.json`, `b_out.json`, `d_out.json`
- DCE model maps (NIfTI when possible; fallback `.npy`)
- ROI spreadsheet output (`.xls`) for ROI-enabled runs
- QC figures for Stage A/B real runs

## Input expectations (DCE)

Config fields are parsed by:
- `/Users/samuelbarnes/code/ROCKETSHIP/python/rocketship/dce_cli.py`
- `/Users/samuelbarnes/code/ROCKETSHIP/python/rocketship/dce_pipeline.py`

Key expectations:
- Dynamic image + ROI/AIF/T1/noise masks are provided (NIfTI path lists)
- `stage_overrides` should provide TR/FA/timing parameters for parity-safe runs
- MATLAB-style `dce_preferences.txt` defaults are loaded automatically from `/Users/samuelbarnes/code/ROCKETSHIP/dce/dce_preferences.txt` when present
- Supported backend values: `auto`, `cpu`, `gpufit`
- Supported AIF curve modes: `auto`, `fitted`, `raw`, `imported`
- Static blood-T1 override for Stage A is available via `stage_overrides.blood_t1_ms` (or `blood_t1_sec`)

Preference precedence:
- explicit `stage_overrides` value
- `dce_preferences.txt` value
- Python built-in fallback default

## What is intentionally not supported in Python port scope

- GUI entrypoints and GUI helper flows
- Manual click-based AIF tools
- ImageJ ROI input (`.roi`)
- MATLAB batch helper queue flow
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
