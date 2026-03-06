# Python Walkthrough

This walkthrough is derived from the Python usage guide in `/Users/samuelbarnes/code/ROCKETSHIP/python/README.md` and focuses on practical run steps.

## Scope

Python entrypoints in this repo:

- DCE CLI: `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_cli.py`
- Parametric T1 CLI: `/Users/samuelbarnes/code/ROCKETSHIP/run_parametric_python_cli.py`
- DCE GUI: `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_gui.py`
- Parametric GUI: `/Users/samuelbarnes/code/ROCKETSHIP/run_parametric_python_gui.py`
- BIDS discovery: `/Users/samuelbarnes/code/ROCKETSHIP/run_bids_discovery.py`

Current workflow emphasis:

- DCE A->B->D pipeline
- Parametric VFA T1 mapping
- Reliability and parity testing while porting from MATLAB

## 1. Environment Setup

Recommended setup:

```bash
cd /path/to/ROCKETSHIP
python3 install_python_acceleration.py
```

This creates `.venv`, installs dependencies, and attempts to install acceleration packages (`pyCpufit`, `pyGpufit`).

Manual setup alternative:

```bash
cd /path/to/ROCKETSHIP
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements_gui.txt
```

## 2. Run DCE CLI

Default run (uses `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_default.json`):

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py
```

Run with explicit config:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py --config tests/python/dce_cli_config.example.json
```

Run with preference file + runtime overrides:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_dce_python_cli.py \
  --config tests/python/dce_cli_config.example.json \
  --dce-preferences /path/to/ROCKETSHIP/dce/dce_preferences.txt \
  --set voxel_MaxFunEvals=100 \
  --set blood_t1_ms=1600
```

Typical outputs:

- `dce_pipeline_run.json`
- `dce_pipeline_events.jsonl`
- Stage checkpoints (`a_out.json`, `b_out.json`, `d_out.json`) when enabled
- DCE parameter maps
- ROI spreadsheet output for ROI-enabled runs

## 3. Run Parametric T1 CLI

Default run (uses `/Users/samuelbarnes/code/ROCKETSHIP/python/parametric_default.json`):

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_parametric_python_cli.py
```

Typical outputs:

- `parametric_t1_run.json`
- `parametric_t1_events.jsonl`
- T1 map outputs
- R-squared map outputs

## 4. Discover BIDS Sessions

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python run_bids_discovery.py \
  --bids-root tests/data/BIDS_test \
  --output-json out/bids_manifest.json \
  --print-json
```

## 5. Run Python GUI

DCE GUI:

```bash
cd /path/to/ROCKETSHIP
source .venv/bin/activate
python run_dce_python_gui.py
```

Parametric GUI:

```bash
cd /path/to/ROCKETSHIP
source .venv/bin/activate
python run_parametric_python_gui.py
```

## 6. Run Tests

All Python tests:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python -m pytest tests/python -q
```

Contract parity tools:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python tests/contracts/generate_python_results.py --output /tmp/python_results.json
.venv/bin/python tests/contracts/compare_with_matlab_baseline.py --python-results /tmp/python_results.json --require-all
```

## 7. Important Runtime Notes

- DCE real Stage-A metadata (TR, FA, frame spacing) must be explicit from sidecar JSON or complete manual override tuple.
- Backend values are `auto`, `cpu`, `gpufit`.
- Stage-D accelerated fit behavior and options are documented in:
  - `/Users/samuelbarnes/code/ROCKETSHIP/docs/dce_options.md`

## 8. Related Project Docs

- `/Users/samuelbarnes/code/ROCKETSHIP/python/README.md`
- `/Users/samuelbarnes/code/ROCKETSHIP/docs/project-management/TODO.md`
- `/Users/samuelbarnes/code/ROCKETSHIP/docs/project-management/ROADMAP.md`
- `/Users/samuelbarnes/code/ROCKETSHIP/docs/project-management/PORTING_STATUS.md`
