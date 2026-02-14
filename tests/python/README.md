# Python Parity Runner

This directory contains Python parity tooling and first model ports.

## Script
- `compare_with_matlab_baseline.py`
- `generate_python_results.py`

## First model port
- Source: `python/rocketship/dce_models.py`
- Function: `model_tofts_cfit`
- Function: `model_patlak_cfit`

Run Python unit tests:

```bash
python3 -m unittest discover -s tests/python -p 'test_*.py'
```

## What it does
- Loads MATLAB baseline outputs from `tests/contracts/baselines/matlab_reference_v1.json`
- Loads contracts/tolerances from `tests/contracts/`
- Compares Python outputs to MATLAB baseline by contract ID
- Prints pass/fail summary per contract

## Generate Python outputs from current ports

```bash
python3 tests/python/generate_python_results.py \
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
python3 tests/python/compare_with_matlab_baseline.py \
  --python-results /tmp/python_results_tofts.json
```

Add `--require-all` to fail if any mapped contract output is missing.
