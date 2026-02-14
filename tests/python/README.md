# Python Parity Runner

This directory contains a lightweight MATLAB-vs-Python comparison runner.

## Script
- `compare_with_matlab_baseline.py`

## What it does
- Loads MATLAB baseline outputs from `tests/contracts/baselines/matlab_reference_v1.json`
- Loads contracts/tolerances from `tests/contracts/`
- Compares Python outputs to MATLAB baseline by contract ID
- Prints pass/fail summary per contract

## Generate template Python results JSON
Use this before implementing Python models:

```bash
python3 tests/python/compare_with_matlab_baseline.py \
  --write-template tests/python/python_results_template.json
```

## Compare Python results against MATLAB baseline

```bash
python3 tests/python/compare_with_matlab_baseline.py \
  --python-results tests/python/python_results_template.json
```

Add `--require-all` to fail if any mapped contract output is missing.
