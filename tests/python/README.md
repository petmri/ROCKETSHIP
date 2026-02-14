# Python Parity Runner

This directory contains Python parity tooling and first model ports.
For a resume-later snapshot, see:
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/PORTING_STATUS.md`

Current focus:
- End-to-end DCE CLI port for parts `A`, `B`, and `D`.
- Default execution: single-process with in-memory stage handoff.
- Deprecated/not being ported: `neuroecon`, GUI batch queue flow, manual click-based AIF tools, email notifications.
- GPUfit support is planned as an optional backend when available; CPU remains the parity baseline.

## Environment setup (macOS)
This project is now configured to use a local `venv` at
`/Users/samuelbarnes/code/ROCKETSHIP/.venv`.

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

## Script
- `compare_with_matlab_baseline.py`
- `generate_python_results.py`

## First model port
- Source: `python/rocketship/dce_models.py`
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
- Source: `python/rocketship/dsc_helpers.py`
- Function: `import_aif`
- Function: `previous_aif`
- Source: `python/rocketship/dsc_models.py`
- Function: `dsc_convolution_ssvd`
- Source: `python/rocketship/parametric_models.py`
- Function: `t2_linear_fast`
- Function: `t1_fa_linear_fit`

Run Python unit tests:

```bash
.venv/bin/python -m unittest discover -s tests/python -p 'test_*.py'
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
