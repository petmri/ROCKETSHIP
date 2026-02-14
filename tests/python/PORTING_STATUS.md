# Python Porting Status

This file is the handoff point for resuming work later.

## Current scope covered
Python implementations currently exist in `/Users/samuelbarnes/code/ROCKETSHIP/python/rocketship/`:

- `dce_models.py`
  - `model_tofts_cfit`
  - `model_patlak_cfit`
  - `model_extended_tofts_cfit`
  - `model_patlak_linear`
  - `model_tofts_fit` (SciPy least-squares)
- `dsc_helpers.py`
  - `import_aif`
  - `previous_aif`
- `parametric_models.py`
  - `t2_linear_fast`
  - `t1_fa_linear_fit`

## Parity status against MATLAB baseline
From `.venv`:

```bash
.venv/bin/python tests/python/generate_python_results.py --output /tmp/python_results.json
.venv/bin/python tests/python/compare_with_matlab_baseline.py --python-results /tmp/python_results.json --require-all
```

Expected summary right now:
- `9 pass`
- `0 fail`
- `0 missing`
- `1 skipped`

Skipped contract:
- `ssvd_deconvolution` (baseline path `dsc.ssvd_deconvolution` is not present in `matlab_reference_v1.json`)

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

- PR: fast single-job checks + synthetic DCE smoke run.
- Push to `dev/master`: heavier matrix (full validation path).

## Next recommended steps
1. Add MATLAB baseline export for `ssvd_deconvolution` and map it in parity runner.
2. Port DSC sSVD/oSVD implementations (or at least sSVD first) and validate against that baseline.
3. Decide whether to improve `model_tofts_fit` confidence intervals beyond placeholder estimate values (current parity passes under `fit_recovery` tolerances).
4. Add a top-level Python package test runner target (e.g., Makefile or npm-style script equivalent) if desired.
