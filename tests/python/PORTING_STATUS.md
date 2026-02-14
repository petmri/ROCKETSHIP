# Python Porting Status

This file is the handoff point for resuming work later.

## Current scope covered
Python implementations currently exist in `/Users/samuelbarnes/code/ROCKETSHIP/python/rocketship/`:

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
- GUI entrypoints and UI helper utilities.
- MATLAB-specific batch helper scripts.

## Execution architecture decisions
- Primary runtime path is a single-process end-to-end pipeline (`A -> B -> D`) with in-memory data flow.
- Intermediate `.mat` handoff files are not part of normal execution.
- Optional stage checkpoint export is allowed for parity/debug only.
- Fitting backend approach:
  - CPU path is required and treated as the parity baseline.
  - GPUfit is optional (`backend=auto|cpu|gpufit`) and can be enabled once installed.

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

- PR: fast single-job checks + synthetic DCE smoke run.
- Push to `dev/master`: heavier matrix (full validation path).

## Next recommended steps
1. Decide if DCE scope should now include reference-region/auxiliary models (`model_FXL_reference_region*`, `model_0`) or move to DSC next.
2. Decide whether to improve confidence interval estimation for Python inverse fits (`model_tofts_fit`, `model_vp_fit`, `model_tissue_uptake_fit`, `model_2cxm_fit`, `model_fxr_fit`) beyond placeholder CI values.
3. Port `DSC_convolution_oSVD` when ready.
