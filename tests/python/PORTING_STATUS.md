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
- ImageJ ROI input support (`.roi`).
- GUI entrypoints and UI helper utilities.
- MATLAB-specific batch helper scripts.

Confirmed in-scope requirements to retain:
- ROI spreadsheet outputs (`.xls`).

## Execution architecture decisions
- Primary runtime path is a single-process end-to-end pipeline (`A -> B -> D`) with in-memory data flow.
- Intermediate `.mat` handoff files are not part of normal execution.
- Optional stage checkpoint export is allowed for parity/debug only.
- Fitting backend approach:
  - CPU path is required and treated as the parity baseline.
  - GPUfit is optional (`backend=auto|cpu|gpufit`) and can be enabled once installed.

## DCE CLI scaffold status
- In-memory pipeline scaffold implemented in:
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/rocketship/dce_pipeline.py`
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/rocketship/dce_cli.py`
  - `/Users/samuelbarnes/code/ROCKETSHIP/run_dce_python_cli.py`
- Stage A status:
  - real implementation path is wired (NIfTI input, ROI/AIF extraction, R1/Cp/Ct generation, QC figure saving)
- Stage B status:
  - real non-GUI implementation path is wired (time restriction, non-interactive AIF selection/fitting, pass-through arrays, QC figure saving)
  - supported AIF modes: `fitted`, `raw`, `imported`
  - stage mode control: `stage_overrides.stage_b_mode = auto|real|scaffold`
- Stage D status:
  - real in-memory implementation path is wired (voxel/ROI model fitting, parameter-map output, `.xls` ROI tables)
  - supported non-GUI models: `tofts`, `ex_tofts`, `patlak`, `tissue_uptake`, `2cxm`, `fxr`, `auc`
  - stage mode control: `stage_overrides.stage_d_mode = auto|real|scaffold`
- Recent parity fixes (MATLAB alignment):
  - Stage A now uses MATLAB-style column-major voxel indexing for `lvind/tumind/noiseind`.
  - Stage D map writeback now uses MATLAB-style linear index mapping.
  - Stage A TR handling now converts `tr_ms -> tr_sec` before R1 calculations.
  - Stage A now auto-converts T1 maps from ms to seconds when magnitudes indicate ms units.
  - Stage A cleanup behavior (`cleanAB`, `cleanR1t`) now mirrors MATLAB logic more closely.
- Dataset parity tests:
  - `tests/python/test_dce_pipeline_parity_metrics.py` adds full-pipeline Tofts Ktrans checks against MATLAB maps using correlation and MSE tolerances.
  - fast path uses `test_data/synthetic/generated/bbb_p19_downsample_x3y3`.
  - full-volume `BBB data p19` path is available behind env gating due runtime.
  - both tests require MATLAB baseline map `processed/results_matlab/Dyn-1_tofts_fit_Ktrans.nii` (not legacy `processed/results`).
  - MATLAB baseline generator: `tests/matlab/generate_dce_tofts_parity_map.m`
- Scope guards enforced by config validation:
  - rejects ImageJ `.roi` input
  - accepts backend `auto|cpu|gpufit`
  - accepts AIF mode `auto|fitted|raw|imported`
- Checkpoint support:
  - writes `a_out.json`, `b_out.json`, `d_out.json` when `checkpoint_dir` is configured
- Example config:
  - `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/dce_cli_config.example.json`

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
1. Add dataset-backed regression checks from `test_data` for Stage D map outputs (summary metrics and tolerant voxel comparisons).
2. Decide whether to port `nested` and `FXL_rr` DCE model flows in the Python CLI or keep them explicitly unsupported.
3. Expand DSC parity work (`DSC_convolution_oSVD`) once DCE dataset regression checks are stable.
