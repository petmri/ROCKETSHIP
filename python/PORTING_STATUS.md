# Python Porting Status

## Snapshot
- Date: 2026-02-20
- Branch: `codex/algorithm-test-suite`
- Commit: working tree (uncommitted)

Current automated baseline:
- Command: `.venv/bin/python -m pytest tests/python -q`
- Result: `105 passed, 12 skipped, 2 xfailed`
- Current non-blocking xfails:
  - `tests/python/test_osipi_pycpufit.py::test_osipi_pycpufit_2cxm_fast`
  - `tests/python/test_osipi_pycpufit.py::test_osipi_pycpufit_tissue_uptake_fast`

## Important Details / Lessons Learned
- `t1_fa_fit` MATLAB-vs-Python contract parity currently gates indices `[0, 1, 2, 5]`:
  - `T1`, `M0`, `r_squared`, `sse`
- CI entries `[3, 4]` are intentionally not gated yet:
  - Python currently returns CI placeholders (`-1`) while MATLAB returns fitted CI values.
  - Contract case uses `compare_indices` in `tests/contracts/parametric_core_contracts.json` to make this explicit.

## Category Definitions
- `done`: implemented and acceptable for current transition goals.
- `primary`: required before merge to `dev` for real-data workflow trials.
- `secondary`: important, but not blocking initial `dev` merge.
- `will not port`: intentionally excluded unless project scope changes.

## MATLAB Feature Audit

| MATLAB area | Representative MATLAB files | Python status | Category | Notes |
|---|---|---|---|---|
| DCE CLI pipeline A/B/D | `run_dce_cli.m`, `dce/A_make_R1maps_func.m`, `dce/B_AIF_fitting_func.m`, `dce/D_fit_voxels_func.m` | Implemented via `python/dce_pipeline.py` + `python/dce_cli.py` | done | Core in-memory CLI flow exists and is tested. |
| DCE primary models | `dce/model_tofts*.m`, `dce/model_patlak*.m`, `dce/model_extended_tofts*.m` | Implemented in `python/dce_models.py` and wired into pipeline | primary | Edge-case regression tests cover low-SNR/non-uniform timer/bounds; strict OSIPI reliability thresholds are merge-gated; backend-consistency checks exist for CPU/CPUfit/GPUfit where available. |
| DCE unstable models | `dce/model_2cxm*.m`, `dce/model_tissue_uptake*.m` | Implemented but still unstable in some parity/reliability paths | secondary | Keep improving; not a blocker for first dev merge unless promoted. |
| DCE optional models | `dce/model_fxr*.m`, `dce/auc_helper.m`, `nested`, `FXL_rr` pathways | Partial (`fxr`, `auc` present; `nested`/`FXL_rr` not executed) | secondary | Decide post-primary whether to fully support or deprecate. |
| DCE Part E post-fit analysis | `dce/fitting_analysis.m`, `dce/compare_fits.m`, `dce/compare_gui.m` | Not ported as workflow | primary | Explicit user priority for transition completeness. |
| DCE legacy GUIDE GUI | `dce/dce.m`, `dce/RUNA.m`, `dce/RUNB.m`, `dce/RUND.m` | Not ported 1:1 | will not port | Python keeps CLI-first flow with modern GUI wrapper instead. |
| DCE neuroecon/email/manual AIF UX | `dce/run_neuroecon_job.m`, manual GUI AIF flows | Not ported | will not port | Previously marked out of scope. |
| Parametric core model math | `parametric_scripts/fitParameter.m` (`t2_linear_fast`, `t1_fa_linear_fit`, `t1_fa_fit`) | Partial in `python/parametric_models.py` | primary | Linear + nonlinear + two-point VFA are implemented; MATLAB contract parity now covers `t2_linear_fast`, `t1_fa_linear_fit`, and `t1_fa_fit` with automated runner checks in Python tests. |
| Parametric T1 mapping workflow | `parametric_scripts/custom_scripts/T1mapping_fit.m`, `parametric_scripts/calculateMap.m` | Partial (`python/parametric_pipeline.py`, `python/parametric_cli.py`, `run_parametric_python_cli.py`) | primary | Linear VFA CLI path is implemented and fixture-tested; GUI and remaining MATLAB-path behavior are still pending. |
| Parametric GUI workflow | `parametric_scripts/fitting_gui.m`, `run_parametric.m` | Partial (`python/parametric_gui.py`, `run_parametric_python_gui.py`) | primary | GUI v1 exists for file selection/run/progress/summary; additional MATLAB behavior and QC parity may still be needed. |
| SI->Concentration conversion | `dce/A_make_R1maps_func.m` signal-to-R1/concentration steps | Partial (`python/dce_pipeline.py` Stage A + `python/dce_signal.py`) | primary | OSIPI SI2Conc reliability test and merge-gate summary runner are in place (`tests/python/test_osipi_si_to_conc_reliability.py`, `tests/python/run_osipi_reliability.py`). |
| DSC core functions | `dsc/import_AIF.m`, `dsc/previous_AIF.m`, `dsc/DSC_convolution_sSVD.m` | Implemented (`python/dsc_helpers.py`, `python/dsc_models.py`) | done | Core parity contract coverage exists. |
| DSC extended workflow | `dsc/dsc_process.m`, `dsc/DSC_convolution_oSVD.m`, `run_dsc.m` | Partial | secondary | Expand after primary DCE + T1 + Part E closure. |
| Script preference coverage | `script_preferences.txt` mapped in audit JSON | Mixed (`supported` + `pending`) | primary | Pending keys should be closed or explicitly dropped before merge decision. |
| Contract and parity tooling | MATLAB baseline export + compare scripts | Active in `tests/contracts/` and pytest parity tests | done | Runner and artifact organization recently improved. |

## Primary Gaps Blocking Dev Merge Trial
1. Python T1 mapping workflow parity with MATLAB usage (remaining behavior + real-data validation).
2. Python Part E analysis workflow availability.
3. Real-data qualification runbook and results for Python workflows.

## Secondary Gaps
- 2CXM and tissue uptake robustness across CPU/CPUfit/GPUfit.
- DSC `oSVD` and broader DSC workflow parity.
- Final decision and cleanup for `nested`/`FXL_rr`.

## Notes for Contributors
- Keep Python code clean and modular; remove dead compatibility layers.
- Prefer explicit units and deterministic behavior in model fitting paths.
- Use `tests/contracts/` and `tests/python/` as the evidence path for all porting claims.
