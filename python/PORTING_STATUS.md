# Python Porting Status

## Snapshot
- Date: 2026-02-22
- Branch: `codex/algorithm-test-suite`
- Commit: working tree (uncommitted)

Current automated baseline:
- Command: `.venv/bin/python -m pytest tests/python -q`
- Result: `139 passed, 13 skipped, 2 xpassed`
- Current xfail-marked non-blocking tests (XPASS on patched local `pycpufit`):
  - `tests/python/test_osipi_pycpufit.py::test_osipi_pycpufit_2cxm_fast`
  - `tests/python/test_osipi_pycpufit.py::test_osipi_pycpufit_tissue_uptake_fast`

Latest qualification packet run:
- Discovery command:
  - `.venv/bin/python run_bids_discovery.py --bids-root tests/data/BIDS_test --output-json out/python_qualification_bids_test_auto/discovery_manifest.json --print-json`
- Qualification command:
  - `.venv/bin/python run_python_qualification.py --bids-root tests/data/BIDS_test --output-root out/python_qualification_bids_test_auto_tol1e6 --backend auto --no-postfit-arrays --print-summary-json`
- Result:
  - `status=ok`, `sessions_discovered=5`, `sessions_passed=5`, `sessions_failed=0`, `blocker_count=0`, `warning_count=1`
  - `backend=auto` selected accelerated `cpufit_cpu` in Stage D for primary models.
  - `ex_tofts` primary-map finiteness gating now passes after adopting accelerated `gpu_tolerance=1e-6` (previous `1e-12` caused excessive max-iteration failures in the Cpufit CPU path).
- Merge packet:
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/QUALIFICATION_MERGE_PACKET.md`
- Historical note:
  - Earlier gated qualification runs failed on accelerated `ex_tofts` finiteness (`cpufit_cpu`) before the tolerance update; Stage-D guarded fallback remains in place as a safety net for all-NaN accelerated outputs.

## Important Details / Lessons Learned
- `t1_fa_fit` MATLAB-vs-Python contract parity currently gates indices `[0, 1, 2, 5]`:
  - `T1`, `M0`, `r_squared`, `sse`
- CI entries `[3, 4]` are intentionally not gated yet:
  - Python currently returns CI placeholders (`-1`) while MATLAB returns fitted CI values.
  - Contract case uses `compare_indices` in `tests/contracts/parametric_core_contracts.json` to make this explicit.
- Part E input contract is now Stage-D NPZ postfit arrays:
  - enable with `stage_overrides.write_postfit_arrays = true`.
  - Part E loader reads `*_postfit_arrays.npz` directly, avoiding `.mat`-format/version issues.
- Qualification warning from 2026-02-20 BIDS-test run:
  - `sub-02downsample_ses-01` required flip-angle metadata trim (`3 -> 2`) to match derivative VFA frames.
- Qualification lesson (updated 2026-02-22):
  - `ex_tofts` accelerated `cpufit_cpu` fits previously failed qualification map-finiteness checks under `gpu_tolerance=1e-12`.
  - Local fix adopted on 2026-02-22: accelerated `gpu_tolerance=1e-6` (shared accelerated solver tolerance) after CPUfit/Cpufit diagnosis identified excessive max-iteration failures from overly tight tolerance.
  - Qualification rerun now passes all 5 BIDS-test sessions with `backend=auto` (`cpufit_cpu`).
  - Stage-D all-nonfinite accelerated-output fallback remains in `python/dce_pipeline.py` as a defensive guard.
- Accelerated-vs-CPU tolerance mapping is not 1:1:
  - Accelerated path uses single `gpu_tolerance` (now default `1e-6`) for CPUfit/GPUfit constrained LM solver.
  - CPU SciPy path uses `tol_fun` -> `ftol` default `1e-12` and `tol_x` -> `xtol` default `1e-6` in `python/dce_models.py`.
- Real-data DCE inputs are now stricter by design:
  - Stage A requires a dedicated AIF ROI mask (or future auto-AIF routine); brain-mask fallback as AIF is rejected.
  - TR/FA/time resolution for real data must come from sidecar JSON or be fully specified in config (no silent defaults).
- CI topology update (2026-02-27):
  - Python-MATLAB parity checks moved to dedicated `parity_checks` job in `.github/workflows/run_DCE.yml`.
  - Parity job now uses `tests/data/ci_fixtures/...` dataset paths and generates required downsample MATLAB baseline maps in-job before running `test_downsample_bbb_p19_tofts_ktrans`.
  - `python_portability` matrix now includes `ubuntu-22.04` in addition to macOS/Windows.
  - Unified `matlab_checks` matrix now targets `R2020b`, `R2022a`, and `latest` for both push and pull-request events (legacy PR-smoke-only MATLAB job removed).
  - Workflow-level concurrency now cancels superseded in-progress CI runs for the same workflow/ref.
- DCE baseline auto-detection port status (updated 2026-02-24):
  - Python Stage A now supports explicit baseline-end detectors via `stage_overrides.steady_state_auto_method`:
    - `legacy_sobel` (from `dce_auto_aif` baseline-end path)
    - `piecewise_constant` (from `find_end_ss` branch implementation)
    - `glr` and `tv` (ported from `synthetic_dce` `ismrm_submit/end_baseline_detect.py`)
  - Manual `steady_state_end` still takes precedence; if no manual end and no auto method are provided, Python now defaults to `legacy_sobel`.
- Synthetic phantom GT reliability checks now exist for calibration/guarding:
  - `tests/python/test_phantom_gt_reliability.py` reconstructs T1 in-test and compares T1 + primary DCE maps against `rawdata/.../gt` ground-truth maps for `sub-05phantom`/`sub-06phantom`/`sub-07phantom`.
  - Tolerances are region- and model-specific (`tests/data/BIDS_test/phantom_gt_mae_tolerances.json`) because model assumptions intentionally bias some tissue classes.
  - Initial tolerance profile is derived from local CPU + `cpufit_cpu` runs; MATLAB calibration can be added to the same workflow later.
  - Current profile is explicitly marked provisional (`gate_ready=false`) while phantom performance triage is in progress.
- Phantom GT troubleshooting status (2026-02-23):
  - Timing metadata and Stage-A conversion metadata (`TemporalResolution`, relaxivity, hematocrit) mismatches were fixed and are no longer the dominant phantom error source.
  - Nonlinear T1 fitting is now the default for Python qualification/parametric workflows and substantially improves phantom T1 accuracy.
  - Low-noise diagnostic phantom `sub-08phantom` (extra VFA flip angle) shows good T1 + AIF agreement but still large DCE parameter bias, supporting model-mismatch diagnosis (`2cxm`-generated phantoms vs simpler fit models).
  - Detailed notes and temporary phantom-only diagnostic behavior are documented in `tests/PHANTOM_GT_QUALIFICATION_STATUS.md`.

## Category Definitions
- `done`: implemented and acceptable for current transition goals.
- `primary`: required before merge to `dev` for real-data workflow trials.
- `secondary`: important, but not blocking initial `dev` merge.
- `will not port`: intentionally excluded unless project scope changes.

## MATLAB Feature Audit

| MATLAB area | Representative MATLAB files | Python status | Category | Notes |
|---|---|---|---|---|
| DCE CLI pipeline A/B/D | `run_dce_cli.m`, `dce/A_make_R1maps_func.m`, `dce/B_AIF_fitting_func.m`, `dce/D_fit_voxels_func.m` | Implemented via `python/dce_pipeline.py` + `python/dce_cli.py` | done | Core in-memory CLI flow exists and is tested. |
| DCE primary models | `dce/model_tofts*.m`, `dce/model_patlak*.m`, `dce/model_extended_tofts*.m` | Implemented in `python/dce_models.py` and wired into pipeline | primary | Edge-case regression tests cover low-SNR/non-uniform timer/bounds; strict OSIPI reliability thresholds are merge-gated; backend-consistency checks exist for CPU/CPUfit/GPUfit where available. Qualification includes explicit primary-model map-finiteness gating plus guarded acceleration fallback when accelerated output has no usable finite primary parameters. Current BIDS-test qualification passes with accelerated `cpufit_cpu` after adopting shared accelerated tolerance `gpu_tolerance=1e-6`. |
| DCE unstable models | `dce/model_2cxm*.m`, `dce/model_tissue_uptake*.m` | Implemented but still unstable in some parity/reliability paths | secondary | Keep improving; not a blocker for first dev merge unless promoted. |
| DCE optional models | `dce/model_fxr*.m`, `dce/auc_helper.m`, `nested`, `FXL_rr` pathways | Partial (`fxr`, `auc` present; `nested`/`FXL_rr` not executed) | secondary | Decide post-primary whether to fully support or deprecate. |
| DCE Part E post-fit analysis | `dce/fitting_analysis.m`, `dce/compare_fits.m`, `dce/compare_gui.m` | Partial (`python/dce_postfit_analysis.py`, `tests/python/run_dce_postfit_analysis.py`) | primary | Statistical core is ported (supported-model checks, SSE extraction, f-test, AIC/relative-likelihood, ROI CSV writers, voxel map reconstruction) plus reproducible JSON/CSV/NPY artifact writers (`run_ftest_analysis`, `run_aic_analysis`), NPZ-based runner path (`load_dce_fit_stats_from_npz`) using Stage-D postfit exports (`write_postfit_arrays`), and optional plot/stat-summary outputs for troubleshooting; remaining gap is workflow qualification on external real cohorts and end-user handoff docs. |
| DCE legacy GUIDE GUI | `dce/dce.m`, `dce/RUNA.m`, `dce/RUNB.m`, `dce/RUND.m` | Not ported 1:1 | will not port | Python keeps CLI-first flow with modern GUI wrapper instead. |
| DCE neuroecon/email/manual AIF UX | `dce/run_neuroecon_job.m`, manual GUI AIF flows | Not ported | will not port | Previously marked out of scope. |
| Parametric core model math | `parametric_scripts/fitParameter.m` (`t2_linear_fast`, `t1_fa_linear_fit`, `t1_fa_fit`) | Partial in `python/parametric_models.py` | primary | Linear + nonlinear + two-point VFA are implemented; MATLAB contract parity now covers `t2_linear_fast`, `t1_fa_linear_fit`, and `t1_fa_fit` with automated runner checks in Python tests. |
| Parametric T1 mapping workflow | `parametric_scripts/custom_scripts/T1mapping_fit.m`, `parametric_scripts/calculateMap.m` | Partial (`python/parametric_pipeline.py`, `python/parametric_cli.py`, `run_parametric_python_cli.py`) | primary | VFA linear/nonlinear/two-point fit types are available with fixture + BIDS-based naming/integrity tests; optional B1-scaled FA support (`b1_map_file` or auto-detected `B1_scaled_FAreg.nii(.gz)`), MATLAB-style TR fallback (`script_preferences.txt` key `tr`, or explicit `script_preferences_path`), MATLAB-style `odd_echoes` frame selection, and XY Gaussian smoothing (`xy_smooth_sigma` / `xy_smooth_size`) are implemented. |
| Parametric GUI workflow | `parametric_scripts/fitting_gui.m`, `run_parametric.m` | Partial (`python/parametric_gui.py`, `run_parametric_python_gui.py`) | primary | GUI v1 exists for file selection/run/progress/summary; additional MATLAB behavior and QC parity may still be needed. |
| SI->Concentration conversion | `dce/A_make_R1maps_func.m` signal-to-R1/concentration steps | Partial (`python/dce_pipeline.py` Stage A + `python/dce_signal.py`) | primary | OSIPI SI2Conc reliability test and merge-gate summary runner are in place (`tests/python/test_osipi_si_to_conc_reliability.py`, `tests/python/run_osipi_reliability.py`). |
| DSC core functions | `dsc/import_AIF.m`, `dsc/previous_AIF.m`, `dsc/DSC_convolution_sSVD.m` | Implemented (`python/dsc_helpers.py`, `python/dsc_models.py`) | done | Core parity contract coverage exists. |
| DSC extended workflow | `dsc/dsc_process.m`, `dsc/DSC_convolution_oSVD.m`, `run_dsc.m` | Partial | secondary | Expand after primary DCE + T1 + Part E closure. |
| Script preference coverage | `script_preferences.txt` mapped in audit JSON | Mixed (`supported` + `pending`) | primary | Pending keys should be closed or explicitly dropped before merge decision. |
| Contract and parity tooling | MATLAB baseline export + compare scripts | Active in `tests/contracts/` and pytest parity tests | done | Runner and artifact organization recently improved. |

## Primary Gaps Blocking Dev Merge Trial
1. Python T1 mapping workflow parity with MATLAB usage (remaining behavior + external real-data validation).
2. Python Part E analysis workflow external real-data qualification and handoff.
3. Qualification hardening for primary DCE models on broader cohorts/backend combinations:
   - current BIDS-test packet passes with `gpu_tolerance=1e-6`; next step is confirming the same behavior on additional real cohorts and CUDA/GPUfit runtimes.

## Secondary Gaps
- 2CXM and tissue uptake robustness across CPU/CPUfit/GPUfit.
- DSC `oSVD` and broader DSC workflow parity.
- Final decision and cleanup for `nested`/`FXL_rr`.

## Notes for Contributors
- Keep Python code clean and modular; remove dead compatibility layers.
- Prefer explicit units and deterministic behavior in model fitting paths.
- Use `tests/contracts/` and `tests/python/` as the evidence path for all porting claims.
