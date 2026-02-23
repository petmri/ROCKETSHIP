# Python Transition TODO

## Objective
Finish the Python transition to the point that it can be merged to `dev` and tested on real-data workflows, with clear quality gates and maintainable code.

## Primary (Blockers for Dev-Branch Trial)
1. Parametric maps and T1 fitting workflow
- [x] Port remaining workflow behavior from `parametric_scripts/custom_scripts/T1mapping_fit.m` and required `calculateMap` path components.
- [x] Add Python CLI entrypoint for T1 mapping workflow with clear config schema and run summary output.
- [x] Add Python GUI support for T1 fitting workflow (file selection, run controls, progress, QC).
- [x] Add real-data tests for T1 output integrity and expected file naming.
- [x] Add fixture-backed tests for T1 output integrity and expected file naming.
- [x] Add OSIPI T1 mapping reliability checks beyond current linear-only coverage (nonlinear and two-FA comparators).
- [x] Add OSIPI signal-intensity to concentration verification tests.
- [x] Integrate OSIPI SI-to-concentration thresholds into merge-gate reporting.
- [x] Add MATLAB-vs-Python parity test for non-linear VFA T1 synthetic reference.
- [x] Add contract-runner integration for non-linear VFA T1 parity.

2. DCE primary model readiness (`patlak`, `tofts`, `ex_tofts`)
- [x] Tighten parity/reliability thresholds and make primary model checks strict merge gates.
- [x] Ensure backend consistency across `cpu`, `cpufit`, and `gpufit` where available.
- [x] Add regression tests for known edge cases (bounds, low SNR, non-uniform timer inputs).
- [ ] Implement automatic DCE baseline-window detection (`steady_state_end`, and start if needed) for Stage A by copying MATLAB behavior; current default-first-2-frames behavior is not sufficient for real data and can silently bias maps.
- [x] Add qualification gating for non-finite primary-model parameter maps (see `python/qualification.py` and `tests/python/test_python_qualification.py`).
- [x] Resolve qualification blocker from `ex_tofts` non-finite accelerated maps by falling back to next backend/CPU when accelerated output has no usable finite primary parameters (`python/dce_pipeline.py`, `tests/python/test_dce_pipeline.py`).
- [x] Adopt accelerated DCE `gpu_tolerance=1e-6` default (was `1e-12`) after CPUfit/Cpufit max-iteration diagnosis; verified with full Python test suite and `run_python_qualification.py` on 5-session `tests/data/BIDS_test`.

3. Part E post-fitting analysis
- [x] Port required workflow from `dce/fitting_analysis.m`, `dce/compare_fits.m`, and supporting analysis helpers.
- [x] Implement reproducible Python outputs for ROI/voxel fit review used in current workflows.
- [x] Add automated tests for analysis outputs and plotting/stat summary generation.

4. Real-data workflow qualification
- [x] Run end-to-end Python DCE + T1 workflows on representative real datasets.
- [x] Record blocker issues and classify them as fix-now vs post-merge follow-up.
- [x] Prepare merge packet: command recipes, known differences, troubleshooting notes.
- Qualification artifacts (2026-02-20):
  - `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto/discovery_manifest.json`
  - `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto/qualification_summary.json`
  - `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto_gated/qualification_summary.json`
  - `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto_gated_fallback/qualification_summary.json`
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/QUALIFICATION_MERGE_PACKET.md`
- Latest local qualification rerun (2026-02-22):
  - `backend=auto` (`cpufit_cpu`) on 5-session `tests/data/BIDS_test` passed (`sessions_failed=0`, `blocker_count=0`) after accelerated tolerance default update to `gpu_tolerance=1e-6`.

5. Synthetic phantom images example datasets qualification
- [x] Get real NII BIDs data and replace matrix with all 0/1
- [x] Insert synthetic DCE curves into real nii files, maintaining original headers just replace all SI
- [x] Save ground truth Ktrans, vp, etc. files for synthetic datasets
- [x] Compare fit values to ground truth
- [ ] Acheive reasonable passing tolerances (~10-20% bias values)
- [ ] Separate implementation-error vs model-mismatch error by generating matched-model phantom datasets (`tofts`, `ex_tofts`, `patlak`) in `synthetic_dce`; current phantoms are primarily `2cxm`-generated and simpler fits are expected to be biased.
- [x] Add synthetic phantom ground-truth reliability checks (region/model-specific MAE tolerances) for `sub-05phantom`/`sub-06phantom`/`sub-07phantom`, with T1 reconstructed in-test before DCE fitting.
- Phantom GT reliability calibration (2026-02-22):
  - New test file: `tests/python/test_phantom_gt_reliability.py` (qualification-gated; CPU + auto/cpufit paths).
  - Initial tolerance profile written to `tests/data/BIDS_test/phantom_gt_mae_tolerances.json` from local CPU + cpufit summary runs.
  - Tolerance profile is currently marked `gate_ready=false` (provisional exploratory profile while phantom performance triage is in progress).
  - MATLAB-based phantom GT calibration is still pending and can be folded into the same tolerance/profile workflow later.
- Phantom GT diagnostic status (2026-02-23):
  - `sub-08phantom` added (very low noise + extra VFA flip angle) to isolate T1/noise effects.
  - AIF timing/scaling metadata pass-through to Stage A (`TemporalResolution`, relaxivity, hematocrit) is now fixed and used from JSON.
  - Nonlinear T1 fitting substantially improved phantom T1 maps, but DCE parameter bias remains large.
  - Most likely remaining dominant source is model mismatch (phantoms generated with `2cxm`, evaluated with `tofts`/`ex_tofts`/`patlak`).
  - See `tests/PHANTOM_GT_QUALIFICATION_STATUS.md`.

5. Update license to GPL-3 before `dev` merge
- [x] Add license file

## Secondary (Important, Not Blocking First Dev Merge)
1. Include robust example datasets, synthetic only
- [ ] In BIDS struture generate 3 subjects (low, medium, high SNR) with 2 meassurements (identical parameters with unique noise)
2. Other
- [ ] switch the T1 fitting over to CPUfit/GPUfit for speed up
- [x] Improve `2cxm` and `tissue_uptake` stability/accuracy is OSIPI.
- [ ] Improve `2cxm` and `tissue_uptake` stability/accuracy for real data.
- [ ] Expand DSC support beyond current core (`DSC_convolution_oSVD` and workflow-level parity).
- [ ] Decide final status of `nested` and `FXL_rr` (full support vs explicit non-support and cleanup).

## Will Not Port (Unless Scope Changes)
- neuroecon execution path
- legacy email notification flow
- manual click-based MATLAB AIF tooling
- ImageJ `.roi` compatibility path
- legacy MATLAB queue/prep GUI flows that are not needed in Python workflows

## GPUfit / CPUfit Handoff Items (for external accelerator project)
Current observed accelerator issues to hand off / track:
- `test_osipi_pycpufit_2cxm_fast`:
  - `vp` absolute error too large (`actual=0.070445478`, `expected=0.02`, tolerance `0.01857123`).
- `test_osipi_pycpufit_tissue_uptake_fast`:
  - non-finite output (`vp=nan`) in fast backend path.
- `TOFTS_EXTENDED` / `ex_tofts` qualification issue (historical, resolved in current local workflow):
  - Upstream CPUfit/Cpufit fixes plus ROCKETSHIP accelerated `gpu_tolerance=1e-6` default resolved BIDS-test qualification failures on `cpufit_cpu`.
  - Keep repro package below as regression archive for upstream.
- Repro package (regression archive / external handoff reference):
  - `/Users/samuelbarnes/code/ROCKETSHIP/tests/contracts/handoffs/cpufit_tofts_extended/`
  - Runner:
    - `.venv/bin/python tests/contracts/handoffs/cpufit_tofts_extended/run_cpufit_tofts_extended_repro.py --json`
  - Included payloads:
    - `bids_short_timer_repro.npz` (failing short-timer Stage-B voxel)
    - `osipi_control_repro.npz` (passing OSIPI control)
  - Historical observed output before upstream/local fix:
    - BIDS payload: `cpufit state=2, iterations=0` while CPU reference parameters are finite.
    - OSIPI payload: `cpufit state=0` with close agreement to CPU reference.

Requested handoff topics:
- [ ] Improve constrained fit robustness for multi-parameter DCE models (`2cxm`, `tissue_uptake`).
- [ ] Ensure deterministic handling/reporting of failed fits (no silent NaN propagation).
- [ ] Verify bound handling and initialization behavior consistency across GPUfit/CPUfit implementations.
- [ ] Provide backend diagnostics that can be surfaced directly in Python test failure messages.
- [ ] Verify CUDA/GPUfit runtime behavior for recent CPUfit/CUDA-side `TOFTS_EXTENDED` and `2CXM` fixes on a CUDA-capable machine.

Current handling in main suite:
- `2cxm` and `tissue_uptake` fast-backend OSIPI tests are marked `xfail` (secondary-goal, non-blocking) while remaining visible.
- On the current patched local `pycpufit`, these xfail-marked tests may `XPASS`; keep non-blocking status until broader coverage is complete.

## Synthetic_DCE Handoff Items (external project)
- Import segmentation image with tissue classes
- Output DCE images where each segmentation class maps to a unique tissue generation class (just provide tissue parameters in code)
- Option to add motion, use inverse motion correction matrixes
- Output ground truth maps (Ktrans, etc)

## Recently Completed (Condensed)
- Contract parity tooling moved to `tests/contracts/`.
- `run_dce_parity.py` restored with corr/MSE/MAE summary output.
- Benchmark runner renamed to `tests/python/run_dce_benchmark.py`.
- Script-preference audit completed with support/pending/drop classification metadata.
- Dataset-backed parity expanded beyond Tofts maps (including ROI `.xls` checks).
- Python parametric T1 pipeline/CLI scaffold added (`run_parametric_python_cli.py`, `python/parametric_pipeline.py`, `python/parametric_cli.py`).
- OSIPI SI-to-concentration source data and peer result tables imported into `tests/data/osipi/`.
- OSIPI SI-to-concentration reliability test added (`tests/python/test_osipi_si_to_conc_reliability.py`).
- OSIPI SI-to-concentration merge-gate reporting runner added (`tests/python/run_osipi_reliability.py`) and wired in CI.
- Primary DCE edge-case regression tests added for non-uniform timer, low-SNR fits, and custom bounds (`tests/python/test_dce_models.py`).
- Primary DCE OSIPI thresholds tightened to strict peer-max limits and integrated into merge-gate reporting (`tests/python/test_osipi_dce_reliability.py`, `tests/python/run_osipi_reliability.py`).
- Primary DCE backend consistency checks added across CPU/CPUfit/GPUfit where available (`tests/python/test_osipi_backend_consistency.py`).
- Accelerated DCE default tolerance updated to `gpu_tolerance=1e-6` (shared accelerated solver tolerance) after CPUfit/Cpufit max-iteration diagnosis; full `tests/python` suite and 5-session BIDS qualification rerun pass.
- Synthetic phantom GT reliability helper/runner added (`tests/python/phantom_gt_helpers.py`, `tests/python/run_phantom_gt_reliability.py`) with region-labeled MAE summaries and per-backend tolerance profile support.
- Phantom GT runner now includes compact AIF diagnostics, `--subject` filtering, and explicit phantom metadata alignment notes (baseline-images diagnostic override; provisional tolerance gating).
- Parametric T1 GUI v1 added (`run_parametric_python_gui.py`, `python/parametric_gui.py`) with run controls, event progress, and summary/artifact display.
- Parametric T1 real-data naming/integrity tests added for BIDS-based multifile and stacked inputs (`tests/python/test_parametric_pipeline.py`).
- Parametric pipeline now supports nonlinear and two-point VFA fit types in addition to linear, with tiny-fixture integration tests (`python/parametric_pipeline.py`, `tests/python/test_parametric_pipeline.py`).
- Parametric pipeline now supports optional B1-scaled flip-angle fitting (`b1_map_file` explicit or auto-detected `B1_scaled_FAreg.nii(.gz)`) with integration coverage (`python/parametric_pipeline.py`, `tests/python/test_parametric_pipeline.py`).
- Parametric pipeline now supports MATLAB-style TR fallback from `script_preferences.txt` (`tr`) when sidecar TR is unavailable, including explicit `script_preferences_path` config support and integration coverage (`python/parametric_pipeline.py`, `tests/python/test_parametric_pipeline.py`).
- Parametric pipeline now supports MATLAB-style `odd_echoes` frame selection and optional XY Gaussian smoothing (`xy_smooth_sigma` / `xy_smooth_size`), including integration coverage and GUI/config wiring (`python/parametric_pipeline.py`, `python/parametric_gui.py`, `tests/python/test_parametric_pipeline.py`).
- Part E statistical core port started: Python helpers now cover MATLAB-style model support checks, SSE extraction, f-test, AIC/relative-likelihood comparison, ROI CSV outputs, voxel-vector-to-volume reconstruction, and artifact writers for reproducible JSON/CSV/NPY outputs (`python/dce_postfit_analysis.py`, `tests/python/test_dce_postfit_analysis.py`).
- Stage D optional Part E array export added via `stage_overrides.write_postfit_arrays` (`*_postfit_arrays.npz`), with NPZ loader/runner path in `python/dce_postfit_analysis.py` and `tests/python/run_dce_postfit_analysis.py`, plus regression coverage in `tests/python/test_dce_postfit_analysis.py` and `tests/python/test_dce_pipeline.py`.
- Part E analysis outputs now include statistical summary fields and optional PNG plots (F-test p-value histograms, AIC best-model/likelihood distributions) with automated coverage (`python/dce_postfit_analysis.py`, `tests/python/test_dce_postfit_analysis.py`).
