# Python Transition Completed

## Purpose
Archive historical completed work only.

Do not track open items in this file; active work belongs in `docs/project-management/TODO.md`.

Completed items moved from `TODO.md` on 2026-03-05 to keep the active backlog short.

## Completed Recent Updates (2026-03-02)
- [x] Batch DCE config assembly now prefers per-session DCE metadata JSON for `tr`/`fa`, and avoids template `tr`/`fa` defaults unless explicitly passed via `--set`.
- [x] Batch mode now forces `dce_metadata_path` to the current session sidecar by default (prevents template test-fixture metadata paths from leaking into real-data runs unless explicitly overridden via `--set dce_metadata_path=...`).
- [x] DCE frame spacing now has strict metadata behavior: it must come from sidecar JSON (`time_resolution_sec`, `TemporalResolution`, MATLAB-style `RepetitionTime` when `RepetitionTimeExcitation` exists, `AcquisitionDuration`, or `TriggerDelayTime/nReps`) or explicit config override (`time_resolution_sec`/`time_resolution`); missing values are a hard error.
- [x] Python Stage A now supports MATLAB-style `start_t`/`end_t` timepoint clipping (1-based frame window) before concentration conversion.
- [x] Python Stage A/B auto injection timing now mirrors MATLAB CLI auto behavior.
- [x] Batch template hardcoded `start_injection[_min]`/`end_injection[_min]` values are stripped unless explicitly passed via `--set`, so auto injection is the default batch behavior.
- [x] MATLAB legacy Sobel parity fix completed.
- [x] Stage-D batch parity diagnostics completed on clean-reference `RUNNER_DATA` packet (`sub-1101743/{ses-01,ses-02}`).

### Scan Parameter Policy (No Silent Defaults)
- [x] Policy set: there are no implicit defaults for scan parameters in Python workflows.
- [x] Required scan parameters must come from metadata or explicit user config.
- [x] Missing required scan parameters now hard-fail.

## Completed Primary Items

### 1. Parametric maps and T1 fitting workflow
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

### 2. DCE primary model readiness (`patlak`, `tofts`, `ex_tofts`)
- [x] Tighten parity/reliability thresholds and make primary model checks strict merge gates.
- [x] Ensure backend consistency across `cpu`, `cpufit`, and `gpufit` where available.
- [x] Add regression tests for known edge cases (bounds, low SNR, non-uniform timer inputs).
- [x] Implement automatic DCE baseline-window detection methods for Stage A (`steady_state_auto_method` = `legacy_sobel` and `piecewise_constant`), with explicit method selection and manual `steady_state_end` precedence.
- [x] Validate the current default DCE baseline auto method (`legacy_sobel`) on representative real datasets (MATLAB comparison + qualification impact), and keep `legacy_sobel` as default for now.
- [x] Add qualification gating for non-finite primary-model parameter maps.
- [x] Resolve qualification blocker from `ex_tofts` non-finite accelerated maps by falling back to next backend/CPU when accelerated output has no usable finite primary parameters.
- [x] Adopt accelerated DCE `gpu_tolerance=1e-6` default (was `1e-12`) after CPUfit/Cpufit max-iteration diagnosis; verified with full Python test suite and `run_python_qualification.py` on 5-session `tests/data/BIDS_test`.

### 3. Part E post-fitting analysis
- [x] Port required workflow from `dce/fitting_analysis.m`, `dce/compare_fits.m`, and supporting analysis helpers.
- [x] Implement reproducible Python outputs for ROI/voxel fit review used in current workflows.
- [x] Add automated tests for analysis outputs and plotting/stat summary generation.

### 4. Real-data workflow qualification
- [x] Run end-to-end Python DCE + T1 workflows on representative real datasets.
- [x] Record blocker issues and classify them as fix-now vs post-merge follow-up.
- [x] Prepare merge packet: command recipes, known differences, troubleshooting notes.
- [x] Latest local qualification rerun (2026-02-22) passed on 5-session `tests/data/BIDS_test` with `backend=auto` (`cpufit_cpu`) after accelerated tolerance default update.

### 5. Synthetic phantom images example datasets qualification (completed subset)
- [x] Get real NII BIDS data and replace matrix with all 0/1.
- [x] Insert synthetic DCE curves into real NIfTI files, maintaining original headers.
- [x] Save ground-truth Ktrans, vp, etc. files for synthetic datasets.
- [x] Compare fit values to ground truth.
- [x] Add synthetic phantom ground-truth reliability checks (region/model-specific MAE tolerances) for `sub-05phantom`/`sub-06phantom`/`sub-07phantom`, with T1 reconstructed in-test before DCE fitting.

## Completed Secondary Items
- [x] Improve `2cxm` and `tissue_uptake` stability/accuracy in OSIPI benchmark/reliability subsets (real-data hardening remains tracked in `TODO.md`).

## Completed Handoff Items
- [x] Resolve `PATLAK` Cpufit/Cpufit real-data divergence in multi-fit constrained runs (RUNNER_DATA ses-02 payload; all fits currently report `CONVERGED` but parameter agreement is poor).

## Completed Condensed Milestones
- [x] Contract parity tooling moved to `tests/contracts/`.
- [x] `run_dce_parity.py` restored with corr/MSE/MAE summary output.
- [x] Benchmark runner renamed to `tests/python/run_dce_benchmark.py`.
- [x] Script-preference audit completed with support/pending/drop classification metadata.
- [x] Dataset-backed parity expanded beyond Tofts maps (including ROI `.xls` checks).
- [x] Python parametric T1 pipeline/CLI scaffold added (`run_parametric_python_cli.py`, `python/parametric_pipeline.py`, `python/parametric_cli.py`).
- [x] OSIPI SI-to-concentration source data and peer result tables imported into `tests/data/osipi/`.
- [x] OSIPI SI-to-concentration reliability test added (`tests/python/test_osipi_si_to_conc_reliability.py`).
- [x] OSIPI SI-to-concentration merge-gate reporting runner added (`tests/python/run_osipi_reliability.py`) and wired in CI.
- [x] Primary DCE edge-case regression tests added for non-uniform timer, low-SNR fits, and custom bounds (`tests/python/test_dce_models.py`).
- [x] Primary DCE OSIPI thresholds tightened to strict peer-max limits and integrated into merge-gate reporting (`tests/python/test_osipi_dce_reliability.py`, `tests/python/run_osipi_reliability.py`).
- [x] Primary DCE backend consistency checks added across CPU/CPUfit/GPUfit where available (`tests/python/test_osipi_backend_consistency.py`).
- [x] Accelerated DCE default tolerance updated to `gpu_tolerance=1e-6` and validated with full `tests/python` suite and BIDS qualification.
- [x] Synthetic phantom GT reliability helper/runner added (`tests/python/phantom_gt_helpers.py`, `tests/python/run_phantom_gt_reliability.py`).
- [x] Phantom GT runner enhanced with compact AIF diagnostics and explicit phantom metadata alignment notes.
- [x] Parametric T1 GUI v1 added (`run_parametric_python_gui.py`, `python/parametric_gui.py`).
- [x] Parametric T1 real-data naming/integrity tests added for BIDS-based multifile and stacked inputs (`tests/python/test_parametric_pipeline.py`).
- [x] Parametric pipeline now supports nonlinear and two-point VFA fit types in addition to linear.
- [x] Parametric pipeline now supports optional B1-scaled flip-angle fitting (`b1_map_file` explicit or auto-detected `B1_scaled_FAreg.nii(.gz)`).
- [x] Parametric pipeline now requires TR from VFA sidecar metadata (`RepetitionTime`) or explicit `tr_ms`.
- [x] Parametric pipeline now supports MATLAB-style `odd_echoes` frame selection and optional XY Gaussian smoothing.
- [x] Part E statistical core port started, including reproducible JSON/CSV/NPY artifacts.
- [x] Stage D optional Part E array export added via `stage_overrides.write_postfit_arrays` with NPZ loader/runner path and regression coverage.
- [x] Part E analysis outputs now include statistical summaries and optional PNG plots.
- [x] CI workflow separated Python-only checks from MATLAB-backed parity checks and expanded portability/Matlab matrices.

## Archived From PORTING_STATUS (Moved 2026-03-05)

Historical snapshot entries moved out of active status view:
- [x] Python DCE timing metadata resolution update (2026-03-03): JSON frame-spacing branches (`RepetitionTime` with `RepetitionTimeExcitation`, `AcquisitionDuration`, `TriggerDelayTime/n_reps`, and existing `time_resolution_sec` / `TemporalResolution`).
- [x] Python DCE timing/injection parity update (2026-03-03): MATLAB-style `start_t`/`end_t` clipping and auto injection behavior with legacy Sobel parity.
- [x] Removal of Python runtime scan-parameter fallbacks from `script_preferences.txt` (2026-03-03).
- [x] Batch DCE per-session metadata-preference update with Stage-A metadata provenance fields (2026-03-02).

Historical lessons/details moved from active status section:
- [x] Qualification warning (2026-02-20): `sub-02downsample_ses-01` flip-angle metadata trim (`3 -> 2`) for derivative VFA frame match.
- [x] Qualification lesson (2026-02-22): accelerated `ex_tofts` finiteness issue resolved after adopting `gpu_tolerance=1e-6`; guarded fallback retained.
- [x] PATLAK accelerated backend caveat (2026-03-03) with 2026-03-05 closure update and removal of temporary PATLAK handoff package.
- [x] CI topology update (2026-02-27): dedicated `parity_checks`, updated portability matrix, unified MATLAB matrix, and workflow concurrency cancellation.
- [x] DCE baseline auto-detection port status (2026-02-24): `legacy_sobel`, `piecewise_constant`, `glr`, `tv` with manual-end precedence.
- [x] Phantom GT troubleshooting status (2026-02-23): timing/conversion mismatch fixes and model-mismatch diagnosis context.
