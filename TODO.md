# Python Transition TODO

## Objective
Finish the Python transition to the point that it can be merged to `dev` and tested on real-data workflows, with clear quality gates and maintainable code.

## Primary (Blockers for Dev-Branch Trial)
1. Parametric maps and T1 fitting workflow
- [ ] Port remaining workflow behavior from `parametric_scripts/custom_scripts/T1mapping_fit.m` and required `calculateMap` path components.
- [x] Add Python CLI entrypoint for T1 mapping workflow with clear config schema and run summary output.
- [x] Add Python GUI support for T1 fitting workflow (file selection, run controls, progress, QC).
- [ ] Add real-data tests for T1 output integrity and expected file naming.
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

3. Part E post-fitting analysis
- [ ] Port required workflow from `dce/fitting_analysis.m`, `dce/compare_fits.m`, and supporting analysis helpers.
- [ ] Implement reproducible Python outputs for ROI/voxel fit review used in current workflows.
- [ ] Add automated tests for analysis outputs and plotting/stat summary generation.

4. Real-data workflow qualification
- [ ] Run end-to-end Python DCE + T1 workflows on representative real datasets.
- [ ] Record blocker issues and classify them as fix-now vs post-merge follow-up.
- [ ] Prepare merge packet: command recipes, known differences, troubleshooting notes.

## Secondary (Important, Not Blocking First Dev Merge)
- [ ] Improve `2cxm` and `tissue_uptake` stability/accuracy across all fit backends.
- [ ] Expand DSC support beyond current core (`DSC_convolution_oSVD` and workflow-level parity).
- [ ] Decide final status of `nested` and `FXL_rr` (full support vs explicit non-support and cleanup).

## Will Not Port (Unless Scope Changes)
- neuroecon execution path
- legacy email notification flow
- manual click-based MATLAB AIF tooling
- ImageJ `.roi` compatibility path
- legacy MATLAB queue/prep GUI flows that are not needed in Python workflows

## GPUfit / CPUfit Handoff Items (for external accelerator project)
Current observed issues from `.venv/bin/python -m pytest tests/python -q` on 2026-02-20:
- `test_osipi_pycpufit_2cxm_fast`:
  - `vp` absolute error too large (`actual=0.070445478`, `expected=0.02`, tolerance `0.01857123`).
- `test_osipi_pycpufit_tissue_uptake_fast`:
  - non-finite output (`vp=nan`) in fast backend path.

Requested handoff topics:
- [ ] Improve constrained fit robustness for multi-parameter DCE models (`2cxm`, `tissue_uptake`).
- [ ] Ensure deterministic handling/reporting of failed fits (no silent NaN propagation).
- [ ] Verify bound handling and initialization behavior consistency across GPUfit/CPUfit implementations.
- [ ] Provide backend diagnostics that can be surfaced directly in Python test failure messages.

Current handling in main suite:
- `2cxm` and `tissue_uptake` fast-backend OSIPI tests are marked `xfail` (secondary-goal, non-blocking) while remaining visible.

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
- Parametric T1 GUI v1 added (`run_parametric_python_gui.py`, `python/parametric_gui.py`) with run controls, event progress, and summary/artifact display.
