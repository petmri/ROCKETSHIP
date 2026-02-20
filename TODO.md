# Python Transition TODO

## Objective
Finish the Python transition to the point that it can be merged to `dev` and tested on real-data workflows, with clear quality gates and maintainable code.

## Primary (Blockers for Dev-Branch Trial)
1. Parametric maps and T1 fitting workflow
- [ ] Port workflow behavior from `parametric_scripts/custom_scripts/T1mapping_fit.m` and required `calculateMap` path components.
- [ ] Add Python CLI entrypoint for T1 mapping workflow with clear config schema and run summary output.
- [ ] Add Python GUI support for T1 fitting workflow (file selection, run controls, progress, QC).
- [ ] Add fixture + real-data tests for T1 output integrity and expected file naming.

2. DCE primary model readiness (`patlak`, `tofts`, `ex_tofts`)
- [ ] Tighten parity/reliability thresholds and make primary model checks strict merge gates.
- [ ] Ensure backend consistency across `cpu`, `cpufit`, and `gpufit` where available.
- [ ] Add regression tests for known edge cases (bounds, low SNR, non-uniform timer inputs).

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

## Recently Completed (Condensed)
- Contract parity tooling moved to `tests/contracts/`.
- `run_dce_parity.py` restored with corr/MSE/MAE summary output.
- Benchmark runner renamed to `tests/python/run_dce_benchmark.py`.
- Script-preference audit completed with support/pending/drop classification metadata.
- Dataset-backed parity expanded beyond Tofts maps (including ROI `.xls` checks).
