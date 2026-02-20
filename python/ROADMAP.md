# Python Transition Roadmap

## Mission
Deliver a Python implementation that is complete enough to merge into `dev` and run real-data workflows with confidence, while keeping code quality high enough for long-term academic maintenance.

## Delivery Target
The transition is "dev-branch ready" when all items below are true:
1. Primary DCE models (`tofts`, `patlak`, `ex_tofts`) are reliable across CPU + acceleration backends and pass strict gating tests.
2. Parametric T1 fitting workflow is available in Python (CLI + GUI), including map generation and expected outputs.
3. Part E post-fitting analysis workflow is available in Python with equivalent outputs needed by current users.
4. Real-data dry runs succeed on representative datasets (not only synthetic/fixture data).
5. Documentation and runbooks are updated so users can execute Python workflows without reading MATLAB source.

## Engineering Constraints
- Python path is pre-production: correctness and maintainability are higher priority than preserving legacy implementation details that are not scientifically required.
- Favor readable, unit-explicit, modular code suitable for academic extension.
- Remove dead Python code and unused compatibility layers once replacement paths are validated.

## Primary Workstreams

### 1. Parametric Maps and T1 Fitting (Primary)
Scope:
- Port `parametric_scripts/custom_scripts/T1mapping_fit.m` behavior needed for routine workflows.
- Port required `calculateMap`/`fitParameter` pathway components for T1 map production.
- Provide Python GUI support for T1 fitting workflow (file selection, parameter controls, run status, QC preview).

Current status:
- CLI v1 for VFA T1 mapping is implemented (`run_parametric_python_cli.py`, `python/parametric_cli.py`, `python/parametric_pipeline.py`) with linear/nonlinear/two-point fit support.
- Optional B1-scaled FA handling is implemented in the Python parametric pipeline (explicit `b1_map_file` or auto-detected `B1_scaled_FAreg.nii(.gz)` in VFA directory).
- MATLAB-style TR fallback is implemented for parametric T1 when sidecar TR is absent (`script_preferences.txt` key `tr`, with optional explicit `script_preferences_path`).
- MATLAB-style `odd_echoes` selection and XY Gaussian smoothing (`xy_smooth_sigma` / `xy_smooth_size`) are implemented in the Python parametric pipeline.
- GUI v1 is implemented for parametric T1 (`run_parametric_python_gui.py`, `python/parametric_gui.py`).
- Fixture-backed and BIDS-based integration tests for map outputs and naming are in place (`tests/python/test_parametric_pipeline.py`).
- OSIPI reliability coverage now includes linear, nonlinear, and two-FA T1 checks (`tests/python/test_osipi_t1_reliability.py`).
- Remaining work is map-level parity hardening and real-data workflow validation.

Required outputs:
- T1 map files in expected naming/location patterns.
- run summary + QC artifacts comparable to existing MATLAB usage.

### 2. DCE Core Model Hardening (Primary)
Scope:
- Stabilize and tighten gating for `tofts`, `patlak`, and `ex_tofts`.
- Keep backend behavior consistent across `cpu`, `cpufit`, and `gpufit` when available.
- Keep dataset-backed parity and OSIPI reliability tests as merge gates for these models.

Current status:
- Edge-case regression coverage exists for low-SNR data, non-uniform timer spacing, and custom fit bounds in `tests/python/test_dce_models.py`.
- Strict OSIPI reliability thresholds for primary DCE models are now merge-gated via `tests/python/test_osipi_dce_reliability.py` and `tests/python/run_osipi_reliability.py`.
- Backend-consistency checks now cover CPU vs CPUfit/GPUfit (where available) in `tests/python/test_osipi_backend_consistency.py`.
- Remaining work is real-data backend qualification in end-to-end workflows.

Required outputs:
- strict pass/fail criteria on primary model parity and reliability tests.
- backend-specific diagnostics available from test runners.

### 3. DCE Part E Post-Fitting Analysis (Primary)
Scope:
- Port necessary functionality from `dce/fitting_analysis.m`, `dce/compare_fits.m`, and related analysis helpers.
- Support ROI and voxel-level post-fit review used in current analysis workflows.

Current status:
- Initial non-GUI statistical core is now in Python (`python/dce_postfit_analysis.py`) with MATLAB-style model support checks, SSE extraction, f-test, AIC/relative-likelihood comparison, ROI CSV writers, and voxel-map reconstruction helpers.
- Reproducible output helpers are available for Part E statistics (`run_ftest_analysis`, `run_aic_analysis`) that emit JSON/CSV/NPY artifacts for ROI and voxel review.
- Stage D now supports optional Part E array export (`stage_overrides.write_postfit_arrays=true`) to `*_postfit_arrays.npz`, including model metadata, timer/AIF curves, voxel+ROI fit outputs, voxel indexing, and residual arrays.
- The Python Part E runner (`tests/python/run_dce_postfit_analysis.py`) now consumes these NPZ artifacts directly (no `.mat` dependency).
- Part E outputs now include statistical summaries plus optional plot artifacts (F-test p-value histogram, AIC model-count/likelihood histograms).
- Unit coverage for this core, NPZ-based input path, and plot/stat-output generation is in place (`tests/python/test_dce_postfit_analysis.py`, `tests/python/test_dce_pipeline.py`).
- Remaining work is full workflow handoff and real-data qualification to replace MATLAB Part E usage end-to-end.

Required outputs:
- analysis summaries/plots needed for downstream interpretation.
- reproducible analysis execution without MATLAB GUI dependencies.

### 4. Real-Data Workflow Qualification (Primary)
Scope:
- Validate end-to-end DCE + T1 Python workflows on representative real datasets.
- Document known differences from MATLAB where scientifically acceptable.

Required outputs:
- qualification report for `dev` merge decision.
- issue list split into blocker vs follow-up.

### 5. OSIPI Verification Expansion (Primary)
Scope:
- Keep OSIPI verification additive and explicitly sourced from local mirrors:
  - `/Users/samuelbarnes/code/DCE-DSC-MRI_CodeCollection`
  - `/Users/samuelbarnes/code/DCE-DSC-MRI_TestResults`
- Maintain T1 verification coverage for linear, nonlinear, and two-FA peer comparisons.
- Add SI-to-concentration reliability checks as explicit merge evidence for signal-conversion math.

Required outputs:
- OSIPI-labeled T1 + SI-to-concentration reliability tests under `tests/python/`.
- Data provenance and source commit references documented in `tests/data/osipi/README.md`.
- Peer-aligned error summaries integrated into transition reporting docs and CI merge-gate reporting (`tests/python/run_osipi_reliability.py`).
- Explicit MATLAB-vs-Python parity coverage for primary parametric fitters (including non-linear VFA T1) in automated test paths and contract-runner checks.

## Secondary Workstreams
- Refine `2cxm` and `tissue_uptake` numerical stability and backend consistency.
- Expand DSC beyond current core parity (`oSVD`, larger workflow coverage).
- Revisit `nested` and `FXL_rr` only after primary workflow completion.

## Not Planned for Port (Unless Scope Changes)
- neuroecon execution path
- legacy email notification flow
- manual click-driven MATLAB AIF tools
- ImageJ `.roi` compatibility path
- legacy MATLAB queue/prep GUI flows not required by Python-first workflows

## Sequence to Merge
1. Close primary DCE reliability gates.
2. Complete remaining T1 workflow gaps (GUI + residual MATLAB behavior).
3. Complete OSIPI T1 + SI-to-concentration verification expansion.
4. Complete Part E analysis workflow.
5. Run real-data qualification and resolve blockers.
6. Merge to `dev` for broader user testing.
