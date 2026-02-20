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

Required outputs:
- T1 map files in expected naming/location patterns.
- run summary + QC artifacts comparable to existing MATLAB usage.

### 2. DCE Core Model Hardening (Primary)
Scope:
- Stabilize and tighten gating for `tofts`, `patlak`, and `ex_tofts`.
- Keep backend behavior consistent across `cpu`, `cpufit`, and `gpufit` when available.
- Keep dataset-backed parity and OSIPI reliability tests as merge gates for these models.

Required outputs:
- strict pass/fail criteria on primary model parity and reliability tests.
- backend-specific diagnostics available from test runners.

### 3. DCE Part E Post-Fitting Analysis (Primary)
Scope:
- Port necessary functionality from `dce/fitting_analysis.m`, `dce/compare_fits.m`, and related analysis helpers.
- Support ROI and voxel-level post-fit review used in current analysis workflows.

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
2. Complete T1 CLI + GUI workflow.
3. Complete Part E analysis workflow.
4. Run real-data qualification and resolve blockers.
5. Merge to `dev` for broader user testing.
