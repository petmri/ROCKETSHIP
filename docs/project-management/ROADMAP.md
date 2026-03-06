# Python Transition Roadmap

## Purpose
Define long-horizon direction, merge criteria, and workstream sequencing.

This file should not carry day-to-day task lists or historical change logs.

## Mission
Deliver a Python implementation that is mature enough for `dev` merge and representative real-data workflows, while preserving scientific correctness and maintainable architecture.

## Dev-Merge Readiness Criteria
1. Primary DCE models (`tofts`, `patlak`, `ex_tofts`) are reliable across CPU and accelerated backends under merge-gated tests.
2. Parametric T1 workflow is operational in Python (CLI + GUI) with expected outputs and qualification evidence.
3. DCE Part E post-fit analysis is available in Python with reproducible outputs used by downstream interpretation.
4. End-to-end real-data qualification succeeds on representative cohorts with blocker/follow-up classification.
5. Documentation and runbooks are sufficient for Python-first execution without MATLAB source dependency.

## Guiding Constraints
- Python path is pre-production: correctness, numerical reliability, and maintainability are prioritized over historical MATLAB UX symmetry.
- Metadata-first acquisition parameters: per-session sidecars are the default source of truth; template overrides are explicit opt-in.
- Injection-window behavior should default to Stage-A/Stage-B auto timing unless users intentionally provide manual overrides.
- Remove dead or redundant compatibility layers once replacement pathways are validated.

## Primary Workstreams

### 1. Parametric Maps and T1
- Complete remaining parity gaps needed for production-like T1 mapping usage.
- Keep map output contracts, QC artifacts, and interface behavior stable across CLI/GUI paths.
- Expand evidence on external/real cohorts beyond fixture-only confidence.

### 2. DCE Primary Model Reliability
- Maintain strict gating for `tofts`, `patlak`, and `ex_tofts` reliability/parity.
- Keep CPU vs accelerated backend behavior explainable with explicit diagnostics.
- Resolve remaining backend-specific drift that affects real-data confidence.

### 3. DCE Part E Workflow
- Finish end-to-end Python handoff for post-fit analysis workflows currently performed in MATLAB.
- Ensure reproducible ROI/voxel statistics artifacts and qualification-ready outputs.

### 4. Real-Data Qualification
- Broaden qualification from local packet validation to additional representative cohorts.
- Confirm backend behavior on CUDA-capable environments where acceleration is part of target workflows.
- Maintain merge packet quality: command recipes, caveats, and explicit blocker boundaries.

### 5. OSIPI Verification
- Keep OSIPI evidence additive and provenance-tagged.
- Maintain T1 and SI-to-concentration reliability as explicit evidence lines for merge decisions.

## Secondary Workstreams
- Improve `2cxm` and `tissue_uptake` robustness and backend consistency.
- Expand DSC beyond current core parity (`oSVD` and broader workflow coverage).
- Decide support posture for `nested` and `FXL_rr` after primary workflow closure.

## Not Planned (Unless Scope Changes)
- neuroecon execution path
- legacy email notification flow
- manual click-driven MATLAB AIF tools
- ImageJ `.roi` compatibility path
- legacy MATLAB queue/prep GUI flows not required by Python-first workflows

## Delivery Sequence
1. Keep primary DCE reliability/consistency gates healthy.
2. Close remaining T1 workflow parity and validation gaps.
3. Complete Part E qualification and handoff artifacts.
4. Expand real-data qualification breadth and backend coverage.
5. Merge to `dev` for broader user testing.
