# Python Porting Status

## Purpose
Capture the current measurable state of the transition.

Use this file for present-tense status only: latest test/qualification outcomes, open blockers, and active risks.
Do not maintain long task lists here (use `TODO.md`) and do not archive historical completion logs here (use `COMPLETED.md`).

## Snapshot (2026-03-05)

### Automated Baseline
- Command: `.venv/bin/python -m pytest tests/python -q`
- Result: `139 passed, 13 skipped, 2 xpassed`
- Non-blocking accelerated model cases are still tracked as xfail/XPASS sensitive in:
  - `tests/python/test_osipi_pycpufit.py::test_osipi_pycpufit_2cxm_fast`
  - `tests/python/test_osipi_pycpufit.py::test_osipi_pycpufit_tissue_uptake_fast`

### Latest Qualification Packet
- Qualification target: `tests/data/BIDS_test`
- Result: `status=ok`, `sessions_discovered=5`, `sessions_passed=5`, `sessions_failed=0`, `blocker_count=0`, `warning_count=1`
- Backend path: `backend=auto` selected accelerated `cpufit_cpu` for primary Stage-D fits.
- Merge packet reference: `docs/project-management/projects/qualification/QUALIFICATION_MERGE_PACKET.md`

## Current Blockers (Aligned to TODO)
1. Synthetic phantom GT tolerance hardening is still open.
2. Matched-model phantom generation is still needed to separate implementation error from model-mismatch effects.

## Current Active Risks
1. Accelerated backend behavior remains model/session dependent outside the primary passing packet.
2. CUDA/GPUfit runtime verification coverage is still limited.
3. T1 and Part E need broader external-cohort qualification evidence before merge confidence is complete.

## Current Technical State Notes
- Accelerated DCE tolerance default is `gpu_tolerance=1e-6`; this unblocked prior accelerated `ex_tofts` finiteness failures seen with tighter settings.
- Stage-D fallback protections remain active for all-nonfinite accelerated outputs.
- Part E contract input is NPZ (`stage_overrides.write_postfit_arrays=true`), avoiding prior `.mat` compatibility friction.
- Real-data Stage-A parameter policy remains strict: no silent scan-parameter defaults.

## Immediate Next Status Checkpoints
1. Re-run qualification after phantom matched-model data generation and tolerance tuning.
2. Capture CUDA-capable backend verification results for accelerated paths.
3. Confirm whether current xfail/XPASS accelerated secondary-model tests should be promoted, retained, or split by backend.
