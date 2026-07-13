# Python Porting Status

## Purpose
Capture the current measurable state of the transition.

Use this file for present-tense status only: latest test/qualification outcomes, open blockers, and active risks.
Do not maintain long task lists here (use `TODO.md`) and do not archive historical completion logs here (use `COMPLETED.md`).

## Snapshot (2026-07-12)

### Automated Baseline
- Command: `.venv/bin/python -m pytest tests/python -q`
- Result: `179 passed, 11 skipped, 2 xfailed`
- The OSIPI secondary-model reliability checks (`2cxm`, `tissue_uptake`/`2cum`) are now promoted to real passing gates: the two fast CPUfit cases (`test_osipi_pycpufit_2cxm_fast`, `test_osipi_pycpufit_tissue_uptake_fast`) run in the default suite, and the full-sweep reliability cases (`test_osipi_dce_reliability.py::test_osipi_2cxm_reliability_delay0_against_reference_values`, `::test_osipi_2cum_reliability_delay0_against_reference_values`) pass under `--osipi-slow` (0 nonfinite failures; within OSIPI peer max-abs-error). 2CXM `ve`/`fp` clear peer tolerance by a thin margin (~1e-5), so watch for cross-platform sensitivity.
- The 2 remaining `xfailed` are the phantom-GT provisional-tolerance checks (`test_phantom_gt_reliability.py`), which stay xfail pending matched-model phantom generation.

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
4. Real-data parity can still regress when dataset-backed Python checks drift from the MATLAB reference baseline-generation policy.
5. The deprecated parity flag aliases (`--run-multi-model-backend-parity`/`--mm-parity`, `--parity-required-models`, `--parity-require-all-models`/`--all-models`) are still defined in `conftest.py` for back-compat, even though CI and `tests/python/run_dce_parity.py` now use the `--parity-suite` selector exclusively. They should be removed once nothing external depends on them. (`--run-parity`/`--run-full-parity` are already gone — those tests are default-on.)

## Current Technical State Notes
- Accelerated DCE tolerance default is `gpu_tolerance=1e-6`; this unblocked prior accelerated `ex_tofts` finiteness failures seen with tighter settings.
- Stage-D fallback protections remain active for all-nonfinite accelerated outputs.
- Part E contract input is NPZ (`stage_overrides.write_postfit_arrays=true`), avoiding prior `.mat` compatibility friction.
- Real-data Stage-A parameter policy remains strict: no silent scan-parameter defaults.
- The Python Stage-B fitted AIF path now includes MATLAB-style six-parameter timing (`A, B, c, d, t_base_end, t0_exp`).
- Downsample Tofts parity for the `sub-10bbbdownsample` fixture (formerly `ci_fixtures/dce/bbb_p19_downsample_x3y3`, consolidated into `tests/data/BIDS_test`) was restored after aligning the dataset-backed Python parity fixture with post-`8ef4988` MATLAB auto baseline/injection timing (`steady_state_auto_method=find_end_ss`, `auto_find_injection=1`).
- Stage-D Tofts fit-preference parity and near-identical Stage-B plasma AIF output (once timing policy is aligned) are now covered by the committed dataset-backed checks (`tests/python/test_dce_pipeline_parity_metrics.py::test_bbb_p19_region_parity` and `::test_bbb_p19_roi_xls_parity`).

## Immediate Next Status Checkpoints
1. Re-run qualification after phantom matched-model data generation and tolerance tuning.
2. Capture CUDA-capable backend verification results for accelerated paths.
3. OSIPI reliability gating was reworked to remove a near-circular limit (done): the DCE reliability + fast-backend tests now hard-gate on OSIPI's official published acceptance tolerances (`tests/data/osipi/reference/osipi_official_tolerances.json`, transcribed from OSIPI `DCEmodels_data.py`), and the peer-error spread (`osipi_peer_error_summary.json`) is a non-gating signal. Root cause: the peer pool *includes* the LEK/Edinburgh implementation ROCKETSHIP ports, so for `2cxm`/`tissue_uptake` our fit reproduces LEK and the peer `max` tracked our own error to ~4 sig figs — a self-referential gate. Provenance fully verified against the OSIPI testing-framework paper (van Houdt et al., MRM 2023, doi:10.1002/mrm.29826): DRO ground truth byte-identical to OSIPI @ `23d3714` (Manning et al., MRM 2021); the peer JSON is now reproducible in-repo — all per-contributor result CSVs are committed under `reference/{dce,dsc}_models_results/` + the existing T1/SI2Conc dirs, and `generate_peer_error_summary.py` recomputes it to machine precision. The summary generator was fixed and emits provenance, a dual-gate accuracy table, and per-case GT-vs-fit tables.
4. Remove the deprecated parity flag aliases (`--run-multi-model-backend-parity`/`--mm-parity`, `--parity-required-models`, `--parity-require-all-models`/`--all-models`) from `conftest.py` now that CI and `tests/python/run_dce_parity.py` have been migrated to the `--parity-suite` selector.
