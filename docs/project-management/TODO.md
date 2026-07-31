# Python Transition TODO

## Purpose
Track only active, actionable tasks.

Do not log historical completions here. Record finished work in `docs/project-management/COMPLETED.md`.
Keep strategic sequencing in `docs/project-management/ROADMAP.md` and current measurable state in `docs/project-management/PORTING_STATUS.md`.
Larger feature requests should be logged in `docs/project-management/projects/feature-request/new_features.md`.

## Secondary Active Work (Non-Blocking First Dev Merge)

### 1. Synthetic Example Dataset Coverage
- [ ] Generate BIDS-structured synthetic set with three SNR tiers (low/medium/high), each with two repeated measurements using identical parameters and independent noise. Some of this work done but needs a decision on large data distribution before commit (see below).
- [ ] Reach acceptable MAE tolerance windows for primary maps (targeting approximately 10-20% region/model bias where scientifically reasonable).
- [ ] Isolate implementation error vs expected model mismatch by generating matched-model phantom datasets in `synthetic_dce` (`tofts`, `ex_tofts`, `patlak` generation variants).

### 2. Large Test/Parity Data Distribution
- [ ] Decide how to distribute large test/parity fixtures (~240 MB now, growing) outside the
      main git repo. Options + recommendation (Release assets now, DataLad later) in
      `docs/project-management/projects/large-data-distribution/large_data_distribution.md`.

### 3. Modeling and Workflow Follow-Ups
- [ ] Migrate `tests/data/BIDS_test` onto the production BIDS layout and naming so
      `tests/python/run_dce_benchmark.py` resolves it again without a `--dataset-root` switch.
      The benchmark now assumes `sourcedata/raw/<sub>/<ses>` plus
      `derivatives/<dceprep-*>/<sub>/<ses>` and the `desc-` entity names
      (`desc-AIF_T1map`, `space-DCEref_desc-brain_mask`); the fixture still uses `rawdata/`
      and the older `label-` names, so the default invocation fails until it is converted.
      Old-layout compatibility was deliberately not kept.
- [ ] Port MATLAB's contrast-agent relaxivity auto-selection to Python. `run_dce_cli.m:110-127`
      reads `InstitutionName`/`ManufacturersModelName` and `AcquisitionDateTime` from the DCE JSON
      and picks 5.7 (MultiHance, USC pre-2017-10-01) or 3.4 (Dotarem) unless
      `force_use_default_relaxivity` is set. `dce_pipeline._resolve_timing_metadata` only honours an
      explicit `relaxivity` key in the JSON and otherwise hardcodes 3.4
      (`python/dce_pipeline.py:1936-1940`), so the two pipelines silently disagree on any USC
      pre-Oct-2017 study. Measured on `sub-203103` (`AcquisitionDateTime` 2017-03-09,
      `InstitutionName` USCINI): MATLAB 5.7, Python 3.4. **Ktrans and vp are invariant** — `Ct` and
      `Cp` both scale with relaxivity, so the ratio cancels in the fit — which is why map-based
      parity never caught it. It shows up only in absolute-concentration outputs: measured
      `python_sse / matlab_sse` = 2.8106 across 643,576 voxels, exactly `(5.7/3.4)^2`.
      `dcedynamicCt` is affected the same way.
- [ ] Evaluate moving T1 fitting onto CPUfit/GPUfit for performance improvements.
- [ ] Finish Python non-fit pipeline performance: make the Stage-A/B QC figures opt-out
      (0.62 s unconditionally, and 68% of in-scope time on the small single-slice fixtures).
      The per-voxel seeding (round 1) and the `model_*_cfit` curve functions (round 2) are
      done. Profiles, results, the bit-exactness gotchas and the peak-memory/multiprocessing
      findings are in
      `docs/project-management/projects/python-pipeline-performance/python_pipeline_performance.md`.
- [x] Improve `2cxm` and `tissue_uptake` stability/accuracy on real data.
- [ ] Expand DSC support beyond current core (`DSC_convolution_oSVD` and broader workflow parity).
- [ ] Decide final status of `nested` and `FXL_rr` (full support vs explicit non-support with cleanup).
- [ ] End steady-state long term: implement a more robust end_ss estimation method
      (update AutoAIF neural net) consistent across all models/datasets. 
- [x] End steady-state Short term: AIF sidecar 
- [x] End steady-state medium term: evaluate the 4 algorithms, port the winner to MATLAB. Done and archived (`docs/project-management/projects/archived/steady-state-tv-default/`);
      the `tv`-default rollout is committed (`a9d78b6`) and `pytest -m parity` passes
      (all gated checks, no hand-curated exceptions -- 12/12 at the time, 42/42 after the
      2026-07-29 gate-scope review below; see
      `docs/project-management/projects/archived/batch-parity/batch_parity.md`).
- [ ] **QoF-aware ROI stats.** Exclude unreliable voxels (χ²_ν > `qof_chi2_max`) from voxelwise
      ROI parameter rollups, and/or report a reliable-fraction per ROI, so nonsense voxels stop
      polluting ROI means. This was the original motivation for the QoF work and is the one piece
      never built -- the metric, the maps, the pipeline hook and the parity filter are all landed
      and shipped. Background:
      `docs/project-management/projects/archived/batch-parity/quality_of_fit.md`.
- [ ] Decide whether `shrink_sigma` (eBayes σ² moderation + prior-predictive clamp) should become
      the pipeline default rather than opt-in. Needs broader `RUNNER_DATA` evidence than the one
      session it was validated on. Background:
      `docs/project-management/projects/archived/batch-parity/sigma_estimators.md`.

### 4. Parity Testing Gaps (carried over from the archived batch-parity project)
DCE parity itself is done -- **42/42** gated voxelwise checks and 4/4 ROI-xls checks pass with no
hand-curated exceptions. These two gaps are about what the suite *cannot currently catch*; C and D
from the original list were dropped as won't-do. Full rationale:
`docs/project-management/projects/archived/batch-parity/batch_parity.md` -> "Testing gaps".
- [x] **Gate-scope review (2026-07-29).** The gated set went from 12 checks (tofts+patlak, Ktrans
      only) to 42 (tofts/patlak/ex_tofts, *every* parameter, all three regions, cpu and auto),
      under one rule with no per-model/parameter/region carve-outs. Two hand-rolled masks
      (`ktrans_upper_exclude`, `ve_ktrans_min`) collapsed into the single bound-pinning
      identifiability filter that `test_backend_equivalence.py` already used -- which is what
      made ex_tofts gateable (worst check 0.807 -> 0.9998). Scatter is now gated on RMSE
      normalized by the reference RMS: the previous absolute bound would have passed a 1737%
      error on patlak Ktrans. `tissue_uptake` and `2cxm` remain reported-only, each with the
      measurement (taken *with* the filter applied) recorded in `UNGATED_MODEL_REASONS`.
      Verified by injecting four regressions and confirming each fails.
- [x] **A. Stage-B AIF contract gate.** Built 2026-07-29: `tests/python/test_stage_b_aif_parity.py`
      gates `Cp_use`/`CpROI`/`Stlv_use`/`timer`, the `step` window, `start_time`/`end_time`/
      `max_index` and the AIF fit coefficients `[A B c d t_base_end t0_exp]` against a committed
      MATLAB payload (`Dyn-1_stage_b_aif.json`, written by
      `tests/matlab/helpers/write_stage_b_aif_contract.m`; regenerate with `'stageBOnly', true`).
      Python matches MATLAB to ~1e-8 relative; gates at `rel_mae ≤ 1e-5` / `rel_max_abs ≤ 1e-4` /
      `corr ≥ 0.99999`, timing to 1e-6 min, exact on indices. Verified to fail on a one-frame
      injection shift, a 0.1% `Cp_use` rescale (corr stays 1.0 — the point) and a moved peak.
      MATLAB-side drift folded into `check_matlabref_map_drift.py`; CI step added.
- [x] **B. Backend-equivalence gate.** Built 2026-07-29:
      `tests/python/test_backend_equivalence.py`, on 2000 voxels of Stage-B arrays frozen from
      `RUNNER_DATA/sub-1101743/ses-01` (332 KB; `freeze_stage_b_backend_fixture.py`). Gates
      `cpu` vs `cpufit_cpu` and vs `gpufit_cuda` on `patlak`/`tofts`/`ex_tofts`; skips with
      reason when a backend is absent. Enforced in CI by the `backend_equivalence` job, which
      installs the backends via `install_python_acceleration.py`; runners have no CUDA, so the
      `cpufit` half gates and the `gpufit` half skips itself.
      **This corrected the premise.** The "cpufit/gpufit diverges from MATLAB" reading was
      measuring bound-pinned voxels: on this fixture ex_tofts Ktrans reads corr 0.23 over all
      voxels but **0.9998** once voxels with a parameter pinned at a bound are excluded, and
      the backends' SSE agree to ~1e-6 relative. So the gate is three-part — equivalence on the
      identifiable subset, objective (SSE) agreement over all voxels, and bound-hit symmetry —
      rather than a naive whole-sample correlation, which would have had to be set so loose it
      gated nothing. Verified to fail on a 20% Ktrans bias, inflated SSE, and skewed bound-hit
      rates.
- [x] **Install `pycpufit` in CI** so gap B's gate is enforced there. Done via a dedicated
      `backend_equivalence` job running `install_python_acceleration.py --no-matlab --no-gui`,
      which pulls the prebuilt pyCpufit/pyGpufit from the `ironictoo/Gpufit` release
      (`py3-none-any`, ctypes-loaded `.so`, no hard CUDA link). Kept out of `python_checks` so a
      flaky external download cannot fail the main Python job — and it is now the only
      end-to-end exercise of the installer.

## External Accelerator Handoff (Open Items Only)

### GPUfit / CPUfit Backend
The `E=Ktrans/Fp` reparam + O(N) convolution + analytic-Jacobian fix is **done and verified on
both cpufit and CUDA gpufit** (all 5 OSIPI sweeps pass on each, incl. all 24 2CXM; ~3000× faster
on 2cxm) — see `COMPLETED.md` and
`docs/project-management/projects/osipi-verification/STATUS.md`. Remaining:
- [x] **Verify the reparam kernels on CUDA hardware.** Done in `1f96a68`; re-confirmed 2026-07-29 on this CUDA box (`cuda_available=True`, runtime 12.5 / driver 13.0): `test_osipi_pygpufit.py` is 5/5 green and the `2cxm` `xfail` is gone.
- [x] **Review the `Fp` floor default change** (`2cxm`/`tissue_uptake` `lower_limit_fp` 1e-3→1e-4 in `dce_pipeline._stage_d_fit_prefs`) before dev-merge — it affects all backends (only relaxes the feasible region; physically ~0.6 mL/100mL/min).
- [x] Verify bound handling and initialization consistency across GPUfit/CPUfit implementations. Closed by the archived Stage-D consolidation: bounds and per-voxel seeds for all five accelerated-eligible models now come from one shared `python/dce_fit_backends.py` (`_*_bounds_row` / `assemble_*_candidates`), candidates are clamped into bounds before dispatch (`cb0cc68`), and `tests/python/test_osipi_backend_consistency.py` gates cpu-vs-cpufit-vs-gpufit agreement to 1e-4 relative.
- [ ] Provide backend diagnostics that can be surfaced directly in Python test failure messages.
      Concretely: `dce_fit_backends._run_accelerated` collapses gpufit/cpufit's per-voxel `states`
      to `states == 0` and drops the codes (`python/dce_fit_backends.py:783`), and fills `extra`
      with `None`. Keeping a state histogram (e.g. `MAX_ITERATION` counts) would have named the
      `TOFTS {0: 6329, 1: 952}` non-convergence concentration directly instead of it having to be
      found by hand.

### Synthetic_DCE Generator
- [ ] Import segmentation image with tissue classes.
- [ ] Output DCE images where each segmentation class maps to a unique tissue generation class.
- [ ] Add motion simulation option with inverse motion-correction matrices.
- [ ] Output ground-truth maps (Ktrans, vp, etc.).

## Out of Scope (Unless Scope Changes)
- neuroecon execution path
- legacy email notification flow
- manual click-based MATLAB AIF tooling
- ImageJ `.roi` compatibility path
- legacy MATLAB queue/prep GUI flows not needed in Python workflows
