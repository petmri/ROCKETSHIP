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
- [ ] **Apply the single-source-defaults pattern to parametric T1.** Deferred follow-up to
      `docs/project-management/projects/defaults-single-source/PLAN.md`, which covers DCE
      only. `python/parametric_default.json` + `python/parametric_pipeline.py` have the same
      disease the DCE side is being cured of: hardcoded fallbacks in source, a defaults file
      that is also a fixture run config, and tests that resolve from source rather than from
      the shipped file. Do it after the DCE work lands so the resolver (`dce_config.py`, or
      whatever it is named by then) can be reused rather than reinvented.
- [ ] Decide whether accelerated 2CXM is still worth having. Moving every backend onto the 0.1 s
      dense grid (2026-08-02) made cpufit/gpufit fit ~50x more points, and on CPU that erased the
      advantage entirely: cpufit 2cxm went 0.01 s -> 1.72 s on a 24-voxel batch, against 1.70 s
      for python. The numerical win was the point and it landed (backends now agree on Fp to
      0.00% where they differed by 53%), but cpufit 2cxm currently buys nothing.
      **Profiled 2026-08-02, and it is fixable in GPUfit rather than a reason to drop the path.**
      91% of a cpufit run is inside `fit_constrained` (the spline upsampling is ~5%). The 50x point
      count costs ~976x wall time (61 -> 3001 points, 0.0003 s -> 0.2927 s on a 24-voxel batch),
      which factors cleanly into 49x points  x  7.5x LM iterations (mean 7.8 -> 58.4)  x  2.8x per
      point per iteration (25.1 -> 69.6 ns). So the kernel is O(n) *per iteration* only; neither
      other factor is waste to optimize away. The iteration growth is the fit doing real work it
      previously skipped -- at 61 points the objective is nearly flat in Fp, so `tolerance=1e-6` is
      met after 7.8 iterations and all 24 voxels report "converged" while disagreeing with python by
      53% on Fp; on the dense grid 22/24 converge and 2 hit the 200 cap. The per-point cost rises
      then plateaus (25 -> 36 -> 56 -> 60 -> 67 -> 70 -> 70 ns), the shape of a working set leaving
      L1: ~1.7 KB at 61 points against ~84 KB at 3001. Two specific inefficiencies in
      `~/code/GPUfit/Cpufit/lm_fit_cpp.cpp` account for the per-point cost:
        1. `exp_conv_recurrence` calls `std::exp(-kappa*dt)` *inside* the per-point loop, to
           support a non-uniform timer. On the dense grid dt is constant by construction, so the
           decay factor could be computed once per rate instead of once per point -- ~3000 exp
           calls collapsing to 1. This is what python's `_exp_weighted_trapz_uniform` does, and
           it is why vectorized python ties compiled C that has an analytic Jacobian.
        2. `calc_curve_values` calls `calc_values_two_compartment_exchange` and then
           `calc_derivatives_two_compartment_exchange`, and both run the same two
           `exp_conv_recurrence` passes -- the convolutions are computed twice per iteration
           where the derivative pass already produces the values. Both also allocate fresh
           `std::vector<REAL>` buffers of `n_points` on every call (six per LM iteration, ~72 KB
           at 3001 points), which is allocator traffic and cache pressure that grows with n;
           preallocating would attack the 2.8x above as well as the exp count.
      Together that is ~12,000 `std::exp` calls per voxel-iteration against a measured 0.20 ms
      per voxel-iteration, i.e. ~17 ns each, consistent with the transcendentals being nearly
      the whole cost. Fixing both should restore a large cpufit margin.
      gpufit is a separate and worse case: `Gpufit/models/two-compartment_exchange.cuh` computes
      each point with its own `for (i = 1; i <= point_index; i++)` sum, because a gpufit model
      function evaluates one point per thread. That is O(n^2) total work and 4 `exp` per inner
      iteration, so the 50x point count costs ~2500x there, not 50x. A sequential recurrence does
      not map onto gpufit's one-thread-per-point structure, so this is architectural -- it would
      need a prefix-scan formulation or a model that evaluates a whole curve per thread. Do not
      assume the extra points simply parallelize away. Measure on CUDA before deciding.
- [ ] Fp is not identifiable from 2CXM at typical DCE temporal resolution, and unifying the
      backends did not change that -- it cannot. On 24 synthetic voxels with 5 s frames, median
      relative Fp error is ~150% on both backends, because the plasma MTT (1-2 s) is far below
      the frame rate and upsampling cannot recover information the acquisition never recorded.
      Ktrans/ve/vp recover to roughly 4%/3%/12%. Consider whether Fp should be reported at all
      at this resolution, or gated on a frame-rate check.
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
      `force_use_default_relaxivity` is set. **The silent-disagreement half of this is now
      fixed** (defaults single-source, 2026-08-12): Python no longer carries the hardcoded 3.4
      fallback, and `relaxivity` is a `required` key resolved image sidecar -> run config ->
      error, so a study that would previously have been processed at the wrong relaxivity now
      stops instead. What remains is the *convenience* half — whether Python should infer the
      agent from `InstitutionName`/`AcquisitionDateTime` the way MATLAB does, or keep insisting
      the value be recorded in the sidecar (which `dce2bids` already writes). Note the failure
      this originally described was invisible to map parity: **Ktrans and vp are invariant**,
      because `Ct` and `Cp` both scale with relaxivity and the ratio cancels in the fit. It
      showed up only in absolute-concentration outputs — measured on `sub-203103`
      (`AcquisitionDateTime` 2017-03-09, `InstitutionName` USCINI), MATLAB 5.7 vs Python 3.4
      gave `python_sse / matlab_sse` = 2.8106 across 643,576 voxels, exactly `(5.7/3.4)^2`, and
      `dcedynamicCt` was affected the same way. So do not re-gate this on maps.
- [ ] Evaluate moving T1 fitting onto CPUfit/GPUfit for performance improvements.
- [ ] Finish Python non-fit pipeline performance: make the Stage-A/B QC figures opt-out
      (0.62 s unconditionally, and 68% of in-scope time on the small single-slice fixtures).
      The per-voxel seeding (round 1) and the `model_*_cfit` curve functions (round 2) are
      done. Profiles, results, the bit-exactness gotchas and the peak-memory/multiprocessing
      findings are in
      `docs/project-management/projects/python-pipeline-performance/python_pipeline_performance.md`.
- [ ] Expand DSC support beyond current core (`DSC_convolution_oSVD` and broader workflow parity).
- [ ] Decide final status of `nested` and `FXL_rr` (full support vs explicit non-support with cleanup).
- [ ] End steady-state long term: implement a more robust end_ss estimation method
      (update AutoAIF neural net) consistent across all models/datasets. 
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
- [ ] **Regenerate the `matlabref` tree on a MATLAB host.** Needs doing, and needs hardware this
      was not run on. `tests/data/BIDS_test/derivatives/matlabref/` is fit through
      `FXLfit_generic`, which reads `dce/dce_preferences.txt`, so the `voxel_MaxFunEvals`
      50 -> 200 change moves it — measurably, for `ex_tofts`/`tissue_uptake`/`2cxm` on noisy
      data, though not for `tofts`/`patlak`. Do it in its own commit so the diff reads as a
      settings change rather than parity drift. The contract baseline
      (`tests/contracts/baselines/matlab_reference_v1.json`) is *not* affected and needs no
      action — already verified by regenerating it and passing `check_baseline_drift.py`. Full
      measurement archived under 2026-08-12 in `COMPLETED.md`.

## External Accelerator Handoff (Open Items Only)

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
