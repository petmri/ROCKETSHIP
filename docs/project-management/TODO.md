# Python Transition TODO

## Purpose
Track only active, actionable tasks.

Do not log historical completions here. Record finished work in `docs/project-management/COMPLETED.md`.
Keep strategic sequencing in `docs/project-management/ROADMAP.md`. For current measurable state, run the test suite — there is no status document.
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
- [ ] Decide whether to gate `2cxm` Fp reporting on frame rate and/or fit quality. Fp is
      recoverable at typical DCE frame rates — 0.8% median error at 5 s frames on the OSIPI DRO,
      see `COMPLETED.md` (2026-08-12) — but it degrades sharply outside that: 567% at 10 s
      frames, and at 5 s frames the p90 error runs 8.5% / 34.9% / 81.8% / 975% as SNR falls
      through ~410 / ~40 / ~20 / ~8, so the median hides a heavy tail. A frame-rate check plus a
      QoF/SNR caveat on Fp would keep the reported value honest; pairs naturally with the
      QoF-aware ROI stats item below. Note the numbers above are best-case: the DRO is generated
      by the exact model being fitted, with no model mismatch, AIF error or motion.
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
- [ ] Remove the deprecated parity flag aliases from `conftest.py`
      (`--run-multi-model-backend-parity`/`--mm-parity`, `--parity-required-models`/`--req-models`,
      `--parity-require-all-models`/`--all-models`). CI and `tests/python/run_dce_parity.py` use
      the `--parity-suite` selector exclusively; the aliases are kept only for back-compat and
      should go once nothing external depends on them. (Carried over from the deleted
      `PORTING_STATUS.md`, which was the only place this was recorded.)
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
- [ ] For install_python_acceleration.py. Remove "python" from the name as it installs acceleration for matlab too. Second, when it doesn't detect matlab have it print a warning, but continue and complete successfully. Don't make the user re-run with --skip-matlab
-[ ] Redo the python terminal outputs so they are user readable. Make shorter, format properly. 
- [ ] Decide the fate of the MATLAB `script_preferences.txt` options the port has never ruled
      on either way: `drift_files`, `drift_global`, `aif_rr_type`, `fxl_rr`, `injection_time`,
      `force_use_default_relaxivity`, `nested`, `xy_smooth_size`, `roi_list`, `fit_voxels`.
      Each needs porting or an explicit decision not to. Carried over from the script
      preferences audit, retired 2026-08-16 — it was the only record that these are open.

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
