# Steady-state-end default: piecewise_constant/find_end_ss -> tv/find_end_ss_tv

## Status: archived (2026-07-22) -- core work done; residual parity-gate findings moved to batch-parity

> **Archived note:** the algorithm work below (MATLAB port, numeric validation, both
> languages' default flip, two unrelated MATLAB bugfixes, matlabref regeneration) is
> complete and verified. The still-open parity-gate residual (this doc's "Blocked"
> section) has been folded into the ongoing parity-tracking doc, since it's the same
> class of issue already tracked there: see
> `project-management/projects/archived/batch-parity/batch_parity.md` ("Update (2026-07-22):
> tv-default steady-state rollout re-triggers this, plus new residuals") for current
> status and next steps. This file is kept as a historical snapshot of how the `tv`
> default was chosen and built; do not treat it as the live tracking doc going forward.

## Why

`tests/python/run_baseline_end_reliability.py` (a new on-demand diagnostic, see below) was
run against 224 real AIFArtist-rated sessions (`derivatives/AIFArtist` +
`bids_ppg/sourcedata/raw`, USC-PPG network share). AIF-mask-restricted signal (matching
what production actually uses -- confirmed by reading both `A_make_R1maps_func.m` and
`_run_stage_a_real` in `dce_pipeline.py`):

| Algorithm | Accuracy | MSE (frames^2) |
|---|---|---|
| `piecewise_constant` (MATLAB's current default, `find_end_ss`) | 0.0% | 225.98 |
| `legacy_sobel` (`dce_auto_aif`) | 0.9% | 1.74 |
| `glr` | 88.4% | 0.18 |
| `tv` | **88.8%** | 0.18 |

`tv` and `glr` are Python-only (ported from an external script; never existed in MATLAB).
Decision: port `tv` to MATLAB, then flip both languages' defaults together, scoped to
`dev` only (`master` is 201 commits behind and has no `python/` dir / no `find_end_ss.m`
at all -- structurally different codebase in this area, out of scope).

## Done and verified

- **`dce/find_end_ss_tv.m`** -- new MATLAB port of `_tv_baseline_end`
  (`python/dce_pipeline.py:1286-1381`). Numerically validated against Python on 6 curves
  (5 synthetic covering every branch + the real `sub-10bbbdownsample` AIF curve): **exact
  match on `end_ss` in all 6 cases**. Permanent regression test added:
  `tests/python/test_find_end_ss_tv_matlab_parity.py` (skips if MATLAB isn't on PATH; all
  6 cases pass).
- `dce/A_make_R1maps_func.m`: `steady_state_time == -2` now calls `find_end_ss_tv` instead
  of `find_end_ss` (minimal-diff swap, mirroring the original `find_end_ss`-adoption
  commit `b120076`).
- `python/dce_pipeline.py`'s `_resolve_baseline_window` default fallback: `tv` instead of
  `piecewise_constant`. Config defaults updated to match: `python/dce_default.json`,
  `python/dceprep_default.json`, `tests/python/dce_cli_config.example.json`.
- `tests/python/test_dce_pipeline.py`: renamed/updated the one test that asserted the
  implicit default (`..._defaults_to_tv_when_no_options_set`); every other
  `piecewise_constant` mention in that file is an explicit override and still valid.
- `tests/python/test_dce_pipeline_parity_metrics.py`'s `_make_config` override changed
  `piecewise_constant` -> `tv` (keeps Python explicitly matching MATLAB's new real
  default, same pattern as before).
- **Two unrelated, pre-existing MATLAB bugs found and fixed** while regenerating
  reference maps (both confirmed independent of this change -- reproduced identically
  with the *old* `find_end_ss` algorithm before being fixed):
  - `dce/FXLfit_generic.m`'s Patlak/GPU branch had dead, half-implemented "ANIMAL study"
    code (commit `912a774`) requiring an `xdata{1}.ID` field that normal callers
    (including this repo's own fixture-regeneration script) never set, and which
    referenced an `ids` variable from a `xlsread(...)` call that was itself commented
    out -- unconditionally broken, not reachable-but-dead. Removed; restored simple
    uniform GPU-fit weighting.
  - Same branch called `gpufit_constrained(..., constraint_types, ...)` but only
    `constraint_type` (singular) was ever defined -- every other model branch in the same
    file (tofts/ex_tofts/tissue_uptake/2cxm, 4 occurrences) consistently uses the
    singular name. Fixed the one mismatched call site.
  - Regenerated `sub-10bbbdownsample` matlabref maps for all 5 models
    (`tofts,patlak,ex_tofts,tissue_uptake,2cxm`) using the documented command in
    `tests/README.md`; this only succeeded after both bugs above were fixed (Patlak's
    GPU path is unconditionally hit whenever MATLAB's GPU fitting is available, which it
    is in this environment).
- Full non-parity/non-MATLAB suite: `pytest tests/python -q -m "not parity"` (excluding
  the two MATLAB-requiring files above) -- **188 passed**, no regressions from any of the
  above.
- **New standalone diagnostic** (not part of CI, on-demand only):
  `tests/python/run_baseline_end_reliability.py` +
  `tests/python/baseline_end_reliability_helpers.py` -- walks a BIDS derivatives tree for
  AIF-mask JSON sidecars carrying human-rated `SteadyStateEndTimeIndex` (0-based, as
  written by AIFArtist -- converted to the 1-based `end_ss_1b` convention on read), pairs
  each with the raw dynamic series, runs all 4 detectors, and writes per-session PNGs +
  an accuracy/MSE summary. Documented in `tests/README.md`.

## Blocked: `pytest -m parity` does not pass with the new default

Two failures, found by rerunning the full suite after regenerating the `tv`-based
reference maps:

### 1. `patlak_ktrans_brain` (region_parity) -- very likely NOT a new problem

`corr=0.032` (cpu backend) / `corr=0.878` (auto/accelerated backend) vs. MATLAB,
threshold 0.95. GM and WM regions both pass fine for patlak; tofts passes in brain+WM.

**This looks like the same already-documented, already-"tabled" single-voxel
non-identifiability issue**, not a new regression from `tv`:
`project-management/projects/stage-d-fit-consolidation/STAGE_D_FIT_CONSOLIDATION_PLAN.md`
("Known residual, tabled") and the `parity-whole-brain-roi-noise` memory note both
describe `patlak_ktrans_brain_auto_vs_cpu` on this exact fixture collapsing to
near-zero correlation, root-caused to a single voxel where `vp` saturates its upper
bound (CPU finds `Ktrans=0.512261, SSE=8339.5`; gpufit finds a genuinely worse local
optimum, `Ktrans=0.0, chi-square=10231.9`) -- confirmed not a flat-manifold tie, not an
iteration-budget issue, and not fixed by giving the accelerated backend proper
per-voxel seeding (already tried). GM/WM already exclude this voxel and are already
perfect (corr=1.0). The planned mitigation recorded there: **a GM/WM-style gating
exception for patlak+brain** (report-only, like `tofts`+`gm` already is), not a fitter
change.

The `tv` window most likely just shifted this one already-unstable voxel enough to push
the *gated* `_vs_matlab` checks over the threshold too (previously only the *ungated*
`_auto_vs_cpu` check was known to be bad). **Fastest next step: apply the
already-decided fix** (add `patlak`+`brain` to the reported-only exclusions in
`test_dce_pipeline_parity_metrics.py`, mirroring the existing `tofts`+`gm` exception)
and rerun -- this was not done as part of this session per an explicit decision to stop
and hand off rather than continue recalibrating, but it's a small, low-risk, already-
justified change, not a new investigation.

### 2. `tofts_roi_xls` and `tissue_uptake_roi_xls` (roi_xls_parity) -- not yet explained

- `tofts_roi_xls`: `mae=0.032908, max_abs_err=0.105665` (limit `0.03`)
- `tissue_uptake_roi_xls`: `mae=0.029032, max_abs_err=0.118740` (limit `0.05`)
- `ex_tofts_roi_xls` and `patlak_roi_xls` both pass fine.

No pre-existing writeup found for these two. Needs fresh investigation -- start by
checking whether the `tv`-derived window shifted the ROI-average baseline/injection
timing enough to matter for these two models specifically (both fit from an ROI-mean
curve, unlike the per-voxel region-parity check above), and whether it's a handful of
voxels/timepoints or a systematic shift.

## Current repo state (nothing git-committed)

All changes above exist only in the working tree. `pytest -m parity` **fails** in this
state -- do not merge/commit without resolving the two items above (or explicitly
re-scoping, e.g. reverting just the default flip per the "keep tv as opt-in only" option
that was on the table and not chosen).

## Reproduce

```bash
# Regenerate matlabref maps (only needed if find_end_ss_tv.m or FXLfit_generic.m change again):
S=tests/data/BIDS_test; sub=sub-10bbbdownsample; ses=ses-01
matlab -batch "addpath('tests/matlab'); generate_dce_tofts_parity_map( \
  'dynamicPath', '$S/rawdata/$sub/$ses/dce/${sub}_${ses}_DCE.nii', \
  'aifRoiPath', '$S/derivatives/$sub/$ses/dce/${sub}_${ses}_label-AIF_mask.nii', \
  'brainRoiPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_label-brain_mask.nii', \
  't1MapPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_space-DCEref_T1map.nii', \
  'noiseRoiPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_label-noise_mask.nii', \
  'outputRoot', '$S/derivatives/matlabref/$sub/$ses/dce', 'models', {'tofts', 'patlak', 'ex_tofts', 'tissue_uptake', '2cxm'});"

# MATLAB<->Python port parity (should stay green regardless of the above):
pytest tests/python/test_find_end_ss_tv_matlab_parity.py -v

# The failing gate:
pytest tests/python -m parity -q
```
