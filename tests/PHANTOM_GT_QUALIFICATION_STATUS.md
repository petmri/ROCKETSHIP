# Phantom GT Qualification Status

## Scope
This document tracks the current status of synthetic phantom ground-truth (GT) qualification work driven by:

- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/run_phantom_gt_reliability.py`
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/phantom_gt_helpers.py`
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/test_phantom_gt_reliability.py`

The current goal is diagnosis and reliability characterization, not final gating.

## Phantom Datasets (Known Facts)

Tracked phantom subjects in `tests/data/BIDS_test/rawdata`:
- `sub-05phantom`
- `sub-06phantom`
- `sub-07phantom`
- `sub-08phantom` (diagnostic low-noise case)

Ground-truth assets are stored under each session `rawdata/.../gt/` and include:
- GT maps: `T1`, `Ktrans`, `ve`, `vp`, `fp`
- GT tissue-class segmentation (`desc-gtTissueClass_mask`)
- GT AIF mask (`desc-gtAIFMask_mask`)
- GT AIF time series (`desc-gtAIF_timeseries.txt` + `.json`)

Current phantom tissue labels used by tests:
- `1 = muscle_fat`
- `2 = brain`
- `3 = vessel`

Key metadata now available in phantom DCE/GT sidecars (and used by ROCKETSHIP Stage A):
- `TemporalResolution`
- `Relaxivity_per_mM_per_s`
- `Hematocrit` (phantoms currently use `0.0`)
- `AIFConcentrationKind` (`plasma`)
- GT `BaselineImages`

`sub-08phantom` specifics (diagnostic case):
- very low DCE noise (high SNR)
- extra VFA flip angle (`15 deg`, so 4 VFA images total)
- intended to separate T1/noise effects from DCE model mismatch effects

## Tests / Checks Completed

### 1. Phantom GT data organization + canonical naming
- GT folders standardized for `sub-05` to match `sub-06/07/08` naming scheme.
- Canonical GT filenames now used (`desc-gt*`).

### 2. Phantom GT reliability helper and runner
- Added summary runner with region-wise voxelwise error metrics:
  - `tests/python/run_phantom_gt_reliability.py`
- Added helper for:
  - T1 reconstruction in-test
  - DCE fitting
  - GT loading
  - AIF diagnostics
  - tolerance profile generation

### 3. T1 quality improvements for phantom qualification
- Phantom T1 reconstruction switched to nonlinear VFA fitting (`t1_fa_fit`) instead of linear VFA.
- This removed catastrophic linear-fit outliers that dominated MAE.

### 4. DCE metadata strictness + pass-through fixes
- ROCKETSHIP now requires explicit DCE frame spacing (`TemporalResolution`) and no longer infers it from `RepetitionTime`.
- ROCKETSHIP Stage A now reads `relaxivity` and `hematocrit` from phantom DCE JSON sidecars.

### 5. AIF diagnostics and comparison
- Runner now prints compact AIF comparison tables.
- AIF diagnostics compare GT AIF vs ROCKETSHIP Stage-A/B curves and report:
  - timing agreement
  - shape correlation
  - MAE/bias
  - peak ratio
  - baseline-window behavior

### 6. Low-noise diagnostic phantom (`sub-08phantom`)
- Added low-noise phantom with 4 VFA angles to test whether poor DCE fit accuracy is mainly due to noise/T1 quality.
- Observed outcome: T1 and AIF look good, but DCE parameter bias remains high for several parameters/regions.

## Problems Ruled Out (or Substantially Reduced)

### Timing metadata / frame spacing
- Previously a real bug: frame spacing could be incorrectly inferred from sequence TR.
- Fixed by requiring explicit `TemporalResolution` / `time_resolution_sec`.
- This was causing silent fitting errors on phantom DCE runs.

### Missing phantom conversion metadata
- Previously ROCKETSHIP used default `relaxivity=3.4` and `hematocrit=0.45`, which did not match synthetic phantoms.
- Phantom DCE JSON now carries explicit `Relaxivity_per_mM_per_s`, `Hematocrit`, and `AIFConcentrationKind`.
- ROCKETSHIP Stage A now consumes those values automatically.

### AIF blood/plasma mismatch
- Diagnostic comparison showed large error when comparing ROCKETSHIP plasma `Cp` to GT without consistent metadata assumptions.
- Generator metadata was updated and ROCKETSHIP pass-through now aligns plasma assumptions (`AIFConcentrationKind=plasma`, `Hematocrit=0.0`).
- AIF agreement is now good enough that it is not the main remaining error source.

### T1 linear-fit instability as primary driver
- Linear VFA T1 fitting produced catastrophic outliers.
- Nonlinear T1 greatly improved T1 performance.
- T1 is still imperfect voxelwise, but no longer the main explanation for the large DCE parameter bias.

## Remaining Problems

### DCE fit parameter bias remains large
- `Ktrans`, `ve`, `vp` bias percentages are often still very large in phantom GT summaries.
- This persists even after:
  - DCE timing fix
  - relaxivity/hematocrit pass-through fix
  - nonlinear T1 fitting
  - phantom baseline alignment

### Most likely dominant cause: model mismatch
- Phantom generation currently uses a `2cxm` forward model (in `synthetic_dce`), while the phantom GT runner primarily evaluates:
  - `tofts`
  - `ex_tofts`
  - `patlak`
- Large systematic bias is expected when fitting simpler models to `2cxm`-generated curves, especially voxelwise and in certain tissue classes.

### Percent metrics can overstate error for near-zero GT regions
- `%` bias/MAE metrics can become very large when GT medians are near zero (especially some brain-region parameters).
- This is a reporting effect on top of real fitting bias.

## Temporary / Diagnostic Behavior (Intentional, Documented)

### Phantom-only baseline override in helper
- `tests/python/phantom_gt_helpers.py` currently sets Stage A baseline window from GT AIF `BaselineImages` when present.
- This is a phantom diagnostic alignment step so ROCKETSHIP baseline handling matches the generator metadata.
- It is not the general solution for real data.
- Real-data solution remains TODO: port MATLAB automatic baseline-window detection into Python Stage A.

### Phantom tolerance profile is provisional (not a gate)
- `tests/data/BIDS_test/phantom_gt_mae_tolerances.json` is currently exploratory.
- `tests/python/test_phantom_gt_reliability.py` now marks tolerance assertions as `xfail` when `gate_ready=false`.
- Use the runner for diagnostic review until performance is understood and tolerances are recalibrated.

## Lessons Learned

1. Metadata correctness matters as much as model math.
- Silent timing or conversion defaults can completely invalidate DCE fits without obvious errors.

2. Synthetic test packets need explicit provenance metadata.
- `TemporalResolution`, relaxivity, hematocrit, AIF kind, and baseline images should be exported with the phantom data.

3. T1 and AIF debugging are necessary but not sufficient.
- Good T1 + good AIF does not guarantee good DCE parameter recovery if the fit model differs from the generation model.

4. Separate “diagnostic phantoms” from “gating phantoms”.
- `sub-08phantom` is useful for debugging but should not silently change current tolerance-based gating until the gate is recalibrated.

## Recommended Next Steps

1. Generate matched-model phantom sets (`tofts`, `ex_tofts`, `patlak`) in `synthetic_dce` to isolate implementation error from model-mismatch bias.
2. Keep `sub-05/06/07/08` as stress tests and model-mismatch characterization datasets.
3. Recalibrate phantom GT tolerances only after the expected-model-vs-fit behavior is clearly separated.
4. Port MATLAB Stage A baseline auto-detection so phantom-only baseline override can be retired.
