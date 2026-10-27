# Test-Retest Validation of ROCKETSHIP 2.0

## Goal
Use scan-rescan data to show that the Python pipeline (2.0) measures DCE parameters on real
patients **at least as repeatably as MATLAB 1.x**. It should also show that processing
contributes only a small share of the scan-rescan variance.

## What this can and cannot show
In vivo data has no ground truth, so test-retest measures **precision, not accuracy**.
Accuracy stays with the OSIPI DROs (`tests/data/osipi/`) and the phantom ground truth
(`projects/phantom-gt/`). Test-retest covers what those cannot: stability on real
acquisitions with real noise, motion, AIF variation and metadata quirks.

Observed scan-rescan variance = biology + acquisition + processing. ROCKETSHIP controls only
the last, so the claims are about that term and about MATLAB vs Python, never about absolute
truth.

## Data (surveyed 2026-10-27)
`DCE_BIDS` below is the lab's DCE BIDS share (raw data under `DCE_BIDS/sourcedata/raw`).
`participants.tsv` marks a usable pair with `test_retest=1`: test-retest design, exactly two
sessions, DCE in both. Every subject's `*_sessions.tsv` gives `days_from_first` and the
`test_retest_role` of each session.

| Collection (TCIA) | Pairs | Rescan interval | Scanner | Dynamic | Frame spacing source | T1 |
|---|---|---|---|---|---|---|
| RIDER Neuro MRI (glioma) | 19 | 1–2 days | Siemens 1.5 T | 65 frames × 4.76 s, FA 25° | `RepetitionTime` (with `RepetitionTimeExcitation`) | VFA 5/10/20/30° |
| QIN-GBM Treatment Response | 49 | 1–8 days (median 4) | Siemens Trio 3 T | 60 frames × 6.01 s, FA 10°, dual echo | `RepetitionTime` (with `RepetitionTimeExcitation`) | VFA 2–30° |
| QIN Prostate Repeatability | 15 | 3–14 days (median 9) | GE 3 T (MR750w, Signa HDxt) | 40–54 frames, FA 12/15° | `TriggerDelayTime / n_frames` | **no VFA** |

- **No sidecar in the cohort has `VolumeTiming`.** Frame spacing comes from the scalar
  MATLAB-convention fields, identically in both pipelines.
- **QIN-GBM DCE is dual-echo.** Each session has an echo-1 and an echo-2 magnitude series
  (`run-01`/`run-02`), and some sessions also have phase series. Some sessions also have
  their VFA volumes filed under `dce/`. The staging step picks echo 1 explicitly; never let
  `*DCE.json` discovery choose on these raw folders.
- **Prostate** has no T1 mapping and is not brain, so DCEPrep does not stage it. It is out of
  scope for phase 1 (see open questions).

### Existing processing (`DCE_BIDS/derivatives`)
- **Staged inputs**: `sourcedata/raw_dceprep_testretest`, 111 sessions, built by the
  `in-house_toolbox` script `dce_stage_dceprep_testretest.py`. `staging_manifest.tsv`
  records per session the chosen series, VFA set, field strength, relaxivity (3.8 at 3 T,
  4.3 at 1.5 T), both raters' AIF masks, the baseline end, and flags. Pairs it covers:
  - **43 with AIF masks from both raters (AC, WN): 24 QIN-GBM + 19 RIDER.**
  - 25 more QIN-GBM subjects are staged with one session and no AIF masks.
- **AIF masks**: `derivatives/AIFArtist`, two raters per session. AC used an automatic
  baseline end; WN set the baseline-end frame by hand. Rater and baseline method are
  therefore **confounded** in an AC-vs-WN comparison (see comparison 2).
- **Prior runs** (25 QIN-GBM subjects, 50 sessions):
  - `dceprep-testretest` (July): **MATLAB 1.x** outputs (`A_dceR1info.mat`,
    `dce_patlak_fit.mat`).
  - `dceprep-testretest2` (October): Python outputs.
  - `existing_runs_review/existing_runs_session_metrics.csv` already tabulates WM/GM Patlak
    Ktrans, vp, AIF fit R² and AIF peak per session for both runs.
  - They differ in DCEPrep version and AIF setup as well as pipeline, so they are a
    **descriptive** MATLAB-vs-Python comparison, not the primary one.
- **Rater runs** `dceprep-testretest_AC` / `_WN` (2026-10-27, ROCKETSHIP Python
  `7127152` via DCEPrep `e455970`; eLabFTW experiment 19):
  - Preprocessing finished for 75 of 86 AC cases. The 11 failures are DCEPrep errors
    (`ZeroDivisionError`, empty-median `StatisticsError`, one inconsistent VFA
    `RepetitionTime`), not ROCKETSHIP errors.
  - No model fits exist yet apart from two trial `ex_tofts` maps under WN.

## Design: three comparisons

1. **Python vs MATLAB repeatability (primary claim).** Run both pipelines on every session
   with matched settings. Compare scan-rescan repeatability per model × region × parameter.
   Treat it as a **non-inferiority** test: Python's wSD is no worse than MATLAB's by more
   than a margin fixed before unblinding (proposed: the upper 95% CI of the wSD ratio
   Python/MATLAB < 1.1; to be agreed).
   - "Matched" means the **same staged inputs and the same AIF masks** from the rater runs:
     run MATLAB `run_dce_cli` on the DCEPrep outputs the Python fits use, with its
     `script_preferences.txt` set to the Python run config's values.
   - The July/October pair cannot serve: DCEPrep version and AIF setup changed with the
     pipeline.
2. **Processing-only repeatability (variance budget).** Re-run single sessions while varying
   only processing choices, one factor at a time:
   - backend (`cpu` / `cpufit` / `gpufit`)
   - `steady_state_auto_method`
   - `start_t` ± 1 frame
   - AIF mask eroded/dilated by one voxel
   - AIF rater (AC vs WN masks). To separate rater from baseline method, cross them: AC
     masks with WN's manual baseline end, and WN masks with auto-detection.
   - fitted vs raw AIF

   Report each factor's spread relative to the scan-rescan wSD. A factor whose processing
   spread is a sizeable fraction of the rescan wSD is the next robustness target.
3. **External plausibility.** Compare scan-rescan repeatability with published values for
   the same protocol, model and region. This catches the case where both pipelines agree
   and are both noisy.

## Metrics (QIBA conventions)
- **wSD** and the **repeatability coefficient** RC = 2.77 × wSD: the smallest change
  between two scans that counts as real.
- **wCV** for parameters that are strictly positive and well above 0 (tumor Ktrans, ve,
  vp), computed from log-transformed values. For near-zero parameters (BBB Patlak Ktrans),
  report absolute wSD only; wCV is unstable near zero.
- **ICC(2,1)** (two-way random, absolute agreement): between-subject vs within-subject
  variance.
- **Bland–Altman**: bias, 95% limits of agreement, and a check that the differences do not
  grow with the mean value.
- **Uncertainty**: bootstrap CIs resampled **by subject**, not by ROI or voxel.

The unit of analysis is the **ROI median** per subject × session × region × model ×
parameter. Voxel-wise agreement needs registration between sessions, which adds its own
error; it is a secondary, descriptive analysis only.

## Analysis rules fixed before looking at results
- **Exclusion criteria**, applied blind to the outcome:
  - excessive motion (threshold TBD)
  - AIF fit adjusted R² below a threshold (`fit_rsquared_cp_adj` in `b_out.json`)
  - reliable fraction of the χ²_ν QoF map below a threshold (`write_qof_maps`)
  - failed or missed injection

  Log every exclusion with its reason.
- **Settings are fixed across sessions and pipelines.** Record them in one run config
  committed with the analysis:
  - AIF mode, baseline method, blood T1, hematocrit, relaxivity
  - `start_t`/`end_t`, model flags, backend
- **Metadata audit.** Diff Stage-A `metadata_sources` between paired sessions. Flag any pair
  whose TR/FA/time resolution came from different sidecar fields, including `VolumeTiming`
  in one session and not the other.
- **Held-out split.** Develop on a subset of subjects and confirm on the rest. Any pipeline
  change made during the work is judged only on the held-out subjects.

## Tooling to build
- [ ] `python/repeatability.py`: pair sessions by subject. Input is a tidy table of ROI
      medians (subject, session, pipeline, model, region, parameter, value). Output is the
      metrics above with bootstrap CIs. Unit-tested on synthetic pairs with a known wSD.
- [ ] Runner: batch both pipelines over the cohort, collect ROI medians from the Part E /
      ROI `.xls` outputs, call `repeatability.py`, and write a report (tables +
      Bland–Altman and per-subject paired plots).
  - `run_dce_bids_batch.py` covers Python batch processing.
  - The MATLAB side runs `run_dce_cli` per session.
  - Start the plots from `python/plot_ktrans_correlation.py`.
- [ ] Processing-variance runner for comparison 2 (one session, one factor at a time).
- [ ] Synthetic sanity check: the repeated-noise synthetic set in `TODO.md` (Secondary 1)
      should give the wSD it was generated with. This validates the statistics code before
      it touches real data.

## Dependencies and risks
- **Data distribution**: real test-retest data cannot go in git. CI coverage, if any, waits
  on `projects/large-data-distribution/`. Until then this runs locally against the data
  root, and only derived tables/reports are committed (no PHI).
- **Known Python/MATLAB divergences** will show up here and need to be explained, not
  tuned away:
  - backend drift on real data
  - `VolumeTiming` precedence differences (`docs/dce_options.md`)
  - MATLAB's `script_preferences.txt` relaxivity fallback
- **Small cohorts**: CIs on wSD ratios will be wide. Report them honestly rather than
  claiming equivalence from point estimates.

## Open questions
Answered by the data survey above: cohort size and rescan intervals, scanners and protocol,
`VolumeTiming` (absent), layout (BIDS + DCEPrep staging), existing MATLAB results (July run
only; not matched to the current inputs). Still open:
- [ ] **Primary endpoint**: which model × region × parameter carries the 2.0 claim? GM/WM
      label maps exist; Patlak Ktrans in normal-appearing WM/GM is the working assumption.
      Are tumor (enhancing-lesion) masks available or planned for RIDER/QIN-GBM?
- [ ] **Pooling**: analyse RIDER (1.5 T) and QIN-GBM (3 T) separately, or pooled with
      collection as a factor? Proposed: separately, with pooled results reported as
      secondary.
- [ ] **QIN-GBM without AIF masks**: 25 subjects are staged with one session and no masks.
      Draw masks for both sessions, use auto-AIF for them, or leave them out?
- [ ] **Prostate (15 pairs)**: include in a later phase with an assumed fixed T10, or
      exclude? If included, check whether `TriggerDelayTime` spans `n` or `n-1` frame
      intervals (about 2% on the time axis). That shifts accuracy but not repeatability.
- [ ] **DCEPrep failures** (11/86 AC cases): fix and re-run before the analysis, or treat
      as exclusions? Fixing is cleaner, since these are tooling failures, not data quality.
- [ ] **Non-inferiority margin** and exclusion thresholds: agree before unblinding.
- [ ] **Development/held-out split**: proposed split by subject within each collection,
      stratified, about 1/3 development. Fix the seed and record the split in this folder.

## Status
- [x] Approach drafted (2026-10-27).
- [x] Data surveyed and the plan filled in (2026-10-27).
- [ ] Remaining open questions answered.
- [ ] Statistics module + synthetic validation.
- [ ] Rater runs fitted (Python), and MATLAB run on the same staged inputs.
- [ ] Processing-variance budget.
- [ ] Held-out confirmation and report.
