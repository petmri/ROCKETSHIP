# Test-Retest Validation of ROCKETSHIP 2.0

## Goal
Use scan-rescan data to show that the Python pipeline (2.0) measures DCE parameters on real
patients **at least as repeatably as MATLAB 1.x**. It should also show that processing
contributes only a small share of the scan-rescan variance.

## Where the work lives
- **This folder (git, branch `test-retest-validation`)**: the plan, proposals, open
  questions and status (this file), `prereg.json`, run settings, the MATLAB driver and
  `matlab_arm.md`, `power_simulation.py`. Open work is listed once, in
  `project-management/TODO.md`.
- **DCEPrep branch `fix/vfa-norm-failure-diagnostics`**: the DCEPrep fixes.
- **`DCE_BIDS/derivatives/rocketship-testretest/` on the data share**: fits, tables,
  reports, QC, logs, the ad hoc check scripts, and everything that names a subject.
  Its `README.md` indexes the folder and `PROVENANCE.md` maps outputs to commits.

Nothing else holds results. Subject IDs go on the share only, never in git.

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
- **Third rater, SG** (in `derivatives/AIFArtist` by 2026-11-16): AIF masks for 49 sessions,
  45 with a hand-marked baseline end, and **tumour ROIs for 35 sessions**. Complete pairs:
  22 for the AIF (18 RIDER, 4 QIN-GBM), 15 for the tumour ROI (all RIDER). SG also left a
  flags file: 50 sessions flagged, 49 of them QIN-GBM (45 "Poor AIF", 3 "Bad data",
  1 "Missing baseline") and 1 RIDER ("Poor AIF"). See proposals 11 and 12.
- **Prior runs** (25 QIN-GBM subjects, 50 sessions):
  - `dceprep-testretest` (July): **MATLAB 1.x** outputs (`A_dceR1info.mat`,
    `dce_patlak_fit.mat`).
  - `dceprep-testretest2` (October): Python outputs.
  - `existing_runs_review/existing_runs_session_metrics.csv` already tabulates WM/GM Patlak
    Ktrans, vp, AIF fit R² and AIF peak per session for both runs.
  - They differ in DCEPrep version and AIF setup as well as pipeline, so they are a
    **descriptive** MATLAB-vs-Python comparison, not the primary one.
- **Rater runs** `dceprep-testretest_AC` / `_WN`, as found on 2026-11-16 (their
  `PROVENANCE.md` files are the record; eLabFTW experiment 20):
  - Re-run on 2026-10-28/29 with DCEPrep `fdb2944` (branch `registration-mm-smoothing`,
    not merged) and ROCKETSHIP `7127152` (branch `case-cli-aif-mode`, not merged). The first
    run (2026-10-27, DCEPrep `e455970`, 75 of 86) is kept beside them as
    `*_pre-regfix_2026-10-28`.
  - Preprocessing: **85 of 86 sessions** per rater. See "DCEPrep failures" below.
  - Patlak and extended Tofts fits exist for 85 (AC) and 84 (WN) sessions, raw AIF curve,
    blood T1 fixed per field strength.
  - Decisions already taken in those runs, before any rule in this plan was fixed: one
    motion-corrupted VFA dropped at staging, one WN session's fit set aside for a
    non-enhancing AIF, and AIF voxels without a valid VFA T1 given the cohort blood T1
    (11 AC and 6 WN sessions). See proposal 4.
- **In-house repeatability summary** `dceprep-testretest_analysis` (2026-10-29): Python-only
  scan-rescan and inter-rater tables from the rater runs. It contains no MATLAB results, but
  it means Python-only repeatability has been looked at. Not opened for this work.
- **Validation fits** `rocketship-testretest/python_AC`, `python_WN`, `python_SG`
  (2026-11-16): the Python arm fitted by `run_testretest_fits.py`, see "Status". AC and WN
  are fitted from the rater runs. There is no DCEPrep run for SG; its inputs are derived
  from the AC run (`rocketship-testretest/dceprep-derived_SG`, see "Rater SG" below).

## DCEPrep failures: root causes (2026-11-16)
The 11 failures of the 2026-10-27 run, from the logs and the data. Session-level detail
(IDs, QC tables) is in `DCE_BIDS/derivatives/rocketship-testretest/NOTES_2026-11-16.md`;
subject IDs stay out of this repository.

| Sessions | Error | Root cause | State |
|---|---|---|---|
| 5 | `ZeroDivisionError` in `VFA_norm.py` | VFA-to-DCE registration slid along z, so the white-matter mask and the registered VFA did not overlap at all | fixed in DCEPrep `fdb2944` (smoothing in mm, every voxel sampled) |
| 4 | `StatisticsError: no median for empty data` in `VFA_norm.py` | same, with partial overlap: no slice had the 1000 WM voxels the histogram fit needs | fixed in `fdb2944` |
| 1 | `fslmaths: images of different sizes` in the DCE worker | the empty-slice cleanup removed a slice mid-slab and rewrote the masks while the DCE worker read them | fixed in `e455970` and `fdb2944` (trim only slab edges, with `fslroi`) |
| 1 | `Inconsistent RepetitionTime across raw VFA sidecars` | a real protocol deviation, see below | **open: decision needed** |

- The two `VFA_norm.py` errors were symptoms. The script now stops with a message that
  names the file, the voxel counts and the likely cause (DCEPrep branch
  `fix/vfa-norm-failure-diagnostics`, with the repository's first tests).
- **The remaining session is not a tooling failure.** Its second session was acquired with
  a fifth VFA (20°) at TR 7.4 ms against 7.3 ms for the others, where its first session has
  four flips. ROCKETSHIP's check tolerates 0.1 ms and trips only through floating-point
  rounding. Fitted with one TR the T1 comes out about 0.8% low. More important, the VFA slab
  of that session sits about 13 mm above the DCE slab, a third of the DCE slab has no VFA
  data, and the five flips are registered up to 7 mm apart. Proposal 2.
- **A crash-free run is not a correct one.** A new check
  (`scripts/vfa_registration_qc.py` on the DCEPrep branch) compares the transforms of the
  flips of one session, which should nearly agree because the VFAs are acquired back to
  back. Median disagreement is 0.7 mm and the 95th percentile 2.9 mm, but one session that
  now preprocesses cleanly has its lowest flip 12.8 mm from the other three; at a 3 mm
  limit 15 of 86 sessions are flagged, mostly through the lowest flip angle. Proposal 1.
- DCEPrep cannot be re-run on the analysis workstation as it stands (no HD-BET, empty
  DCEPrep venv), so no session was re-preprocessed for this work.

## Rater SG without a DCEPrep run (2026-11-16)
DCEPrep cannot run on the analysis workstation, and its preprocessing depends on the rater
in one place only: the dynamic series is masked by brain-or-AIF, so an AIF voxel outside
the brain mask survives only in the run of the rater who drew it.
`run_testretest_derive_rater.py` uses that. For each session SG drew, it links the AC
run's preprocessed files, moves SG's mask onto the AC run's grid, and writes the AIF T1
map, the baseline sidecar and the tumour ROI. A session is derived only if every AIF
voxel has signal in the AC run's series.

- The regridding reproduces the AC run's own AIF masks exactly on all 86 sessions.
- **36 of SG's 49 sessions are derived (14 complete pairs: 12 RIDER, 2 QIN-GBM).** 12 need
  a real DCEPrep run, because SG's AIF lies partly or wholly outside what the AC run kept,
  and 1 is the session that fails preprocessing. A DCEPrep run for SG would bring the
  complete pairs to 21 or 22.
- SG's tumour ROIs do not depend on the AIF and are written for all 35 sessions
  (`rocketship-testretest/regions_SG`). With `collect --regions-root` and
  `regions_tumor.json` they apply to every rater's fits: regions `tumor`, `WM_nontumor`
  and `GM_nontumor` next to `WM` and `GM`.
- Check of the method: rater WN's inputs derived from the AC run the same way, fitted,
  and compared with the fits from WN's own DCEPrep run (73 sessions). The AIF masks are
  identical in all 73. WM-median Patlak Ktrans agrees within 0.1% in 53 sessions and
  within 1% in 67; the largest difference is 4.8%. The fits are not identical because the
  two DCEPrep runs are not: their T1 maps differ (WM median by up to 2.7%, median 0.02%)
  although nothing about the T1 map depends on the rater, and the Ktrans difference
  follows the T1 difference (Spearman 0.42). So a derived run stands in for a real one
  to within DCEPrep's own run-to-run variation. See proposal 15.

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
- [x] `python/repeatability.py`: wSD, RC, log-scale wCV, ICC(2,1), Bland–Altman, subject
      bootstrap, wSD-ratio non-inferiority, processing SD, stratified split, session
      pairing. Unit-tested on synthetic pairs with a known wSD
      (`tests/python/test_repeatability.py`).
- [x] Runner, Python side: `run_testretest_fits.py` fits a rater run from one config
      (`testretest_run_config.json`) into a separate root; `run_testretest_analysis.py`
      collects ROI medians, audits metadata, and runs the analysis in `synthetic`,
      `shuffled` or `unblinded` mode (`python/testretest_study.py`). The report has the
      tables plus Bland–Altman and paired plots per group.
  - ROI medians are taken from the parameter maps inside the DCEPrep WM/GM masks, not from
    the ROI `.xls` (which fits the ROI-mean curve, not the voxels).
  - [ ] MATLAB side: set up to run on another machine, not yet run.
        `run_testretest_export_matlab_jobs.py` writes a job list from the Python fits and
        `matlab/run_testretest_matlab.m` runs it (`matlab_arm.md`). The driver is untested.
  - [x] Other raters without a DCEPrep run: `run_testretest_derive_rater.py`.
- [x] Processing-variance runner: `run_testretest_processing_variance.py` with
      `processing_factors.json`. Tested on the tiny fixture; not yet run on the cohort.
- [x] Blinding: `prereg.json` holds every decision as null; the unblinded mode and the
      variance budget's comparison with the rescan wSD are refused until it is complete
      and signed off.
- [ ] Synthetic sanity check at the image level: the repeated-noise synthetic set in
      `TODO.md` (Secondary 1) still does not exist. The statistics are validated on
      generated tables (`run_testretest_analysis.py selfcheck`), not through the pipeline.

## Dependencies and risks
- **Data distribution**: real test-retest data cannot go in git. CI coverage, if any, waits
  on `projects/large-data-distribution/`. Until then this runs locally against the data
  root; derived tables, reports and anything that names a subject stay on the share
  (`rocketship-testretest/`). Commits carry code, settings and counts only.
- **Known Python/MATLAB divergences** will show up here and need to be explained, not
  tuned away:
  - backend drift on real data
  - `VolumeTiming` precedence differences (`docs/dce_options.md`)
  - MATLAB's `script_preferences.txt` relaxivity fallback
- **Small cohorts**: CIs on wSD ratios will be wide. Report them honestly rather than
  claiming equivalence from point estimates.

## Proposals (2026-11-16, none applied)
Nothing below changes a pre-set decision; each is for the open questions that follow.

1. **Register the VFA set with one transform** (DCEPrep). Each flip is registered to the
   DCE on its own, and the lowest flip angle, with the least contrast, is the one that
   strays. One transform for the set removes the failure mode; the consistency check then
   guards it. It changes every T1 map, so it needs a full preprocessing re-run, and it
   should land together with DCEPrep#14 (bias-corrected VFAs into the z-normalization),
   before the MATLAB arm is run. Until then: decide a limit for the check (3 mm flags 15 of
   86 sessions; one session is at 12.8 mm) and whether exceeding it excludes the pair.
2. **The session that still fails.** Options: (a) exclude the pair as an acquisition
   deviation (VFA slab off the DCE slab, extra flip, different TR); (b) re-stage it with
   the four flips of its first session, register with the shared transform of proposal 1,
   and keep it only if VFA coverage of the brain mask is adequate. Proposed rule for (a),
   blind to outcome: exclude a pair when a session's VFA slab covers less than a set
   fraction of the DCE brain mask. Separately, ROCKETSHIP's TR check should name the flips
   and state its tolerance, and per-flip TR in the T1 fit would make the case fittable.
3. **Match the VFA flip set within a pair.** Two pairs have different flip sets in their
   two sessions (the session above, and the one that lost a motion-corrupted flip at
   staging). Dropping the same flip from the partner session keeps the T1 estimate
   comparable between test and retest. The metadata audit flags both.
4. **Bring earlier decisions under the pre-registered rules.** The dropped VFA, the
   set-aside WN session and the AIF T1 patch were decided during the rater runs. Record
   them in `exclusions_manual.tsv` with their reasons, or reverse them. The validation
   fits apply none of them beyond what is in the staged inputs, so the WN session with the
   non-enhancing AIF is not set aside there; check its row in `fit_manifest.tsv`.
5. **Decision interval for non-inferiority.** In simulation at 19 and 24 pairs the
   percentile bootstrap's 95% interval covers the true ratio 90–92% of the time, so its
   upper limit is too low too often; Pitman's interval for correlated variances covers
   94–96%. Proposed: decide on the Pitman interval and report the bootstrap beside it.
   Also state whether "upper 95% CI" means the upper limit of a two-sided 95% interval
   (assumed in the code) or a one-sided 95% bound.
6. **Margin and power.** Chance of concluding non-inferiority when the pipelines are
   truly equal (Pitman 95% upper limit below the margin; simulation, 1500 replicates).
   "Shared" is the fraction of rescan variance common to both pipelines.

   | Pairs | Shared | Margin 1.1 | Margin 1.25 |
   |---|---|---|---|
   | 19 (RIDER) | 0.80 / 0.95 / 0.99 | 0.08 / 0.22 / 0.73 | 0.31 / 0.78 / 1.00 |
   | 24 (QIN-GBM) | 0.80 / 0.95 / 0.99 | 0.09 / 0.26 / 0.85 | 0.38 / 0.89 / 1.00 |
   | 43 (pooled) | 0.80 / 0.95 / 0.99 | 0.18 / 0.48 / 0.98 | 0.64 / 0.99 / 1.00 |

   A 1.1 margin can only be met if the two pipelines' test-retest differences nearly
   coincide. Reproduce with `power_simulation.py` in this folder. Analysing the cohorts separately roughly halves the power of the pooled
   analysis, which bears on the pooling question.
7. **AIF-fit exclusion rule.** With the raw AIF curve there is no AIF fit, so
   `fit_rsquared_cp_adj` does not exist for these runs. Use the fitted AIF for the primary
   analysis, replace the rule (for example a minimum AIF enhancement), or set it to
   `not_applied`.
8. **"Normal-appearing" needs lesion masks.** The DCEPrep WM/GM masks of these glioma
   patients include tumour and oedema. Rater SG's tumour ROIs (35 sessions, 15 RIDER
   pairs) are now on the fit grid and the collector offers `WM_nontumor` and `GM_nontumor`
   (`regions_tumor.json`); QIN-GBM has no complete pair with a tumour ROI. The endpoint
   needs to say which regions carry the claim.
9. **MATLAB arm.** Decide which code is "1.x" (the `1.2` tag or the MATLAB side of the
   validation commit). The driver that replaces `run_dce_cli` targets the latter
   (`matlab_arm.md`).
10. **cpufit as a backend level.** `backend` accepts `auto`, `cpu`, `gpufit`; cpufit is only
    reached through `auto` without CUDA. An explicit value would let the variance runner
    cover it.

11. **SG's flags.** SG flagged 49 QIN-GBM sessions, 45 of them for a poor AIF, and drew an
    AIF for only 12 of the flagged sessions. That is a rater's judgement on almost the
    whole QIN-GBM cohort, made without seeing any outcome, and it bears on the pooling
    question and on the exclusion rules ("Bad data" and "Missing baseline" map onto the
    plan's failed-or-missed-injection rule). Decide how the flags enter
    `exclusions_manual.tsv`, if at all.
12. **A DCEPrep run for SG.** The derived SG run covers 14 complete pairs; a real run would
    cover 21. Preprocess SG on the DCEPrep machine when convenient, and until then treat SG
    results as coming from the derived subset.
13. **Baseline frame convention.** The staging script copies AIFArtist's
    `SteadyStateEndTimeIndex` unchanged; its own docstring says AIFArtist counts from 0 and
    ROCKETSHIP from 1, so hand-marked baselines (WN, SG) end one frame early. The derived
    SG run copies it the same way, to stay consistent with WN. Decide whether to add one
    everywhere before the fits of record.
14. **What MATLAB does where Python auto-detected the baseline** (`'python'` or `'auto'`
    in the MATLAB driver): pass Python's frame, or let MATLAB's detector choose. All of
    rater AC is auto-detected, so for AC this decides whether the detector is part of the
    comparison.

15. **DCEPrep is not reproducible run to run, and that leaks into the rater comparison.**
    The AC and WN runs preprocessed the same data separately and their T1 maps differ by
    up to 2.7% in the WM median. An AC-versus-WN comparison between those runs therefore
    mixes the rater with preprocessing noise. Comparing raters on one preprocessing (the
    derived-run route, or one DCEPrep run with all raters' masks) removes it. For
    identical repeat runs DCEPrep's README asks for one ITK thread; decide whether the
    fits of record need that, and add "DCEPrep re-run" as a factor of the variance budget.
16. **The pure-CPU backend is too slow to be a variance-budget level on whole-brain
    voxel fits.** One session did not finish in nine minutes (the GPU takes about 15 s).
    Run the backend factor on a sub-sampled ROI, or on `cpufit` once it can be selected
    (proposal 10).

17. **Perturb the AIF mask differently.** The manual AIFs are a few voxels, so one-voxel
    in-plane erosion empties them, and dilation reaches voxels DCEPrep zeroed outside the
    brain-or-AIF mask. Options: drop each voxel in turn (leave-one-out), or perturb within
    the voxels the run kept; for dilation, preprocess with a dilated mask so the extra
    voxels keep their signal.

## Open questions
Answered by the data survey above: cohort size and rescan intervals, scanners and protocol,
`VolumeTiming` (absent), layout (BIDS + DCEPrep staging), existing MATLAB results (July run
only; not matched to the current inputs). Still open:
- [ ] **Primary endpoint**: which model × region × parameter carries the 2.0 claim? GM/WM
      label maps exist; Patlak Ktrans in normal-appearing WM/GM is the working assumption.
      Tumour ROIs now exist from rater SG for 15 RIDER pairs and no QIN-GBM pair, so a
      tumour endpoint, or WM/GM with the tumour removed, is possible for RIDER only
      (proposal 8).
- [ ] **Pooling**: analyse RIDER (1.5 T) and QIN-GBM (3 T) separately, or pooled with
      collection as a factor? Proposed: separately, with pooled results reported as
      secondary.
- [ ] **QIN-GBM without AIF masks**: 25 subjects are staged with one session and no masks.
      Draw masks for both sessions, use auto-AIF for them, or leave them out?
- [ ] **Prostate (15 pairs)**: include in a later phase with an assumed fixed T10, or
      exclude? If included, check whether `TriggerDelayTime` spans `n` or `n-1` frame
      intervals (about 2% on the time axis). That shifts accuracy but not repeatability.
- [ ] **DCEPrep failures**: 10 of the 11 are fixed (see "DCEPrep failures"). Still to
      decide: the one remaining session (proposal 2), a limit for the registration
      consistency check and what exceeding it means (proposal 1), and whether to
      re-preprocess with a shared VFA transform before the MATLAB arm.
- [ ] **Non-inferiority margin** and exclusion thresholds: agree before unblinding. Enter
      them in `prereg.json`; the code refuses the real analysis until it is complete
      (proposals 5–7).
- [ ] **MATLAB**: where it will run, and which code is "1.x" (proposal 9).
- [ ] **Which rater** carries the primary claim: AC, WN or SG. SG covers fewer pairs
      (proposal 12).
- [ ] **Development/held-out split**: proposed split by subject within each collection,
      stratified, about 1/3 development. Fix the seed and record the split in this folder.

## Status
- [x] Approach drafted (2026-10-27).
- [x] Data surveyed and the plan filled in (2026-10-27).
- [x] DCEPrep failures root-caused (2026-11-16): 10 of 11 fixed, 1 awaiting a decision;
      diagnostics, tests and a registration QC check on DCEPrep branch
      `fix/vfa-norm-failure-diagnostics`.
- [x] Statistics module + synthetic validation (2026-11-16).
- [x] Rater runs fitted, Python arm (2026-11-16), all at ROCKETSHIP `318180d`, Patlak and
      extended Tofts with QoF maps, in `rocketship-testretest/`: `python_AC` 85 sessions,
      `python_WN` 85, `python_SG` 36 (derived inputs). The one session without a fit is
      the one that fails preprocessing. Each folder has `fit_manifest.tsv` and
      `provenance.json`; `PROVENANCE.md` beside them records both repositories' commits.
  - Repeat fits of AC and WN on the same inputs and code are not bit-identical (GPU), but
    the difference is confined to at most 40 voxels a session: Patlak Ktrans WM/GM
    medians are unchanged in all 340 cases, and no ROI median of either model moved by
    more than 0.2%.
- [ ] MATLAB run on the same staged inputs: **not run.** MATLAB is not installed on the
      analysis workstation; the arm is set up for another machine (`matlab_arm.md`): job
      lists `rocketship-testretest/matlab_jobs_{AC,WN,SG}.json`, driver
      `matlab/run_testretest_matlab.m` (untested).
- [x] Rater SG brought in without a DCEPrep run (2026-11-16): 36 of 49 sessions, tumour
      ROIs for all 35.
- [x] Analysis code, blinded (2026-11-16): tested on synthetic tables and on the
      label-shuffled real table of all three raters (`rocketship-testretest/tables`, 3530
      ROI medians, regions WM, GM, tumour and WM/GM without tumour). No real repeatability
      result has been computed or viewed.
- [ ] Remaining open questions answered and `prereg.json` signed off.
- [ ] Processing-variance budget: runner piloted on four subjects of rater AC (eight
      sessions, all factors but the backend, proposal 16), 112 refits, none failed. Not
      run on the cohort, and its comparison with the rescan wSD stays locked. The pilot
      showed the AIF-mask factor barely works on hand-drawn AIFs (proposal 17): eroding
      left no voxel in all 8 sessions, and dilation reached voxels the run had zeroed in 6
      of 8. Another rater's mask had no signal in the AC run in 1 of 8 sessions.
- [ ] Held-out confirmation and report.
