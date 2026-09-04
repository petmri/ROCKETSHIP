# Detecting non-steady-state leading frames

Status: implemented in Python (`python/dce_transient.py`), off by default.
Prototype and evidence: `~/Code/chop`, September 2025.

## Problem

The first frames of a dynamic series are acquired before the magnetisation reaches steady
state. Runs have historically trimmed a fixed two frames (`start_t: 3`), arrived at by
escalation: one frame at first, then two when oscillations still caused trouble, then
walk-back logic where two left the baseline too short. A fixed count is wrong in both
directions on a heterogeneous dataset, so `start_t_auto_method: transient` detects it
per series instead. The fixed count remains available and still wins when set.

Naming hazard: "steady state" here is the *magnetisation* steady state. ROCKETSHIP's
`steady_state_start`/`steady_state_end` mean the pre-contrast baseline window, which is a
different thing at a different point in the series.

## Evidence

Evaluated on the USC-PPG DCE dataset, 938 series across ~650 subjects
(`/media/network_mriphysics/USC-PPG/bids_ppg/sourcedata/raw`), 64 frames each, mostly
320x320x14 coronal VIBE. All 938 mean-signal curves and per-slice profiles were cached, so
the numbers below are reproducible without re-reading the images.

**The volume mean alone is not sensitive enough.** Judging frame 0 by its mean elevation
flags 235 of 938 series. Adding the banding channel flags 521. The 286 extra are series whose
first frame is visibly banded while its mean sits within noise — in the starkest case
(`sub-467368_ses-01`) the mean deviation is 1.5σ while banding is 32x the plateau level. The
mechanism explains why: uneven k-space partition weighting redistributes signal across slices
more than it changes the total, so it largely cancels in the mean.

**Banding metric.** Per frame: each slice divided by its own plateau mean (cancels anatomy
exactly), detrended along the slice axis with a 5-slice moving average, RMS of the residual,
expressed as a multiple of the plateau frames' own RMS. The detrend width matters — the
observed period is 3-4 slices, not alternating slices, so a second difference (the obvious
first choice) attenuates it by about half.

**Threshold.** Measuring a frame that should be settled (frame 1, in series with late
arrival) with the same out-of-sample normalisation gives the null: median 1.4, p90 2.7, p99
4.5. `start_t_auto_osc_z` is 5.0, just outside that, and was confirmed by review of borderline
cases. Human review of 40 series agreed with the mean-channel verdicts before the banding
channel was added; the banding threshold was set by inspection of the review figures rather
than by a labelled set, so it remains the weakest-evidenced constant here.

**Distribution at the shipped defaults:** chop 0 for 420 series, 1 for 513, 2 for 5, and 6
declined. Under the previous fixed `start_t: 3` every one of these would have lost two frames.

**Sequential threshold.** A second transient frame is detectable only in context: across
chop-1 series the frame-1 mean deviation centres on zero (median -0.1σ), but five series sit
at 2.0-3.6σ, clearly separated from that null yet under a flat threshold of 4. Hence
`start_t_auto_z` 4.0 to trigger and `start_t_auto_z_ext` 2.0 to extend. Banding at frame 1 is
near 1.0 even in those cases — the residual is a smooth mean offset, not banding — which is
why the two channels have different jobs.

## The arrival dependency, and why the detector declines

Both channels measure frame 0 against a plateau of remaining pre-arrival frames, so both are
only as good as the arrival estimate. Where arrival is mislocated late, the plateau is drawn
from post-contrast frames and the *real* baseline reads as the transient — the failure chops
the entire baseline away, which is worse than doing nothing.

Observed on `sub-1101409_ses-02`: a single spike frame at 31 (703 against neighbours of 540)
out-jumped the true wash-in at frame 3, because the real arrival rises over three frames
(~30 per smoothed step) while the spike is instantaneous (56). Arrival came back as frame 30,
the plateau was post-contrast, and all three true baseline frames were trimmed.

`start_t_auto_max_baseline` (8) catches this: 932 of 938 series have arrival at or before
frame 7. The 6 beyond it — 2 spike-hijacked, 2 non-enhancing series with no arrival at all,
2 others — return no answer and a reason rather than a chop. Every chop-3 result in the
dataset was this failure; with the guard the chop-3 bucket is empty.

Improving arrival detection itself is deliberately out of scope here (owned separately).

## Open questions

- Three series (`sub-1101879_ses-02`, `sub-203708_ses-01`, `sub-1102028_ses-02`) have a
  correct arrival but a frame 0 that sits 3-5σ *below* the plateau while still banded, so the
  banding channel trims them. Whether a low-but-banded first frame should be trimmed has not
  been decided; the mean channel is one-sided (high) by design and would not have fired.
- The 6 declined series are all arguably QC failures rather than chop problems. Whether a
  non-enhancing series should fail a run outright is a pipeline-level question, untouched here.
- Thresholds are evidenced on one dataset from one scanner and protocol.

## Implementation notes

`python/dce_transient.py` holds the detector; `_resolve_auto_start_t` in
`python/dce_pipeline.py` wires it into Stage A, before the timepoint trim, reusing the
existing `steady_state_auto_method` detector for arrival. Noise σ comes from
`dce_sigma.successive_difference_sigma` — the whole series has far more samples than the
three or four baseline frames, and lag-1 differencing is blind to slow drift. No new noise
code was added; `dce_sigma` was already the shared implementation.

MATLAB is not covered: this is a Python-only feature by request.
