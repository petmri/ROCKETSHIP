# Batch Parity Status (MATLAB vs Python DCE)

## Status: archived (2026-07-28) — all four numbered issues closed; two testing gaps moved to TODO.md

> **Archived note:** every issue tracked in this document is resolved. The DCE parity gate
> stands at **12/12 gated voxelwise checks** and **4/4 ROI-xls checks** passing, with **no
> hand-curated exceptions left** — the last one (tofts+GM) was retired 2026-07-28 once the
> Stage-B AIF fix made it unnecessary. Sub-docs `aif_fitting_parity.md`, `quality_of_fit.md`
> and `sigma_estimators.md` are archived alongside this one.
>
> **What stayed live** (moved to `project-management/TODO.md`, do not track them here):
> testing gaps **A** (Stage-B AIF contract gate) and **B** (backend-equivalence gate), and
> **QoF-aware ROI stats**. Gaps **C** (dense-ROI companion) and **D** (dedicated CI parity job)
> were dropped as won't-do, as were the QoF/σ refinements listed in the sub-docs.
>
> This file is a historical snapshot of how DCE parity was achieved. Do not treat it as the
> live tracking doc.

_Last reviewed: 2026-07-28._

## Scope
Parity between the MATLAB reference pipeline and the Python port for DCE parameter maps
(Ktrans/ve/vp/fp), on two fronts:
- the committed `sub-10bbbdownsample` fixture (`test_bbb_p19_region_parity`,
  `test_bbb_p19_roi_xls_parity`), and
- the end-to-end batch pipeline (`run_dce_bids_batch.py`) on real `RUNNER_DATA` sessions.

**Gated set:** tofts + patlak **Ktrans only** (cpu & auto vs MATLAB), across all three regions
(brain/GM/WM). Everything else (ex_tofts/tissue_uptake/2cxm, non-Ktrans params, backend
auto-vs-cpu) is reported-only — not identifiable on this fixture. Suite layout in
`tests/README.md`.

**Final gate state (2026-07-28): 12/12 gated voxelwise checks pass, with no hand-curated
exceptions.** The last exception — tofts+GM reported-only, on the grounds that tofts Ktrans was
non-identifiable there — was retired once the Stage-B AIF fix lifted it to `corr` 0.980 (cpu) /
0.992 (auto) against a 0.95 floor. Worth noting what that says about the original diagnosis: GM
non-identifiability was real but was *not* the reason the numbers disagreed. Lowest gated value
now is 0.9751 (`tofts_ktrans_brain_cpu_vs_matlab`). ROI-xls: 4/4.

**Workstreams (all closed)**
- **Per-voxel quality-of-fit (QoF) reliability metric** — the general solution to the recurring
  "noisy/non-conforming voxel pollutes parity and analysis" problem that the Spearman swap and
  hand-curated ROIs only partially mask. **Parity side landed + committed (`153728e`)**: reduced-χ²
  (estimator B) filters the gate at absolute χ²_ν ≤ 6.0 (resolved issue #1). **No MATLAB-side QoF**
  (single-sided Python-CPU mask is a data-quality filter — see `quality_of_fit.md`). Pipeline hook,
  σ-outlier robustification and `RUNNER_DATA` validation all landed; **QoF-aware ROI stats** is the
  one piece never built and is now tracked in `TODO.md`.
  Full record: [`quality_of_fit.md`](quality_of_fit.md); σ detail in
  [`sigma_estimators.md`](sigma_estimators.md).
- **Stage-B AIF fitting parity** — the largest single contributor to the remaining gap, and the
  root cause of issue #2. Full record: [`aif_fitting_parity.md`](aif_fitting_parity.md).

## Current State

### Done / committed
- **Steady-state end = `tv` (default, both languages)** — commit `a9d78b6`. Ported to MATLAB
  (`dce/find_end_ss_tv.m`), numerically verified against Python
  (`tests/python/test_find_end_ss_tv_matlab_parity.py`). Replaced the old
  `piecewise_constant`/`find_end_ss` default, which scored ~0% vs `tv`'s ~88-93% on 224
  AIFArtist-rated real sessions. Full history:
  `projects/archived/steady-state-tv-default/STATUS.md`.
- **Stage-D fit-backend consolidation** — all five models migrated; archived
  (`projects/archived/stage-d-fit-consolidation/`).
- **Gated Ktrans checks use Spearman (rank) correlation, not Pearson** — commit `49cbe34`. A
  single high-leverage/non-identifiable voxel no longer collapses the metric; a genuine
  regression still drives Spearman toward 0/negative. Thresholds unchanged
  (`model_ktrans_corr_min=0.95`). Mirrors the same swap in `check_matlabref_map_drift.py`
  (`ddc35c0`). Effect: 9/10 gated checks pass (was 8/10);
  `patlak_ktrans_brain_auto_vs_matlab` flipped to pass (Spearman `0.9736` vs Pearson `0.8779`).
- **Gated/reported split** — obsoletes the former "ex_tofts required-failure" and
  weighted-AIF-regresses-ex_tofts TODOs: ex_tofts is no longer a gate.
- **QoF reduced-χ² masking in the parity gate** — commit `153728e`. `python/dce_sigma.py`
  (estimator B σ + reduced χ²) + `python/dce_qof.py` (per-voxel σ/χ²_ν maps + `reliable_mask`);
  `test_bbb_p19_region_parity` excludes voxels with χ²_ν > 6.0 (calibrated on cross-backend
  divergence). Resolves issue #1 (patlak+brain 0.937→0.990) on principled grounds. 27 unit tests.
  Not yet wired into the pipeline as a normal output — see `quality_of_fit.md` remaining work.

### Alignment verified (still holds)
- Stage-A arrays (`timer`, `Ct`, and the **measured** `CpROI`) match MATLAB to floating-point
  noise; the Patlak core fitter and the CPU / linear-CPU backends match MATLAB almost exactly on
  sampled voxels. Stage-D is in near-exact parity when fed the same AIF (issue #2 table).
  **`Cp_use` (the *fitted* AIF) is explicitly NOT aligned** — see issue #2 and
  [`aif_fitting_parity.md`](aif_fitting_parity.md). The earlier "`Cp_use` matches to
  floating-point noise" claim predates auto-detected injection timing.
- March-2026 `RUNNER_DATA/sub-1101743` clean-reference check: Python CPU vs MATLAB Patlak Ktrans
  correlated `>0.9999` on active voxels (`|Ktrans|>=1e-5`) in both sessions; lower all-voxel
  correlation (~0.89-0.92) was dominated by near-zero floor voxels, not slope/scale drift.
  _(Predates the steady-state overhaul — directional, re-verify before acting.)_

### Open issues

**1. patlak + `brain` non-identifiability — RESOLVED 2026-07-23 via QoF χ² masking.**
History: with the CPU-regenerated reference (issue #3), `patlak_ktrans_brain_cpu_vs_matlab` passed
(Spearman `corr=0.9732`) but the `auto`/gpufit path still failed —
`patlak_ktrans_brain_auto_vs_matlab` `corr=0.9369` (< `0.95`), the known gpufit-vs-MATLAB-CPU
backend divergence (`parity-backend-divergence`) concentrated at a handful of non-identifiable
voxels. **Fix (implemented):** `test_bbb_p19_region_parity` now filters each region by the
per-model QoF reduced-χ² reliable mask (keep voxels with **χ²_ν ≤ 6.0**, an absolute cutoff
calibrated on real cross-backend divergence — see `quality_of_fit.md` / `sigma_estimators.md`).
Excluding ~5–8% (20 of 237 brain voxels) lifts `patlak_ktrans_brain_auto_vs_matlab` to `corr=0.990`
and **all 10 gated checks pass** — the principled replacement for a hand-curated exception (the
canonical QoF target). Deep-dive on the underlying single-voxel mechanism kept below.

**Update 2026-07-28 — QoF is no longer load-bearing for this gate.** The claim that
`ROCKETSHIP_PARITY_QOF_CHI2_MAX=0` reproduces the `0.9369` failure was true when written but is
now stale: after the Stage-B AIF fix (`aif_fitting_parity.md` S11), running with QoF **disabled**
gives `patlak_ktrans_brain_auto_vs_matlab` `corr=0.9678` — a pass. QoF still helps materially
(0.9678 → 0.9940) and stays enabled for the margin, but the underlying disagreement it was built
to mask was mostly the fitted AIF, not voxel quality. Two lessons worth carrying: the QoF filter
was calibrated against a divergence whose dominant cause was a bug elsewhere, and a metric that
"fixes" a failure is not thereby diagnosing it.

**2. `tofts_roi_xls` / `tissue_uptake_roi_xls` ROI-average failures — ROOT-CAUSED 2026-07-24:
the Stage-B AIF fit differs between the two languages.** Full write-up:
[`aif_fitting_parity.md`](aif_fitting_parity.md).

Two configuration mismatches were masking a real algorithm difference, and are now fixed
(commit `192d3c1`, which also regenerated the baseline): (a) `generate_dce_tofts_parity_map.m`
pinned `startInjectionMin=0.5 /
endInjectionMin=0.7` while the Python fixture auto-detected them, so the two sides never shared a
Stage-B input — the generator now defaults both to `-1` (auto); (b) **both** pipelines converted
`end_ss` (a 1-based *frame number*) to minutes as `end_ss * dt` instead of `(end_ss - 1) * dt`
(`dce/B_AIF_fitting_func.m:72`, `python/dce_pipeline.py`), landing the injection window a full
frame late and forcing the fitted AIF to zero on a frame already carrying contrast. `end_ss` is
otherwise produced and consumed correctly as the last baseline frame on both sides. Also fixed
alongside: the same frame→time error in `dce/dce_auto_aif.m`, and an unassigned
`start_injection`/`end_injection` in `A_make_R1maps_func.m`'s explicit-`steadyStateTime` branch
(that path errored out entirely).

With those fixed and the baseline regenerated, `test_bbb_p19_region_parity` passes and the
**entire** remaining ROI-xls gap is the fitted AIF: substituting MATLAB's `Cp_use` into Python's
own Stage-D drops every model to passing (`ex_tofts` 1.2e-5, `patlak` 1.4e-5, `tofts` 0.024,
`tissue_uptake` 0.026). The two AIFs differ at exactly one frame — the first contrast frame:
measured `2.3421`, Python `2.1363`, MATLAB `0.9819` — because MATLAB fits `t0_exp` as a free
parameter (landing at 1.064 min) while Python's default `aif_biexp_timing_method = "legacy_sobel"`
holds it fixed at 0.792. Both sides zero-weight every frame through the AIF peak, so that value is
unconstrained by data on either side. Note this supersedes the "Stage-B `Cp_use` matches
MATLAB to floating-point noise" claim under "Alignment verified" below, which predates
auto-detected injection timing.

**RESOLVED 2026-07-27 (S1-S11 of `aif_fitting_parity.md`).** Both languages now run the same
5-parameter fit with `t_base_end` supplied by Stage A and `delta` reparameterised, uniform
weighting, `aif_Robust = off`, and `find_end_ss_tv` / `steady_state_auto_method = "tv"` as the
baseline-end detector on both sides. Baseline regenerated. All four ROI-xls gates pass:
`tofts` 0.001440, `ex_tofts` 0.000013, `patlak` 0.000013, `tissue_uptake` 0.002235 (ex-`Fp`,
see S8) — a ~100x improvement on `ex_tofts`/`patlak` over where this issue started. The detector
choice was settled on 280 human-rated sessions rather than on the fixture: `tv` 95.0% against
`biexp_fit`'s 74.6%, and S11's R3 records why goodness-of-fit cannot be used to choose between
them.

**3. MATLAB CI maps were zero-width — root-caused and FIXED 2026-07-23 (working tree).**
The `_ci_metrics()` diagnostics (`ci_norm_absdiff_median/p95`, `prop_py_outside_matlab_ci`;
reported-only) were non-functional because every MATLAB `*_ci_low`/`*_ci_high` map was
`0.0`. **Root cause:** commit `a9d78b6` regenerated the maps on a GPU machine, and the gpufit
path explicitly zero-pads CI columns (`dce/FXLfit_generic.m:813-815`) — only the CPU
`fit()`/`confint()` path produces real CIs. (History confirms: at `313d3b6` the CI maps had
2832 non-zero voxels; `a9d78b6` silently switched to GPU.) **Fix (done, uncommitted):**
regenerated all 5 models with `force_cpu=1` (the CPU path is also the correct gold standard —
gpufit diverges from MATLAB CPU, see `parity-backend-divergence`); `force_cpu` reverted to `0`
afterward. CI maps now have real widths (2500–2700 positive-width Ktrans voxels/model), and the
CI-aware metrics are live: e.g. `patlak_ktrans_wm` `ci_norm_absdiff_p95=0.057, prop_out=0.000`;
`tofts_ktrans_brain` `ci_norm_absdiff_p95=1.89, prop_out=0.19` (tofts's larger CI-relative
spread is the expected non-identifiability signal). The CPU-reference restoration also improved
`patlak_ktrans_brain_cpu_vs_matlab` (see issue #1). **Note:** this rewrote the committed
reference maps (Ktrans/ve/vp/fp shift from gpufit→CPU `fit()` values). **Regen recipe:**
set `force_cpu=1`, run the `generate_dce_tofts_parity_map` command in
`projects/archived/steady-state-tv-default/STATUS.md`, revert `force_cpu`; ~51 min single-core.

**Both follow-ups closed 2026-07-28:**
- **The generator now refuses to run with `force_cpu ~= 1`** (`generate_dce_tofts_parity_map.m`,
  precondition check before Stage A) with an error naming the remedy, so a GPU-generated baseline
  cannot silently recur. It *checks* rather than *sets* the pref on purpose: `dce_preferences.txt`
  is a tracked CRLF file and a generator that rewrites it is its own corruption hazard.
- **`prop_py_outside_matlab_ci` now excludes zero-width intervals**, as `ci_norm_absdiff_median`
  already did. This immediately exposed the same bug on the Python side: gpufit produces no CIs,
  so every `auto` row had all-zero Python CI widths and `prop_matlab_outside_py_ci` was reporting
  a fake `1.0` ("100% disagreement") that was really "no data". Both now report `NaN` plus a
  `n_zero_ci_width` / `n_zero_py_ci_width` count.

**4. Batch-processing regression coverage — CLOSED 2026-07-28 as won't-do.** Never built beyond
the design in "Testing gaps" below. Superseded in practice: the ROI-xls gate plus the 12 gated
voxelwise checks cover the map-level regressions this was meant to catch, and the two gaps judged
still worth building (A and B) moved to `TODO.md`.

## Deep-dive: single-voxel patlak non-identifiability (kept — hard-won, not cheaply reproducible)
Root-cause chain (fully isolated, `sub-10bbbdownsample`, 237-voxel sparse `brain` sample):
1. Switching Python's steady-state window from a hardcoded `[1,2]` test override to
   MATLAB-matching auto-detection (correct fix) widened the true Ktrans range from ~0.014 to
   ~0.51 max. Isolated: steady-state-auto alone reproduces the regression; injection-timing does
   not. (Commits `3c17ff3…`→`66fd795…`; table in the stage-d-consolidation plan's Motivation.)
2. The widened range exposed a real bug: patlak's gpufit path had **zero per-voxel seeding** (one
   fixed, data-blind `initial_value_ktrans` for all voxels) and no multi-start, unlike the CPU
   path (seeded per-voxel from the closed-form linear-Patlak estimate). Fixed in the Stage-D
   consolidation (`python/dce_fit_backends.py`): both backends now seed per-voxel from the same
   linear estimate, expanded into x1/x10/x100 candidates.
3. That fixed the majority, but **one voxel of 237** still explains the residual near-zero
   Pearson correlation (which the Spearman swap then de-fanged). Its linear seed is degenerate
   (ktrans0=-0.637, vp0=15.39; vp's upper bound is 1.0), so x1/x10/x100 gives **zero diversity**
   (all candidates collapse to the same bounds-clipped start on both backends).
4. `vp` saturates its upper bound (1.0) on both backends. Pinned there, CPU (float64 scipy `trf`)
   reaches Ktrans=0.5123, SSE=8339.5; gpufit (float32) reaches Ktrans=0.0, chi-square=10231.9 —
   **CPU's objective is objectively lower/better**, so gpufit lands in a genuinely worse local
   optimum near the boundary. Ruled out iteration/tolerance budget (10×/10,000× changed nothing;
   gpufit reports `state=0` converged early).

**Open questions for whoever picks this up:** does gpufit/cpufit's float32 LM step-acceptance
near a bound differ from scipy's float64 `trf`? Should candidate assembly clamp/reject an
out-of-bounds or sign-flipped linear seed? Should a bound-hit auto-escalate to the random
log-uniform multi-start that rescues `2cxm`/`tissue_uptake`? (Bound-hit detection is also a QoF
signal — see `quality_of_fit.md`.)

**Also open (compressed from earlier notes):** a gpufit **tofts** non-convergence concentration
was seen in phantom qualification (`gpufit_cuda`, `TOFTS {0: 6329, 1: 952}` where state `1 =
MAX_ITERATION`, vs `PATLAK {0: 7280, 1: 1}`) — same backend-precision-near-a-hard-fit family;
tracked with phantom work in `projects/phantom-gt/`.

See also memory notes: `parity-whole-brain-roi-noise`, `parity-backend-divergence`,
`parity-tofts-gm-nonidentifiable`, `noisy-data-parity-philosophy`.

## Testing gaps + plan to close (dispositioned 2026-07-28)
Gaps that let CPU-vs-CPUfit divergence and weighted-AIF side effects slip through: parity gates
compare only final maps (not Stage-B `Cp_use` as a first-class contract); no required
`cpu`-vs-`cpufit_cpu` backend-equivalence test on a real-data checkpoint; sparse-ROI sampling
with no dense-ROI cross-check to separate true drift from mask instability.

**A and B moved to `project-management/TODO.md` and are tracked there. C and D are won't-do.**

- **A. Stage-B AIF contract gate** — lock `Cp_use`/`step`/`baseline`/`max_index` against a
  reference payload; gate on MAE/corr, not existence. **The highest-value gap, and S11 proved
  it retroactively:** issue #2 is exactly the failure this gate was designed to catch, it went
  undetected for months because parity compares only final maps, and a cross-language `Cp_use`
  check would have caught it on day one. There is still no such test — verified 2026-07-28.
  **→ TODO.md.**
- **B. Backend-equivalence gate** — frozen Stage-B arrays from a `RUNNER_DATA` fixture; required
  `cpu` vs `cpufit` map metrics for `patlak`/`tofts`/`ex_tofts`; skip-with-reason if
  `pycpufit` absent. Backed by the measured cpufit/gpufit-vs-MATLAB divergence
  (`parity-backend-divergence`). **→ TODO.md.**
- **C. Dense-ROI companion** — ~~keep the sparse suite, add dense-ROI metrics~~ **DROPPED.**
  Largely subsumed: the QoF χ² filter now handles the noisy-voxel half of what dense-ROI
  cross-checking was for, and the GM/WM/brain regions already give three mask sizes per model.
- **D. CI integration** — ~~dedicated nightly parity job~~ **DROPPED.** CI infrastructure is a
  separate decision from parity correctness, and the suite runs on demand via `-m parity` today.
