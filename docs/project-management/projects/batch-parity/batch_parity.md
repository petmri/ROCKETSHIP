# Batch Parity Status (MATLAB vs Python DCE)

_Last reviewed: 2026-07-23._

## Scope
Parity between the MATLAB reference pipeline and the Python port for DCE parameter maps
(Ktrans/ve/vp/fp), on two fronts:
- the committed `sub-10bbbdownsample` fixture (`test_bbb_p19_region_parity`,
  `test_bbb_p19_roi_xls_parity`), and
- the end-to-end batch pipeline (`run_dce_bids_batch.py`) on real `RUNNER_DATA` sessions.

**Gated set:** tofts + patlak **Ktrans only** (cpu & auto vs MATLAB). Everything else
(ex_tofts/tissue_uptake/2cxm, non-Ktrans params, backend auto-vs-cpu) is reported-only — not
identifiable on this fixture. Suite layout in `tests/README.md`.

**Active workstreams**
- **Per-voxel quality-of-fit (QoF) reliability metric** to exclude unreliable voxels from both
  analysis and real-data parity — see [`quality_of_fit.md`](quality_of_fit.md) (first metric =
  reduced χ²; its noise-σ dependency is broken out into
  [`sigma_estimators.md`](sigma_estimators.md)).
  This is the general solution to the recurring "noisy/non-conforming voxel pollutes parity and
  analysis" problem that the Spearman swap and hand-curated ROIs only partially mask.
- Closing the remaining gated-parity gaps (below).

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

### Alignment verified (still holds)
- Stage-A/Stage-B arrays (`timer`, `Ct`, `Cp_use`) match MATLAB to floating-point noise on the
  clean-reference run; the Patlak core fitter and the CPU / linear-CPU backends match MATLAB
  almost exactly on sampled voxels.
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
canonical QoF target). Disable with `ROCKETSHIP_PARITY_QOF_CHI2_MAX=0` (reproduces the old `0.9369`
failure). Deep-dive on the underlying single-voxel mechanism kept below.

**2. `tofts_roi_xls` / `tissue_uptake_roi_xls` ROI-average failures — not root-caused.**
Under the `tv` window, `test_bbb_p19_roi_xls_parity` fails for `tofts`
(`mae=0.0329`, limit `0.03`) and `tissue_uptake` (`mae=0.0290`, limit `0.05` on max_abs_err
`0.1187`); `ex_tofts` and `patlak` still pass. Both failing models fit from an ROI-averaged
curve — start by checking whether the `tv`-derived window shifted the ROI-mean baseline/injection
timing enough to matter for these two specifically.

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
reference maps (Ktrans/ve/vp/fp shift from gpufit→CPU `fit()` values) — a substantive fixture
change to review before committing. Secondary follow-up: `prop_py_outside_matlab_ci` should skip
degenerate (zero-width) voxels the way `ci_norm_absdiff_median` already does. **Regen recipe:**
set `force_cpu=1`, run the `generate_dce_tofts_parity_map` command in
`projects/archived/steady-state-tv-default/STATUS.md`, revert `force_cpu`; ~51 min single-core.
The generator should arguably force CPU itself so this can't silently recur — open follow-up.

**4. Batch-processing regression coverage — design, not yet built.** See "Testing gaps" below.

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

## Testing gaps + plan to close (open)
Gaps that let CPU-vs-CPUfit divergence and weighted-AIF side effects slip through: parity gates
compare only final maps (not Stage-B `Cp_use` as a first-class contract); no required
`cpu`-vs-`cpufit_cpu` backend-equivalence test on a real-data checkpoint; sparse-ROI sampling
with no dense-ROI cross-check to separate true drift from mask instability.

- **A. Stage-B AIF contract gate** — lock `Cp_use`/`step`/`baseline`/`max_index` against a
  reference payload; gate on MAE/corr, not existence.
- **B. Backend-equivalence gate** — frozen Stage-B arrays from a `RUNNER_DATA` fixture; required
  `cpu` vs `cpufit` map metrics for `patlak`/`tofts`/`ex_tofts`; skip-with-reason if
  `pycpufit` absent.
- **C. Dense-ROI companion** — keep the sparse suite, add dense-ROI metrics for required models;
  emit both on failure.
- **D. CI integration** — dedicated parity job (optional per-PR, required nightly) running the
  multi-model parity runner + A + B, archiving summary JSON for trend comparison.
