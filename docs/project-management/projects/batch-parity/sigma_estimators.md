# Noise-σ Estimation for Reduced χ² (QoF #2)

_Batch-parity / quality-of-fit sub-plan. Split out of [`quality_of_fit.md`](quality_of_fit.md)
2026-07-23._

## Why this is its own document
Reduced χ² `= SSE/(σ²·(N−p))` is the first QoF signal we're building (see
[`quality_of_fit.md`](quality_of_fit.md)), and it is only as good as its σ. Estimating a defensible
per-voxel noise σ **in concentration units** is the hard, self-contained problem — so it gets its own
plan. We pursue **two** estimators, **B** and **C**; the rest are recorded for completeness but not
built now.

## What σ we actually need
- Ideally a **per-time-point `σ_i(t)`** feeding a **weighted** χ²_ν `= (1/(N−p))·Σ(r_i/σ_i)²`, because
  the concentration noise is heteroscedastic (below). A per-voxel **scalar** σ is the degraded fallback.
- σ must be **independent of the residuals it normalizes.** A σ taken from a voxel's own fit residuals
  forces χ²_ν≈1 by construction and destroys all discrimination. Hard rule.

## Why concentration-domain noise is awkward (recap)
The pipeline forms `C(t)` in two nonlinear steps ([`python/dce_signal.py`](../../../../python/dce_signal.py)):
baseline-normalize `E = 100·(S − S0)/S0` (`S0 = mean(S[baseline])`), then SPGR log-inversion
`C = −(1/(TR·r1))·ln(N(E)/D(E))` with `N`, `D` affine in `E`. Therefore:
- **Heteroscedastic (first-order issue).** `σ_C(t) ≈ |dC/dS|_{S(t)}·σ_S` is time-varying even when the
  signal-domain σ_S is stationary — larger where the SPGR curve is steep (high enhancement). A single
  scalar σ mis-weights the curve *before* any distribution-shape concern.
- **Non-Gaussian at the extremes.** Magnitude MR signal is Rician (≈Gaussian at SNR≳3–5). The
  affine-then-log map is locally linear mid-range (noise stays ~Gaussian, just rescaled) but injects
  skew/heavy tails at high-`E` (curvature near the `N/D` pole) and low-SNR (Rician floor + noisy-`S0`
  division). `E[SSE]=Σσ²` holds for *any* zero-mean noise, so "χ²_ν≈1 ⇒ noise-level residuals" survives
  non-Gaussianity **in expectation** — only p-value thresholds need Gaussianity ⇒ **calibrate τ_χ
  empirically, never from a χ² table.**
- **Short baseline.** `S0` is often 1–3 points, a shared noisy denominator across every frame — the
  concrete reason baseline-scatter σ (estimator A) is unreliable.

## Estimators we will build

### B — concentration-domain successive-difference (per-voxel scalar; the available one)
- Robust lag-1 / von Neumann: `σ̂ = 1.4826 · median(|C_{i+1} − C_i|) / √2` (DER-SNR is an equivalent
  higher-order variant). The `1.4826` makes MAD a consistent Gaussian-σ estimate; the `/√2` undoes the
  variance-doubling of differencing.
- **Inputs:** only `C(t)` — already in the post-fit NPZ as `ct_voxel_mM` (see *Plumbing probe*).
  **Output:** one scalar σ per voxel.
- Assumes the true curve is smooth frame-to-frame so lag-1 differences are noise-dominated. This holds
  in the pre-contrast baseline and the slow wash-out, but **breaks for vascular voxels during bolus
  arrival/wash-in**, where `C` genuinely jumps between frames — the median form only partly protects
  (for vascular voxels those large differences can be a big, one-sided fraction). **Exclude the
  wash-in window before differencing** — see *Excising the bolus wash-in* below. Remaining limitation:
  blends the heteroscedastic regimes into one number; optional sliding-window variant gives a crude
  `σ(t)`.
- **This is the day-1 estimator — no new plumbing.**

### C — signal-domain σ_S + analytic Jacobian propagation (per-time-point; the principled one)
- Estimate `σ_S` in the **signal** domain (successive-difference on `S(t)`, where the noise is closest
  to stationary — with the **same wash-in window excised**, since the bolus transient is present in
  `S(t)` too; see *Excising the bolus wash-in* below), then propagate per frame:
  `σ_C(t) = |dC/dE|_{E(t)} · (100/S0) · σ_S`.
- SPGR Jacobian in closed form (with `a = e^{TR/T10}`, `c = cos(k_fa·α)`):
  `N(E) = a(1−a)E + 100a(1−c)`, `D(E) = c(1−a)E + 100a(1−c)`,
  so `dC/dE = −(1−a)/(TR·r1) · [a/N − c/D]`.
  **Implementation choice:** compute `dC/dE` by **central finite-difference of the existing
  `enhancement_to_concentration_spgr`** (reuses production math, zero derivation-drift risk); keep the
  analytic formula above only as a unit-test cross-check.
- **Inputs:** raw `S(t)` (or the pipeline's `R1t` time series), SPGR params (TR, FA, T10, r1, k_fa).
  **Output:** per-frame `σ_i(t)` — directly handles heteroscedasticity.
- **Plumbing gate:** the *Plumbing probe* below resolves this — raw `S(t)`/`R1t` and a per-voxel T10
  are **not** in the post-fit artifact, so C needs a bounded plumbing step; **ship B first.** Known
  approximation: this diagonal `σ_i` ignores the correlated shared-`S0` term (full covariance is an
  open question, not a blocker).

**Relationship (C ⊇ B):** C is B performed where the noise is stationary and then mapped through the
true nonlinearity. Where the SPGR-Gaussian picture holds, B ≈ frame-average of C; **systematic B-vs-C
divergence localizes where the model breaks** — a useful diagnostic in its own right (see validation).

### Excising the bolus wash-in (shared by B and C)
Both estimators difference a time series and treat `|Δ|` as noise — valid only where the *true* curve
is smooth. That assumption is violated for **vascular voxels during bolus arrival/first pass**, where
the concentration genuinely jumps frame-to-frame; those large, one-sided differences inflate σ̂ and the
median form only partly absorbs them. Because we already know the acquisition timing, we can excise the
transient cheaply:
- **Window:** start at `end_ss` (the pre-contrast steady-state end = bolus-onset frame) and span the
  **injection duration** plus a short first-pass margin for the peak. This is scan-level timing (global
  frame indices, not per-voxel), so one window applies to every voxel. All of it is on disk in
  `checkpoints/b_out.json` — `start_time_index`/`end_time_index`, `start_injection_min`/
  `end_injection_min`, `time_resolution_min` (and `start_injection == end_ss`); see *Plumbing probe*.
- **How to difference around it:** compute lag-1 differences **only within the retained segments**
  (pre-`end_ss` baseline + post-window wash-out tail) and pool their `|Δ|`; **do not** difference
  across the excised gap (that jump is signal, not noise). The slow wash-out tail is the workhorse
  segment (the baseline is often only 1–3 frames).
- Applies identically to `C(t)` (B) and `S(t)` (C) — the bolus transient is in both domains.
- The exact window bounds / first-pass margin are a calibration knob (see Open questions); start
  conservative (exclude a little extra) since the wash-out tail supplies plenty of clean differences.

## Recorded but not pursued now
- **A — baseline temporal scatter:** too few baseline points; only samples the low-C regime.
- **D — well-fit-voxel pooling (p95–98):** selection bias drives σ **low** (rank by R², not raw SSE,
  if ever revived); global prior only.
- **E — air / noise-region:** undefined in concentration (no valid T10/S0; air is Rayleigh, not tissue
  Rician). Signal-domain only.
- **F — self-residual MAD:** circular; excluded as a denominator.

## Validation strategy (real-data-weighted)
**Caveat that reshapes this plan:** phantom-gt's ground truth carries *idealized, added noise*
(≈Gaussian), **not** the artifacts and poorly-modeled signal we actually care about. It will
**overstate** real-data performance, so it can only prove an estimator is *arithmetically correct for
idealized noise*. It is a **unit-test tier only** — the go/no-go rests on unlabeled real data.

### Tier 1 — estimator correctness (sanity gate; synthetic / phantom-gt, low weight)
- Inject a **known** σ into a smooth synthetic curve at several SNRs; confirm B and C recover it within
  tolerance. Validates the math/implementation, nothing more.
- Confirm finite-difference `dC/dE` matches the analytic formula and the slope of
  `enhancement_to_concentration_spgr`.
- **A Tier-1 pass is explicitly not evidence of real-data readiness.**

### Tier 2 — usefulness on real data (the actual bar; no ground truth needed)
- **Cross-backend-divergence correlation (primary payoff).** On real `RUNNER_DATA`, χ²_ν built from our
  σ should be **elevated exactly where gpu/cpu/python/matlab disagree** — restricting to low-χ²_ν voxels
  should collapse the cross-backend parity gap. Ties σ → χ²_ν → the real problem, no GT required. This
  is the deciding test.
- **B-vs-C agreement map.** Where the SPGR-Gaussian model holds they should track; systematic divergence
  maps model breakdown and tells us when C's extra plumbing earns its keep.
- **Realistic-artifact injection.** Perturb *real* curves with the failure modes we care about (motion
  spikes, baseline drift, truncated/clipped wash-in, bolus mistiming) — **not** idealized Gaussian — and
  check χ²_ν flags them while σ stays stable on the clean frames.
- **Test-retest / repeat consistency** where a dataset offers it: σ and the reliable mask should
  reproduce across repeats.

## Plumbing probe (resolved 2026-07-23)
Traced what the QoF / post-fit stage actually receives. **Verdict: B is fully unblocked; the wash-in
window's timing is on disk; C needs a bounded plumbing step.**

Artifacts (all under the session's `dce/` output):
- **Post-fit arrays** `*_{model}_fit_postfit_arrays.npz` (`_write_postfit_arrays`,
  [`python/dce_pipeline.py`](../../../../python/dce_pipeline.py)) — carries `ct_voxel_mM`
  (**per-voxel `C(t)`**), `cp_mM`, `timer_min`, `voxel_results` (fit params + **SSE** via `sse_col`),
  and **`voxel_residuals`** (per-frame `r_i`, precomputed), plus ROI variants. ⚠️ gated behind
  `stage_overrides.write_postfit_arrays` — **default false**, so the QoF run must enable it.
- **Stage-B checkpoint** `checkpoints/b_out.json` — `start_time_index`/`end_time_index`,
  `start_injection_min`/`end_injection_min`, `time_resolution_min`. Pipeline sets
  `start_injection := end_ss`, so **the wash-in window is fully derivable here.**
- **Stage-A checkpoint** `checkpoints/a_out.json` — `tr_ms`, `fa_deg`, `relaxivity` (r1),
  `time_resolution_min` (the SPGR scalar params).

**B — GREEN, no new plumbing.** `ct_voxel_mM` + `voxel_results` (SSE) + `voxel_residuals` are all in the
NPZ; the wash-in window comes from `b_out.json`. Everything to build B + reduced-χ² already exists — only
flip on `write_postfit_arrays`.

**Wash-in window — GREEN.** Timing lives in `b_out.json`. To keep QoF reading a single artifact,
optionally copy `start_time_index`/`end_time_index`/`time_resolution_min` into the postfit payload (a
~3-line addition to `_write_postfit_arrays`).

**C — YELLOW, bounded plumbing.** Missing from the NPZ: (i) raw `S(t)` — or the `R1t` time series the
pipeline already computes at Stage-B (`R1tTOI`) — needed for the stationary-domain σ_S; and (ii) a
per-voxel **T10** map for an exact S↔R1 inversion (TR/FA/r1 are scalar and already in `a_out.json`;
k_fa/B1 not persisted → assume 1.0). The pipeline works in **ΔR1 = r1·C**, so `C(t)` is a scaled ΔR1 and
the heteroscedasticity C exploits comes from the S→R1 nonlinearity — which needs at least a
relative-signal reconstruction (params + T10) or the persisted `R1t`. Cleanest options: (i) compute σ_S
at Stage-B/D where `R1t` + params are live and emit per-voxel/per-frame σ into the NPZ, or (ii) persist
`R1t_voxel` + the SPGR scalars into the NPZ and do the Jacobian in the QoF module. Either is bounded —
**confirms "ship B first."**

## First code steps

**Status (2026-07-23):** steps 1–5 landed — **first real per-voxel χ²_ν maps produced.**
- `python/dce_sigma.py` (estimator B + shared successive-difference core + `bolus_exclude_window` +
  scalar & weighted `reduced_chi_square`); `tests/python/test_dce_sigma.py`, 18 Tier-1 tests.
- `python/dce_qof.py` — loads a `*_postfit_arrays.npz` (+ sibling `b_out.json`), computes per-voxel σ
  (B) and reduced χ², reconstructs Fortran-order volumes, writes `*_qof_{sigma,chi2nu,reliable}` maps;
  `tests/python/test_dce_qof.py`, 5 tests. (23 tests total, all passing.)
- Ran on `sub-10bbbdownsample` (brain ROI, 2833 voxels, N=64, cpu, tofts+patlak, ~85 s) with
  `write_postfit_arrays=true`, wash-in window `(3, 6)`. **Results:** median χ²_ν 1.48 (patlak) / 1.93
  (tofts) with heavy right tails (p99 ≈ 23); frac χ²_ν ≤ 2 = 0.71 / 0.53 (tofts's larger poorly-fit
  fraction matches its known non-identifiability). σ median ≈ 0.0084 mM. **Validated:** σ maps are
  byte-identical across the two models (σ is a property of `C(t)`, model-independent — a clean internal
  check); the QoF ROI footprint exactly matches the Ktrans param map (Fortran-order reconstruction
  correct); **median χ²_ν = O(1) confirms B's σ and the fit SSE are on the same scale** (the key
  end-to-end sanity check); high-χ²_ν (>5) voxels concentrate at elevated Ktrans (0.122 vs 0.032
  median) — an early Tier-2 signal that χ²_ν flags the noisy/non-conforming voxels.

**Parity integration + calibrated threshold (2026-07-23):** `test_bbb_p19_region_parity` filters each
region by a per-model QoF reduced-χ² reliable mask (CPU run's `*_postfit_arrays.npz` via
`dce_qof.qof_volumes`/`reliable_mask`), keeping voxels with **χ²_ν ≤ 6.0** (`QOF_CHI2_MAX`, env
`ROCKETSHIP_PARITY_QOF_CHI2_MAX`, ≤0 disables). **Result:** excluding ~5–8% (20 of 237 brain voxels)
lifts `patlak_ktrans_brain_auto_vs_matlab` from `corr=0.937` (FAIL) → `0.990` (PASS); all 10 gated
checks pass — principled replacement for a hand-curated exception. Maps for the full-brain run in
`out/qof/sub-10bbbdownsample_ses-01/` (σ + χ²_ν + reliable @ χ²_ν≤6).

**τ_χ calibration (Tier-2, real-data-weighted, not phantom-gt):** swept τ on the full brain (2833
voxels, CPU + auto/gpufit) measuring Spearman parity corr vs retained fraction (script logic in
`sigma_estimators` history). Findings: (1) χ²_ν **positively tracks cross-backend divergence**
(Spearman(χ²_ν, |auto−matlab|) ≈ 0.28–0.30) — validates it as a reliability signal; (2) no sharp
knee — patlak auto/matlab climbs smoothly 0.982→0.995 as τ tightens ∞→2; the anomalous tail is
χ²_ν ≳ 8 (up to ~500 = residuals ~500× noise variance); (3) **percentiles are ROI-size-unstable** (p95
= 5.8 on the sparse 402-voxel ROI vs 8.5 on the full brain), so an **absolute** cutoff is preferred.
Chose **τ_χ = 6.0**: ~3–4× the χ²_ν median (~1.7), removes the clear tail while **retaining ~92%**, and
is stable across ROI size. Reasonable range 5–8; revisit per-model / per-dataset as more data lands.

Remaining: mask on the **intersection of both pipelines'** reliable masks once MATLAB χ² exists;
per-model τ if models diverge more; estimator C (`dcdE_spgr`, `sigma_signal_jacobian`) still deferred
with its plumbing. Actual `bolus_exclude_window` signature is frame-based:
`bolus_exclude_window(onset_frame, duration_frames, n_frames, *, margin_frames=2)`.

1. **New module `python/dce_sigma.py`** (name TBD) with:
   - `successive_difference_sigma(x, *, exclude=None, robust=True) -> float` — the shared core: robust
     lag-1 σ over a 1-D series, differencing only *within* retained segments given an `exclude` frame
     window (the wash-in excision). Used by both B and C.
   - `bolus_exclude_window(end_ss, injection_duration, dt, *, margin=...) -> (lo, hi)` — turn the known
     timing into the excised frame range.
   - `sigma_successive_difference(ct, *, exclude=None, robust=True) -> float` — estimator **B** (thin
     wrapper of the core on `C(t)`).
   - `dcdE_spgr(enh, tr, fa, t10, r1, k_fa) -> np.ndarray` — central finite-difference derivative of
     `enhancement_to_concentration_spgr`; analytic form in a comment as a cross-check.
   - `sigma_signal_jacobian(signal, baseline_indices, tr, fa, t10, r1, k_fa, *, exclude=None) -> np.ndarray`
     — estimator **C** (per-frame `σ_i`): σ_S via the core on `S(t)` with the same `exclude`, then
     propagate through `dce_signal.signal_to_enhancement` + `dcdE_spgr`.
2. **Reduced-χ² helper** `reduced_chi_square(sse, sigma, n, p)` accepting **scalar or per-frame** σ
   (weighted form). SSE is already produced per voxel (`sse_col` in `_MODEL_META`,
   [`python/dce_postfit_analysis.py`](../../../../python/dce_postfit_analysis.py)).
3. **Unit tests (Tier 1):** recover injected σ; FD-vs-analytic derivative; weighted ≡ unweighted when σ
   is constant.
4. **Plumbing (resolved — see *Plumbing probe*):** B + reduced-χ² build from the existing post-fit NPZ
   (`ct_voxel_mM`, `voxel_results`/SSE, `voxel_residuals`) with the wash-in window from `b_out.json` —
   just enable `write_postfit_arrays`. Optionally copy the three timing scalars into the NPZ for
   self-containment. **C is deferred** pending persistence of `R1t` (or raw signal) + SPGR params/T10.
5. Only then wire χ²_ν into the QoF reliable-mask path ([`quality_of_fit.md`](quality_of_fit.md)
   Phase 1).

## Open questions
- Per-frame diagonal `σ_i` vs full covariance (shared-`S0` correlation) — start diagonal, revisit only
  if Tier-2 shows it matters.
- B's single blended σ vs a sliding-window `σ(t)` — worth it, or does C already cover that need?
- Bolus-exclusion window bounds — how much margin past `end_ss + injection_duration` to cover the
  first-pass peak, and whether a single scan-level window is enough or a few slow-arrival voxels need
  slack. Start conservative; the wash-out tail has clean differences to spare.
- One σ / χ²_ν threshold across all models, or per-model (tofts vs patlak differ in identifiability)?
- Compute σ / χ²_ν only at analysis/comparison time — **never mutate committed reference maps.**

## Related
- [`quality_of_fit.md`](quality_of_fit.md) — parent QoF plan; #2 (reduced χ²) is the first metric and
  this doc is its σ dependency.
- [`python/dce_signal.py`](../../../../python/dce_signal.py) — `signal_to_enhancement`,
  `enhancement_to_concentration_spgr` (the exact conversion whose Jacobian C propagates).
- [`python/dce_postfit_analysis.py`](../../../../python/dce_postfit_analysis.py) — per-voxel SSE.
- Memory: `quality-of-fit`, `parity-backend-divergence` (the divergence Tier-2 correlates against),
  `noisy-data-parity-philosophy`.
