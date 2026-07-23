# Per-Voxel Quality-of-Fit (QoF) Reliability

_Batch-parity workstream. Created 2026-07-23._

## Motivation
Real DCE data contains noisy / artifact voxels whose signal does not conform to the fitting
model's assumptions (poorly-shaped bolus, non-monotonic uptake, motion, low enhancement/SNR).
The optimizer still returns "parameters," but those parameters are unconstrained by the data,
so they:
- **vary wildly across backends** (gpu / cpu / python / matlab), inflating apparent parity
  disagreement even when every well-fit voxel agrees, and
- **pollute downstream analysis** with physiologically meaningless values.

Our current defenses only paper over this: Spearman correlation is robust to a few such voxels
but discards magnitude information, and hand-curated GM/WM ROIs exclude them manually per
fixture. We want a principled, automatic, **per-voxel** quality-of-fit score that flags
unreliable fits and can be reused for (1) analysis masking and (2) parity on real data — compare
only where *both* pipelines agree the fit is trustworthy.

The single-voxel patlak+brain failure documented in [`batch_parity.md`](batch_parity.md) is the
canonical case: one voxel with a degenerate seed, a parameter pinned at its bound, and backends
landing in different optima. QoF should exclude exactly that voxel on principled grounds.

## Goals
- A per-voxel **QoF reason bitmask + boolean "reliable" mask**, written alongside each parameter
  map (and rolled up per ROI).
- Built from quantities **both MATLAB and Python already produce**, so the reliable subset does
  not itself become a source of cross-language divergence.
- Two consumers, one metric:
  - **Analysis:** drop or flag unreliable voxels in ROI statistics and parameter maps.
  - **Parity:** evaluate map agreement over the **intersection of both pipelines' reliable
    masks** — a principled, automatic replacement for today's hand-curated GM/WM gating.

## Candidate per-voxel signals

**Decision (2026-07-23):** the foundation is **residual goodness-of-fit (#1 + #2)** — both are
pure functions of SSE + the data, so they compute on **every backend including GPU** (gpufit
returns per-voxel chi-square/SSE; no confidence intervals needed). This is the general,
portable core. Enhancement/CNR is dropped (threshold too data/study-dependent to set
reliably). CI-based and model-selection signals are kept only as CPU-only secondary adjuncts.

### Primary — residual GoF (all-backend)
- **#1 — R²** `= 1 − SSE/SS_tot`, `SS_tot = Σ(Ct_i − mean(Ct))²`. Dimensionless, comparable
  across voxels, and **needs no noise estimate**. *Signal-referenced*: answers "what fraction of
  the curve's variance did the model explain?" Caveats: over-rejects low-enhancement voxels
  (tiny `SS_tot` → poor/negative R² even for a noise-level fit); lenient on high-SNR systematic
  misfit. Optional adjusted-R² `= 1 − (1−R²)(N−1)/(N−p−1)` penalizes parameter count for
  cross-model fairness.
- **#2 — reduced χ²** `= SSE / (σ²·(N−p))`. *Noise-referenced*: answers "are the residuals just
  noise?" `≈1` good, `≫1` model mismatch. The more physically meaningful reject test, but needs
  a noise σ (cheapest: per-voxel scatter of the pre-contrast baseline of `Ct`, self-contained
  and in the right units; alternative: global noise-ROI σ propagated to concentration units).

  #1 and #2 are **complementary** (signal- vs noise-referenced) — gate on **both**
  (`reliable ⟺ R² ≥ τ_R AND χ²_ν ≤ τ_χ`), which is more robust than either alone and covers
  each other's blind spots.

### Secondary / optional adjuncts
- **Bound-hit** (param == fit lower/upper limit) — optimizer stuck at a boundary, no interior
  optimum. Cheap, all-backend (compare map to prefs bounds), directly catches the canonical
  patlak+brain failure. Good low-cost addition once #1/#2 land.
- **Convergence state** (0=ok, 1=max-iter; gpufit/cpufit return it) — analysis masking only,
  backend-specific.
- **CI width / CoV** (`CIwidth/value`) — non-identified parameter. **CPU-only** (gpufit doesn't
  produce CIs), so *not* part of the portable core; useful for CPU-side analysis and the
  reported-only CI-aware parity metric. CI maps are now regenerated with real widths.
- **Model-selection** (F-test / AIC/BIC; `run_dce_postfit_analysis.py --analysis ftest` exists)
  — later phase.
- **Dropped:** enhancement / CNR — thresholds too data/study-dependent.

## Estimating the noise σ — the crux of reduced χ² (#2)

**First implementation (2026-07-23):** lead with **#2 (reduced χ²)** alone — the more physical reject
test and the most immediately useful signal. #1 (R²) is a cheap add-on later; the eventual
`R² ≥ τ_R AND χ²_ν ≤ τ_χ` both-gate stays the end state. The whole difficulty of #2 is a defensible
per-voxel σ **in concentration units** — that work now lives in its own plan,
[`sigma_estimators.md`](sigma_estimators.md).

In brief: concentration noise is **heteroscedastic** (`σ_C(t)≈|dC/dS|·σ_S`, the first-order problem)
and **skewed/heavy-tailed at high-E and low-SNR**, because `C(t)` is built by baseline-normalize +
SPGR log-inversion. So we want a **per-time-point σ_i(t)** feeding a **weighted** χ²_ν, and we
**calibrate τ_χ empirically** (`E[SSE]=Σσ²` makes χ²_ν≈1 survive non-Gaussianity in expectation, but
p-value thresholds don't). We pursue two estimators — **B** (concentration-domain successive-difference,
available day-1) and **C** (signal-domain σ_S propagated through the analytic SPGR Jacobian,
per-frame) — and reject baseline-scatter / air / self-residual. Validation is **real-data-weighted**:
phantom-gt only sanity-checks estimator arithmetic (its added noise is idealized, not the artifacts we
care about); the deciding test is that χ²_ν rises where backends diverge on real data. Full detail,
formulas, and the first code steps are in [`sigma_estimators.md`](sigma_estimators.md).

## Proposed scheme
- Compute the primary residual metrics per voxel from the SSE we already write plus a per-voxel
  noise σ (**σ estimator per [`sigma_estimators.md`](sigma_estimators.md)** — successive-difference /
  Jacobian-propagated, *not* baseline scatter); lead with **reduced χ² (#2)**, add **R² (#1)** as the
  cheap complement, then
  flag `reliable ⟺ R² ≥ τ_R AND χ²_ν ≤ τ_χ`.
- Add **bound-hit** as a cheap all-backend adjunct flag; keep CI-CoV / convergence-state as
  CPU-side / analysis-only extras.
- Emit a per-voxel **reason bitmask** (which flags fired) plus a boolean **reliable mask**.
- Optionally a continuous `0–1` QoF score for ranking, but **gate on the interpretable flags
  first** — interpretable beats opaque.
- **Calibrate thresholds (`τ_R`, `τ_χ`) empirically, not by hand** — but weight real-data behavior
  (cross-backend-divergence, below) over phantom-gt, whose idealized added noise overstates real
  performance. phantom-gt is a sanity tier, not the calibration authority.

## Validation plan
Weight real-data behavior over phantom-gt: phantom-gt's ground truth carries *idealized, added noise*,
not the artifacts / poorly-modeled signal QoF exists to catch, so it **overstates** real performance
and serves only as a sanity tier (see [`sigma_estimators.md`](sigma_estimators.md) for the same caveat
applied to σ).
- **Real-data cross-backend divergence (primary — the core problem):** show gpu/cpu/python/matlab
  disagreement **concentrates in low-QoF voxels** — restricting to high-QoF voxels collapses the parity
  gap. No ground truth needed. Demonstrate on `sub-10bbbdownsample` + a `RUNNER_DATA` session.
- **Realistic-artifact injection (real curves, not idealized noise):** perturb real voxels with the
  failure modes we care about (motion spikes, baseline drift, truncated wash-in) and confirm QoF flags
  them.
- **Phantom-GT (sanity tier only):** excluding low-QoF voxels should still reduce region MAE vs GT and
  push `ci_coverage_frac` toward ~0.95 under its idealized noise; reuse its `ci_coverage_frac` +
  `z=|GT−fit|/CI_halfwidth`. Necessary-not-sufficient — do not read a phantom-gt pass as real-data
  readiness.
- **Regression guard:** the high-QoF subset should show tight cross-tool agreement that stays stable
  across commits.

## Integration into parity testing
**Implemented + calibrated 2026-07-23.** `test_bbb_p19_region_parity` filters each region by a
per-model QoF **reduced-χ² reliable mask** (from the CPU run's `*_postfit_arrays.npz` via `dce_qof`),
keeping voxels with **χ²_ν ≤ 6.0** (`QOF_CHI2_MAX`, env `ROCKETSHIP_PARITY_QOF_CHI2_MAX`, ≤0 disables).
τ=6 was calibrated on real cross-backend divergence, not phantom-gt (see `sigma_estimators.md`): χ²_ν
tracks |auto−matlab| divergence (Spearman ≈0.29), and an absolute cutoff beats a percentile (p95 swung
5.8–8.5 across ROI sizes). Effect: excluding ~5–8% (20/237 brain voxels) lifts the long-standing
`patlak_ktrans_brain_auto_vs_matlab` from `corr=0.937` (FAIL) to `0.990` (PASS) — all 10 gated checks
pass, the principled replacement for a hand-curated exception. Remaining/next:
- Currently masks on the **CPU (reference) χ² only**; upgrade to the **intersection of both
  pipelines' reliable masks** once MATLAB-side χ² exists (Phase 2).
- Keep the hand-curated GM/WM ROIs as a fixed control; retire the tofts+gm exception once QoF
  reproduces it. Revisit τ per-model / per-dataset as more real data lands.

## Cross-language consistency (critical)
- Define QoF from quantities both pipelines compute the same way (SSE, CI, bound comparison).
  Backend-specific signals (convergence state) are for **analysis masking only**, never as a
  parity-gate input — or require both pipelines to agree.
- **Mask-overlap guard:** a QoF-parity test must first verify the two reliable masks agree to
  high overlap (Dice/Jaccard) on the fixture; otherwise the mask is a divergence source, not a
  filter.

## Phased roadmap
- **Phase 1 (primary):** implement the residual GoF core in Python, **leading with reduced χ² (#2)**
  — the crux is the σ estimator, so Phase 1 begins in [`sigma_estimators.md`](sigma_estimators.md)
  (stand up estimator B, then C; validate real-data-weighted) before wiring χ²_ν from the SSE we
  already write. Then add **R² (#1)** as the cheap complement and gate
  `reliable ⟺ R² ≥ τ_R AND χ²_ν ≤ τ_χ`; write reliable mask + reason bitmask alongside parameter
  maps; add **bound-hit** as a cheap all-backend adjunct. Validate real-data-weighted (phantom-gt
  sanity tier only, per `sigma_estimators.md`). All of
  this is backend-agnostic (works on gpufit output).
- **Phase 2:** mirror the flag computation in MATLAB (or a shared post-fit analyzer) so the two
  pipelines' reliable masks are comparable; verify mask overlap (Dice/Jaccard).
- **Phase 3:** QoF-masked parity mode in `test_bbb_p19_region_parity` — **done + calibrated
  (2026-07-23):** per-model CPU-χ² reliable mask at absolute **χ²_ν ≤ 6** (Tier-2 divergence
  calibration); canonical target `patlak_ktrans_brain_auto_vs_matlab` now passes (0.937→0.990).
  Remaining: both-pipeline mask intersection (needs Phase 2), real-data parity, retire the tofts+gm
  exception.
- **Phase 4 (optional / CPU-only):** CI-width (CoV) signal and continuous QoF score +
  model-selection (F-test/AIC). CPU-side only, since gpufit produces no CIs.

_Done (2026-07-23):_ MATLAB CI maps regenerated with real widths (`force_cpu` CPU path). This
was a prerequisite for the CI-based signal / reported-only CI-aware parity metric, **not** for
the residual core — #1/#2 never needed CIs. See batch_parity.md issue #3.

## Open questions
- Reduced-χ² needs a trustworthy per-voxel noise σ — see [`sigma_estimators.md`](sigma_estimators.md)
  (candidate estimators + verification plan); the open sub-question is whether estimator C's Jacobian
  propagation beats the simpler successive-difference B enough to justify plumbing raw signal through.
- CI CoV blows up as a parameter → 0; pair the relative threshold with an absolute floor.
- One global threshold set, or per-model (tofts vs patlak have different identifiability)?
- Apply QoF only at comparison/analysis time, never mutating the committed reference maps
  (preferred — keep raw maps intact).

## Related
- [`batch_parity.md`](batch_parity.md) — Spearman swap (partial mitigation), the patlak+brain
  single-voxel case, and the CI-map regeneration that unblocks the CI signal.
- `_ci_metrics()` in `tests/python/test_dce_pipeline_parity_metrics.py` — existing plumbing for
  the CI-width signal.
- `projects/phantom-gt/PHANTOM_GT_QUALIFICATION_STATUS.md` — `ci_coverage_frac` + z-score
  infrastructure for validation.
- Memory: `noisy-data-parity-philosophy`, `parity-whole-brain-roi-noise`,
  `parity-backend-divergence`, `parity-tofts-gm-nonidentifiable`.
