# Per-Voxel Quality-of-Fit (QoF) Reliability

## Status: archived (2026-07-28) — built, validated and shipped; ROI stats moved to TODO.md

> **Archived note:** the metric exists and works. Estimator-B σ with eBayes variance moderation,
> per-voxel reduced χ², the `reliable` mask, the parity-gate filter (χ²_ν ≤ 6.0) and the
> pipeline hook that writes `*_qof_{sigma,chi2nu,reliable}` maps on a normal run are all landed
> and covered by 27 unit tests. Validated on real `RUNNER_DATA` against controls.
>
> **One item stayed live and moved to `docs/project-management/TODO.md`: QoF-aware ROI stats** —
> excluding unreliable voxels from ROI parameter rollups. That was the original motivation for
> the whole workstream and is the one piece never built.
>
> **Batch integration needs no work** (verified 2026-07-28): `write_qof_maps` already flows
> through `run_dce_bids_batch.py --set write_qof_maps=true` and through `--config-template`;
> `_to_bool` accepts the string form. Only a convenience flag would be new, and it is not worth
> a doc entry.
>
> **Dropped as won't-do:** estimator C (per-frame heteroscedastic σ), the B-vs-C agreement map,
> the reason bitmask, the bound-hit flag, R² (#1) as a second gate, and the eBayes dof /
> per-model-prior refinements. Estimator B is sufficient and validated; the rest are refinements
> to something that already works.
>
> **Read the caveat in "Integration into parity testing" below alongside `batch_parity.md`'s
> 2026-07-28 update:** QoF is no longer load-bearing for the parity gate. The failure it was
> calibrated against was mostly the fitted AIF, not voxel quality.

_Batch-parity workstream. Created 2026-07-23. Archived 2026-07-28._

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
- Built from the **concentration curve `C(t)` and the fit SSE** — quantities driven by the shared
  *input data*, so the reliable subset is a **data-quality** judgement, not a fitter-specific one
  (see "Why single-sided masking is valid").
- Two consumers, one metric:
  - **Analysis:** drop or flag unreliable voxels in ROI statistics and parameter maps.
  - **Parity:** evaluate map agreement over the **Python-CPU reliable mask** — a principled,
    automatic replacement for today's hand-curated GM/WM gating.

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
- **Real-data cross-backend divergence (primary — the core problem): CONFIRMED (2026-07-23).**
  Restricting to high-QoF voxels collapses the cpu-vs-gpufit gap on both the `sub-10bbbdownsample`
  fixture and real `RUNNER_DATA/sub-1101743` — and a random-exclusion control shows the gain is the
  QoF signal, not voxel attrition (see "Remaining — parity" above). No ground truth needed.
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
pass, the principled replacement for a hand-curated exception. Remaining parity work is in
"Status & remaining work" below.

## Why single-sided (Python-CPU) masking is valid
**Decision (2026-07-23): no MATLAB-side QoF.** The earlier plan wanted the parity mask built from the
*intersection of both pipelines' reliable masks*, with a Dice/Jaccard overlap guard — which implied a
MATLAB mirror of the QoF computation. We drop that, because the reliable mask is a **data-quality**
judgement, not a fitter judgement:
- **σ comes only from `C(t)`** — the concentration curve is the shared input data, and σ is
  model- and backend-independent (verified: σ maps are byte-identical across models). A voxel with a
  noisy/artifact curve is unreliable no matter who fits it.
- **The χ²_ν numerator (Python-CPU SSE) matches MATLAB-CPU closely** (>0.97; both use the `fit()`
  path), so the Python-CPU χ² is a faithful stand-in for "is this voxel fittable."
So a single-sided Python-CPU mask excludes voxels the *data* can't support, not voxels *Python*
happens to fit poorly — no MATLAB mirror needed. Backend-specific signals (convergence state) stay
analysis-only, never a parity-gate input.

## Status & remaining work
**Done (committed `153728e`, 2026-07-23):**
- Estimator B (concentration-curve noise σ) + wash-in exclusion + reduced χ² — `python/dce_sigma.py`.
- QoF map builder `python/dce_qof.py`: per-voxel σ / χ²_ν volumes + `reliable_mask`.
- Parity gate: `test_bbb_p19_region_parity` filters each region at absolute **χ²_ν ≤ 6.0**
  (calibrated on real cross-backend divergence). Canonical patlak+brain failure resolved; all 10
  gated checks pass. 27 unit tests.

**Closed 2026-07-28:**
- **tofts+gm gate re-enabled and the hand-curated exception retired.** The item below asked to
  re-enable it "if QoF reproduces it." QoF did not, and did not need to: the Stage-B AIF fix
  (`aif_fitting_parity.md` S11) lifted `tofts_ktrans_gm` to corr 0.980 (cpu) / 0.992 (auto)
  against a 0.95 floor with **zero** QoF-excluded voxels in that region. The gate is now 12/12
  with no exceptions. Note the implication: the exception was blamed on GM non-identifiability,
  which is real, but was not what made the numbers disagree.
- **Batch integration** — already works via `--set write_qof_maps=true`; nothing to build.

**Remaining — parity (validation/cleanup; the gate is already live):**
- **RUNNER_DATA validation — DONE (2026-07-23), confirmed.** `sub-1101743/ses-01` (slice 11, 6935
  voxels, Python cpu vs gpufit; shrunk χ²). QoF **χ²≤6 filtering specifically removes the
  divergence-driving voxels**: cpu-vs-gpufit Spearman lifts tofts 0.990→0.995 and patlak 0.973→0.982
  while keeping ~90%. **Controls prove it's the QoF signal, not voxel attrition:** same-fraction
  *random* exclusion gives *zero* gain (0.990/0.973 unchanged, ±0.0005), and dropping low-|Ktrans|
  voxels doesn't help (patlak worse). Caveat: per-voxel Spearman(χ², |cpu−auto|) is weak (0.15 tofts /
  −0.02 patlak) — the χ²↔divergence link is **tail-concentrated**, not smoothly monotonic (exactly
  what a reliability filter exploits). One slice / one session — widen to more sessions to generalize.
- Re-enable the **tofts+gm** gate with QoF on and retire that hand-curated exception if QoF
  reproduces it.
- One-line note (done, above) that single-sided masking is intentional — replaces the dropped
  mask-overlap guard.

**Remaining — data analysis:**
1. **Pipeline hook — DONE (2026-07-23).** `dce_pipeline._write_qof_maps` writes
   `*_qof_{sigma,chi2nu,reliable}.nii.gz` next to the param maps on a normal run, computed from the
   in-memory `ct_source` + SSE + injection timing (no NPZ round-trip), via
   `dce_qof.compute_qof_arrays` with `shrink_sigma=True`. **Opt-in** `write_qof_maps` pref (default
   false); `qof_chi2_max` pref (default 6.0) sets the reliable threshold. Both added to
   `dce_default.json`/`dceprep_default.json`. Best-effort (never fails the fit).
2. **Batch integration:** `run_dce_bids_batch.py` should pass `write_qof_maps` through so batch runs
   emit QoF maps per session (prefs already flow via `stage_overrides` / config template — verify + a
   convenience flag).
3. **QoF-aware ROI stats:** exclude unreliable voxels from voxelwise ROI parameter rollups (and/or
   report reliable-fraction per ROI) so nonsense voxels stop polluting ROI means — the original
   motivation.
4. *(optional)* a `dce_cli.py` flag for `write_qof_maps` (works today via config/`stage_overrides`).

**Done since (2026-07-23):**
- **σ outlier robustification — eBayes variance moderation** (`dce_sigma.eb_moderate_variance`):
  motion voxels' inflated σ was suppressing χ²_ν and slipping the filter; now σ² is shrunk toward an
  **inverse-gamma** prior (empirically the best fit — gamma was worst) with a **prior-predictive
  clamp** that flags contaminated σ without adding a second user threshold. Wired into
  `compute_qof(shrink_sigma=True)` and the parity gate. See [`sigma_estimators.md`](sigma_estimators.md).

**Optional / not on the "usable" critical path:**
- **Estimator C** (heteroscedastic per-frame σ) — accuracy refinement; B is sufficient and validated.
- **Reason bitmask + bound-hit flag; R² (#1)** as the complementary gate — interpretable exclusions.
- **Why median χ²_ν ≈ 1.5–1.9 > 1** — σ bias vs mild misfit; affects only absolute χ²_ν
  interpretation, not the (empirically-calibrated) reliability filter.

## Open questions
- One global τ_χ, or per-model / per-dataset (tofts vs patlak differ in identifiability)? Currently one
  absolute τ=6.0 works for both on the fixture.
- Apply QoF only at comparison/analysis time, never mutating the committed reference maps
  (preferred — keep raw maps intact).
- σ-outlier robustification approach (shrinkage prior distribution) — see `sigma_estimators.md`.

## Related
- [`batch_parity.md`](batch_parity.md) — Spearman swap (partial mitigation), the patlak+brain
  single-voxel case, and the CI-map regeneration that unblocks the CI signal.
- `_ci_metrics()` in `tests/python/test_dce_pipeline_parity_metrics.py` — existing plumbing for
  the CI-width signal.
- `projects/phantom-gt/PHANTOM_GT_QUALIFICATION_STATUS.md` — `ci_coverage_frac` + z-score
  infrastructure for validation.
- Memory: `noisy-data-parity-philosophy`, `parity-whole-brain-roi-noise`,
  `parity-backend-divergence`, `parity-tofts-gm-nonidentifiable`.
