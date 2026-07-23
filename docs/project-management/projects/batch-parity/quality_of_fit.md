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

## Proposed scheme
- Compute the two primary residual metrics per voxel (**R²**, **reduced χ²**) from the SSE we
  already write plus a per-voxel baseline-noise σ; flag `reliable ⟺ R² ≥ τ_R AND χ²_ν ≤ τ_χ`.
- Add **bound-hit** as a cheap all-backend adjunct flag; keep CI-CoV / convergence-state as
  CPU-side / analysis-only extras.
- Emit a per-voxel **reason bitmask** (which flags fired) plus a boolean **reliable mask**.
- Optionally a continuous `0–1` QoF score for ranking, but **gate on the interpretable flags
  first** — interpretable beats opaque.
- **Calibrate thresholds (`τ_R`, `τ_χ`) on ground truth (phantom-gt), not by hand.**

## Validation plan
- **Phantom-GT (ground truth known):** excluding low-QoF voxels should reduce region MAE vs GT
  and push `ci_coverage_frac` toward ~0.95; QoF rank should track `|GT − fit|`. phantom-gt
  already computes `ci_coverage_frac` and standardized error `z=|GT−fit|/CI_halfwidth` — reuse it.
- **Real-data cross-backend divergence (the core problem):** show gpu/cpu/python/matlab
  disagreement **concentrates in low-QoF voxels** — i.e. restricting to high-QoF voxels collapses
  the parity gap. Demonstrate on `sub-10bbbdownsample` + a `RUNNER_DATA` session.
- **Regression guard:** the high-QoF subset should show tight cross-tool agreement that stays
  stable across commits.

## Integration into parity testing
- Add a **QoF-masked parity mode**: for real-data / whole-brain checks, compute corr/rmse over
  the **intersection of both pipelines' reliable masks** instead of (or alongside) the current
  sparse brain ROI.
- Keep the hand-curated GM/WM ROIs as a fixed control until QoF masking is validated to agree
  with them, then retire the patlak+brain and tofts+gm exceptions once QoF reproduces them
  automatically.

## Cross-language consistency (critical)
- Define QoF from quantities both pipelines compute the same way (SSE, CI, bound comparison).
  Backend-specific signals (convergence state) are for **analysis masking only**, never as a
  parity-gate input — or require both pipelines to agree.
- **Mask-overlap guard:** a QoF-parity test must first verify the two reliable masks agree to
  high overlap (Dice/Jaccard) on the fixture; otherwise the mask is a divergence source, not a
  filter.

## Phased roadmap
- **Phase 1 (primary):** implement the residual GoF core in Python — **R² (#1)** and
  **reduced χ² (#2)** per voxel from the SSE we already write plus a per-voxel baseline-noise σ;
  gate `reliable ⟺ R² ≥ τ_R AND χ²_ν ≤ τ_χ`; write reliable mask + reason bitmask alongside
  parameter maps; add **bound-hit** as a cheap all-backend adjunct. Validate/calibrate on
  phantom-gt. All of this is backend-agnostic (works on gpufit output).
- **Phase 2:** mirror the flag computation in MATLAB (or a shared post-fit analyzer) so the two
  pipelines' reliable masks are comparable; verify mask overlap (Dice/Jaccard).
- **Phase 3:** QoF-masked parity mode in `test_bbb_p19_region_parity` + real-data parity; retire
  the hand-curated ROI exceptions once QoF reproduces them (canonical target:
  `patlak_ktrans_brain_auto_vs_matlab`).
- **Phase 4 (optional / CPU-only):** CI-width (CoV) signal and continuous QoF score +
  model-selection (F-test/AIC). CPU-side only, since gpufit produces no CIs.

_Done (2026-07-23):_ MATLAB CI maps regenerated with real widths (`force_cpu` CPU path). This
was a prerequisite for the CI-based signal / reported-only CI-aware parity metric, **not** for
the residual core — #1/#2 never needed CIs. See batch_parity.md issue #3.

## Open questions
- Reduced-χ² needs a trustworthy per-voxel noise σ — is the noise-ROI estimate good enough, or
  do we need a local/spatial estimate?
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
