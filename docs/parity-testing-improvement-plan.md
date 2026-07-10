# Parity Testing Improvement Plan

Status: draft for review · Branch: `parity-testing-improvements` · Date: 2026-07-09

This plan covers three workstreams for the DCE Python↔MATLAB parity tests:
**§2** incorporate curated GM/WM ROIs and narrow the gated scope; **§3** metric set +
MATLAB confidence-interval metrics; **§5** simplify the test switches and rewrite the README.

**Implementation status (landed 2026-07-09):** §3 metrics (RMSE+Corr gate, CI metrics), §2 GM/WM
regions + narrowed gate, and the tofts-GM investigation are done and verified. Phase 2 default-on
standard suite (`test_bbb_p19_region_parity`) + `--parity-suite=allmodels` are green; full default
`pytest tests/python` passes. §5 landed as deprecate-don't-break: `--parity-suite` selector,
`--parity-thresholds` JSON (`tests/python/parity_thresholds_default.json`), README rewrite. Removal
of the deprecated flags + CI migration to `--parity-suite` is deferred and tracked in `PORTING_STATUS.md`.

**Scope decisions (this revision):**
- **Dropped** the former item 1 (fit-bound outlier handling / no-fit-fill hygiene) — background
  is already correctly excluded, so it is not worth a dedicated workstream.
- **Dropped** the former item 4 (OSIPI ground-truth workstream) — the OSIPI reliability tests
  already provide the correctness layer and need no changes here. They remain the reason the
  bbb_p19 parity suite only needs to test *agreement*, not *correctness*.
- **Gated scope for bbb_p19 parity:** gate only on **Patlak and Tofts, Ktrans parameter only**,
  and include that slim gated subset in the **standard (default) test suite**.
- **Optional, never gated:** **ex_tofts, tissue_uptake, 2cxm** move to an opt-in mode that runs
  and reports metrics but never fails the build (they are not identifiable on bbb_p19).
- **GM/WM ROIs are added alongside** the existing whole-brain ROI, not a replacement.

---

## 0. Key findings (context)

1. **Background is correctly excluded.** `processed/T1_brain_roi.nii` encodes background as
   **-2** (and some -1); the metric mask uses `roi > 0`, keeping ~2834 tissue voxels. The
   whole-brain population is heavy-tailed (tofts Ktrans CV ≈ 2.8, ex_tofts ≈ 6.8) with voxels
   pinned at the `Ktrans = 2.0` fit bound — which is why ex_tofts/2cxm/tissue_uptake look
   unstable and are moved to optional rather than gated.

2. **Confidence intervals exist on BOTH sides.** MATLAB writes
   `Dyn-1_<model>_fit_<param>_ci_low.nii` / `_ci_high.nii`; Python's `MODEL_LAYOUTS[...].param_names`
   include `*_ci_low` / `*_ci_high` and `_write_param_maps` writes every param, so Python emits
   the same CI maps. CI-based metrics need no new fitting work.

3. **Curated ROIs are in place.** `processed/T1_gm_roi.nii` (50 voxels, tofts-Ktrans CV 0.75)
   and `processed/T1_wm_roi.nii` (119 voxels, CV 0.29) are tight real-tissue masks with coherent
   dynamic range, stored as T1-valued maps (nonzero = in-ROI).

4. **Switch surface is large.** ~45 conftest options; six `--run-*` enable-flags, three
   model-selection flags, five root overrides, ~20 `--parity-*-corr-min/mse-max` knobs, most
   with a terse alias. README is ~360 lines / 16 sections.

---

## 2. GM/WM ROIs + narrowed gated scope

**Goal:** gate a small, reliable subset on well-behaved tissue, and run everything else as
reported-only diagnostics.

**Gated (standard suite, runs by default):**
- Models/param: **tofts & patlak, Ktrans only**, Python-vs-MATLAB (cpu & auto).
- Regions: **patlak gates brain+GM+WM**; **tofts gates brain+WM, GM reported-only**.
- Gate metrics: **RMSE and Corr** (see §3).

> **Tofts-GM exception (investigated 2026-07-09).** Tofts Ktrans genuinely disagrees with
> MATLAB in the GM ROI (corr ~0.75, rmse ~0.021, 16% of voxels outside MATLAB's 95% CI),
> localized to a spatially-clustered patch of ~8 noisy, weakly-enhancing voxels. This is
> **not a Python bug**: Python's tofts SSE is equal-or-lower than MATLAB's in 49/50 GM voxels
> (8/8 of the diverging ones), with near-identical SSE across 2–4× different Ktrans — i.e. the
> objective is flat along Ktrans (non-identifiable). ve agrees because it is constrained by the
> steady-state level. Decision: gate tofts on brain+WM only; GM reported-only. Follow-up: that
> GM patch may be a CSF-adjacent/partial-volume region worth excluding when the ROI is next revised.

**Reported-only (never gated):**
- Non-Ktrans parameters of tofts/patlak (ve, vp).
- All parameters of **ex_tofts, tissue_uptake, 2cxm** — run only under `--parity-suite=allmodels`,
  metrics logged + written to the summary JSON, but no assertions.

**Implementation:**
- Extend `_dataset_paths` to expose `roi_gm` and `roi_wm` alongside `roi`.
- Add a **region dimension** to the checks; label metrics `<model>_<param>_<region>_<cmp>`
  (e.g. `tofts_ktrans_gm_cpu_vs_matlab`).
- GM/WM are dense (50/119 voxels) → run them **fully**, no `--roi-stride` subsampling; keep
  stride only for whole-brain diagnostic runs.
- Split the current monolithic multi-model test so the **gated tofts/patlak-Ktrans checks run
  by default** (fast: downsample fixture + committed `results_matlab` baselines), while the
  optional models stay behind the opt-in mode.
- Provenance: document GM/WM ROI origin + selection criteria in the DCE fixture README.

---

## 3. Metric set + MATLAB confidence-interval metrics

**Final reported metric set (drop MAE and p95_abs_err):**

| metric | definition | role |
|---|---|---|
| **RMSE** | `sqrt(mean((py−matlab)²))` | **gated** |
| **Corr** | Pearson `r(py, matlab)` | **gated** |
| **CI-normalized difference** | `abs(py − matlab) / (ci_high − ci_low)` per voxel; report median + p95 | reported |
| **Proportion outside CI** | fraction of voxels where `py ∉ [ci_low, ci_high]` (both directions) | reported |

- **Remove** `mae` and `p95_abs_err` from the reported line and the summary JSON.
- **Gating** stays on **RMSE and Corr** (the current gate asserts on `mse`; switch it to `rmse`
  — thresholds convert as `rmse_max = sqrt(mse_max)`, so behavior is equivalent at first).
- **CI-normalized difference** ties disagreement to *fit identifiability*: where a parameter is
  poorly constrained (wide CI), a large absolute difference is correctly discounted.
- **Proportion outside CI** is a clean interpretable scalar; a good parity keeps most voxels
  inside. Report both directions (py-in-matlab-CI and matlab-in-python-CI).

**Notes:**
- CI definition is confirmed consistent: both MATLAB and Python report a **95% CI**, so
  `ci_width` is directly comparable across the two.
- Guard `ci_width → 0` (degenerate/failed CI): bucket zero-width voxels separately, no
  divide-by-zero.

**Plan:** add the two CI metrics as **reported-only** values. Per decision, both CI metrics
(including proportion-outside-CI) **stay reported-only and are never gated** — they inform
interpretation, they don't fail the build.

---

## 5. Simplify switches + rewrite README

**Design:** one suite selector, a fixed default gated subset (and a fixed optional-models set,
both hard-coded), and a thresholds file replacing the per-knob options.

### 5a. Enable-flags → single `--parity-suite`

**Remove** (keep as hidden deprecated aliases for one release):
- `--run-parity` / `--parity`
- `--run-full-parity` / `--full-parity`
- `--run-multi-model-backend-parity` / `--mm-parity`
- `--run-runtime-parity`

**Add:** `--parity-suite=<comma-set>` with values:
- `standard` — **default, runs without any flag**: tofts/patlak Ktrans, GM+WM+brain, gated on RMSE+Corr.
- `allmodels` — adds ex_tofts/tissue_uptake/2cxm, **reported-only, never gated**.
- `full` — full-volume fixture (slow).
- `runtime` — Python-vs-MATLAB wall-clock parity (needs MATLAB).
- `all` — union of the above.

(Leave `--run-osipi-slow` and `--run-qualification` as-is — out of scope for this plan.)

### 5b. Model selection → fixed default + one optional flag

**Remove:**
- `--parity-required-models` / `--req-models`
- `--parity-cpu-optional-models` / `--cpu-opt-models`
- `--parity-require-all-models` / `--all-models`

**Add:** nothing — both sets are **fixed in code**, no new CLI knob:
- Gated set: `{tofts, patlak}` × `{Ktrans}`.
- Reported-only set: `{ex_tofts, tissue_uptake, 2cxm}` (all params), run only when
  `--parity-suite` includes `allmodels`. Never gated.

### 5c. Thresholds → one file

**Remove** the ~20 knobs:
`--parity-downsample-ktrans-corr-min/-mse-max`, `--parity-downsample-ve-*`,
`--parity-full-*`, `--parity-model-*`, `--parity-cpu-auto-*`, `--parity-ex-tofts-ktrans-corr-min`,
`--parity-ve-ktrans-min`, `--parity-ktrans-upper-exclude`.

**Add:**
- `--parity-thresholds <path.json>` — single JSON with the gate thresholds, keyed by
  region/model/param, using **`rmse_max` and `corr_min`** (not `mse_max`). Ship a documented
  default file in `tests/python/`.
- Any residual masking constants (e.g. the old `ve_ktrans_min`) move into that JSON, not CLI.

### 5d. Root overrides + misc → canonical names

- Keep one canonical name each; drop the alias twins: keep `--dataset-root` (drop `--ds-root`),
  `--roi-stride` (drop `--stride`), `--full-root` (drop `--fr-root`).
- Keep `--parity-summary-dir`, `--runtime-parity-matlab-cmd`,
  `--runtime-parity-max-python-over-matlab-ratio`, and the runtime/qualification root overrides.

### 5e. README rewrite (`tests/README.md`, target ~150 lines)

- Lead with a **"How do I run X?"** command table (one row per intent → exact command).
- Single **"Parity suites"** section: the `--parity-suite` selector, the gated-vs-reported
  split, and the fixed tofts/patlak-Ktrans gate.
- **"Thresholds"** section pointing at the JSON default.
- **"Reference / provenance"** appendix (datasets + GM/WM ROI origin).
- Fold the current 6+ parity sections into the above.

---

## Sequencing

**Phase 1 — metrics + regions (reported first):**
1. §3 swap gate `mse`→`rmse`; drop `mae`/`p95_abs_err`; add CI-normalized diff + proportion-outside as reported.
2. §2 wire GM/WM ROIs as reported regions; narrow the gated set to tofts/patlak Ktrans.
→ Re-run; inspect RMSE/Corr + CI context on GM/WM/brain.

**Phase 2 — default-suite + optional split:**
3. Make the gated tofts/patlak-Ktrans checks part of the **standard suite** (default-on).
4. Move ex_tofts/tissue_uptake/2cxm behind `--parity-suite=allmodels`, reported-only.

**Phase 3 — ergonomics:** §5a–5d switch consolidation (with deprecated aliases) + §5e README rewrite.

---

## Resolved decisions

1. **CI definitions match** — both MATLAB and Python report a 95% CI; `ci_width` is directly comparable.
2. **CI metrics stay reported-only** — neither CI-normalized difference nor proportion-outside-CI is ever gated.
3. **Suite value renamed** — `multimodel` → `allmodels`.
4. **Standard suite runs the pipeline by default** — the gated tofts/patlak-Ktrans checks are part of the default `pytest` run (downsample fixture, a few seconds).
5. **Gated models** — both **tofts and patlak** (Ktrans only), per the scope decision above.
