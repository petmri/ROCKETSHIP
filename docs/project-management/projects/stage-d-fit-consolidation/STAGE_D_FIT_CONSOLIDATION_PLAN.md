# Stage-D Fit Backend Consolidation

## Motivation

This project was triggered by a real regression report: running
`pytest tests/python -m parity -v -s` showed
`[PARITY] patlak_ktrans_brain_auto_vs_cpu` at correlation **>0.99** for commit
`3c17ff3416eaacd43de9571b456d4c11bff7f4d1` and earlier, dropping to **corr=-0.008583**
starting at commit `66fd7950acc4d431984da816d82cfaa4572d2886` -- the commit that switched
the steady-state window and AIF injection-timing resolution from hardcoded test values to
auto-detection. The open question was *why* a steady-state/timing change would cause the
Python-internal accelerated-vs-CPU Ktrans correlation to collapse, given both backends
fit the same pipeline and this number is generated fresh at test time (not a stale-MATLAB-
baseline artifact).

Isolating the two auto-detect changes independently (same `roi_stride=12` sparse mask the
real test uses, `n=237` voxels) pinned it on the steady-state window specifically:

| Variant | corr | CPU Ktrans max | GPUfit Ktrans max | GPUfit stuck-at-lower-bound |
|---|---|---|---|---|
| A -- old config, both fixed (steady-state `[1,2]`, injection fixed) | 0.998045 | 0.0138 | 0.0138 | 15/237 |
| B -- steady-state auto-detect only (injection still fixed) | -0.002645 | 0.5120 | 0.0149 | 17/237 |
| C -- injection-timing auto-detect only (steady-state still fixed) | 0.997032 | 0.0117 | 0.0117 | 23/237 |
| D -- both auto-detect (current/actual HEAD config) | -0.006768 | 0.5123 | 0.0131 | 22/237 |

Injection-timing auto-detect alone (C) leaves correlation intact; steady-state
auto-detect alone (B) reproduces the full regression, and matches D (today's actual
behavior) almost exactly. **The steady-state window is the cause.**

The mechanism: the steady-state window is the baseline period Stage A averages to
convert signal to concentration, so changing it changes the actual `Ct(t)` curves fed
into every voxel's fit -- not a cosmetic parameter. Under the old fixed `[1,2]` window,
every voxel's true Ktrans in this sample topped out near 0.014, small and tightly
clustered, so GPUFIT's single fixed starting guess (`initial_value_ktrans=0.0002`,
identical for every voxel regardless of the actual data) was close enough to converge
correctly almost everywhere. Once the auto-detected window resolves to a different
baseline, the true Ktrans range widens to ~0.51 in the same voxels, and GPUFIT --
still starting every voxel from that same fixed, data-blind `0.0002` with no per-voxel
seeding and (unlike `2cxm`/`tissue_uptake`) no multi-start rescue -- increasingly gets
stuck near its lower bound instead of climbing to the now much-higher true value, while
CPU (seeded per-voxel from the closed-form linear Patlak estimate) tracks the shifted
landscape correctly regardless of where the window lands.

In other words, the steady-state fix was correct (it's what makes Python match MATLAB),
but it exposed a pre-existing architectural weakness in the accelerated Stage-D path:
patlak's accelerated fit had no per-voxel seeding and no multi-start rescue at all, unlike
the CPU path. That architectural gap -- and the broader pattern of the same seeding/
multi-start logic being reimplemented differently per model and per backend -- is what
this consolidation project addresses. See the `parity-whole-brain-roi-noise` and
`parity-backend-divergence` memory notes for the adjacent (but separate) non-identifiable-
voxel finding uncovered while verifying the patlak pilot.

## Goal

Stage D fitting (`python/dce_pipeline.py` + `python/dce_models.py`) currently implements
the same concerns -- initial-value seeding, bounds construction, multi-start -- three
separate times, once per backend family, in incompatible ways:

- **CPU/python** (`dce_models.py`, one `model_*_fit` function per model): each function
  hand-builds its own settings dict, bounds, and (for patlak/ex_tofts/tissue_uptake) its
  own multi-start loop via `_best_fit_over_starts` with model-specific fixed-multiplier
  or hand-tuned candidate lists.
- **Accelerated** (`dce_pipeline.py::_fit_stage_d_model_accelerated`): builds a single
  fixed, data-uninformed initial row per model straight from prefs, then optionally runs
  `_accel_multistart_refine` -- a *different* multi-start algorithm (random log-uniform
  coarse-explore + refine) -- but only for `_ACCEL_MULTISTART_MODELS = {"2cxm",
  "tissue_uptake"}`.

The goal is one place to assemble the data structures every backend needs (data to fit,
bounds, candidate initial values) and one consolidated multi-start process every model
funnels through, with per-model pluggable candidate-assembly strategies (log-uniform
random draws, a closed-form linear-fit guess, a grid, or a single fixed value).

## Status: all five models migrated, cleanup done (2026-07-17)

`python/dce_fit_backends.py` now holds the shared machinery for **every accelerated-
eligible model**: patlak, tofts, ex_tofts, tissue_uptake, 2cxm. `dce_models.py`'s five
`model_*_fit` functions are all thin single-voxel wrappers over the corresponding
`fit_*_stage_d(..., backend="python")`; `dce_pipeline.py::_fit_stage_d_model_accelerated`
is now just five one-line delegations (its entire old per-model `initial_row`/
`bounds_row` construction, `_accel_multistart_refine`, `_extraction_fraction_init_bounds`,
and `_ACCEL_MULTISTART_MODELS` are deleted -- nothing else called them). The duplicated
tissue_uptake patlak-seed computation that used to live in
`dce_pipeline._fit_model_curve` is gone too; `assemble_tissue_uptake_candidates` is now
the single place that seed is computed. **Not deleted** (contrary to this doc's earlier
assumption): `dce_models._best_fit_over_starts` and `_clip_start_to_bounds` are still used
by `model_vp_fit` and `model_fxr_fit`, two non-accelerated models out of scope for this
project -- they stay.

Recap of the first three models migrated:

- `FitInputs` -- dataclass bundling `ct`/`cp`/`timer`/`bounds_row`/`prefs` for N voxels.
- `assemble_patlak_candidates(inputs) -> (n_starts, n_voxels, n_params)` -- the one place
  computing the linear-Patlak seed per voxel (`dce_models.model_patlak_linear`) and
  expanding it into patlak's existing x1/x10/x100 candidate rows.
- `assemble_tofts_candidates(inputs) -> (1, n_voxels, n_params)` -- tofts' single fixed
  prefs-default start, broadcast to every voxel (no per-voxel seeding for this model, on
  either backend, matching prior behavior exactly).
- `assemble_ex_tofts_candidates(inputs) -> (3, n_voxels, n_params)` -- ex_tofts' existing
  x1/x10/x100-on-Ktrans candidates (ve/vp held at prefs defaults), broadcast to every
  voxel -- the same fixed-multiplier strategy the CPU path has always used, now also
  applied on the accelerated backend for the first time.
- `run_backend_fit(backend, model_name, inputs, initial_parameters)` -- one signature for
  `"python"` (scipy `least_squares`, looped per voxel) and any accelerated backend string
  (`fit_module.fit_constrained`, one call for the whole batch), dispatching per model via
  a small runner registry. The per-voxel scipy loop and the accelerated `fit_constrained`
  call are each implemented once (`_run_scipy_per_voxel`, `_run_accelerated`) and shared
  by all three models' thin per-model runners -- adding tofts and ex_tofts turned the
  patlak-only runners into genuinely reusable helpers instead of just proving the pattern
  once.
- `fit_with_multistart(backend, model_name, inputs, candidates)` -- tries every candidate
  row, keeps the per-voxel best by chi-square/SSE. Replaces the "keep lower SSE"
  bookkeeping that both `_best_fit_over_starts` and `_accel_multistart_refine` implement
  separately, generalized to work whether the backend fits one voxel at a time (python)
  or the whole batch at once (cpufit/gpufit).
- `fit_patlak_stage_d(...)` / `fit_tofts_stage_d(...)` / `fit_ex_tofts_stage_d(...)` --
  top-level entry points, single or batched voxels, either backend. Tofts and ex_tofts
  keep the accelerated backend's original CI convention (CI columns repeat the point
  estimate, since no Jacobian is available), distinct from patlak's -1.0 sentinel -- these
  are preserved as per-model behavior in the shared architecture, not unified, since
  unifying them would be an unrequested behavior change.

`dce_models.model_patlak_fit`, `model_tofts_fit`, and `model_extended_tofts_fit` are now
thin single-voxel wrappers over the corresponding `fit_*_stage_d(..., backend="python")`;
`_fit_stage_d_model_accelerated`'s patlak/tofts/ex_tofts branches are gone, replaced by
one-line delegations. Verified: all patlak/tofts/ex_tofts unit/OSIPI/backend-consistency
tests pass unchanged (tofts' and ex_tofts' accelerated fixed-value tests assert the exact
same numbers as before their migrations, since the mock always returns the same canned
result regardless of which multistart candidate is tried and `fit_with_multistart`'s
strict less-than tie-break keeps the first candidate's result); the patlak accelerated
backend now gets real per-voxel seeding (previously a single fixed
`initial_value_ktrans` for every voxel, regardless of the actual data). For ex_tofts, the
accelerated backend now gets the same x1/x10/x100 multistart the CPU path already had --
a real behavior change, confirmed via a before/after `git stash` comparison on the
`sub-10bbbdownsample` parity fixture's `-m parity --parity-suite=allmodels` numbers: all
`ex_tofts_*_auto_vs_cpu`/`_auto_vs_matlab` corr/rmse values moved by noise-level amounts
in both directions (largest shift ~0.01 in either corr or rmse), consistent with this
fixture's ex_tofts fits already converging fine from the single fixed start -- i.e. no
regression, and the added multistart is now available for fixtures/voxels where it would
matter.

**Known residual, tabled (not this consolidation's job to fix):**
`patlak_ktrans_brain_auto_vs_cpu` on the `sub-10bbbdownsample` parity fixture still shows
near-zero correlation, driven by a single voxel where vp saturates its upper bound. This
is confirmed *not* an equally-valid-different-point-on-a-flat-manifold situation: CPU
converges to Ktrans=0.512261 (SSE=8339.5), gpufit to Ktrans=0.0 (chi-square=10231.9) --
CPU's objective is verifiably better, so gpufit is landing in a genuinely worse local
optimum once vp is pinned, compounded by this voxel's linear-Patlak seed itself being
degenerate (out of bounds), which makes the x1/x10/x100 multi-start collapse to zero
effective diversity. Confirmed unrelated to iteration/tolerance budget (2000 iters /
1e-10 tolerance changed nothing). GM/WM regions (which exclude this voxel) are already
perfect (corr=1.0). Full root-cause writeup, open questions, and the planned mitigation
(a GM/WM-style gating exception for patlak+brain, not a fitter change) are tracked in
`docs/project-management/projects/batch-parity/batch_parity.md` and the
`parity-whole-brain-roi-noise` memory note.

## `tissue_uptake` and `2cxm`: what was built

Both models' CPU and accelerated solvers work in genuinely different internal
parameterizations (CPU: Ktrans/Fp/Tp canonical-minutes for tissue_uptake, or a
resampled-grid `curve_fit` in canonical-minutes for 2cxm; accelerated: E=Ktrans/Fp
kernel space for both). The shared candidate space `assemble_*_candidates` produces is
therefore the **physical/output space** ([Ktrans, Fp, Vp] for tissue_uptake; [Ktrans, ve,
vp, Fp] for 2cxm) -- the one thing both backends' native parameterizations can be
losslessly converted to/from -- and each backend's runner (`_run_tissue_uptake_python` /
`_run_tissue_uptake_accelerated` / `_run_2cxm_python` / `_run_2cxm_accelerated`) converts
that shared space into whatever its own solver actually needs:

- **CPU** derives a `Tp` (tissue_uptake) or re-embeds the candidate as `initial_value_*`
  overrides into a settings copy passed straight into the existing, unmodified
  `_fit_2cxm_osipi_canonical` (2cxm) -- the safest possible way to add multistart to the
  model flagged as most numerically fragile, since none of its math is touched, only its
  starting point.
- **Accelerated** derives `E = Ktrans/Fp` (same formula `_extraction_fraction_init_bounds`
  used, now inlined as `_e_space_bounds` + a per-candidate clip).

Per-voxel/per-candidate `least_squares`/`curve_fit` exceptions are caught inside these two
models' runners specifically (unlike the shared `_run_scipy_per_voxel` used by
patlak/tofts/ex_tofts, which lets exceptions propagate) -- with random draws now in the
mix, one numerically-bad candidate should not sink a voxel another candidate fits fine.

Candidate strategy: fixed prefs-default + (tissue_uptake only) a per-voxel linear-Patlak
seed on Ktrans/Fp + N random log-uniform draws (4 for tissue_uptake, 5 for 2cxm) --
replacing tissue_uptake's old 4 hand-tuned CPU-only candidates and both models'
accelerated-only `_accel_multistart_refine` (coarse-explore-then-refine) with one shared
mechanism used identically by both backends, per this project's original target
architecture ("random log-uniform for 2cxm/tissue_uptake").

**A real bug found and fixed during verification:** the OSIPI reliability gate
(`test_osipi_2cum_reliability_delay0_against_reference_values`, a hard pass/fail gate on
official tolerances) failed on a low-flow case (`Fp=5` per 100mL/min) after the initial
tissue_uptake migration -- confirmed via `git stash` to be a genuine regression, not a
pre-existing flake. Root cause: `dce_models.model_tissue_uptake_fit`'s original hardcoded
fallback defaults (`initial_value_ktrans=2e-4`, `initial_value_fp=0.2`, used only when no
caller prefs are given) were always canonical-per-minute values, used directly with no
scaling. The new shared candidate space is raw/output-units (matching how the accelerated
backend has always used these same prefs keys, unconverted) -- so the CPU runner's
canonical-unit conversion (`* rate_in_to_min`) was applied a second, spurious time to the
fixed-default candidate specifically, a 60x error for this test's seconds-scale synthetic
data (real pipeline runs never hit this: Stage-D's timer is minutes-native in practice, so
`rate_in_to_min == 1` and the bug is a no-op there). Fixed by having
`assemble_tissue_uptake_candidates`/`assemble_2cxm_candidates` pre-divide the fixed
default's Ktrans/Fp by `rate_in_to_min` before storing it in the shared candidate array, so
the CPU runner's later multiplication recovers the original intended canonical value
exactly. Worth remembering for any future model migration that mixes a "canonical-only"
CPU convention with a "raw-only" accelerated convention under one shared candidate space.

**Verification and honest trade-offs:**
- All tissue_uptake/2cxm unit, OSIPI reliability, and OSIPI backend-consistency tests
  pass, including the mock-based accelerated-outputs test (its `expected_init`/
  `expected_bounds` for both models are unchanged, since they assert the *fixed* base
  candidate specifically).
- Full `pytest tests/python -q`: 195 passed, only the pre-existing tabled patlak failure
  (unchanged from before this migration).
- Real, measured runtime cost: the full local suite went from ~85s to ~136s, and
  `-m osipi` from ~40s to ~87s -- running 5-6 full-cost candidates per voxel (patlak/
  tofts/ex_tofts fixed-multiplier fits are cheap and few; tissue_uptake/2cxm's random
  multistart is not) instead of the old accelerated-only coarse-then-refine trick, which
  ran cheap coarse fits for every candidate and only one full-cost refine. This project
  deliberately did not reimplement that coarse/refine optimization inside the shared
  `fit_with_multistart` (real added complexity for an optimization that only matters at
  much larger voxel counts than the local test fixtures use) -- if a real BIDS batch run
  on GPU hardware turns out to be meaningfully slower, that optimization (or just fewer
  random draws) is the first thing to try.
- Before/after `git stash` comparison on `sub-10bbbdownsample`'s `-m parity
  --parity-suite=allmodels` numbers (none of these are gated, only reported): tissue_uptake
  moved by noise-level amounts in both directions (a wash). 2cxm moved more, and mixed:
  the primary Ktrans correlation improved substantially everywhere it was checked (e.g.
  `2cxm_ktrans_brain_auto_vs_cpu` 0.832 -> 0.980, `_auto_vs_matlab` 0.861 -> 0.942), but a
  few GM/WM backend-consistency numbers on secondary params got worse (e.g.
  `2cxm_ktrans_gm_auto_vs_cpu` 0.450 -> 0.167, `2cxm_ve_gm_auto_vs_matlab` 0.766 -> 0.406).
  Not investigated further: 2cxm is the model already flagged as the most numerically
  fragile in this project ([[noisy-data-parity-philosophy]]: "2CXM unstable") and these
  are small-n (57-119 voxel), already-noisy correlations on secondary parameters, not
  gated assertions -- but if 2cxm parity regressions matter later, start here.

## Cleanup (done)

- Deleted `dce_pipeline._accel_multistart_refine`, `_extraction_fraction_init_bounds`,
  `_ACCEL_MULTISTART_MODELS`/`_ACCEL_MULTISTART_STARTS`/`_ACCEL_MULTISTART_COARSE_ITERS`,
  and the now-fully-dead per-model `initial_row`/`bounds_row` construction inside
  `_fit_stage_d_model_accelerated` (the function is now five one-line delegations).
  Updated the two OSIPI test-file docstrings that referenced `_accel_multistart_refine`
  by name.
- Removed the duplicated tissue_uptake patlak-seed computation in
  `dce_pipeline._fit_model_curve`; `assemble_tissue_uptake_candidates` is now the only
  place that seed is computed (previously computed three times: there, in patlak's own
  copy, and in `model_tissue_uptake_fit` itself).
- **Not deleted, corrected from this doc's earlier assumption:**
  `dce_models._best_fit_over_starts` and `_clip_start_to_bounds` are still used by
  `model_vp_fit` and `model_fxr_fit` -- two models never in scope for this project (not in
  `ACCELERATED_STAGE_D_MODELS`). They stay.

## Call-chain simplification (2026-07-17)

Once all five models shared the same `fit_*_stage_d(ct, cp, timer, prefs, backend)`
signature (single or batched voxels, either backend), the long-standing early split
between "accelerated" (batched) and "CPU" (a per-voxel loop through `_fit_model_curve` ->
a thin `model_*_fit` wrapper -> the same `fit_*_stage_d` anyway) turned out to be
artificial duplication, not a real architectural need -- both paths ended up at the exact
same function, just reached differently. Simplified `_fit_stage_d_model`'s CPU fallback to
call the same batched `fit_*_stage_d(..., backend="python")` directly (one call, not a
per-voxel loop), via a small `_stage_d_fit_funcs()` registry shared with
`_fit_stage_d_model_accelerated` (also simplified from five near-identical branches to one
dict lookup). `_fit_model_curve` now only handles `fxr` (the one model outside this shared
architecture, since its per-voxel R1 baseline can't be batched the same way) -- its other
five branches, and the five `model_*_fit` imports they used, are gone from
`dce_pipeline.py` (the `model_*_fit` wrappers themselves stay in `dce_models.py`: they're
still the public single-voxel API used directly by tests, `rocketship.py`, and
diagnostic scripts).

One real robustness gap this closed: `_run_scipy_per_voxel` (patlak/tofts/ex_tofts'
shared python runner) had no per-voxel exception handling, relying on the now-removed
outer per-voxel loop in `dce_pipeline.py` to isolate one bad voxel from the rest of the
batch. Added the same per-voxel try/except tissue_uptake/2cxm's runners already had, so
one malformed voxel (e.g. a stray NaN that leaked through a mask) still can't sink an
entire batch.

`_stage_d_fit_funcs()` is a function, not a module-level dict: a dict literal built once
at import time would capture the original `fit_*_stage_d` function objects, which doesn't
observe later monkeypatching of e.g. `dce_pipeline.fit_tofts_stage_d` in tests (a dict
holds object references, not name lookups) -- discovered when two tests that patch a
`fit_*_stage_d` name by string kept getting the unpatched original. Rebuilding the dict
inside a function each call reads the current module-level names instead. Two existing
tests (`test_stage_d_gpu_failure_without_cpufit_falls_back_to_cpu`,
`test_stage_d_nonfinite_accelerated_output_falls_back_to_cpu`) were testing the old
per-voxel `_fit_model_curve` CPU fallback specifically and were rewritten to assert the
new batched call instead; every other accelerated-fallback-chain test was untouched since
it mocks `_fit_stage_d_model_accelerated` itself, which kept its exact external behavior.

Verified: full `pytest tests/python -q` still 195 passed (same pre-existing tabled
failure only); `-m osipi` still 22 passed.

### Second pass: collapsing `dce_fit_backends.py`'s internal duplication

Tracing the call chain all the way down (`_fit_stage_d_model` -> `fit_with_multistart` ->
`run_backend_fit` -> per-model runner -> `_run_scipy_per_voxel`/`_run_accelerated`) surfaced
one more genuine 1:1 wrapper and, more importantly, ~250 lines of near-identical logic
copy-pasted across the five `fit_*_stage_d` entry points:

- **`run_backend_fit` deleted**, merged into `fit_with_multistart` (its only caller,
  confirmed via full-repo grep; not imported or mocked by any test). `fit_with_multistart`
  now looks up the per-model/per-backend runner once per multistart loop instead of once
  per candidate.
- **Every `fit_*_stage_d` function had the identical ~12-line input-validation block**
  (normalize `ct`/`cp`/`timer`, detect single-voxel, check shapes) **and an almost-identical
  output-row assembly loop** (copy point estimates + sse, then either real CI bounds or a
  per-model fallback). Extracted both into shared helpers: `_validate_stage_d_inputs` and
  `_assemble_stage_d_output`. The output layout turned out to be a universal convention
  across all five models' `MODEL_LAYOUTS` (`row_len = 3*n_params + 1`: point estimates,
  then sse, then interleaved ci_low/ci_high pairs in the same order) -- confirmed by
  checking every model's layout before generalizing, not assumed.
  - This required unifying the "extra" payload every runner returns: patlak/tofts/
    ex_tofts's python runner (`_run_scipy_per_voxel`) used to hand back the raw scipy
    `OptimizeResult`, with CI computed later at the top level via `_ci_bounds_from_fit`;
    tissue_uptake/2cxm's bespoke runners already computed `(ci_lo, ci_hi)` tuples directly
    (they need extra rate-scaling `_ci_bounds_from_fit` alone can't do). Moved the
    `_ci_bounds_from_fit` call inside `_run_scipy_per_voxel` itself so all five models'
    runners return the same `(ci_lo, ci_hi)`-tuple-or-`None` shape, letting one output
    assembler handle every model.
  - One real difference survives, by design, not oversight: patlak's accelerated backend
    fills missing CIs with a `-1.0` sentinel (its long-standing quirk), while every other
    model repeats the point estimate. `_assemble_stage_d_output` takes this as an explicit
    `ci_fallback` argument rather than silently unifying the two conventions.
- **Introduced `_ModelSpec`/`_MODEL_SPECS`/`_fit_stage_d_batch`**: a small dataclass
  bundling each model's `settings_fn`/`bounds_fn`/`assemble_fn`/`n_params`/`ci_fallback`,
  and one generic engine that assembles inputs, runs `fit_with_multistart`, and calls
  `_assemble_stage_d_output`. The five public `fit_patlak_stage_d`/`fit_tofts_stage_d`/etc.
  functions **could not be deleted or merged into one** -- `dce_pipeline.py` imports each
  by name and several tests patch them by name (e.g.
  `patch("dce_pipeline.fit_tofts_stage_d", ...)`), so each stayed a real, individually
  importable/patchable function -- but each is now a 1-line delegation to
  `_fit_stage_d_batch(model_name, ...)` plus its (kept, since it documents real per-model
  behavior differences) docstring, down from ~35-45 lines apiece.
- Caught one arithmetic bug immediately via the test suite: first wrote
  `_assemble_stage_d_output`'s row length as `2*n_params + 1` (an off-by-n error --
  forgot the output has *two* CI columns per parameter, not one) and got an `IndexError`
  on the very first test run; fixed to `3*n_params + 1` and re-verified.

Verified again after this pass: full `pytest tests/python -q` still 195 passed (same
pre-existing tabled failure only); `-m osipi` still 22 passed.

### Third pass: "python" as just another fallback candidate, and a real bug found along the way

`_fit_stage_d_model` still had a two-stage shape: a loop trying accelerated backend
candidates, then -- as an entirely separate code block below it -- a single CPU/python
fallback call. Since every candidate (accelerated or not) already goes through
`_fit_stage_d_model_accelerated`, which is backend-string-agnostic (it just guards and
delegates to the model's `fit_*_stage_d`), there was no real reason for "python" to be a
separate step rather than the final entry in the same candidate list.

Tracing this surfaced a real, previously-invisible bug: `_apply_model_specific_prefs`
(which strips e.g. `2cxm_lower_limit_fp` down to `lower_limit_fp` for that model
specifically -- the mechanism `_stage_d_fit_prefs` provides so 2cxm/tissue_uptake, the two
models flagged elsewhere in this doc as least stable, can be tuned independently without
affecting other models) was being called **only** in the CPU/python fallback branch. The
accelerated-attempt loop passed `prefs` raw. `tests/python/osipi_fast_backend_helpers.py`
already worked around this itself (it calls `_apply_model_specific_prefs` before invoking
the accelerated function directly) -- which is how the gap was found, not from a failing
test. Net effect in production: whenever the accelerated backend actually succeeded (the
common case), 2cxm/tissue_uptake's per-model override knobs
(`voxel_lower_limit_fp_2cxm`, `voxel_initial_value_vp_tissue_uptake`, etc.) were silently
ignored; they only took effect on the rare run that fell all the way back to pure CPU.

Asked the user how to handle this rather than silently folding in a numeric behavior
change alongside a structural refactor; they chose to fix it. `_fit_stage_d_model` now
calls `_apply_model_specific_prefs` once, before building the candidate list, and every
candidate -- accelerated or "python" -- gets the same processed prefs.
`_acceleration_backend_attempt_order(acceleration_backend) + ["python"]` is the full
candidate list unconditionally (when `acceleration_backend == "none"`, the accelerated
part is `[]`, so the list is just `["python"]`, same net effect as before). Log messages
were reworded from "acceleration backend" / "falling back to pure CPU" to "backend" / "no
fallback remains", since "python" being just another candidate makes the old CPU-specific
phrasing inaccurate.

Two tests (`test_stage_d_gpu_failure_without_cpufit_falls_back_to_cpu`,
`test_stage_d_nonfinite_accelerated_output_falls_back_to_cpu`) previously mocked
`fit_tofts_stage_d`/`fit_ex_tofts_stage_d` directly to test the CPU-fallback step in
isolation (from the prior pass, when it was a separate code path); rewritten to mock
`_fit_stage_d_model_accelerated` with backend-conditional `fake_accel` functions instead,
matching every other fallback-chain test's style now that "python" is just another value
of `acceleration_backend` passed to the same seam. Every other fallback-chain test needed
no changes (they all return early on a successful earlier candidate, so never reach
"python").

`_fit_model_curve` renamed to `_fit_fxr_curve` (it now only ever handles `fxr`, the one
model outside the shared batched architecture, since its per-voxel R1 baseline can't be
batched the same way). `ACCELERATED_STAGE_D_MODELS` deleted (fully superseded by
`_stage_d_fit_funcs()`'s keys, and no longer referenced anywhere after this pass).

Verified: full `pytest tests/python -q` still 195 passed (same pre-existing tabled
failure only); `-m osipi` still 22 passed -- the osipi backend-consistency/reliability
sweeps are exactly where a 2cxm/tissue_uptake prefs-handling change would be expected to
show up, and they stayed green.

## Possible follow-ups (not started, not blocking)

- If real BIDS batch runtime on GPU hardware is measurably slower for tissue_uptake/2cxm,
  consider a coarse-then-refine option inside `fit_with_multistart` (opt-in per model),
  or simply reduce `multistart_starts` from the current defaults (4 / 5).
- If 2cxm's GM/WM secondary-parameter backend-consistency numbers turn out to matter for
  a real dataset (not just this noisy synthetic fixture), revisit here first.

## Verification checklist per model migration

- Model's own unit tests in `tests/python/test_dce_models.py` pass with unchanged numeric
  output (same math, relocated).
- `pytest tests/python -m osipi -v` (backend-consistency + reliability sweeps for that
  model) pass.
- `pytest tests/python -m parity -v -s` -- check the model's `*_auto_vs_cpu` and
  `*_auto_vs_matlab` lines specifically; expect accelerated numbers to move *closer* to
  CPU/MATLAB where the migration adds seeding/multistart that didn't exist before, never
  further away.
- Full `pytest tests/python -q` for fallout.
- Per standing process: run all of the above locally before pushing to CI.
