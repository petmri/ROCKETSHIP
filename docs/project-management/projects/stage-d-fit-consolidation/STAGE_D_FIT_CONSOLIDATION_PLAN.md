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

## Status: patlak + tofts + ex_tofts migrated (2026-07-17)

`python/dce_fit_backends.py` now holds the shared machinery, proven end-to-end on
**patlak, tofts, and ex_tofts**:

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

## Remaining work (not started)

Extend the same three pieces -- assembler, backend runner, `fit_with_multistart` -- to
the remaining two accelerated-eligible models, in this order (each is a candidate for
its own pass, verified against the full local test/parity suite before moving to the next):

### 1. `tissue_uptake`
- CPU (`dce_models.model_tissue_uptake_fit`): canonical-unit conversion
  (`_canonical_time_context`/`_merge_prefs_in_canonical_units`, minutes + per-minute
  rates) *before* candidate assembly, then 4 hand-tuned candidates plus a 5th seeded from
  `model_patlak_linear` (computed a **third** time here, after patlak's own copy and
  `dce_pipeline._fit_model_curve`'s tissue_uptake branch -- consolidating this seed
  computation into one shared helper is a direct win from this migration).
- Accelerated: already in `_ACCEL_MULTISTART_MODELS`, using `_accel_multistart_refine`'s
  random log-uniform coarse-explore + refine in E-space (`_extraction_fraction_init_bounds`
  maps Ktrans/Fp to E=Ktrans/Fp). The assembler for this model needs to support a
  genuinely different candidate strategy (random draws, not fixed multipliers) plus the
  canonical-unit/E-space transforms -- this is the first model to exercise that
  flexibility in the new architecture, and the main design risk: confirm
  `fit_with_multistart`'s generic "try every candidate, keep best" loop still gets the
  coarse-then-refine performance benefit `_accel_multistart_refine` currently has for
  large voxel counts, or accept the simpler "run every candidate at full cost" and
  benchmark whether it matters in practice.

### 2. `2cxm`
- CPU: single-path OSIPI-canonical fit (`dce_models._fit_2cxm_osipi_canonical`), not
  `_best_fit_over_starts`-based -- no multi-start on CPU today at all. Accelerated: same
  E-space + random-log-uniform multistart as tissue_uptake. Migrate last since it's the
  most structurally different CPU path (a single canonical fit function rather than a
  residual+`least_squares` pair) and the most numerically fragile model per existing
  parity notes ([[noisy-data-parity-philosophy]]: "2CXM unstable").

### 3. Cleanup (only once both are migrated)
- Delete `dce_models._best_fit_over_starts` and `dce_pipeline._accel_multistart_refine`,
  `_extraction_fraction_init_bounds` (folded into the tissue_uptake/2cxm assemblers), and
  the now-fully-dead `_fit_stage_d_model_accelerated` per-model `initial_row`/`bounds_row`
  construction and `_ACCEL_MULTISTART_MODELS` set.
- Remove the duplicated tissue_uptake patlak-seed computation in
  `dce_pipeline._fit_model_curve` (lines ~3121-3151 as of this writing) once
  `assemble_tissue_uptake_candidates` owns that seed.

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
