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

## Status: patlak pilot complete (2026-07-17)

`python/dce_fit_backends.py` now holds the shared machinery, proven end-to-end on
**patlak only**:

- `FitInputs` -- dataclass bundling `ct`/`cp`/`timer`/`bounds_row`/`prefs` for N voxels.
- `assemble_patlak_candidates(inputs) -> (n_starts, n_voxels, n_params)` -- the one place
  computing the linear-Patlak seed per voxel (`dce_models.model_patlak_linear`) and
  expanding it into patlak's existing x1/x10/x100 candidate rows.
- `run_backend_fit(backend, model_name, inputs, initial_parameters)` -- one signature for
  `"python"` (scipy `least_squares`, looped per voxel) and any accelerated backend string
  (`fit_module.fit_constrained`, one call for the whole batch).
- `fit_with_multistart(backend, model_name, inputs, candidates)` -- tries every candidate
  row, keeps the per-voxel best by chi-square/SSE. Replaces the "keep lower SSE"
  bookkeeping that both `_best_fit_over_starts` and `_accel_multistart_refine` implement
  separately, generalized to work whether the backend fits one voxel at a time (python)
  or the whole batch at once (cpufit/gpufit).
- `fit_patlak_stage_d(ct, cp, timer, prefs, backend)` -- top-level entry point, single or
  batched voxels, either backend.

`dce_models.model_patlak_fit` is now a thin single-voxel wrapper over
`fit_patlak_stage_d(..., backend="python")`; `_fit_stage_d_model_accelerated`'s patlak
branch is gone, replaced by a one-line delegation to `fit_patlak_stage_d(...,
backend=acceleration_backend)`. Verified: all patlak unit/OSIPI/backend-consistency
tests pass unchanged; the accelerated backend now gets real per-voxel seeding (previously
a single fixed `initial_value_ktrans` for every voxel, regardless of the actual data).

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
the other four accelerated-eligible models, in this order (each is a candidate for its
own pass, verified against the full local test/parity suite before moving to the next):

### 1. `tofts` (lowest risk -- no behavior change expected)
- Single fixed candidate today on both backends (CPU: `model_tofts_fit`'s direct
  `least_squares` call, no multi-start; accelerated: fixed `initial_value_ktrans`/`_ve`
  from prefs). `assemble_tofts_candidates` just needs to produce a `(1, n_voxels, 2)`
  array from the prefs defaults -- a mechanical move, not a new strategy. Good first
  target to prove the pattern generalizes beyond patlak's linear-seeded case without any
  numeric risk.

### 2. `ex_tofts`
- CPU (`dce_models.model_extended_tofts_fit`): 3 fixed-multiplier candidates
  (`x1`/`x10`/`x100` on Ktrans, ve/vp held at defaults), via `_best_fit_over_starts` --
  same pattern as patlak's old CPU path, no linear-regression seed. Port directly:
  `assemble_ex_tofts_candidates` builds the same 3 rows (no per-voxel seed needed, unlike
  patlak).
- Accelerated: single fixed start today, not in `_ACCEL_MULTISTART_MODELS`. Migrating
  will give it the same x1/x10/x100 multistart the CPU path already has -- a real
  behavior change on the accelerated side, expected to only improve accuracy (same
  direction as the patlak fix), but re-verify ex_tofts parity numbers specifically.

### 3. `tissue_uptake`
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

### 4. `2cxm`
- CPU: single-path OSIPI-canonical fit (`dce_models._fit_2cxm_osipi_canonical`), not
  `_best_fit_over_starts`-based -- no multi-start on CPU today at all. Accelerated: same
  E-space + random-log-uniform multistart as tissue_uptake. Migrate last since it's the
  most structurally different CPU path (a single canonical fit function rather than a
  residual+`least_squares` pair) and the most numerically fragile model per existing
  parity notes ([[noisy-data-parity-philosophy]]: "2CXM unstable").

### 5. Cleanup (only once all four are migrated)
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
