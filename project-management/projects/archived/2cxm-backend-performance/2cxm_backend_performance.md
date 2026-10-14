# 2CXM Stage-D Backend Performance (python vs cpufit vs gpufit)

Status: **root cause found, two fixes written and verified**. GPUfit's 2CXM was slower than
CPUfit's mainly because the CUDA kernel's inner loop accumulates in `double` by accident, and
secondarily because the CUDA line search recomputes the model for fits that already accepted a
step. Together the fixes make gpufit **21.8x** faster end-to-end and turn it from 4.8x slower
than cpufit into 4.6x faster.

Measured 2026-08-12 on `barnes-research` (RTX 5090, driver 610.43.02, CUDA runtime 13.3;
cpufit and python are single-threaded, gpufit has the whole GPU).

## What was measured

`dce_fit_backends.fit_2cxm_stage_d` — the whole Stage-D 2CXM unit the pipeline calls,
including the 0.1 s dense-grid upsample on the accelerated path and the 6-candidate
multistart on every path.

Data: the 24 OSIPI 2CXM DRO cases (`2cxm_sd_0.001_delay_0.csv`), decimated from their native
0.5 s sampling to a **5 s frame time**, so the dense grid is the same ~50x upsample a real DCE
acquisition sees: **60 acquired points -> 2951 dense points**. Voxel counts above 24 replicate
the 24 cases with fresh Gaussian noise at the dataset's own sd, so replicate voxels are
distinct fits rather than identical work.

Harness: `bench_2cxm.py` / `diag_gpufit.py` (scratch, not committed; recreate from this doc).

## Results

Wall clock for the whole Stage-D call, best of 3 (2400-voxel rows are single runs). The two
gpufit fix columns are cumulative: "float" is fix 1 alone, "both" adds fix 2.

| voxels | python | cpufit | gpufit (shipped) | gpufit (float) | gpufit (both) |
|-------:|-------:|-------:|-----------------:|---------------:|--------------:|
| 24     | 2.43 s (101.2 ms/vox) | **0.62 s (25.8)** | 13.28 s (553.5) | 2.03 s (84.6) | 1.69 s (70.2) |
| 240    | 23.73 s (98.9) | 6.62 s (27.6) | 57.85 s (241.1) | 4.37 s (18.2) | **3.50 s (14.6)** |
| 2400   | not run | 71.09 s (29.6) | 340.29 s (141.8) | 24.30 s (10.1) | **15.58 s (6.5)** |

Net gpufit speedup: 7.9x / 16.5x / **21.8x**. Against cpufit, gpufit goes from 4.8-8.7x slower
to 1.9x faster at 240 voxels and 4.6x faster at 2400; cpufit still wins at 24 voxels, where the
GPU is launch-overhead bound rather than work bound.

All backends agree on the fits: median SSE and median Ktrans match to 4-5 significant figures
across every cell above. (Python left 2 of 240 voxels unfitted; both accelerated backends fit
all of them.) Fix 2 is bit-identical to fix 1 alone on every cell -- it removes redundant work
without changing the arithmetic, which is the point.

Reading of the shipped columns:

- **cpufit is not "buying nothing" on this host** — it is 3.6-3.9x faster than python at every
  size, contradicting the earlier note in `TODO.md` (measured 2026-08-02 on a different box,
  where cpufit merely tied python). The two known cpufit inefficiencies are still unfixed, so
  that margin is not the ceiling.
- **gpufit was 4.8-8.7x slower than a single CPU core**, and its per-voxel cost was still
  falling at 2400 voxels (553 -> 241 -> 142 ms/voxel), i.e. it was not merely
  occupancy-starved at small batch sizes.

## Why gpufit was slower than cpufit

Three things separate the two backends. Only the second one turned out to matter.

### 1. The CUDA kernel is O(n^2) where cpufit is O(n) — real, but survivable

`Cpufit/lm_fit_cpp.cpp:exp_conv_recurrence` evaluates the two exponential convolutions with an
O(n) recurrence (1 `std::exp` per point per rate). A gpufit model function evaluates *one point
per thread*, so `Gpufit/models/two-compartment_exchange.cuh:74` cannot use a recurrence and
instead runs its own `for (i = 1; i <= point_index; i++)` sum with 4 `exp` per inner iteration.
At 2951 dense points that is ~1475x more arithmetic per curve evaluation.

Measured scaling, per fit-iteration at 24 fits, over 369 -> 2951 points (8x):

| backend | 369 pts | 738 | 1476 | 2951 | exponent |
|---------|--------:|----:|-----:|-----:|---------:|
| cpufit  | 26.9 us | 35.2 | 96.4 | 248.3 | ~1.06 (linear) |
| gpufit  | 75.9 us | 228.1 | 602.1 | 1840.5 | ~1.53 |

gpufit's exponent sits between 1 and 2 because the thread count also grows with n_points; the
*work* is quadratic, the *wall time* is partly absorbed by parallelism. This is architectural
and cannot be fixed without a prefix-scan formulation or a whole-curve-per-thread model — but
after fix #2 below, gpufit beats cpufit by 2.9x at 2400 voxels **while still doing the O(n^2)
work**. So the quadratic kernel was never the reason gpufit lost.

### 2. The inner loop silently runs in double precision — this was the whole gap

```
Gpos  += 0.5 * spacing * (Cp[i] * ep_i + Cp[i - 1] * ep_p);
```

`0.5` is a `double` literal. It promotes the entire right-hand side to double, so all four
accumulators (`Gpos`, `Gppos`, `Gneg`, `Gpneg`) are accumulated in double and narrowed back to
float on store — inside the hottest loop in the fit. `Cpufit`'s equivalent uses `0.5f`
throughout and stays in float.

Confirmed in SASS (`nvcc -O3 -arch=sm_120`, isolated compile of the model header). The inner
loop body is 145 instructions and contains:

- 4 `MUFU.EX2` — the four exponentials, as expected
- **1 `DMUL` + 4 `DFMA` + 13 `F2F` conversions** — the accidental double math
- 64 `NOP` — scheduling stalls, consistent with waiting on the FP64 pipe

On a consumer Blackwell part (RTX 5090) FP64 runs at **1/64** of FP32 rate, so five FP64 ops
per inner iteration dominate everything else in the loop.

Isolated kernel timing, same launch shape gpufit uses (value + full Jacobian, one thread per
(fit, point)):

| variant | 24 fits | 240 fits |
|---------|--------:|---------:|
| shipped (`0.5`)   | 4.965 ms/eval | 26.223 ms/eval |
| float (`0.5f`)    | 0.354 ms/eval | 1.736 ms/eval  |
| speedup           | **14.0x**     | **15.1x**      |

End-to-end, rebuilding `libGpufit.so` with float literals (see "Patch" below) reproduces it:
13.28 -> 2.03 s at 24 voxels (6.5x), 57.85 -> 4.37 s at 240 (13.2x), 340.29 -> 24.30 s at 2400
(**14.0x**). `test_osipi_pygpufit.py`, `test_osipi_backend_consistency.py` and
`test_backend_equivalence.py` all pass on the patched library, and fitted parameters match the
shipped kernel to 4-5 significant figures (not bit-identical — float accumulation order
changed).

`Gpufit/models/tissue_uptake.cuh:64-66` had the identical bug (three accumulators rather than
four) and is fixed the same way; its SASS now contains zero FP64 instructions.
`tofts.cuh`/`tofts_extended.cuh`/`patlak.cuh` write `/ 2`, which stays in float, so they are clean.

tissue_uptake is worth reporting separately because its accelerated path does **not** build the
0.1 s dense grid — it fits on the acquired timebase, so n_points, and therefore its exposure to
the quadratic kernel, comes entirely from the acquisition. 240 voxels of the OSIPI 2CUM DRO set:

| timebase | n_points | cpufit | gpufit (shipped) | gpufit (fixed) | speedup |
|----------|---------:|-------:|-----------------:|---------------:|--------:|
| 5 s frames        | 60  | **0.017 s** | 0.150 s | 0.062 s | 2.4x |
| 0.5 s (native)    | 600 | 0.248 s | 1.532 s | **0.162 s** | **9.4x** |

So the fix moves gpufit tissue_uptake from 6.2x slower than cpufit to 1.5x faster at 600 points,
while at 60 points the problem is too small for the GPU either way. Median SSE and median Ktrans
are **bit-identical** between the shipped and fixed libraries in all four rows.

### 3. The CUDA line search recomputes the model for fits that already accepted a step

`Gpufit/lm_fit_cuda.cu:222` loops `backtracking_index` 0..8 and calls `calc_curve_values()` +
`calc_chi_squares()` every time, with no early exit. Cpufit's equivalent
(`Cpufit/lm_fit_cpp.cpp:2481`) `break`s at the first step that improves chi-square. So every LM
iteration on CUDA costs **9 full curve + Jacobian evaluations**.

Constrained vs unconstrained (1 evaluation per iteration), per fit-iteration at 24 fits:

| backend | constrained | unconstrained | ratio |
|---------|------------:|--------------:|------:|
| cpufit                | 252.3 us | 41.7 us | 6.0x |
| gpufit (shipped)      | 1843.2 us | 170.8 us | 10.8x |
| gpufit (float fix)    | 605.1 us | 20.9 us | **29x** |

Arithmetic check that this accounts for the observed cost: at 24 fits the shipped constrained
path costs 1840 us/fit-iteration x 24 fits = 44.2 ms per LM iteration for the batch; 9 x the
isolated kernel's 4.965 ms/eval = 44.7 ms. The constrained path *is* the nine evaluations. An
nsys profile of the 240-voxel run says the same thing directly: `cuda_calc_curve_values` is
**92.4%** of all GPU time, at 9573 launches against 1069 for `cuda_calculate_hessians` (one per
LM iteration) — exactly 9.0 model evaluations per LM iteration.

**Fix.** `backtrack_accepted` already exists in `GPUData` and `cuda_update_parameters_trial`
already honours it (`cuda_kernels.cu:1110`), so an accepted fit's parameters are held fixed for
the rest of the search — but `cuda_calc_curve_values` did not honour it, and so recomputed that
fit's unchanged curve and Jacobian up to 8 more times. Threading the same mask into that one
kernel is pure waste elimination, and the measured results are bit-identical.

Only `cuda_calc_curve_values` was changed, deliberately. `cuda_calculate_chi_squares` does the
same redundant work but is 1.2% of GPU time, and its reduction calls `sum_up_floats`, which
contains `__syncthreads()`; a partial-block early return there would be a divergence hazard
whenever `n_fits_per_block > 1`. (The pre-existing `finished[fit_index]` return in that kernel
already has that shape — worth a separate look upstream, but not something to widen here.)
`cuda_calc_curve_values` has no `__syncthreads()`, so an early return on part of a block is safe.

Result: model-evaluation GPU time drops 3.93 s -> 3.00 s at 240 voxels, and the fastest launches
fall from 207 us to 3.9 us (batches where every fit is skipped). End-to-end that is 1.25x at 240
voxels and 1.56x at 2400 — the win grows with batch size, since more fits accept early and the
saved work converts directly into time once the GPU is saturated. It is not larger because LM
frequently rejects a step at *every* scale, and those fits legitimately run all 9 trials on both
backends; after this change gpufit and cpufit do comparable numbers of model evaluations.

### Not the cause

- **Precision build flag.** `DOUBLE_PRECISION:BOOL=OFF` — the library is a single-precision
  build. The FP64 in the loop is purely the C++ literals.
- **PTX JIT.** `CMAKE_CUDA_ARCHITECTURES=75;80;86;89;90;100;120;120-virtual` — native sm_120
  SASS is present, nothing is JIT-compiled at load.
- **Small-batch occupancy alone.** Per-fit-iteration cost falls 12196 -> 1084 us going from 1
  to 1536 fits and then flattens, so the GPU is saturated by ~384 fits; even fully saturated,
  the shipped kernel was 4.4x slower per fit-iteration than one CPU core.
- **Per-iteration host sync.** `evaluate_iteration` does one device->host read per LM iteration
  (`lm_fit_cuda.cu:453`), tens of microseconds against tens of milliseconds of kernel time.

## Patch

Applied to `~/Code/Gpufit` (working tree only, **not committed**) and built into
`~/Code/Gpufit-bin/Gpufit/libGpufit.so`. 34 insertions, 13 deletions across 5 files:

```
Gpufit/models/two-compartment_exchange.cuh  0.5 -> (REAL)0.5f (4 accumulators + kpos/kneg +
                                            dkpos/dkneg), 2 -> (REAL)2.0f, 1e-12 -> (REAL)1e-12f
Gpufit/models/tissue_uptake.cuh             same, for its 3 accumulators + eps
Gpufit/cuda_kernels.cu / .cuh               cuda_calc_curve_values gains an optional
                                            `int const * skip` mask (NULL = no skipping)
Gpufit/lm_fit.h / lm_fit_cuda.cu            LMFitCUDA::calc_curve_values(int const * skip = 0);
                                            the backtracking loop passes backtrack_accepted_
```

Both model headers carry a comment at the accumulation lines saying why the literals must stay
`REAL` — this is the failure mode that produces correct answers 14x too slowly, so it will not
survive a casual edit otherwise.

Validation on the patched library, all green:

- Gpufit's Boost suite via `ctest` — 5/5 passed (`Gpufit_Test_Error_Handling`,
  `Gpufit_Test_Patlak`, `Cpufit_Gpufit_Test_Consistency`, `Cpufit_Test_ToftsCpu`,
  `Cpufit_Test_PatlakCpu`). Note what this does and does not cover: `Consistency` exercises the
  **solver** change on linear/gauss_1d/gauss_2d_elliptic (`GPUFIT_USE_BASE_MODELS=ON`), which is
  the right check for a change in `cuda_calc_curve_values` since it touches every model — but no
  Boost test touches 2cxm or tissue_uptake. Those two are covered by the MRI parity example and
  the ROCKETSHIP tests below. Also worth knowing: `Gpufit/tests/` contains eight more test
  sources (Gauss 2D, Brown-Dennis, Fletcher-Powell, ...) that are only wired up behind
  `USE_BASE_MODELS`/`USE_GAUSS2D`, so `ctest` is a thinner suite than the directory suggests.
- Gpufit's own `Gpufit_Cpufit_MRI_Parity` example — PASS on all six MRI models
  (patlak, tofts, tofts_extended, tissue_uptake, 2cxm, T1_FA_exponential), 128/128 jointly
  converged each, no state mismatches. The 2cxm and tissue_uptake `max |param rel diff|` values
  (0.00794723 and 0.00187197) are **unchanged to six digits** from the pre-fix library.
- ROCKETSHIP `test_osipi_pygpufit.py`, `test_osipi_pycpufit.py`,
  `test_osipi_backend_consistency.py`, `test_backend_equivalence.py` — 14 passed (all five
  accelerated models, plus gpufit vs cpu on 2000 real Stage-B voxels).

The installed `pygpufit` in ROCKETSHIP's `.venv` was **left untouched**; the patched library was
tested via a scratch copy on `PYTHONPATH`. To adopt it, copy the rebuilt `libGpufit.so` over
`.venv/lib/python3.14/site-packages/pygpufit/libGpufit.so`, or rebuild the pyGpufit wheel — note
that `install_python_acceleration.py` pulls the prebuilt wheel from the `ironictoo/Gpufit`
release, so a release build is what actually reaches CI and other machines.

## Open follow-ups

1. Land the fixes upstream and cut a Gpufit release, then run `install_python_acceleration.py`
   so ROCKETSHIP picks up the new wheel. Nothing was installed locally — the venv's `pygpufit`
   is still the shipped build, and every measurement above used a scratch copy on `PYTHONPATH`.
2. Add a regression guard for the literal class of bug: a SASS check for `DFMA`/`DMUL`/`DADD`
   in the model kernels. A reintroduced double literal is otherwise invisible — correct
   results, 14x slower.
3. Consider adding 2cxm/tissue_uptake cases to the Boost suite, and wiring up the eight test
   sources in `Gpufit/tests/` that currently only build behind `USE_BASE_MODELS`/`USE_GAUSS2D`.
4. Consider a values-only model evaluation for the line search. 8 of the 9 trials only need
   chi-square, but `cuda_calc_curve_values` always computes the full Jacobian; in the 2CXM
   kernel the `Gppos`/`Gpneg` accumulators exist solely for derivatives (the four `exp` calls
   are shared, so the saving is the FMA chains, maybe 25-30% of the loop). This is a
   Gpufit-wide interface change, so it needs its own design pass.
5. `cuda_calculate_hessians` is now the second-largest GPU cost (5.7%) and is badly
   parallelized: `n_unique_values * n_hessians_per_block` = 30 threads per block, each looping
   serially over all `n_points`.
6. The two known cpufit inefficiencies are still open and unmeasured: `std::exp` inside the
   per-point loop where the dense grid has constant `dt`, and `calc_curve_values` running the
   value and derivative passes back to back so both recompute the same two convolutions.
7. Decide the routing policy. On current numbers cpufit wins at 24 voxels, gpufit wins from
   roughly 100 voxels up (1.9x at 240, 4.6x at 2400); fixing 6 would move that crossover.
