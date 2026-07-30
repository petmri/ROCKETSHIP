# Python DCE Pipeline Performance (non-fit processing)

## Scope

Everything in the Python DCE pipeline *except* the kinetic-model fitting itself, which is
already accelerated through CPUfit/GPUfit. This document records the profile that motivated
the first round of vectorization and what is left.

## Method

`cProfile` plus accumulating wall-clock timers around `run_dce_pipeline`, backend `auto`
resolving to `cpufit_cpu` (Apple M5, 10 cores, 26 GB, no CUDA).

Every committed DCE fixture is single-slice, so at 20k ROI voxels the fixed costs dominate
and the per-voxel costs hide. The measurements below therefore also use a throwaway fixture
that tiles `sub-01original` to 8 slices (160,408 ROI voxels) to stand in for a full-brain
study. That fixture is deliberately **not** committed -- it carries no new information, only
more voxels, and `tests/data/` is already a distribution problem (see
`../large-data-distribution/large_data_distribution.md`).

## Baseline profile, 160k voxels, tofts+patlak, default output settings

| call site | secs | % run |
|---|---|---|
| CPUfit `fit_constrained` (out of scope) | 7.67 | 46% |
| `assemble_*_candidates` -- per-voxel linear-Patlak seeding | 6.70 | 40% |
| Stage A total | 1.72 | 10% |
| — QC figures | 0.55 | |
| — `_clean_ab` + `_clean_r1` + baseline-rescale loops | ~0.86 | |
| — NIfTI load | 0.19 | |
| `_assemble_stage_d_output` | 0.22 | 1% |
| `_write_param_maps` | 0.14 | 1% |
| Stage B total (nearly all QC figure) | 0.08 | 0.5% |
| **In-scope Python (total − CPUfit)** | **8.90** | **54%** |

The Python around the fitter cost more than the fitter. Every hotspot had the same shape: a
loop over voxels doing scalar or tiny-array work, when all voxels share one `cp`/`timer` and
the data is already an `(n_time, n_voxels)` matrix.

## Round 1 (landed)

Vectorized, each verified bit-identical to the code it replaced:

| hotspot | before | after | speedup |
|---|---|---|---|
| `model_patlak_linear` per-voxel seeding | 6511 ms | 83 ms | 78x |
| `_clean_r1` | 372 ms | 49 ms | 7.7x |
| Stage-A baseline rescale loops | 358 ms | 38 ms | 9.5x |
| `_clean_ab` | 133 ms | 48 ms | 2.8x |
| `_assemble_stage_d_output` | 95 ms | 4.4 ms | 21.5x |

End-to-end at 160k voxels, tofts+patlak: **16.35 s -> 8.91 s**, with in-scope Python
**8.90 s -> 1.30 s**. CPUfit is now 86% of the run.

Equivalence evidence: every Stage-A/B/D output array bit-identical across 10 subject x model
combinations on the accelerated backend, all 9 models on the pure-Python backend, and 24,190
individual voxels of scalar-vs-batch seeding comparison. Locked in by
`tests/python/test_dce_models.py::test_patlak_linear_batch_is_bit_identical_to_scalar`.

### Bit-exactness gotchas found along the way

Both are load-bearing and easy to "simplify" back into a difference:

- `.sum(axis=0)` blocks its reduction once an array is wide enough, which reorders the
  summation away from the scalar loop's left-to-right order. `_sequential_sum_over_time` in
  `python/dce_models.py` accumulates row by row instead, at the same cost.
- `np.power(x, 2.0)` routes to libm `pow`, matching Python's `x ** 2` on a float. `x ** 2` on
  a numpy *array* becomes `x*x`, which differs in the last ULP.

## Round 2 (landed): batched curve evaluation

The post-fit path re-evaluates every fitted voxel's forward curve in Python, whatever backend
did the fitting -- gpufit/cpufit return parameters, not curves. So this cost is untouched by
acceleration, and grows as a share of the run the faster the backend gets.

`_exp_weighted_cumulative_trapz_batch` promotes `lam` to a vector and keeps the time loop:
n_time vector steps instead of n_voxels x n_time scalar ones, with the recurrence order
unchanged. On top of it sit
`model_{tofts,extended_tofts,patlak,tissue_uptake,2cxm,fxr}_cfit_batch`, and
`_compute_fit_residuals` collapses to one batched call. `patlak` needs no recurrence at all,
since its integral is voxel-independent.

Measured at 160,408 voxels, tofts, `write_postfit_arrays=1 write_qof_maps=1`:

| | before | after |
|---|---|---|
| `_compute_fit_residuals` | 4.697 s | **0.076 s** (62x) |
| End-to-end, post-fit flags on | 10.995 s | **6.418 s** (1.71x) |
| End-to-end, post-fit flags off (default) | 2.630 s | 2.649 s (unchanged) |

Equivalence: bit-identical Stage-A/B/D arrays *and* written NIfTIs across all six models
(`sub-01original`; `sub-11tiny` for `fxr`), plus scalar-vs-batch agreement over an exhaustive
grid of pathological parameters. Locked in by
`test_cfit_batch_matches_scalar_on_pathological_grid` and `..._on_random_params`.

It does **not** speed up the pure-Python fit path: `_run_scipy_per_voxel` calls
`least_squares` one voxel at a time with scalar parameters, so batching across voxels does not
apply there. That would need a batched Levenberg-Marquardt, a much larger project.

### Where the scalar and batch paths can silently diverge

The batch functions must reject exactly the voxels the scalar functions raise on, because
Python floats raise where numpy returns inf/NaN. Four cases, each handled as a mask:

- float division by exact zero -> `ZeroDivisionError` (numpy: inf)
- `math.exp` past `log(DBL_MAX)` -> `OverflowError` (numpy: inf); hence the `overflowed`
  mask the kernel returns
- squaring a *finite* float past `DBL_MAX` -> `OverflowError` (numpy: inf). `inf ** 2`
  returns inf in both, so the mask is conditioned on a finite base
- `math.sqrt` of a negative -> `ValueError` (numpy: NaN)

Getting one of these wrong turns a rejected voxel into a plausible-looking number, which is
why the tests assert the NaN pattern exactly and the values only loosely. Last-ULP differences
between libm and numpy are explicitly not gated -- they carry no scientific meaning here.

## Remaining, not yet done

### Make the Stage-A/B QC figures opt-out
0.55 s (Stage A) + 0.07 s (Stage B) unconditionally, and 68% of in-scope time on the small
single-slice fixtures. Wants a `write_qc_figures` override alongside the existing
`write_param_maps` / `write_qof_maps` flags. `np.median(r1_toi, axis=1)` over every voxel is
itself a meaningful slice of that. Deliberately deferred.

### Stage-A peak memory
Stage A returns its arrays to Stage B by reference, not by copy (`np.shares_memory` is true
for `Ct`/`Sttum`/`R1tTOI`/`deltaR1TOI`), so nothing is duplicated across the stage boundary.
Retained is 333 MB at 160k voxels, in four `(n_time, n_voxels)` float64 arrays; the AIF-side
arrays are negligible at 160 AIF voxels.

Peak RSS is the figure that matters, and it scales cleanly: 0.34 GB at 20k voxels, 1.37 GB at
160k, i.e. ~7.3 kB/voxel above a 78 MB import baseline, so a 500k-voxel study lands near 4 GB.
It scales with `n_time` too -- a 725-frame study such as `sub-09phantom` is the sharper edge,
not slice count.

This is a ceiling rather than a slowdown: you fit in RAM or you swap. Fixing it properly means
changing the stage-boundary `arrays` contract, which also feeds checkpoints, the parity harness
and GUI event payloads -- a design change, not an optimization.

### Multiprocessing over voxel chunks
Worth more than it first appears. CPUfit is **single-threaded** -- no OpenMP in `Cpufit/`, and
measured at 1.00 of 10 cores busy (wall 7.35 s vs cpu 7.32 s) -- so chunking voxels across
processes would parallelize the now-dominant fit cost, not just the 1.3 s of remaining Python.

The caveat is that this is a CPU-fallback fix, not a Python-overhead fix: on a CUDA box GPUfit
already fits all voxels in parallel and process-chunking buys almost nothing. Adding OpenMP
inside Cpufit is the better version of the same win -- no Python plumbing, no arrays to ship to
workers, and the MATLAB binding benefits too. Tracked in
`../feature-request/new_features.md`.
