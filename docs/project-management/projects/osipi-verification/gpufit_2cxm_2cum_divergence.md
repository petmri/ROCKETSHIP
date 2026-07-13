# Accelerated 2CXM / 2CUM fit behavior on OSIPI (cpufit / gpufit)

*Reference for ROCKETSHIP's OSIPI verification: why the accelerated (cpufit/gpufit)
multi-compartment fits diverged, what fixed most of it, and the residual 2CXM limitation.
This started as a bug report for the [Gpufit](https://github.com/gpufit/Gpufit) project; the
core solver bug is now patched upstream, so it is kept here as internal reference behind the
`2cxm`/`2cum` test xfail reasons.*

## Status (current)

- **False-convergence solver bug — fixed upstream.** Gpufit `dev` `3db5b4d` ("Fix false
  CONVERGED on rejected step in constrained LM solver") plus `607f127` ("better global
  convergence" for patlak/tissue_uptake/2cxm). ROCKETSHIP runs the rebuilt `pyCpufit 1.4.1`.
  This alone removed most of the failures.
- **2CUM (`TISSUE_UPTAKE`) — resolved in ROCKETSHIP** via a backend-agnostic multi-start
  (`dce_pipeline._accel_multistart_refine`): re-fit only the voxels that pin vp/Fp to a bound,
  from a few perturbed starts, keep the lowest chi-square. Passes the OSIPI gate on cpufit
  (and gpufit by the same code path; CUDA still to be confirmed on hardware).
- **2CXM (`TWO_COMPARTMENT_EXCHANGE`) — still precision/parameterization-limited.** ~6 of 24
  OSIPI cases remain outside tolerance on the float32 accelerated path even with multi-start.
  The float64 python backend passes and is the reference. Tracked in `TODO.md`.

## Affected models

| Gpufit model | ROCKETSHIP name | Params fit | OSIPI result (cpufit, post-patch) |
| --- | --- | --- | --- |
| `TOFTS` | tofts | Ktrans, ve | ✅ within tolerance |
| `TOFTS_EXTENDED` | ex_tofts | Ktrans, ve, vp | ✅ within tolerance |
| `PATLAK` | patlak | Ktrans(→PS), vp | ✅ within tolerance |
| `TISSUE_UPTAKE` | 2cum | Ktrans, vp, Fp | ✅ within tolerance (needs multi-start) |
| `TWO_COMPARTMENT_EXCHANGE` | 2cxm | Ktrans, ve, vp, Fp | ❌ ~6/24 cases (see below) |

## What was wrong, and what fixed it

Before the patch, the constrained LM solver reported `FitState == CONVERGED` on steps it had
actually rejected, so 2CXM and 2CUM fits halted at a degenerate point — plasma volume `vp`
pinned to its lower bound and plasma flow `Fp` inflated up to ~35× (e.g. true Fp 40 → 1411
mL/100mL/min) — with no error flag, so the wrong values propagated silently. `3db5b4d` fixes
the false-success report; `_accel_multistart_refine` then rescues the remaining vp-pinned
voxels for 2CUM.

Max absolute error over the OSIPI sweep, as a multiple of the OSIPI acceptance tolerance
(`a_tol + r_tol·|ref|`; passes at < 1.0), cpufit (float32):

| Model · param | before patch | after patch + multi-start |
| --- | ---: | ---: |
| 2cxm · ve | 1.0 | 1.01 ❌ |
| 2cxm · vp | 3.96 | 2.47 ❌ |
| 2cxm · Fp | **152** | 0.18 ✅ |
| 2cxm · PS | 7.4 | 4.34 ❌ |
| 2cum · vp | 3.2 | 0.63 ✅ |
| 2cum · Fp | **277** | 0.18 ✅ |
| 2cum · PS | 2.4 | 0.95 ✅ |

The gross `Fp` inflation (152×, 277× tol) — the false-convergence signature — is gone. What
remains for 2CXM is a subtler `vp`/`ve`/`PS` error: `vp` under-shoots rather than pinning to
the bound (e.g. true 0.10 → 0.038–0.064), the classic `vp`↔`Fp` near-degeneracy of the 2CXM
at high SNR.

## Why 2CXM still fails — and why it is *not* an initialization problem

- **Python passes with a single fixed start.** `model_2cxm_fit` (python) uses one fixed
  start — no multi-start, no warm-start — and passes every OSIPI case. So the accelerated
  failure is not about initial values.
- **What python has that the accelerated path lacks:** (1) **float64** (accelerated is
  float32), and (2) a **better-conditioned reparameterization** — python fits the extraction
  fraction `E = Ktrans/Fp ∈ (0,1)` instead of raw `Fp`, which tames the `vp`↔`Fp` degeneracy.
  Both live inside the solver/model, not the initial values.
- **Initialization strategies do not rescue it.** Multi-start: 2cxm 7→6 bad cases (marginal).
  Warm-start from a linear Patlak fit makes 2cxm *worse* (7→17) — Patlak is an *irreversible*
  model, so its intercept/slope are biased seeds for reversible exchange and push `vp` toward
  the bound. (The same warm-start is unnecessary for 2CUM, which multi-start alone fixes.)

## Options for fixing accelerated 2CXM (open — see TODO.md)

1. **Double-precision cpufit/gpufit build** (`DOUBLE_PRECISION=ON`) — targets the float32
   ill-conditioning directly. Needs a matching double-precision `pyCpufit`/`pyGpufit` wheel
   and float64 plumbing through ROCKETSHIP's accelerated path (currently hardcoded float32).
2. **Reparameterize the compiled 2CXM model** to fit `E = Ktrans/Fp` like python — a C++
   change in the Gpufit fork, applied to both cpufit and gpufit.
3. **Keep python (float64) as the reference for 2CXM** (current) — cpufit/gpufit 2CXM stays
   xfail / `--osipi-slow`; python is used for production 2CXM.

## Exact fit configuration

`fit_constrained` call (`pyCpufit`/`pyGpufit`):

- **Precision:** `float32` for data, `user_info` (time + AIF), initial parameters and constraints.
- **Estimator:** `EstimatorID.LSE` · **Constraints:** `ConstraintType.LOWER_UPPER` on every parameter
- **Tolerance:** `1e-6` · **Max iterations:** `200`
- **2CXM initial values** `[Ktrans, ve, vp, Fp]` = `[2e-4, 0.15, 0.02, 0.35]` (internal units; Fp reported = internal · 6000 mL/100mL/min)
- **2CXM bounds:** Ktrans `[1e-7, 2.0]`, ve `[0.05, 1.0]`, vp `[1e-3, 1.0]`, Fp `[1e-3, 20.0]`
- **Multi-start** (`_accel_multistart_refine`): suspect voxels (fit failed, or vp/Fp pinned to a bound)
  are re-fit from `(Fp_scale, vp_scale)` perturbations of the fixed start; the lowest chi-square
  candidate that converged is kept, so it can never degrade a good base fit.

## Reproduction

Data is public and citable (OSIPI DRO by M. Thrippleton,
[mjt320/DCE-functions](https://github.com/mjt320/DCE-functions); Manning et al., MRM 2021,
[doi:10.1002/mrm.28833](https://doi.org/10.1002/mrm.28833); testing framework van Houdt et al.,
MRM 2023, [doi:10.1002/mrm.29826](https://doi.org/10.1002/mrm.29826)).

```bash
# per-backend accuracy report (python vs cpufit vs gpufit) + figures
.venv/bin/python tests/data/osipi/reference/generate_osipi_summary.py

# the 2CXM full-sweep gate (xfail), with a per-case out-of-tolerance breakdown
.venv/bin/python -m pytest tests/python/test_osipi_pycpufit.py::test_osipi_pycpufit_2cxm_sweep \
    --osipi-slow --runxfail -rA
```

The 2CXM DRO used is `tests/data/osipi/dce_models/2cxm_sd_0.001_delay_0.csv`
(columns `vp, ve, fp, ps` = ground truth; `C_t` = tissue curve; `cp_aif` = AIF; `t` = time).

## How ROCKETSHIP handles it today

- **2CUM:** backend-agnostic multi-start → passes the OSIPI gate on cpufit/gpufit.
- **2CXM:** cpufit/gpufit full-sweep test is `xfail` + `--osipi-slow`; the float64 python
  backend is the default/reference and passes.
- Per-backend numbers and the accuracy-by-backend table live in
  `docs/project-management/projects/osipi-verification/osipi_summary.md`.
