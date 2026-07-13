# Accelerated 2CXM / 2CUM fit behavior on OSIPI (cpufit / gpufit)

*Reference for ROCKETSHIP's OSIPI verification: why the accelerated (cpufit/gpufit)
multi-compartment fits diverged, what fixed most of it, and the residual 2CXM limitation.
This started as a bug report for the [Gpufit](https://github.com/gpufit/Gpufit) project; the
core solver bug is now patched upstream, so it is kept here as internal reference behind the
`2cxm`/`2cum` test xfail reasons. The Gpufit-side notes and harness live in
`~/code/Gpufit/bug/` (`FINDINGS.md`, `experiments.py`, `probe_hard_cases.py`).*

## Status (current)

- **False-convergence solver bug — fixed upstream.** Gpufit `dev` `3db5b4d` ("Fix false
  CONVERGED on rejected step in constrained LM solver") plus `607f127` ("better global
  convergence"). ROCKETSHIP runs the rebuilt `pyCpufit 1.4.1`. This removed the gross
  failures on most cases.
- **The residual failures are caused by the caller's `Fp` initial guess — *not* float32
  precision.** The Gpufit harness shows a `DOUBLE_PRECISION` build hits the *same*
  degenerate minima, so precision is not the cause and a double-precision build is not a
  fix. The default internal `Fp` start (`0.35`) is **50–84× the true internal `Fp`**
  (~0.004–0.007), which drops the LM solver into a wrong basin (`vp` pinned to its bound,
  `Fp` inflated). Any `Fp_init ≤ 0.05` reaches the correct minimum for the hard cases.
- **2CUM (`TISSUE_UPTAKE`) — resolved** via a backend-agnostic random multi-start
  (`dce_pipeline._accel_multistart_refine`, adopted from the Gpufit harness). Passes the
  OSIPI gate on cpufit (and gpufit by the same code path; CUDA still to confirm on hardware).
- **2CXM (`TWO_COMPARTMENT_EXCHANGE`) — mostly resolved; ~6/24 cases remain.** The residual
  misses are all low-flow (`Fp = 5`) cases where `vp` is weakly identifiable at this noise
  level; the float64 python/SciPy reference scatters there too. Planned real fix:
  reparameterize the compiled model to fit `E = Ktrans/Fp` (tracked in `TODO.md`).

## Affected models

| Gpufit model | ROCKETSHIP name | Params fit | OSIPI result (cpufit, post-patch + multi-start) |
| --- | --- | --- | --- |
| `TOFTS` | tofts | Ktrans, ve | ✅ within tolerance |
| `TOFTS_EXTENDED` | ex_tofts | Ktrans, ve, vp | ✅ within tolerance |
| `PATLAK` | patlak | Ktrans(→PS), vp | ✅ within tolerance |
| `TISSUE_UPTAKE` | 2cum | Ktrans, vp, Fp | ✅ within tolerance (needs multi-start) |
| `TWO_COMPARTMENT_EXCHANGE` | 2cxm | Ktrans, ve, vp, Fp | ❌ ~6/24 low-flow cases (see below) |

## What was wrong, and what fixed it

**1. False convergence (solver, fixed upstream).** The constrained LM loop reported
`FitState == CONVERGED` on steps it had actually rejected: when the backtracking search
found no chi-square-reducing step, it restored the base parameters and set
`chi_square = prev_chi_square`, so the convergence test saw a zero change and stopped with
`vp` pinned and `Fp` inflated up to ~35× — no error flag, wrong values propagated silently.
`3db5b4d` gates the chi-square convergence test on an *accepted* step.

**2. Wrong basin from the `Fp` initial (caller-side).** After the solver fix, a few cases
still land on the degenerate minimum. From `probe_hard_cases.py` (live), 2CXM case_3:

```
default Fp_init = 0.35 → CONVERGED chi²=0.040, vp=0.001 (pinned), Fp=220   ❌
Fp_init ≤ 0.05        → CONVERGED chi²=5.9e-4, vp=0.020,          Fp=24.8  ✅ (true 25)
```

The good minimum has **67× lower chi²**, so a keep-lowest-chi² selection recovers it *iff*
one start reaches the low-`Fp` basin. This is why the fix belongs in the caller, not the
solver: the trigger is an initial guess, not solver precision.

Max absolute error over the OSIPI sweep, as a multiple of the OSIPI acceptance tolerance
(`a_tol + r_tol·|ref|`; passes at < 1.0), cpufit (float32):

| Model · param | before patch | after patch + multi-start |
| --- | ---: | ---: |
| 2cxm · ve | 1.0 | 1.67 ❌ |
| 2cxm · vp | 3.96 | 3.57 ❌ |
| 2cxm · Fp | **152** | 0.18 ✅ |
| 2cxm · PS | 7.4 | 6.89 ❌ |
| 2cum · vp | 3.2 | 0.63 ✅ |
| 2cum · Fp | **277** | 0.18 ✅ |
| 2cum · PS | 2.4 | 0.95 ✅ |

The gross `Fp` inflation (152×, 277× tol) — the false-convergence signature — is gone on
both models, and 2CUM now passes every parameter. The 2CXM residual is a `ve`/`vp`/`PS`
scatter concentrated on the low-flow (`Fp = 5`) cases: because we select the multi-start
result by lowest chi-square (no ground truth at fit time) and those cases have a near-flat,
weakly-identifiable `vp`/`ve` valley, chi-square selection can settle on a parameter set
slightly further from truth than the fixed start — which is why a couple of the max errors
tick up rather than down. The `E = Ktrans/Fp` reparameterization is the real fix.

## Why the residual 2CXM cases are hard (not precision, not premature convergence)

- **Not float32.** A `DOUBLE_PRECISION` build shows the same degenerate convergence.
- **Not premature convergence.** After the solver fix these are genuine local minima: at
  the case_3 degenerate point every *single-parameter* perturbation increases chi-square;
  the good minimum needs a *coordinated* move (`vp` up ~20×, `Fp` down ~9×, `Ktrans` down),
  so any local (gradient/KKT) test also reports "converged" there.
- **The residual misses are `Fp = 5` (low-flow) cases** where `vp` is weakly identifiable at
  this noise level — a fundamental identifiability limit shared by the float64 SciPy/python
  reference, not a solver bug.

## Why the python backend does better

Python passes every OSIPI 2CXM case with a **single** fixed start (no multi-start). What it
has that the float32 accelerated path lacks: **float64** and a **better-conditioned
reparameterization** — python fits the extraction fraction `E = Ktrans/Fp ∈ (0,1)` instead
of raw `Fp`, and its effective `Fp` start sits in-basin. Both live inside the solver/model,
not the initial values — which is why the planned fix reparameterizes the compiled model.

*Note on warm start:* seeding `Ktrans`+`vp` from a linear Patlak fit while leaving `Fp` at
the bad `0.35` makes 2CXM *worse* (Patlak is an irreversible model, biased for reversible
exchange). The lever is `Fp`; the random multi-start (which reaches low `Fp` directly) is
simpler and more general than a warm start.

## Multi-start (what ROCKETSHIP does now)

Adopted from the Gpufit harness (`bug/experiments.py`), in
`dce_pipeline._accel_multistart_refine`, applied to `2cxm`/`2cum` on every accelerated
backend (varies only the initial values, so cpufit and gpufit share it):

- Per voxel: the caller's fixed start **+ 8 log-uniform random draws** within the parameter
  bounds (bounds are strictly positive, so draws span the physiological range including
  low `Fp`).
- Each start gets a cheap **coarse fit (30 iterations)**; the lowest coarse chi-square picks
  the basin, and **one full refine (200 iterations)** runs from it.
- The refine replaces the base fit only where it converged and **strictly lowers
  chi-square**, so multi-start can never degrade a good base fit.
- Config: `prefs["accel_multistart"]` (default on), `accel_multistart_starts` (8),
  `accel_multistart_coarse_iters` (30), `accel_multistart_seed` (0, for reproducibility).

## Options for the residual 2CXM cases

1. **Reparameterize the compiled 2CXM model to fit `E = Ktrans/Fp`** (planned) — mirror the
   python backend inside the Gpufit CPU/CUDA fork; applies to both cpufit and gpufit.
2. **Keep python (float64) as the reference for 2CXM** (current) — cpufit/gpufit 2CXM stays
   `xfail` / `--osipi-slow`.

(A double-precision build is **not** an option — the harness shows it does not help.)

## Exact fit configuration

`fit_constrained` call (`pyCpufit`/`pyGpufit`):

- **Precision:** `float32` for data, `user_info` (time + AIF), initial parameters and constraints.
- **Estimator:** `EstimatorID.LSE` · **Constraints:** `ConstraintType.LOWER_UPPER` on every parameter
- **Tolerance:** `1e-6` · **Max iterations:** `200` (coarse multi-start passes use `30`)
- **2CXM fixed start** `[Ktrans, ve, vp, Fp]` = `[2e-4, 0.15, 0.02, 0.35]` (internal units; Fp reported = internal · 6000 mL/100mL/min)
- **2CXM bounds:** Ktrans `[1e-7, 2.0]`, ve `[0.05, 1.0]`, vp `[1e-3, 1.0]`, Fp `[1e-3, 20.0]`

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

# Gpufit-side diagnosis of the Fp-init basin (in ~/code/Gpufit)
.venv/bin/python bug/probe_hard_cases.py
```

The 2CXM DRO used is `tests/data/osipi/dce_models/2cxm_sd_0.001_delay_0.csv`
(columns `vp, ve, fp, ps` = ground truth; `C_t` = tissue curve; `cp_aif` = AIF; `t` = time).

## How ROCKETSHIP handles it today

- **2CUM:** backend-agnostic random multi-start → passes the OSIPI gate on cpufit/gpufit.
- **2CXM:** cpufit/gpufit full-sweep test is `xfail` + `--osipi-slow`; the float64 python
  backend is the default/reference and passes. The planned `E = Ktrans/Fp` reparameterization
  of the compiled model is the real fix (see `TODO.md`).
- Per-backend numbers and the accuracy-by-backend table live in
  `docs/project-management/projects/osipi-verification/osipi_summary.md`.
