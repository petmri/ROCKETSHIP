# Accelerated 2CXM / 2CUM fit behavior on OSIPI (cpufit / gpufit)

*Reference for ROCKETSHIP's OSIPI verification of the accelerated (cpufit/gpufit)
multi-compartment fits: why they diverged, why the backend was slow, and the unified fix
(reparameterize to `E = Ktrans/Fp` with an O(N) convolution + analytic Jacobians) — now
**implemented and verified on cpufit**. Gpufit-side notes/harness live in `~/code/Gpufit/bug/`
(`FINDINGS.md`, `experiments.py`, `probe_hard_cases.py`).*

## Doc TODOs (this project)
- [ ] `osipi_summary.md` is missing the per-case fit values for cpufit and gpufit, and the
      summary accuracy values for gpufit.

## Status (RESOLVED on cpufit; gpufit CUDA pending hardware)

The unified fix is implemented. `2cxm`/`2cum` now fit `E = Ktrans/Fp` with an O(N) exponential
recurrence and analytic Jacobians; the `Ktrans = Fp` pole/sentinel is gone (it is the bound
`E → 1`). **All 5 accelerated Stage-D models now pass the full OSIPI cpufit sweep, including
every 2CXM case** (was ~6/24 low-flow misses), and the backend is dramatically faster.

- **cpufit — done & verified.** Kernel math lives in `~/code/Gpufit/Cpufit/lm_fit_cpp.cpp`
  (the CPU backend has its own C++ models — it does *not* compile the `.cuh` files). Wheel
  rebuilt into `.venv`. `test_osipi_pycpufit.py` passes all five sweeps; `2cxm` is no longer
  xfail. Analytic Jacobian verified to L2 ~1e-8 (`test_reparam_jacobian.py`).
- **gpufit — math done, hardware pending.** `Gpufit/models/two-compartment_exchange.cuh` and
  `tissue_uptake.cuh` carry the same reparam + analytic Jacobian; each column checked against
  central differences of its own forward to L2 ~1e-9 on a host-compiled shim. Not built/run on
  CUDA hardware here, so `test_osipi_pygpufit.py::…_2cxm_sweep` stays `xfail(strict=False)`.
- **Two causes fixed, both required.** (1) The **numerical Jacobian was corrupted near the
  `Ktrans = Fp` singularity** — the reparam + analytic Jacobian removes it (correct gradient).
  (2) The **`Fp` lower bound (`1e-3`/s ≈ 6 mL/100mL/min) excluded the true low-flow value**
  (`Fp = 5` mL/100mL/min ≈ `8.3e-4`/s); it is lowered to `1e-4`/s so low-flow is representable.
  With both, the previously-missed `Fp = 5` cases recover the true params (e.g. case_1:
  ve 0.100/vp 0.021/fp 4.96/ps 0.0500 vs truth 0.1/0.02/5/0.05). *This corrects the earlier
  "weak-identifiability floor" reading below: the float64 python reference passes all 24, so
  the cases were fittable — the accelerated backend just could not reach that Fp.*
- **Multi-start still required** (`dce_pipeline._accel_multistart_refine`, unchanged): without
  it 3/24 `2cxm` + 6/24 `2cum` cases still miss the flow basin. Kept.
- **False-convergence solver bug — fixed upstream** (Gpufit `dev` `3db5b4d` + `607f127`);
  ROCKETSHIP runs the rebuilt `pyCpufit 1.4.1`.

## Affected models

| Gpufit model | ROCKETSHIP | Params (accelerated) | OSIPI cpufit (now) |
| --- | --- | --- | --- |
| `TOFTS` | tofts | Ktrans, ve | ✅ analytic deriv; **O(N) conv** |
| `TOFTS_EXTENDED` | ex_tofts | Ktrans, ve, vp | ✅ analytic deriv; **O(N) conv** |
| `PATLAK` | patlak | Ktrans(→PS), vp | ✅ closed-form linear solve |
| `TISSUE_UPTAKE` | 2cum | **E**, vp, Fp | ✅ **analytic Jac; O(N)**; multi-start |
| `TWO_COMPARTMENT_EXCHANGE` | 2cxm | **E**, ve, vp, Fp | ✅ **analytic Jac; O(N)**; multi-start |

## History (settled): false convergence + wrong `Fp` basin

1. **False convergence (solver, fixed upstream).** The constrained LM loop reported
   `CONVERGED` on *rejected* steps: when backtracking found no chi-square-reducing step it
   restored base parameters and set `chi_square = prev_chi_square`, so the convergence test
   saw zero change and stopped with `vp` pinned and `Fp` inflated up to ~35×, no error flag.
   `3db5b4d` gates the chi-square test on an *accepted* step.
2. **Wrong basin from the `Fp` initial (caller-side).** A few cases still land on a degenerate
   minimum. `probe_hard_cases.py`, 2CXM case_3: default `Fp_init = 0.35` → chi²=0.040, `vp`
   pinned, `Fp=220` ❌; `Fp_init ≤ 0.05` → chi²=5.9e-4, `vp=0.020`, `Fp=24.8` ✅ (true 25). The
   good minimum has 67× lower chi², so keep-lowest-chi² selection recovers it *iff* one start
   reaches the low-`Fp` basin.

Max abs error over the OSIPI sweep as a multiple of the OSIPI tolerance (`a_tol + r_tol·|ref|`;
passes at < 1.0), cpufit float32:

| Model · param | before patch | after patch + multi-start |
| --- | ---: | ---: |
| 2cxm · Fp | **152** | 0.18 ✅ |
| 2cxm · ve / vp / PS | 1.0 / 3.96 / 7.4 | 1.67 / 3.57 / 6.89 ❌ |
| 2cum · vp / Fp / PS | 3.2 / **277** / 2.4 | 0.63 / 0.18 / 0.95 ✅ |

The gross `Fp` inflation (the false-convergence signature) is gone; 2CUM passes every
parameter. The 2CXM residual is a `ve`/`vp`/`PS` scatter on the low-flow (`Fp = 5`) cases.

**Multi-start (what runs today).** From the Gpufit harness (`bug/experiments.py`), in
`dce_pipeline._accel_multistart_refine`, on `2cxm`/`2cum` for every accelerated backend: the
caller's fixed start **+ 8 log-uniform random draws**, each a cheap **coarse fit (30 iters)**
to pick the basin, then **one full refine (200 iters)** from the best; the refine replaces the
base fit only where it converged and strictly lowers chi-square (never degrades a good fit).
Config: `accel_multistart` (on), `accel_multistart_starts` (8), `accel_multistart_coarse_iters`
(30), `accel_multistart_seed` (0). *Warm-starting `Ktrans`+`vp` from linear Patlak while leaving
`Fp` high makes 2CXM worse — the lever is `Fp`, which the random multi-start reaches directly.*

## Root-cause refinement: the numerical Jacobian is broken near `Ktrans = Fp`

Both compiled multi-compartment models convert `(Ktrans, Fp)` to an internal permeability
`PS = Fp·Ktrans/(Fp − Ktrans)`, which has a **pole at `Ktrans = Fp`** (extraction fraction
`E = Ktrans/Fp → 1`). `two-compartment_exchange.cuh` guards it with a discontinuous jump —
`if (Ktrans >= Fp) PS = 1e9;` — and `tissue_uptake.cuh` has **no guard at all** (straight
through the pole to Inf/NaN). Their derivatives are a **5-point finite difference with a fixed
absolute step `h = 1e-4`**. When that stencil lands near the pole it differences *across* the
discontinuity. Measured on the real `fp=5` DRO (internal `Fp ≈ 8.3e-4`, so `h` is **12 % of Fp**,
`±2h` reaches **24 %**), analytic vs the kernel's numeric `∂C/∂Ktrans`:

| E = Ktrans/Fp | numeric (f32, h=1e-4) | true | error |
| ---: | ---: | ---: | --- |
| 0.30 / 0.60 | 260.1 / 162.7 | 260.1 / 162.7 | ✅ ~0 % |
| 0.80 | **−22.3** | +114.6 | wrong sign |
| 0.90 / 0.95 | 1067 / 1040 | 95.4 / 86.8 | ~11–12× too big |
| ≥ 1.00 | ~1000 | ~0 | meaningless (sentinel branch) |

So above `E ≈ 0.75` the solver is handed a gradient of the wrong sign or ~10× magnitude, and
the smaller the true flow the wider that corrupted band (fixed `h` vs shrinking `Fp`) — which is
why the failures concentrate on low-flow cases. (Secondary hazard: a `±2h` perturbation of `ve`
or `vp` can drive the discriminant `(1/Tp+1/Te)² − 4/(Te·Tb)` negative → `sqrt` → `NaN` column.)
This is a *discontinuity*, not round-off, so a `DOUBLE_PRECISION` build does not fix it (consistent
with the harness) — but **analytic derivatives in the `E` parameterization do**, because the pole
becomes the bound `E → 1` and is removed.

> **Correction (post-implementation):** the low-flow misses were *not* a weak-identifiability
> floor. Once the gradient was fixed (analytic Jacobian) **and** the `Fp` floor was lowered so
> `Fp = 5` is representable, all 24 cases recover the true params, matching the float64 python
> reference (which passes all 24). The two fixable causes — corrupted gradient near the pole, and
> an `Fp` lower bound that excluded the true low flow — fully explain the residual.

## Performance: the O(N²) cliff (fixed)

Per-model OSIPI sweep, cpufit (one `fit_constrained` per DRO case, `n_fits=1`), **before vs
after** the O(N) recurrence + analytic Jacobian:

| model | npts | calls/row | ms/row **before** | ms/row **now** |
| --- | ---: | ---: | ---: | ---: |
| patlak (linear solve) | 600 | 1 | 1 | 1.1 |
| ex_tofts | 331 | 1 | 5 | **0.1** |
| tofts | **1321** | 1 | **107** | **0.2** |
| 2cum | 600 | **11** | 3879 | **2.6** |
| 2cxm | 600 | **11** | 12325 | **4.1** |

The two "before" costs, both removed:
- **Convolution was O(N²).** Each model value recomputed `for i in 1..point_index` at every
  timepoint; the exponential recurrence computes the identical integral in **one O(N) pass**.
  That was the whole "slower than python" gap, and it was N-dependent — the tofts-vs-ex_tofts
  mystery was purely timepoint count (1321 vs 331 → ~16× under O(N²)), not fit difficulty. (The
  CPU backend now does one O(N) pass; the CUDA kernels keep the per-point convention — one
  thread per timepoint — but drop the numeric-Jac multiplier below.)
- **2cxm/2cum stacked a 5-point numeric Jacobian** (16 / 12 extra O(N²) evals per iteration) **×
  the 11× multi-start** → minutes. The analytic Jacobian removes the 16×/12× multiplier. (`n_fits=1`
  here is a sweep artifact: production batches all voxels sharing one AIF; the per-fit costs stand.)
- **patlak** is unchanged — pyCpufit routes it to `cpufit_patlak_bounded_linear`, not LM.

The O(N) recurrence + analytic Jacobian below remove both the O(N²) cost and the numeric-Jac
multiplier, and matter in production too (the 11× and numeric Jac otherwise hit every voxel).

## The fix (derived + verified): `E = Ktrans/Fp`, O(N) recurrence, analytic Jacobian

**Reparameterize.** Fit `E = Ktrans/Fp ∈ (0,1)` instead of raw `Ktrans` (recover `Ktrans =
E·Fp`). Then `PS = Fp·E/(1−E)` is smooth on `(0,1)`; the `Ktrans = Fp` pole becomes the bound
`E → 1⁻`, so the `if (Ktrans>=Fp) PS=1e9` sentinel (2cxm) and the unguarded pole (2cum) are
deleted.

**O(N) convolution primitive.** For rate `κ`, per step `k` (`Δ = tₖ − tₖ₋₁`, `decay = e^(−κΔ)`),
with `G₀ = G'₀ = 0`:

```
Gₖ  = decay·Gₖ₋₁ + ½·Δ·(Cpₖ₋₁·decay + Cpₖ)          # ∫₀^tₖ Cp·e^(−κ(tₖ−τ)) dτ  (trapezoid)
G'ₖ = decay·(G'ₖ₋₁ − Δ·Gₖ₋₁ − ½·Δ²·Cpₖ₋₁)           # ∂Gₖ/∂κ   (exact deriv of the discrete Gₖ)
Uₖ  = Uₖ₋₁ + ½·Δ·(Cpₖ₋₁ + Cpₖ)                       # ∫₀^tₖ Cp dτ  (2cum only)
```

Each model value and its full Jacobian is then O(N) (a few of these passes) instead of
O(N²)·(1+params). The messy chain-rule scalars below are computed **once per fit** (O(1)).

**2CUM** — params `(E, vp, Fp)`, single rate `rp = Fp/(vp·(1−E))`; `G,G'` from `rp`:

```
C     = E·Fp·U + Fp·(1−E)·G
∂C/∂E  = Fp·U − Fp·G + Fp·rp·G'
∂C/∂vp = −Fp·(1−E)·rp·G' / vp
∂C/∂Fp = E·U + (1−E)·G + (1−E)·rp·G'
```

**2CXM** — params `(E, ve, vp, Fp)`. Internal scalars once per fit:

```
PS = Fp·E/(1−E);  rp = (PS+Fp)/vp;  re = PS/ve;  rb = Fp/vp
a = rp+re;  c = re·rb;  Δ = √(a²−4c);  Kpos = ½(a+Δ);  Kneg = ½(a−Δ);  Eneg = (Kpos−rb)/Δ
C  = Fp·[ (1−Eneg)·Gpos + Eneg·Gneg ]              # Gpos,G'pos from Kpos ; Gneg,G'neg from Kneg
```

Scalar partials wrt `θ ∈ {E, ve, vp, Fp}` (`∂PS`: `∂E = Fp/(1−E)²`, `∂Fp = E/(1−E)`, else 0):

```
∂rp = ∂PS/vp + 1(θ=Fp)/vp − 1(θ=vp)·(PS+Fp)/vp²
∂re = ∂PS/ve − 1(θ=ve)·PS/ve²
∂rb = 1(θ=Fp)/vp − 1(θ=vp)·Fp/vp²
∂a  = ∂rp+∂re;   ∂c = ∂re·rb + re·∂rb;   ∂Δ = (a·∂a − 2·∂c)/Δ
∂Kpos = ½(∂a+∂Δ);   ∂Kneg = ½(∂a−∂Δ);   ∂Eneg = [(∂Kpos−∂rb)·Δ − (Kpos−rb)·∂Δ] / Δ²
```

Assemble each Jacobian column:

```
∂C/∂θ = Fp·[ ∂Eneg·(Gneg−Gpos) + (1−Eneg)·G'pos·∂Kpos + Eneg·G'neg·∂Kneg ]
        + 1(θ=Fp)·[ (1−Eneg)·Gpos + Eneg·Gneg ]
```

**Verification.** `verify_analytic_jac.py` (in this folder; implements the forward model, the
recurrences, and the formulas above) checks every column against a float64 central difference of the
same discrete forward model on the real low-flow DRO at `E = 0.3, 0.6, 0.85`: **L2 relative error
1e-8–1e-10 on all columns** — including `E = 0.85`, where the current fixed-step numeric Jacobian is
already wrong.

## Implementation plan (status)

**Key correction to the original plan:** the CPU and CUDA models are *not* shared. The cpufit
backend has its own C++ model implementations in `~/code/Gpufit/Cpufit/lm_fit_cpp.cpp`; the
`.cuh` files under `~/code/Gpufit/Gpufit/models/` are the CUDA/gpufit path only. Both were
edited. Caller in `python/dce_pipeline.py`; wheel rebuilt to `.venv` `pyCpufit`.

1. ✅ **O(N) convolution (CPU).** `lm_fit_cpp.cpp` `calc_values_*`/`calc_derivatives_*` rewritten
   to single-pass `Gₖ`/`G'ₖ`/`Uₖ` recurrences for all four conv models. (CUDA keeps the per-point
   convention — one thread per timepoint — so its convolution stays per-point; the numeric-Jac
   multiplier is what dominated there and it is removed.)
2. ✅ **Reparameterize `2cxm`/`2cum` to `E`** (CPU + CUDA). `PS = Fp·E/(1−E)`; the `if(p0>=p3)`
   sentinel and the unguarded pole are deleted.
3. ✅ **Analytic Jacobians** for `2cxm`/`2cum` (CPU + CUDA); 5-point numeric blocks deleted.
   tofts/ex_tofts CPU derivatives also moved to the recurrence.
4. ✅ **Caller (`dce_pipeline.py`).** `2cxm`/`2cum` init/bounds mapped to `E ∈ (0,1)` via
   `_extraction_fraction_init_bounds` (mirrors `dce_models._fit_2cxm_osipi_canonical`); fitted
   `E → Ktrans = E·Fp` on output. **Also lowered the `Fp` floor `1e-3 → 1e-4`/s** so low-flow
   (`Fp = 5`) is representable — the missing piece for the low-flow cases.
5. ✅ **In-repo Jacobian guard:** `tests/python/test_reparam_jacobian.py` (Python mirror of the
   kernel math vs central differences, L2 < 1e-6). Standalone derivation kept in
   `verify_analytic_jac.py`.
6. ✅ **Rebuild + swap** the `pyCpufit` wheel into `.venv` (dylib md5 `4a56ad4f → 0044e3df`).
7. ✅ **Re-verify OSIPI cpufit:** all 5 sweeps pass (2CUM + **all 24 2CXM**); perf probe confirms
   the cliff is gone (see Performance).
8. ✅ **Re-evaluated multi-start:** still needed (without it 3/24 `2cxm` + 6/24 `2cum` miss). Kept
   at 8 random starts.
9. ⬜ **CUDA hardware:** build pyGpufit + run the gpufit sweeps on a CUDA box. Kernel math checked
   host-side (analytic-vs-central L2 ~1e-9) but not built/run with nvcc here. `2cxm` gpufit test
   stays `xfail(strict=False)` until then.
10. ◑ **Docs/tests:** xfail updated (cpufit 2CXM un-xfailed; gpufit reason → hardware-pending),
    this file + `COMPLETED.md` + `TODO.md` updated. `osipi_summary.md` per-case refresh still open
    (Doc TODO above).

## Follow-ups / watch items
- **`Fp` floor change is a production default** (`2cxm`/`tissue_uptake` `lower_limit_fp` 1e-3→1e-4
  in `_stage_d_fit_prefs`). Physically sound (0.6 mL/100mL/min) and only relaxes the feasible
  region, but it affects all backends — review before merge.
- **Done:** the `--osipi-slow` gate was removed entirely — the now-fast `2cxm`/`2cum` sweeps and
  the python reliability fits run in the default OSIPI suite (`pytest -m osipi`, ≈6 s total).

## Exact fit configuration (current, post-reparam)

`fit_constrained` (`pyCpufit`/`pyGpufit`): **float32** data/user_info/params/constraints;
`EstimatorID.LSE`; `ConstraintType.LOWER_UPPER` on every param; tolerance `1e-6`; max iters `200`
(coarse multi-start passes `30`). **2CXM params are now `[E, ve, vp, Fp]`** (`E = Ktrans/Fp`;
recover `Ktrans = E·Fp`). `E` init/bounds are derived from the `Ktrans`/`Fp` prefs by
`_extraction_fraction_init_bounds` (`E_init = Ktrans_init/Fp_init`; `E_lo = Ktrans_lo/Fp_hi`,
`E_hi = Ktrans_hi/Fp_lo`, clipped to `(0,1)`). **2CXM bounds:** ve `[0.05, 1.0]`, vp `[1e-3, 1.0]`,
**Fp `[1e-4, 20.0]`** (floor lowered from `1e-3` so `Fp = 5` mL/100mL/min ≈ `8.3e-4`/s is reachable;
reported `Fp` = internal·6000 mL/100mL/min). 2CUM params are `[E, vp, Fp]` with the same mapping.

## Reproduction

Data is public/citable (OSIPI DRO by M. Thrippleton,
[mjt320/DCE-functions](https://github.com/mjt320/DCE-functions); Manning et al., MRM 2021,
[doi:10.1002/mrm.28833](https://doi.org/10.1002/mrm.28833); framework van Houdt et al., MRM 2023,
[doi:10.1002/mrm.29826](https://doi.org/10.1002/mrm.29826)).

```bash
# per-backend accuracy report (python vs cpufit vs gpufit) + figures
.venv/bin/python tests/data/osipi/reference/generate_osipi_summary.py

# 2CXM full-sweep gate (now passing), per-case out-of-tolerance breakdown on failure
.venv/bin/python -m pytest tests/python/test_osipi_pycpufit.py::test_osipi_pycpufit_2cxm_sweep -rA

# Gpufit-side diagnosis of the Fp-init basin (in ~/code/Gpufit)
.venv/bin/python bug/probe_hard_cases.py
```

2CXM DRO: `tests/data/osipi/dce_models/2cxm_sd_0.001_delay_0.csv` (`vp, ve, fp, ps` = ground
truth; `C_t` = tissue curve; `cp_aif` = AIF; `t` = time). Per-backend numbers live in
`osipi_summary.md`.
