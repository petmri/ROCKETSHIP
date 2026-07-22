# Python Transition TODO

## Purpose
Track only active, actionable tasks.

Do not log historical completions here. Record finished work in `docs/project-management/COMPLETED.md`.
Keep strategic sequencing in `docs/project-management/ROADMAP.md` and current measurable state in `docs/project-management/PORTING_STATUS.md`.

## Primary Blockers (Dev-Merge Critical)

### 1. Synthetic Phantom Qualification
- [ ] Reach acceptable MAE tolerance windows for primary maps (targeting approximately 10-20% region/model bias where scientifically reasonable).
- [ ] Isolate implementation error vs expected model mismatch by generating matched-model phantom datasets in `synthetic_dce` (`tofts`, `ex_tofts`, `patlak` generation variants).

## Secondary Active Work (Non-Blocking First Dev Merge)

### 1. Synthetic Example Dataset Coverage
- [ ] Generate BIDS-structured synthetic set with three SNR tiers (low/medium/high), each with two repeated measurements using identical parameters and independent noise.

### 2. Modeling and Workflow Follow-Ups
- [ ] Evaluate moving T1 fitting onto CPUfit/GPUfit for performance improvements.
- [ ] Improve `2cxm` and `tissue_uptake` stability/accuracy on real data.
- [ ] Expand DSC support beyond current core (`DSC_convolution_oSVD` and broader workflow parity).
- [ ] Decide final status of `nested` and `FXL_rr` (full support vs explicit non-support with cleanup).
- [ ] End steady-state, long term: implement a more robust end_ss estimation method
      (update AutoAIF neural net) consistent across all models/datasets. Short term (AIF
      sidecar) and medium term (evaluate the 4 algorithms, port the winner to MATLAB) are
      done and archived (`docs/project-management/projects/archived/steady-state-tv-default/`);
      the `tv`-default rollout (uncommitted) currently blocks `pytest -m parity` -- see
      `docs/project-management/projects/batch-parity/batch_parity.md` for findings and
      next steps before committing.

## External Accelerator Handoff (Open Items Only)

### GPUfit / CPUfit Backend
The `E=Ktrans/Fp` reparam + O(N) convolution + analytic-Jacobian fix is **done and verified on
cpufit** (all 5 OSIPI sweeps pass incl. all 24 2CXM; ~3000× faster on 2cxm) — see `COMPLETED.md`
and `docs/project-management/projects/osipi-verification/STATUS.md`. Remaining:
- [ ] **Verify the reparam kernels on CUDA hardware.** The `.cuh` kernels (`two-compartment_exchange.cuh`, `tissue_uptake.cuh`) carry the same reparam + analytic Jacobian and are host-verified (analytic-vs-central L2 ~1e-9), but not built/run with nvcc here. Build pyGpufit and run `test_osipi_pygpufit.py` on a CUDA box; the `2cxm` gpufit test is `xfail(strict=False)` until then. Also confirm the false-CONVERGED fix + multi-start (2CUM) on hardware.
- [x] **Review the `Fp` floor default change** (`2cxm`/`tissue_uptake` `lower_limit_fp` 1e-3→1e-4 in `dce_pipeline._stage_d_fit_prefs`) before dev-merge — it affects all backends (only relaxes the feasible region; physically ~0.6 mL/100mL/min).
- [ ] Verify bound handling and initialization consistency across GPUfit/CPUfit implementations (in progress with stage D refactoring).
- [ ] Provide backend diagnostics that can be surfaced directly in Python test failure messages.

### Synthetic_DCE Generator
- [ ] Import segmentation image with tissue classes.
- [ ] Output DCE images where each segmentation class maps to a unique tissue generation class.
- [ ] Add motion simulation option with inverse motion-correction matrices.
- [ ] Output ground-truth maps (Ktrans, vp, etc.).

## Out of Scope (Unless Scope Changes)
- neuroecon execution path
- legacy email notification flow
- manual click-based MATLAB AIF tooling
- ImageJ `.roi` compatibility path
- legacy MATLAB queue/prep GUI flows not needed in Python workflows
