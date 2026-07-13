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

## External Accelerator Handoff (Open Items Only)

### GPUfit / CPUfit Backend
- [ ] **`2cxm` accelerated fit is precision/parameterization-limited.** After the upstream false-CONVERGED fix (Gpufit `dev` `3db5b4d`) and the backend-agnostic multi-start (which resolved `tissue_uptake`/2CUM), the float32 cpufit/gpufit `2cxm` fit still misses ~6/24 OSIPI cases (`vp`↔`Fp` degeneracy). Initialization does not fix it (warm-start from linear Patlak makes it worse); the float64 python backend, which fits `E=Ktrans/Fp`, passes and is the reference. Options: double-precision cpufit/gpufit build, or reparameterize the compiled `2cxm` model to `E=Ktrans/Fp`. See `docs/project-management/projects/osipi-verification/gpufit_2cxm_2cum_divergence.md`.
- [ ] Verify CUDA/GPUfit runtime behavior on CUDA-capable hardware for the false-CONVERGED fix, the backend-agnostic multi-start (2CUM), and the `TOFTS_EXTENDED`/`2CXM` backend fixes (multi-start is verified on cpufit only locally).
- [ ] Verify bound handling and initialization consistency across GPUfit/CPUfit implementations.
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
