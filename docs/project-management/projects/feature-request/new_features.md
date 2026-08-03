# New Feature Requests
-[ ] Turn the noise estimation currently used in pipeline parity testing into a standalone function for general use. This is a small but useful utility that can be used to estimate noise in DCE-MRI voxels, and could be useful for other applications as well (AIF voxel filtering, TV baseline detection, etc.).
- [ ] Implement confidence intervals for GPUfit/CPUfit (currently only available in MATLAB/Python).
- [ ] Add parallelization to CPUfit (OpenMP?)
- [ ] Auto detect when to chop, look at baseline and determine if it is "flat enough" to use the full baseline, or if chopping is required. If chopping is required, determine how many frames to chop. Could also look at osciliations in the z direction to determine if chopping is required.
- [ ] **Take 2CXM residuals at the acquired times rather than on the dense grid.** The model must
      still be evaluated on the 0.1 s grid -- that is what resolves the 1-2 s plasma compartment --
      but the residual can be taken after sampling the prediction back down. Everything in the
      solver then shrinks ~50x: the Jacobian goes from 3001x4 to n_frames x 4, and the SVD inside
      the trust-region step (15% of a 2cxm fit, 5073 calls in a 24-voxel batch) all but disappears,
      as does upsampling ct per voxel. Prototyped 2026-08-02 at **4.0x per voxel per start**, and it
      recovered parameters slightly closer to truth (Fp 0.6017 vs 0.6257 against a truth of 0.600).
      It is also the statistically correct objective: the fit currently treats ~2900 spline-
      interpolated points as independent observations, which `dce_models.fit_2cxm_canonical` already
      has to undo by rescaling the covariance by `dof_interp / dof` -- that hack would be deleted.
      **Blocked on a decision**, not on effort: the compiled cpufit/gpufit kernel evaluates its model
      only at the points it is handed, so it cannot evaluate densely and sample back. Doing this in
      python alone re-splits the backends, which the 2026-08-02 consolidation just closed. Either
      accept the accelerated path as a deliberately lower-fidelity fast option, or drop accelerated
      2cxm (see the related item in `TODO.md` -- on CPU it is currently no faster than python
      anyway). Coarsening the dense grid from 0.1 s to 0.5 s is the alternative that keeps the
      backends identical: quadrature error 0.20% -> 0.78% of curve peak, against an AIF
      reconstruction error of ~25% that no grid choice affects, but worth only ~1.2x on its own.
- [ ] **Analytic Jacobian for the 2CXM fit.** A 2cxm fit averages ~181 forward-model evaluations,
      and with four parameters under a 2-point numerical Jacobian five of every six exist only to
      estimate derivatives. The impulse response is a biexponential whose derivatives with respect
      to all four parameters are closed-form, so an analytic `jac` is available -- messy, but real.
      Worth roughly 2x on its own since it does not touch the 3001-length linear algebra;
      composes with the item above. The same argument applies to the other `least_squares` models,
      whose forward curves are also analytically differentiable.
- [ ] **Compute confidence intervals only for the winning multistart candidate.** `fit_with_multistart`
      runs each candidate through a runner that calls `_ci_bounds_from_fit` per voxel
      (`dce_fit_backends.py:510` and `:640`), then keeps only the best candidate's and discards the
      rest; 2cxm pays the same way through `curve_fit`, which computes `pcov` unconditionally.
      Deferring the CI to a second pass over the winners would remove that waste. **Measured first
      and it is small** -- 3.4% of a patlak run, 2.0% tofts, 1.6% ex_tofts, 0.3% tissue_uptake,
      ~0.4% 2cxm -- so this is a tidiness win, not a performance one. It would need the runners to
      return enough state (or the fit object) to reconstruct the CI later, which is a real interface
      change; weigh that against a few percent before starting.