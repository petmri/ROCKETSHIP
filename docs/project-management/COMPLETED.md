# Python Transition Completed

## Purpose
Archive historical completed work only.

Do not track open items in this file; active work belongs in `docs/project-management/TODO.md`.

Completed items moved from `TODO.md` on 2026-03-05 to keep the active backlog short.

## Completed Recent Updates (2026-08-12)

### Archived from TODO on 2026-08-12 (completed earlier, never moved across)
- [x] Improve `2cxm` and `tissue_uptake` stability/accuracy on real data.
- [x] End steady-state Short term: AIF sidecar 
- [x] End steady-state medium term: evaluate the 4 algorithms, port the winner to MATLAB. Done and archived (`docs/project-management/projects/archived/steady-state-tv-default/`);
      the `tv`-default rollout is committed (`a9d78b6`) and `pytest -m parity` passes
      (all gated checks, no hand-curated exceptions -- 12/12 at the time, 42/42 after the
      2026-07-29 gate-scope review archived under 2026-08-12 in this file; see
      `docs/project-management/projects/archived/batch-parity/batch_parity.md`).

### Parity Testing Gaps (closed 2026-08-12; carried over from the archived batch-parity project)
DCE parity itself is done -- **42/42** gated voxelwise checks and 4/4 ROI-xls checks pass with no
hand-curated exceptions. These two gaps are about what the suite *cannot currently catch*; C and D
from the original list were dropped as won't-do. Full rationale:
`docs/project-management/projects/archived/batch-parity/batch_parity.md` -> "Testing gaps".
- [x] **Gate-scope review (2026-07-29).** The gated set went from 12 checks (tofts+patlak, Ktrans
      only) to 42 (tofts/patlak/ex_tofts, *every* parameter, all three regions, cpu and auto),
      under one rule with no per-model/parameter/region carve-outs. Two hand-rolled masks
      (`ktrans_upper_exclude`, `ve_ktrans_min`) collapsed into the single bound-pinning
      identifiability filter that `test_backend_equivalence.py` already used -- which is what
      made ex_tofts gateable (worst check 0.807 -> 0.9998). Scatter is now gated on RMSE
      normalized by the reference RMS: the previous absolute bound would have passed a 1737%
      error on patlak Ktrans. `tissue_uptake` and `2cxm` remain reported-only, each with the
      measurement (taken *with* the filter applied) recorded in `UNGATED_MODEL_REASONS`.
      Verified by injecting four regressions and confirming each fails.
- [x] **A. Stage-B AIF contract gate.** Built 2026-07-29: `tests/python/test_stage_b_aif_parity.py`
      gates `Cp_use`/`CpROI`/`Stlv_use`/`timer`, the `step` window, `start_time`/`end_time`/
      `max_index` and the AIF fit coefficients `[A B c d t_base_end t0_exp]` against a committed
      MATLAB payload (`Dyn-1_stage_b_aif.json`, written by
      `tests/matlab/helpers/write_stage_b_aif_contract.m`; regenerate with `'stageBOnly', true`).
      Python matches MATLAB to ~1e-8 relative; gates at `rel_mae ≤ 1e-5` / `rel_max_abs ≤ 1e-4` /
      `corr ≥ 0.99999`, timing to 1e-6 min, exact on indices. Verified to fail on a one-frame
      injection shift, a 0.1% `Cp_use` rescale (corr stays 1.0 — the point) and a moved peak.
      MATLAB-side drift folded into `check_matlabref_map_drift.py`; CI step added.
- [x] **B. Backend-equivalence gate.** Built 2026-07-29:
      `tests/python/test_backend_equivalence.py`, on 2000 voxels of Stage-B arrays frozen from
      `RUNNER_DATA/sub-1101743/ses-01` (332 KB; `freeze_stage_b_backend_fixture.py`). Gates
      `cpu` vs `cpufit_cpu` and vs `gpufit_cuda` on `patlak`/`tofts`/`ex_tofts`; skips with
      reason when a backend is absent. Enforced in CI by the `backend_equivalence` job, which
      installs the backends via `install_python_acceleration.py`; runners have no CUDA, so the
      `cpufit` half gates and the `gpufit` half skips itself.
      **This corrected the premise.** The "cpufit/gpufit diverges from MATLAB" reading was
      measuring bound-pinned voxels: on this fixture ex_tofts Ktrans reads corr 0.23 over all
      voxels but **0.9998** once voxels with a parameter pinned at a bound are excluded, and
      the backends' SSE agree to ~1e-6 relative. So the gate is three-part — equivalence on the
      identifiable subset, objective (SSE) agreement over all voxels, and bound-hit symmetry —
      rather than a naive whole-sample correlation, which would have had to be set so loose it
      gated nothing. Verified to fail on a 20% Ktrans bias, inflated SSE, and skewed bound-hit
      rates.
- [x] **Install `pycpufit` in CI** so gap B's gate is enforced there. Done via a dedicated
      `backend_equivalence` job running `install_python_acceleration.py --no-matlab --no-gui`,
      which pulls the prebuilt pyCpufit/pyGpufit from the `ironictoo/Gpufit` release
      (`py3-none-any`, ctypes-loaded `.so`, no hard CUDA link). Kept out of `python_checks` so a
      flaky external download cannot fail the main Python job — and it is now the only
      end-to-end exercise of the installer.

- [x] **Regenerate the `matlabref` tree after the `voxel_MaxFunEvals` 50 -> 200 change.**
      `tests/contracts/baselines/matlab_reference_v1.json` is **not** affected and needs no
      action — verified by regenerating it in MATLAB R2024a and running
      `check_baseline_drift.py`, which passed. It is generated from `default_dce_fit_prefs()`
      (`MaxFunEvals` 2000), not from `dce_preferences.txt`, so production settings never
      reach it. The `matlabref` pipeline tree
      (`tests/data/BIDS_test/derivatives/matlabref/`) *does* read `dce_preferences.txt` via
      `FXLfit_generic`, so it will move. Needs a MATLAB host; do it in its own commit.

      Measured on the parity fixture with production voxel prefs (TolFun 1e-12, TolX 1e-6,
      MaxIter 50), 30 noisy realizations per model per noise level, same noise draws for both
      budgets. Realizations whose fit changed, out of 30:

      | model | sigma 0.02 | 0.05 | 0.10 | worst rel diff |
      |---|---|---|---|---|
      | tofts | 0 | 0 | 0 | — |
      | ex_tofts | 0 | 1 | 7 | 35% |
      | tissue_uptake | 3 | 14 | 20 | 96% |
      | 2cxm | 11 | 20 | 28 | 180% |

      Repeating the comparison at 200 vs 2000 gives 0/30 everywhere except 2cxm at sigma 0.10
      (1/30), so **200 is a convergence plateau and 50 was truncating** — this is a
      correctness fix, not a cosmetic sync. Noise-free data converges at either budget, which
      is why the contracts never detected it. `tofts` and `patlak` are unaffected at every
      noise level. Harness: `tests/matlab/compare_voxel_maxfunevals.m`.

### GPUfit / CPUfit Backend (closed 2026-08-12)
The `E=Ktrans/Fp` reparam + O(N) convolution + analytic-Jacobian fix is **done and verified on
both cpufit and CUDA gpufit** (all 5 OSIPI sweeps pass on each, incl. all 24 2CXM; ~3000× faster
on 2cxm) — see `COMPLETED.md` and
`docs/project-management/projects/osipi-verification/STATUS.md`. Remaining:
- [x] **Verify the reparam kernels on CUDA hardware.** Done in `1f96a68`; re-confirmed 2026-07-29 on this CUDA box (`cuda_available=True`, runtime 12.5 / driver 13.0): `test_osipi_pygpufit.py` is 5/5 green and the `2cxm` `xfail` is gone.
- [x] **Review the `Fp` floor default change** (`2cxm`/`tissue_uptake` `lower_limit_fp` 1e-3→1e-4 in `dce_pipeline._stage_d_fit_prefs`) before dev-merge — it affects all backends (only relaxes the feasible region; physically ~0.6 mL/100mL/min).
- [x] Verify bound handling and initialization consistency across GPUfit/CPUfit implementations. Closed by the archived Stage-D consolidation: bounds and per-voxel seeds for all five accelerated-eligible models now come from one shared `python/dce_fit_backends.py` (`_*_bounds_row` / `assemble_*_candidates`), candidates are clamped into bounds before dispatch (`cb0cc68`), and `tests/python/test_osipi_backend_consistency.py` gates cpu-vs-cpufit-vs-gpufit agreement to 1e-4 relative.
- **Won't do: backend diagnostics surfaced in Python test failure messages.**
      Concretely: `dce_fit_backends._run_accelerated` collapses gpufit/cpufit's per-voxel `states`
      to `states == 0` and drops the codes (`python/dce_fit_backends.py:783`), and fills `extra`
      with `None`. Keeping a state histogram (e.g. `MAX_ITERATION` counts) would have named the
      `TOFTS {0: 6329, 1: 952}` non-convergence concentration directly instead of it having to be
      found by hand.
      Closed unbuilt on 2026-08-12. The `TOFTS {0: 6329, 1: 952}` investigation this was
      meant to shortcut is finished, and `test_osipi_backend_consistency.py` plus the
      `backend_equivalence` CI job now gate the behaviour the histogram would only have
      described. Revisit only if a future non-convergence hunt actually stalls for want of
      the state codes.

## Completed Recent Updates (2026-08-02)
- [x] **One 2CXM implementation, evaluated on the dense grid, on every backend.** Python carried
  two 2CXM forward models and the accelerated kernel a third, and they were used at different
  stages: the python Stage-D fit used the OSIPI 0.1 s resampled grid, the post-fit residuals used
  the MATLAB port on the acquired grid, and cpufit/gpufit fit on the acquired grid too. Measured
  against a closed-form 2CXM, the acquired timebase is the whole problem -- a 5 s frame costs
  15.6% of curve peak with the trapezoid recurrence and 79.8% with a rectangle-rule convolution,
  because the plasma MTT is routinely 1-2 s. The two python formulations were verified
  mathematically identical (bit-identical eigenvalues and amplitudes); only their grid and
  quadrature differed. Consolidated to a single `cxm2_curve` on the 0.1 s spline-upsampled grid,
  used by the fit, the post-fit residuals and (via an upsampled ct/timer/cp feed) the accelerated
  backends. `model_2cxm_cfit`/`model_2cxm_cfit_batch` and the `twocxm_forward` MATLAB contract
  were deleted with them. Quadrature moved from `np.convolve` to the exact-in-the-exponential
  trapezoid recurrence, run as a first-order IIR through `scipy.signal.lfilter`: 5x more accurate
  on this grid (0.20% vs 1.11% of peak) and 35x cheaper. Python 2cxm Stage-D 15.0 s -> 1.7 s
  (8.8x). Backend agreement on synthetic 2CXM data went from median relative differences of
  Ktrans 1.4% / ve 2.2% / vp 22.5% / **Fp 53%** to 0.15% / 0.01% / 0.05% / 0.00%. Caveats in
  `TODO.md`: cpufit 2cxm is no longer faster than python, and Fp remains non-identifiable.

## Completed Recent Updates (2026-07-30)
- [x] **Python non-fit pipeline performance, round 2: batched curve evaluation.** The post-fit
  path re-evaluates every fitted voxel's forward curve in Python regardless of backend
  (gpufit/cpufit return parameters, not curves), so it was untouched by acceleration and grew
  as a share of the run the faster the backend got. Added
  `_exp_weighted_cumulative_trapz_batch` plus `model_*_cfit_batch` for all six models, and
  reduced `_compute_fit_residuals` to one batched call: at 160k voxels with
  `write_postfit_arrays=1 write_qof_maps=1`, residuals 4.70 s -> 0.076 s (62x) and end-to-end
  10.99 s -> 6.42 s; unchanged on the default path, where both flags are off. Bit-identical
  Stage-A/B/D arrays and written NIfTIs across all six models. The subtle part is that Python
  floats *raise* (ZeroDivisionError, OverflowError from `math.exp` and from squaring a finite
  float, ValueError from a negative `math.sqrt`) where numpy returns inf/NaN, so each needs an
  explicit mask; details in
  `docs/project-management/projects/python-pipeline-performance/python_pipeline_performance.md`.

## Completed Recent Updates (2026-07-29c)
- [x] **Python non-fit pipeline performance, round 1.** Profiling showed the Python around the
  fitter cost more than the fitter itself: at 160k ROI voxels (tofts+patlak, default outputs)
  8.90 s of in-scope Python against 7.67 s of CPUfit, 6.70 s of it the per-voxel linear-Patlak
  seeding loop. Vectorized that seeding (new `model_patlak_linear_batch`), `_clean_ab`,
  `_clean_r1`, the two Stage-A baseline-rescale loops, and `_assemble_stage_d_output`:
  16.35 s -> 8.91 s end-to-end, in-scope Python 8.90 s -> 1.30 s, with every Stage-A/B/D output
  array bit-identical across both backends and all 9 models. Remaining work (the `model_*_cfit`
  curve functions, opt-out QC figures) is tracked in `TODO.md`; profile and equivalence
  evidence in
  `docs/project-management/projects/python-pipeline-performance/python_pipeline_performance.md`.

## Completed Recent Updates (2026-07-29b)
- [x] **Both dev-merge-critical Primary Blockers closed.**
  - **GUI status.** Both PySide6 GUIs (`python/dce_gui.py`, `python/parametric_gui.py`)
    verified against the current pipeline by driving their actual `Run` button flow
    headlessly (`QT_QPA_PLATFORM=offscreen`, real `QProcess` subprocess, real fixtures) --
    not just a code read. Found and fixed a real bug in the process: the parametric GUI's
    "Reset Defaults" / default-load path was broken. `parametric_default.json`'s paths are
    authored relative to `python/` (matching `parametric_cli.py`'s
    `ParametricT1Config.from_dict(..., base_dir=config_path.parent)` resolution), but the GUI
    re-serializes the payload into a new config under `output_dir` and was treating relative
    text as REPO_ROOT-relative -- so a stock "click Run" attempt wrote its run config one
    directory *above* the repo root and then failed to find the VFA inputs. The DCE GUI
    happened to work today only by coincidence (its CLI resolves relative paths against the
    subprocess cwd, which the GUI always sets to REPO_ROOT). Fixed by having each GUI resolve
    every path field to an absolute path at the point it's collected from the UI, anchored
    against the correct base for that CLI (`self._config_path.parent` for parametric, REPO_ROOT
    for DCE) -- so "Save Config As" and the run-config it launches are both portable
    regardless of where they end up on disk. Also fixed a `str(None) -> "None"` literal-string
    bug for optional fields (`mask_file`, `b1_map_file`, `checkpoint_dir`, and the run-summary
    display) that would have tried to open a file named `None`. Re-verified both GUIs
    end-to-end after the fix (real Stage A/B/D DCE run and real VFA T1 fit, both `status=ok`).
    Model-flag set (`tofts`/`ex_tofts`/`fxr`/`nested`/`patlak`/`tissue_uptake`/`two_cxm`/`auc`/
    `FXL_rr`) confirmed to still match `dce_pipeline.MODEL_SELECTION_ORDER` exactly.
  - **Path cleanup.** Repo-wide sweep for hardcoded local/user-specific paths. Found none in
    Python source (all path construction is `Path(__file__)`-relative or config-driven).
    Fixed one real portability wart: `tests/python/run_batch_stage_d_diagnostics.py` (an
    on-demand diagnostics script) defaulted `--matlab-bin` to the maintainer's own
    `/opt/homebrew/bin/matlab`, which doesn't exist on Linux, Intel Mac, or a different
    install layout; now defaults to `shutil.which("matlab") or "matlab"` (PATH-resolved,
    still overridable). Normalized five docs (`docs/dce_options.md`, `python/README.md`,
    `docs/wiki/python-walkthrough.md`, `tests/contracts/README.md`,
    `docs/project-management/projects/qualification/QUALIFICATION_MERGE_PACKET.md`) that
    hardcoded the maintainer's real absolute machine path
    (`/Users/samuelbarnes/code/ROCKETSHIP`) to the placeholder convention already used
    elsewhere in the same docs (`/path/to/ROCKETSHIP`). Left alone as legitimate/out-of-scope:
    `tests/python/freeze_stage_b_backend_fixture.py`'s `DEFAULT_RUNNER_DATA` (documented,
    CLI-overridable, established team convention for a not-in-CI regeneration script);
    `dce/run_neuroecon_job.m` (neuroecon is explicitly out of scope); a commented-out
    third-party path in `external_programs/niftitools/tommyscript.m` (dead, unreferenced,
    a former collaborator's personal analysis script -- flagged for the user to decide on
    deletion rather than removed unilaterally).

## Completed Recent Updates (2026-07-28)
- [x] **DCE batch-parity project complete and archived** to
  `docs/project-management/projects/archived/batch-parity/`. MATLAB-vs-Python DCE parity now
  stands at **12/12 gated voxelwise checks** and **4/4 ROI-xls checks**, with **no hand-curated
  exceptions** — the last one (tofts+GM reported-only) was retired once the Stage-B AIF fix made
  it unnecessary (corr 0.980 cpu / 0.992 auto against a 0.95 floor). ROI-xls `max_abs_err`:
  `tofts` 0.001440, `ex_tofts` 0.000013, `patlak` 0.000013, `tissue_uptake` 0.002235 — roughly a
  hundredfold better than where the project started, with `ex_tofts`/`patlak` at the ~1.3e-5 level
  predicted for two pipelines running genuinely the same algorithm. Four numbered issues closed:
  patlak+brain non-identifiability (QoF χ² masking), the Stage-B AIF algorithm difference
  (S1–S12 of `aif_fitting_parity.md`: unified 5-parameter fit, `tv` detector at 95.0% on 280
  human-rated sessions, `aif_Robust` off), zero-width MATLAB CI maps (CPU regeneration; the
  generator now refuses to run with `force_cpu ~= 1`), and batch-regression coverage (won't-do).
  Sub-projects `quality_of_fit.md` (per-voxel reduced-χ² reliability, shipped and validated) and
  `sigma_estimators.md` (estimator B + eBayes variance moderation, accepted) archived alongside.
  Residual open work moved to `TODO.md`: parity testing gaps A (Stage-B AIF contract gate) and B
  (backend-equivalence gate), QoF-aware ROI stats, and the `shrink_sigma` default decision.
  Two silent-wrong-answer hazards were closed in the same pass — `tv`'s no-detection fallback now
  reports a distinct mode instead of a plausible `end_ss = 1`, and the CI parity metrics now
  exclude zero-width intervals instead of scoring them as total disagreement.

## Completed Recent Updates (2026-07-22)
- [x] **Stage-D fit-backend consolidation, archived.** All five accelerated-eligible
  models (`patlak`, `tofts`, `ex_tofts`, `tissue_uptake`, `2cxm`) migrated onto one shared
  `python/dce_fit_backends.py` machinery (`FitInputs`, per-model `assemble_*_candidates`,
  `fit_with_multistart`), replacing three separate, incompatible seeding/multi-start
  implementations (CPU/python, accelerated, and per-model hand-rolled variants). Three
  follow-up cleanup passes collapsed remaining duplication (`_stage_d_fit_funcs()`
  registry, `_validate_stage_d_inputs`/`_assemble_stage_d_output` shared helpers,
  `_ModelSpec`/`_fit_stage_d_batch`), fixed a real prefs-handling bug
  (`_apply_model_specific_prefs` wasn't applied on the accelerated-attempt path, silently
  ignoring 2cxm/tissue_uptake per-model override knobs whenever the accelerated backend
  succeeded), and fixed a tissue_uptake candidate-unit-scaling bug caught by the OSIPI
  gate. Verified throughout: full `pytest tests/python -q` (195 passed) and `-m osipi`
  green after every pass. One known residual (patlak+brain-ROI non-identifiability at a
  `vp` bound) was tabled, not fixed by this project -- folded into the ongoing
  `docs/project-management/projects/archived/batch-parity/batch_parity.md` tracking. Archived to
  `docs/project-management/projects/archived/stage-d-fit-consolidation/`.
- [x] **`tv` steady-state-end detector ported to MATLAB and made the default in both
  languages** (`dce/find_end_ss_tv.m`, `python/dce_pipeline.py`'s `_resolve_baseline_window`
  fallback, both config JSON defaults). Decision driven by a new on-demand diagnostic
  (`tests/python/run_baseline_end_reliability.py`) run against 224 real AIFArtist-rated
  sessions: `tv` ~88-93% accuracy vs. MATLAB's previous default (`piecewise_constant`)
  ~0%. The MATLAB port is verified numerically identical to Python on 6 curves including
  a real one (`tests/python/test_find_end_ss_tv_matlab_parity.py`). Found and fixed two
  unrelated, pre-existing MATLAB bugs in `FXLfit_generic.m`'s Patlak/GPU branch along the
  way (dead "ANIMAL study" ID-matching code; a `constraint_type`/`constraint_types`
  typo), both confirmed independent of the steady-state change. **Not yet committed**:
  flipping the default surfaced parity-gate failures (`patlak_ktrans_brain`,
  `tofts`/`tissue_uptake` ROI-xls) -- root-cause hypothesis and next steps folded into
  `docs/project-management/projects/archived/batch-parity/batch_parity.md` (the brain-ROI one
  looks like the same already-tabled non-identifiability issue from the Stage-D
  consolidation above). Archived to
  `docs/project-management/projects/archived/steady-state-tv-default/`.

## Completed Recent Updates (2026-07-13)
- [x] **Unified accelerated-fit fix (`E=Ktrans/Fp` reparam + O(N) convolution + analytic Jacobians) — implemented and verified on cpufit.** Rewrote the compiled `2cxm`/`2cum` models so parameter[0] is the extraction fraction `E∈(0,1)` (recover `Ktrans=E·Fp`); `PS=Fp·E/(1−E)` is smooth, so the `Ktrans=Fp` pole and the `if(p0>=p3)PS=1e9` sentinel are deleted. The CPU backend (`~/code/Gpufit/Cpufit/lm_fit_cpp.cpp` — which has its *own* C++ models, not the `.cuh` files) now computes value + full Jacobian in a single **O(N) exponential-recurrence** pass (`G`/`G'`/`U`) for all four conv models, replacing the O(N²) per-point convolution and the 5-point numerical Jacobian. The CUDA `.cuh` kernels (`Gpufit/models/two-compartment_exchange.cuh`, `tissue_uptake.cuh`) got the same reparam + analytic Jacobian (host-verified analytic-vs-central L2 ~1e-9; **not yet built/run on CUDA hardware** — gpufit 2CXM stays `xfail(strict=False)`). Caller (`python/dce_pipeline.py`) maps `Ktrans`/`Fp` prefs → `E` init/bounds (`_extraction_fraction_init_bounds`, mirroring the float64 python reference) and `E→Ktrans=E·Fp` on output. **Also lowered the `Fp` floor `1e-3→1e-4`/s** (`2cxm`/`tissue_uptake`), the missing piece that lets low-flow `Fp=5` (≈8.3e-4/s) be represented. Result: **all 5 OSIPI cpufit sweeps pass, including all 24 2CXM cases** (was ~6/24 xfail — the earlier "weak-identifiability floor" reading was wrong; the float64 reference passes all 24), and the O(N²) cliff + numeric-Jac multiplier are gone (tofts 107→0.2, 2cum 3879→2.6, 2cxm 12325→4.1 ms/row). Multi-start still required (kept). Added `tests/python/test_reparam_jacobian.py` (Jacobian guard, L2<1e-6); un-xfailed cpufit 2CXM. Rebuilt `pyCpufit` into `.venv` (dylib md5 `4a56ad4f→0044e3df`). Full derivation/status in `docs/project-management/projects/osipi-verification/STATUS.md`.
- [x] Rebuilt patched `pyCpufit 1.4.1` (Gpufit `dev` `3db5b4d` "Fix false CONVERGED on rejected step" + `607f127` global-convergence) verified in use, and added a **backend-agnostic random multi-start** (`python/dce_pipeline.py:_accel_multistart_refine`, adopted from the Gpufit `bug/experiments.py` harness): per voxel the accelerated Stage-D fit tries the fixed start plus 8 log-uniform draws with a cheap coarse fit, refines from the best basin, and keeps the lowest chi-square (never degrades a good fit; identical for cpufit/gpufit). This resolved `tissue_uptake` (2CUM) on the OSIPI sweep (1 failing case → 0) and promoted the cpufit/gpufit 2CUM sweep from `xfail` to passing. Residual `2cxm` misses are low-flow (`Fp=5`) weakly-identifiable-`vp` cases — the root cause is the `Fp` initial landing in a wrong basin, **not float32** (a `DOUBLE_PRECISION` build shows the same degenerate minima) — tracked in `TODO.md` + `docs/project-management/projects/osipi-verification/STATUS.md`, with an `E=Ktrans/Fp` compiled-model reparameterization planned.
- [x] Investigated why the python backend beats the accelerated path on the stiff models: python 2CUM already multi-starts (+ linear-Patlak seed via `_best_fit_over_starts`), python 2CXM wins on float64 + the `E=Ktrans/Fp` reparameterization (single start). Cross-checked against the Gpufit `bug/` harness (`FINDINGS.md`, `probe_hard_cases.py`): the residual failures are an `Fp`-initial basin problem, and a linear-Patlak warm-start that leaves `Fp` high makes 2CXM *worse*. Regenerated `osipi_summary.md` + figures with per-backend numbers.

## Completed Recent Updates (2026-07-12)
- [x] Real Python fit confidence intervals (`python/dce_models.py`): replaced the placeholder "CI = point estimate" (zero-width) returns with genuine Jacobian-based 95% intervals (`beta ± t(1-alpha/2, dof) * sqrt(diag(MSE * inv(J^T J)))`, the `confint`/`nlparci` equivalent) for `tofts`, `ex_tofts`, `patlak`, `vp`, `tissue_uptake`, `fxr`, and `2cxm`. Derived params follow MATLAB propagation (`tissue_uptake` vp via the Tp CI; `2cxm` Ktrans=E*Fp via the delta method); matches an analytic OLS interval to ~1e-8.
- [x] Phantom GT-in-CI coverage metric (`tests/python/phantom_gt_helpers.py`, `run_phantom_gt_reliability.py`): report per-region `ci_coverage_frac` (fraction of voxels where ground truth falls inside the fit's 95% CI; well-calibrated ~0.95) plus standardized error `z=|GT-fit|/CI_halfwidth` — a scale-free accuracy-under-noise signal that avoids the near-zero-GT `%GT` blow-up. Full CPU sweep shows `ex_tofts` brain Ktrans is calibrated (~0.90-0.98) while `tofts`/`patlak` are systematically biased, and `sub-08` (near-perfect T1) still fails `tofts` — confirming model mismatch over T1 quality. Findings logged in `docs/project-management/projects/phantom-gt/PHANTOM_GT_QUALIFICATION_STATUS.md`.
- [x] Test fixtures consolidated into `tests/data/BIDS_test`; `tests/data/ci_fixtures` removed. `downsample_x2_bids` (byte-identical to `sub-02downsample`) dropped in favor of the latter; `bbb_p19_downsample_x3y3` → `sub-10bbbdownsample` (DCE fit-parity fixture, ROIs now `derivatives/.../desc-*_mask.nii`, MATLAB baselines under `derivatives/matlabref/`); `vfa_small`+`tiny_settings_case` → `sub-11tiny`; unused `sub-03noisyhigh`/`sub-04noisylow` dropped. Every subject-internal file renamed to its real BIDS label. Consuming tests, `dce_default.json`/`dceprep_default.json`, and generation scripts repointed; dataset-level qualification now skips (not fails) sessions lacking preprocessed inputs, and ROI-xls parity compares a canonical tissue token so BIDS mask names align with the frozen MATLAB reference. Added BIDS `participants.tsv/json`, `dataset_description.json`, and a top-level data README.

## Completed Recent Updates (2026-07-10)
- [x] DCE dataset-backed parity reworked (`tests/python/test_dce_pipeline_parity_metrics.py`): evaluate brain/GM/WM regions (pipeline fits a union ROI), gate only Tofts + Patlak `Ktrans` on RMSE + correlation, and report (non-gating) CI-normalized abs-diff (p95) and proportion-outside-CI for every Python-vs-MATLAB parameter. Tofts-GM is reported-only (disagreement is non-identifiability, not a bug). Replaced the previous per-parameter MAE/p95 gates and consolidated four overlapping parity tests plus a dead helper cluster into `test_bbb_p19_region_parity`.
- [x] ROI-only DCE fit mode added (`stage_overrides.fit_voxels=0`): average-then-fit per ROI (matching MATLAB), skipping the per-voxel fit — whole-brain ROI `.xls` parity dropped from ~8 min to ~3 s and is less noise-biased for nonlinear models. Powers `test_bbb_p19_roi_xls_parity`.
- [x] Parity tests are now default-on. Removed the `--run-parity`/`--parity` and `--run-full-parity`/`--full-parity` opt-in flags (and the full-volume parity test). Standard region parity, ROI `.xls` parity, and nonlinear T1-map parity (`test_bids_t1_map_parity_nonlinear`) all run by default; `--parity-suite=allmodels` adds `ex_tofts`/`tissue_uptake`/`2cxm` as reported-only extras, and `--parity-thresholds` accepts a JSON gate-override file.
- [x] CI MATLAB baseline drift guard added (`tests/contracts/check_baseline_drift.py`): `run_DCE.yml` regenerates the MATLAB reference and fails if the committed `matlab_reference_v1.json` no longer matches current MATLAB output, so Python parity is verified against *current* MATLAB rather than a stale snapshot. CI also gained a small-VFA T1-map parity step.
- [x] Python DCE config is now JSON-only by default: removed the `--dce-preferences` CLI flag and flipped `use_dce_preferences` to `false` in `dce_default.json`/`dceprep_default.json`. The `dce_preferences.txt` bridge still works as an explicit opt-in via `stage_overrides`.
- [x] Added an ASCII-art startup banner (`python/banner.py`, printed to stderr) and a single-source version file (`python/version.py`, `__version__ = "1.3"`).

## Completed Recent Updates (2026-03-02)
- [x] Batch DCE config assembly now prefers per-session DCE metadata JSON for `tr`/`fa`, and avoids template `tr`/`fa` defaults unless explicitly passed via `--set`.
- [x] Batch mode now forces `dce_metadata_path` to the current session sidecar by default (prevents template test-fixture metadata paths from leaking into real-data runs unless explicitly overridden via `--set dce_metadata_path=...`).
- [x] DCE frame spacing now has strict metadata behavior: it must come from sidecar JSON (`time_resolution_sec`, `TemporalResolution`, MATLAB-style `RepetitionTime` when `RepetitionTimeExcitation` exists, `AcquisitionDuration`, or `TriggerDelayTime/nReps`) or explicit config override (`time_resolution_sec`/`time_resolution`); missing values are a hard error.
- [x] Python Stage A now supports MATLAB-style `start_t`/`end_t` timepoint clipping (1-based frame window) before concentration conversion.
- [x] Python Stage A/B auto injection timing now mirrors MATLAB CLI auto behavior.
- [x] Batch template hardcoded `start_injection[_min]`/`end_injection[_min]` values are stripped unless explicitly passed via `--set`, so auto injection is the default batch behavior.
- [x] MATLAB legacy Sobel parity fix completed.
- [x] Stage-D batch parity diagnostics completed on clean-reference `RUNNER_DATA` packet (`sub-1101743/{ses-01,ses-02}`).

### Scan Parameter Policy (No Silent Defaults)
- [x] Policy set: there are no implicit defaults for scan parameters in Python workflows.
- [x] Required scan parameters must come from metadata or explicit user config.
- [x] Missing required scan parameters now hard-fail.

## Completed Primary Items

### 1. Parametric maps and T1 fitting workflow
- [x] Port remaining workflow behavior from `parametric_scripts/custom_scripts/T1mapping_fit.m` and required `calculateMap` path components.
- [x] Add Python CLI entrypoint for T1 mapping workflow with clear config schema and run summary output.
- [x] Add Python GUI support for T1 fitting workflow (file selection, run controls, progress, QC).
- [x] Add real-data tests for T1 output integrity and expected file naming.
- [x] Add fixture-backed tests for T1 output integrity and expected file naming.
- [x] Add OSIPI T1 mapping reliability checks beyond current linear-only coverage (nonlinear and two-FA comparators).
- [x] Add OSIPI signal-intensity to concentration verification tests.
- [x] Integrate OSIPI SI-to-concentration thresholds into merge-gate reporting.
- [x] Add MATLAB-vs-Python parity test for non-linear VFA T1 synthetic reference.
- [x] Add contract-runner integration for non-linear VFA T1 parity.

### 2. DCE primary model readiness (`patlak`, `tofts`, `ex_tofts`)
- [x] Tighten parity/reliability thresholds and make primary model checks strict merge gates.
- [x] Ensure backend consistency across `cpu`, `cpufit`, and `gpufit` where available.
- [x] Add regression tests for known edge cases (bounds, low SNR, non-uniform timer inputs).
- [x] Implement automatic DCE baseline-window detection methods for Stage A (`steady_state_auto_method` = `legacy_sobel` and `piecewise_constant`), with explicit method selection and manual `steady_state_end` precedence.
- [x] Validate the current default DCE baseline auto method (`legacy_sobel`) on representative real datasets (MATLAB comparison + qualification impact), and keep `legacy_sobel` as default for now.
- [x] Add qualification gating for non-finite primary-model parameter maps.
- [x] Resolve qualification blocker from `ex_tofts` non-finite accelerated maps by falling back to next backend/CPU when accelerated output has no usable finite primary parameters.
- [x] Adopt accelerated DCE `gpu_tolerance=1e-6` default (was `1e-12`) after CPUfit/Cpufit max-iteration diagnosis; verified with full Python test suite and `run_python_qualification.py` on 5-session `tests/data/BIDS_test`.

### 3. Part E post-fitting analysis
- [x] Port required workflow from `dce/fitting_analysis.m`, `dce/compare_fits.m`, and supporting analysis helpers.
- [x] Implement reproducible Python outputs for ROI/voxel fit review used in current workflows.
- [x] Add automated tests for analysis outputs and plotting/stat summary generation.

### 4. Real-data workflow qualification
- [x] Run end-to-end Python DCE + T1 workflows on representative real datasets.
- [x] Record blocker issues and classify them as fix-now vs post-merge follow-up.
- [x] Prepare merge packet: command recipes, known differences, troubleshooting notes.
- [x] Latest local qualification rerun (2026-02-22) passed on 5-session `tests/data/BIDS_test` with `backend=auto` (`cpufit_cpu`) after accelerated tolerance default update.

### 5. Synthetic phantom images example datasets qualification (completed subset)
- [x] Get real NII BIDS data and replace matrix with all 0/1.
- [x] Insert synthetic DCE curves into real NIfTI files, maintaining original headers.
- [x] Save ground-truth Ktrans, vp, etc. files for synthetic datasets.
- [x] Compare fit values to ground truth.
- [x] Add synthetic phantom ground-truth reliability checks (region/model-specific MAE tolerances) for `sub-05phantom`/`sub-06phantom`/`sub-07phantom`, with T1 reconstructed in-test before DCE fitting.

## Completed Secondary Items
- [x] Improve `2cxm` and `tissue_uptake` stability/accuracy in OSIPI benchmark/reliability subsets (real-data hardening remains tracked in `TODO.md`).

## Completed Handoff Items
- [x] Resolve `PATLAK` Cpufit/Cpufit real-data divergence in multi-fit constrained runs (RUNNER_DATA ses-02 payload; all fits currently report `CONVERGED` but parameter agreement is poor).

## Completed Condensed Milestones
- [x] Contract parity tooling moved to `tests/contracts/`.
- [x] `run_dce_parity.py` restored with corr/MSE/MAE summary output.
- [x] Benchmark runner renamed to `tests/python/run_dce_benchmark.py`.
- [x] Script-preference audit completed with support/pending/drop classification metadata.
- [x] Dataset-backed parity expanded beyond Tofts maps (including ROI `.xls` checks).
- [x] Python parametric T1 pipeline/CLI scaffold added (`run_parametric_python_cli.py`, `python/parametric_pipeline.py`, `python/parametric_cli.py`).
- [x] OSIPI SI-to-concentration source data and peer result tables imported into `tests/data/osipi/`.
- [x] OSIPI SI-to-concentration reliability test added (`tests/python/test_osipi_si_to_conc_reliability.py`).
- [x] OSIPI SI-to-concentration merge-gate reporting runner added (`tests/python/run_osipi_reliability.py`) and wired in CI.
- [x] Primary DCE edge-case regression tests added for non-uniform timer, low-SNR fits, and custom bounds (`tests/python/test_dce_models.py`).
- [x] Primary DCE OSIPI thresholds tightened to strict peer-max limits and integrated into merge-gate reporting (`tests/python/test_osipi_dce_reliability.py`, `tests/python/run_osipi_reliability.py`).
- [x] Primary DCE backend consistency checks added across CPU/CPUfit/GPUfit where available (`tests/python/test_osipi_backend_consistency.py`).
- [x] Accelerated DCE default tolerance updated to `gpu_tolerance=1e-6` and validated with full `tests/python` suite and BIDS qualification.
- [x] Synthetic phantom GT reliability helper/runner added (`tests/python/phantom_gt_helpers.py`, `tests/python/run_phantom_gt_reliability.py`).
- [x] Phantom GT runner enhanced with compact AIF diagnostics and explicit phantom metadata alignment notes.
- [x] Parametric T1 GUI v1 added (`run_parametric_python_gui.py`, `python/parametric_gui.py`).
- [x] Parametric T1 real-data naming/integrity tests added for BIDS-based multifile and stacked inputs (`tests/python/test_parametric_pipeline.py`).
- [x] Parametric pipeline now supports nonlinear and two-point VFA fit types in addition to linear.
- [x] Parametric pipeline now supports optional B1-scaled flip-angle fitting (`b1_map_file` explicit or auto-detected `B1_scaled_FAreg.nii(.gz)`).
- [x] Parametric pipeline now requires TR from VFA sidecar metadata (`RepetitionTime`) or explicit `tr_ms`.
- [x] Parametric pipeline now supports MATLAB-style `odd_echoes` frame selection and optional XY Gaussian smoothing.
- [x] Part E statistical core port started, including reproducible JSON/CSV/NPY artifacts.
- [x] Stage D optional Part E array export added via `stage_overrides.write_postfit_arrays` with NPZ loader/runner path and regression coverage.
- [x] Part E analysis outputs now include statistical summaries and optional PNG plots.
- [x] CI workflow separated Python-only checks from MATLAB-backed parity checks and expanded portability/Matlab matrices.

## Archived From PORTING_STATUS (Moved 2026-03-05)

Historical snapshot entries moved out of active status view:
- [x] Python DCE timing metadata resolution update (2026-03-03): JSON frame-spacing branches (`RepetitionTime` with `RepetitionTimeExcitation`, `AcquisitionDuration`, `TriggerDelayTime/n_reps`, and existing `time_resolution_sec` / `TemporalResolution`).
- [x] Python DCE timing/injection parity update (2026-03-03): MATLAB-style `start_t`/`end_t` clipping and auto injection behavior with legacy Sobel parity.
- [x] Downsample Tofts parity realignment (2026-03-09): used commit `8ef4988` as the reference change point, added MATLAB-vs-Python Stage-A/B diagnostic exporters, ported six-parameter Stage-B AIF timing to Python, and aligned dataset-backed parity fixtures with post-`8ef4988` auto baseline/injection timing.
- [x] Removal of Python runtime scan-parameter fallbacks from `script_preferences.txt` (2026-03-03).
- [x] Batch DCE per-session metadata-preference update with Stage-A metadata provenance fields (2026-03-02).

Historical lessons/details moved from active status section:
- [x] Qualification warning (2026-02-20): `sub-02downsample_ses-01` flip-angle metadata trim (`3 -> 2`) for derivative VFA frame match.
- [x] Qualification lesson (2026-02-22): accelerated `ex_tofts` finiteness issue resolved after adopting `gpu_tolerance=1e-6`; guarded fallback retained.
- [x] PATLAK accelerated backend caveat (2026-03-03) with 2026-03-05 closure update and removal of temporary PATLAK handoff package.
- [x] CI topology update (2026-02-27): dedicated `parity_checks`, updated portability matrix, unified MATLAB matrix, and workflow concurrency cancellation.
- [x] DCE baseline auto-detection port status (2026-02-24): `legacy_sobel`, `piecewise_constant`, `glr`, `tv` with manual-end precedence.
- [x] Phantom GT troubleshooting status (2026-02-23): timing/conversion mismatch fixes and model-mismatch diagnosis context.
