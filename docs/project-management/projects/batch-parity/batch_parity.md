# Batch Parity Status (MATLAB vs Python DCE)

> **Staleness note (2026-07-17):** most of this document predates the steady-state/
> injection-timing overhaul (Python now always auto-detects steady-state end, matching
> MATLAB's `find_end_ss`, with a `SteadyStateEndTimeIndex` AIF-sidecar override for
> fixed/reproducible runs -- see `docs/dce_options.md`) and the Stage-D fit-backend
> consolidation (`docs/project-management/projects/archived/stage-d-fit-consolidation/`,
> archived 2026-07-22 -- all five models migrated, done). The
> "Key Diagnostics and Artifacts" paths below are absolute macOS paths
> (`/Users/samuelbarnes/...`) from a different dev machine and likely don't resolve
> here. The specific Stage-A/B numeric snapshots ("Latest CPU-vs-MATLAB clean-reference
> check") and the "auto vs manual injection window" framing in Outstanding TODO #4
> predate that overhaul and should be re-verified, not assumed current, before acting on
> them. Left as-is rather than rewritten -- flagging per request, not fixing now.

## Scope
Primary tracking for parity work on `RUNNER_DATA/sub-1101743/{ses-01,ses-02}` and related parity fixtures.

Current focus:
- end-to-end batch parity (`run_dce_bids_batch.py`)
- Stage-B AIF fit behavior impact
- Stage-D backend behavior (`cpu` vs `cpufit_cpu`)

## Current Status
### Changes Made
- Stage-A time-window parity:
  - added MATLAB-style `start_t/end_t` frame clipping before concentration conversion.
- Stage-A/Stage-B injection-timing parity:
  - aligned auto behavior to MATLAB logic:
    - `start_injection = end_ss`
    - `end_injection = mean(argmax(AIF voxels))`
  - made `auto_find_injection=1` enforce Stage-A auto timing in Stage B.
- Legacy baseline-end detector parity:
  - matched MATLAB endpoint behavior for moving-average smoothing and Sobel-style derivative scaling in `legacy_sobel`.
- Metadata/time-resolution parity and safety:
  - added MATLAB-compatible JSON branches for frame spacing (`time_resolution_sec`, `TemporalResolution`, `RepetitionTime@RepetitionTimeExcitation`, `AcquisitionDuration`, `TriggerDelayTime/n_reps`).
  - removed script-preference fallback for scan timing; missing required scan metadata now hard-fails.
- Batch-config cleanup:
  - removed implicit hardcoded injection-window defaults from batch template assembly unless explicitly provided via `--set`.
- File-discovery bug fix:
  - prioritized DCE-space brain mask selection over anatomical-space mask collisions.
- Reference-map selection fix:
  - standardized parity comparisons on canonical MATLAB `dce_patlak_fit_*.nii` outputs to avoid mixed-file drift.
- Stage-B AIF fit update:
  - introduced weighted biexponential fitting (MATLAB-style post-peak emphasis), which improved RUNNER_DATA Stage-B alignment (`Cp_use`) in clean-reference parity runs.

### Latest CPU-vs-MATLAB clean-reference check (2026-03-04)
Dataset:
- `RUNNER_DATA/derivatives/dceprep-matlab-cleanref`
- `RUNNER_DATA/derivatives/dceprep-python-batch-cleanref-cpucheck`
- subject/session: `sub-1101743/{ses-01,ses-02}`
- model/backend: Patlak, Python `backend=cpu`

Ktrans map parity (tumor voxels from Python Stage-B `tumind`):
- `ses-01` (all finite voxels): `corr=0.922909`, `slope=0.999566`, `mean_ratio(py/mat)=1.00137`, `mae=3.50e-06`
- `ses-02` (all finite voxels): `corr=0.892135`, `slope=0.999818`, `mean_ratio(py/mat)=1.01294`, `mae=1.05e-05`
- `ses-01` (active voxels `|Ktrans_matlab|>=1e-5`): `corr=0.99999998`, `slope=1.000009`, `mean_ratio=0.999974`
- `ses-02` (active voxels `|Ktrans_matlab|>=1e-5`): `corr=0.99999995`, `slope=1.000011`, `mean_ratio=0.999558`

Interpretation:
- CPU path is numerically aligned with MATLAB for non-floor Ktrans voxels in both sessions.
- Lower all-voxel correlations are dominated by near-zero/floor-heavy voxels, not slope/scale drift in active voxels.

### Confirmed aligned
- Stage-A/Stage-B arrays are numerically aligned between MATLAB and Python for the latest clean-reference run:
  - `timer`, `Ct`, and `Cp_use` match to floating-point noise.
- Patlak core fitter contract is aligned:
  - MATLAB vs Python on identical `(Ct, Cp_use, timer, prefs)` is numerically identical in sampled single-curve and ROI checks.
- CPU Patlak backend alignment:
  - On sampled voxels from clean-reference checkpoints, Python CPU matches MATLAB almost exactly.
- CPU linear Patlak backend alignment:
  - On sampled voxels from clean-reference checkpoints, Python CPU linear fit matches MATLAB almost exactly.

### Confirmed not aligned
- Stage-B weighted AIF fit update currently regresses one required multi-model parity check (below).
- Regression on GPU-accelerated backend behavior observed in qualification test; root cause under investigation.

## Key Diagnostics and Artifacts
- Stage-D diagnostics runner:
  - `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/run_batch_stage_d_diagnostics.py`
- Diagnostics outputs:
  - `/Users/samuelbarnes/code/ROCKETSHIP/out/batch_stage_d_diagnostics_aifw2_10000_20260303.json`
  - `/Users/samuelbarnes/code/ROCKETSHIP/out/batch_stage_d_diagnostics_aifw2_10000_patlakfix_20260303.json`


## Outstanding TODOs
1. Resolve current required parity failure introduced by weighted Stage-B AIF fit.
- Failing suite:
  - `.venv/bin/python tests/python/run_dce_parity.py -s multi-model`
- Required failures currently observed:
  - `ex_tofts_ktrans_cpu_vs_matlab` (`corr=0.193030`, threshold `>=0.85`)
  - `ex_tofts_ktrans_auto_vs_cpu` (`corr=0.188802`, threshold `>=0.98`)
- Causality check completed:
  - clean tracked `HEAD` passes required checks;
  - applying only `python/dce_pipeline.py` from current work reproduces failure;
  - temporarily reverting only `_fit_aif_biexp` weighted-fit block restores pass.

2. Keep weighted AIF fit for now, but define compatibility policy.
- Decide whether to:
  - calibrate parity thresholds/dataset for this known algorithmic change, or
  - add a compatibility mode for parity fixtures while keeping weighted mode for batch runs.

3. Resolve testing failure
- New qualification failure (backend=auto/GPU path):
  - failing test: `tests/python/test_python_qualification.py::test_bids_root_qualification_processes_all_sessions`
  - failing session: `sub-07phantom_ses-01`
  - failing model/params: `tofts:Ktrans`, `tofts:ve`
  - blocker message: `finite_nonzero_ratio=0.869` below threshold `0.900`
  - backend used in failing run: `gpufit_cuda`
  - direct accelerated fit diagnostics show TOFTS non-convergence state concentration:
    - `TOFTS {0: 6329, 1: 952}` where state `1 = MAX_ITERATION`
    - comparison context: `TOFTS_EXTENDED {0: 7203, 1: 78}`, `PATLAK {0: 7280, 1: 1}`

4. Design and implement robust batch-processing regression coverage (new).
- Goal:
  - catch batch parity regressions early across Stage-A/B assembly, input-resolution logic, and Stage-D backend behavior.
- Dataset plan (small but representative):
  - `batch_fixture_small_real`: downsampled real-like case with valid sidecars/AIF/brain mask for end-to-end realism.
  - `batch_fixture_small_synth`: compact synthetic case with known parameter ground truth (for trend checks, not strict MATLAB parity).
  - `batch_fixture_edge_inputs`: tiny case variants for metadata/input-path edge conditions.
- Runtime tiers:
  - `fast` tier (`<60s` total): smoke + high-value parity checks on tiny subsets.
  - `extended` tier (`~2-5 min`): richer parity metrics and backend cross-checks on small fixtures.
  - `nightly` tier: larger reference run and trend-report artifact generation.
- Parameter-range coverage to include:
  - scan params: `TR`, `FA`, frame spacing (short/long), `relaxivity`, `hematocrit`.
  - timing controls: `start_t/end_t`, `steady_state_*`, auto vs manual injection window.
  - model fit controls: primary bounds/init/tolerances (`tofts`, `ex_tofts`, `patlak`).
- Input-method coverage matrix:
  - JSON sidecar resolution via each supported path:
    - `time_resolution_sec`
    - `TemporalResolution`
    - `RepetitionTime` + `RepetitionTimeExcitation`
    - `AcquisitionDuration`
    - `TriggerDelayTime/n_reps`
  - explicit CLI/config overrides (`--set ...`) with precedence checks.
  - expected hard-fail behavior for missing required scan parameters.
- Proposed required checks:
  - Stage-B contract metrics (`Cp_use`, `step`, baseline summary) against locked fixture references.
  - CPU vs MATLAB parity metrics on required maps for primary models.
  - CPU vs CPUfit equivalence thresholds on locked Stage-B checkpoint payloads.
  - deterministic file-discovery assertions (AIF ROI and DCE-space brain mask selection).
- Proposed implementation sequence:
  - Phase 1: add fixtures + Stage-B contract test + input-method matrix test.
  - Phase 2: add CPU-vs-CPUfit checkpoint test and wire into extended parity runner.
  - Phase 3: add CI split (`fast` on PR, `extended/nightly` scheduled) with JSON trend artifacts.

## Tabled: Patlak/GPUfit non-identifiability at a parameter bound (2026-07-17)

Found while building the Stage-D fit-backend consolidation
(`docs/project-management/projects/archived/stage-d-fit-consolidation/`, archived
2026-07-22 -- all five models, including tofts/ex_tofts/tissue_uptake/2cxm, finished
migrating); likely the same class of issue as "Regression on GPU-accelerated backend
behavior observed in qualification test" above. That migration is now fully done and
this issue is still present (reconfirmed 2026-07-22, see the Update section below) --
the mitigation below is the actual remaining work, not blocked on anything else anymore.

**Symptom:** `patlak_ktrans_brain_auto_vs_cpu` / `_auto_vs_matlab`
(`tests/python/test_dce_pipeline_parity_metrics.py::test_bbb_p19_region_parity`, model
`patlak`, region `brain`, `sub-10bbbdownsample` fixture) collapses to corr ~-0.007,
despite `gm`/`wm` regions on the exact same fixture already being perfect (corr=1.0).

**Root cause chain (fully isolated, not guessed):**
1. Switching Python's steady-state window from a hardcoded `[1,2]` test override to
   MATLAB-matching auto-detection (a correct, intentional fix) widened the true Ktrans
   range in this fixture's 237-voxel sparse sample from ~0.014 max to ~0.51 max. Verified
   by isolating steady-state-auto vs injection-timing-auto independently -- steady-state
   alone reproduces the full regression; injection-timing alone does not. Full isolation
   table + implicated commits (`3c17ff3...` -> `66fd795...`) are in the consolidation
   plan's Motivation section.
2. The widened range exposed a real architectural bug: patlak's accelerated (gpufit)
   fit had zero per-voxel seeding (one fixed, data-blind `initial_value_ktrans` for every
   voxel) and no multi-start, unlike the CPU path (seeded per-voxel from the closed-form
   linear-Patlak estimate). Fixed by the Stage-D consolidation's patlak pilot
   (`python/dce_fit_backends.py`): both backends now seed each voxel from the same
   linear estimate, expanded into x1/x10/x100 candidates.
3. That fix resolved the gap for the overwhelming majority of voxels, but **one single
   voxel** (out of 237) still fully explains the residual near-zero correlation --
   Pearson correlation over a small, tightly-clustered-near-zero sample is extremely
   sensitive to one high-leverage outlier.
4. Deep-dived that one voxel directly (captured the exact per-voxel candidates/results
   from a live pipeline run): its linear-regression seed is itself degenerate
   (ktrans0=-0.637, vp0=15.39 -- vp's upper bound is 1.0), so the x1/x10/x100 multiplier
   strategy gives **zero effective diversity** here (all three candidates collapse to
   the same bounds-clipped starting point on both backends).
5. vp saturates its upper bound (1.0) on both backends regardless of candidate. Once vp
   is pinned there, CPU (float64 scipy `trf`) converges to Ktrans=0.512261 with
   SSE=8339.5; gpufit (float32) converges to Ktrans=0.0 with chi-square=10231.9.
   **CPU's objective is objectively lower/better, not merely a different-but-equally-
   valid point on a flat manifold** -- gpufit is landing in a genuinely worse local
   optimum near this boundary.
6. Ruled out iteration/tolerance budget as the cause: rerunning with
   `gpu_max_n_iterations=2000` and `gpu_tolerance=1e-10` (vs. defaults 200/1e-6, a
   10x/10,000x increase) changed nothing -- gpufit reports `state=0` (converged) well
   before that budget, so more budget can't help; the solver believes it's done.

**Open questions for whoever picks this back up:**
- Does GPUfit/CPUfit's internal LM step-acceptance/convergence check behave differently
  in float32 near a bound vs. scipy's float64 `trf`? (Most likely explanation, not yet
  confirmed against the library internals.)
- Should candidate assembly clamp/reject an out-of-bounds or sign-flipped linear seed
  before building the x1/x10/x100 multipliers, so a degenerate seed doesn't silently
  collapse to zero diversity?
- Should a voxel where a parameter lands on its bound automatically escalate to the
  random-log-uniform multi-start (`2cxm`/`tissue_uptake`'s current rescue mechanism)
  rather than the fixed-multiplier strategy?

**Current mitigation plan (not yet implemented):** a GM/WM-style gating exception for
patlak+`brain` in the parity test (matching the existing tofts+`gm` precedent already in
`test_bbb_p19_region_parity`), since this is a non-identifiability/backend-precision
issue, not a fitter bug to chase further right now.

See also: `parity-whole-brain-roi-noise` and `parity-backend-divergence` memory notes.

## Update (2026-07-22): tv-default steady-state rollout re-triggers this, plus new residuals

From the (now-archived) `steady-state-tv-default` initiative
(`docs/project-management/projects/archived/steady-state-tv-default/STATUS.md`): a new
on-demand diagnostic (`tests/python/run_baseline_end_reliability.py`) run against 224 real
AIFArtist-rated sessions showed the `tv` steady-state-end detector far outperforming
MATLAB's current default (`piecewise_constant`/`find_end_ss`): ~88-93% accuracy vs. ~0%.
`tv` was ported to MATLAB (`dce/find_end_ss_tv.m`, verified numerically identical to
Python's `_tv_baseline_end` on 6 curves including a real one -- permanent test
`tests/python/test_find_end_ss_tv_matlab_parity.py`) and wired as both languages' default.
Two unrelated, pre-existing MATLAB bugs in `FXLfit_generic.m`'s Patlak/GPU branch were
found and fixed along the way (dead "ANIMAL study" ID-matching code referencing an
undefined `ids` variable; a `constraint_type`/`constraint_types` typo) -- both confirmed
independent of the steady-state change by reproducing them with the *old* algorithm first.

**As of this note, none of the above is committed** (working-tree only, `dev` branch).

Regenerating `sub-10bbbdownsample`'s matlabref maps under the new `tv` window and rerunning
`pytest -m parity` surfaced:

1. **`patlak_ktrans_brain` now fails on the *gated* `_cpu_vs_matlab` (corr=0.032) and
   `_auto_vs_matlab` (corr=0.878) checks, not just the previously-known ungated
   `_auto_vs_cpu` check documented above.** This is very likely the *same* tabled
   single-voxel non-identifiability issue, just pushed over the gating threshold by the
   window shift -- GM/WM still pass fine (as they always have, since they exclude the
   problem voxel), and nothing above about root cause or mitigation needs revisiting.
   **This is the fastest lead for whoever picks this up**: apply the already-decided GM/WM-
   style gating exception for patlak+brain (still not implemented) and rerun.
2. **New, not yet root-caused:** `tofts_roi_xls` (`mae=0.032908, max_abs_err=0.105665`,
   limit `0.03`) and `tissue_uptake_roi_xls` (`mae=0.029032, max_abs_err=0.118740`, limit
   `0.05`) now fail under `test_bbb_p19_roi_xls_parity`; `ex_tofts_roi_xls` and
   `patlak_roi_xls` both still pass. No prior writeup covers these two -- start by checking
   whether the `tv`-derived window shifted the ROI-mean baseline/injection timing enough to
   matter for these two models specifically (both fit from an ROI-averaged curve, unlike
   the per-voxel check above).

Full repro commands (matlabref regeneration, the exact test invocations) are in the
archived `steady-state-tv-default/STATUS.md`.

## Testing Gap Analysis
The CPU-vs-CPUfit divergence and weighted-AIF side effects were not caught early because:
- Existing parity gates focus on final map parity and do not separately gate Stage-B AIF-fit outputs (`Cp_use`) as a first-class contract.
- No required regression test enforces backend-equivalence (`cpu` vs `cpufit_cpu`) on a real-data checkpoint payload for primary models.
- Multi-model parity checks are sensitive to sparse-ROI sampling; we did not have a secondary dense-ROI cross-check requirement to disambiguate true model drift vs sparse-mask instability when changing AIF fitting.

## Plan To Close Testing Gaps
### A. Add Stage-B AIF contract gate (new)
- Add a regression test that compares Python Stage-B outputs against a locked reference payload for:
  - `Cp_use`, `step`, `baseline`, `max_index`
- Include two test modes:
  - default production mode (current weighted fit)
  - optional compatibility mode (if introduced)
- Gate on quantitative bounds (MAE/corr), not just pass/fail existence.

### B. Add required backend-equivalence check on real-data checkpoint payload (new)
- Add a test using frozen Stage-B arrays (`Ct`, `Cp_use`, `timer`) from `RUNNER_DATA`-derived fixture.
- Required check for primary models (`patlak`, `tofts`, `ex_tofts`):
  - CPU vs CPUfit map metrics (`corr`, `slope`, `mae`) under explicit thresholds.
- If `pycpufit` unavailable, mark skipped with explicit reason.

### C. Strengthen multi-model parity robustness
- Keep current sparse-ROI suite, and add dense-ROI companion metrics for required models.
- On required-failure events, emit both sparse and dense diagnostics to reduce false confidence and speed root-cause analysis.

### D. CI integration
- Add a dedicated parity diagnostic job (non-nightly optional, nightly required) that runs:
  1. `run_dce_parity.py -s multi-model`
  2. new Stage-B AIF contract test
  3. new CPU-vs-CPUfit checkpoint equivalence test
- Archive summary JSON artifacts for trend comparison.

