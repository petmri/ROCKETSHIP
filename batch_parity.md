# Batch Parity Status (MATLAB vs Python DCE)

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

### Confirmed not aligned
- Accelerated Patlak backend (`cpufit_cpu`) is still session-dependent on real data:
  - acceptable in `ses-01`, large drift in `ses-02`.
- Stage-B weighted AIF fit update currently regresses one required multi-model parity check (below).

## Key Diagnostics and Artifacts
- Stage-D diagnostics runner:
  - `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/run_batch_stage_d_diagnostics.py`
- Diagnostics outputs:
  - `/Users/samuelbarnes/code/ROCKETSHIP/out/batch_stage_d_diagnostics_aifw2_10000_20260303.json`
  - `/Users/samuelbarnes/code/ROCKETSHIP/out/batch_stage_d_diagnostics_aifw2_10000_patlakfix_20260303.json`
- Patlak cpufit handoff package:
  - `/Users/samuelbarnes/code/ROCKETSHIP/tests/contracts/handoffs/cpufit_patlak_batch/`

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

3. Upstream cpufit Patlak investigation remains open.
- Continue handoff with reproducible payloads in `tests/contracts/handoffs/cpufit_patlak_batch/`.

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

## Practical Guidance Until Fixes Land
- For parity-critical comparisons, prefer `backend=none` (CPU path) when evaluating against MATLAB references.
- Treat accelerated Patlak (`cpufit_cpu`) outputs as non-authoritative on real-data parity until upstream fix is validated.
