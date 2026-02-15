# Test Roadmap

## Completed
- Core MATLAB unit test scaffold for DCE/DSC/parametric algorithms.
- Parity contract definitions and tolerance profiles.
- MATLAB parity baseline export script for deterministic synthetic fixtures.
- CI wiring for unit/integration plus PR synthetic smoke checks.
- Python parity runner and baseline comparison tooling.
- Initial Python ports for DCE/DSC helper/parametric core models.
- Expanded DCE forward/inverse parity coverage through VP, tissue uptake, 2CXM, and FXR models.
- Python in-memory DCE CLI pipeline (`A -> B -> D`) with optional stage checkpoints.
- Real Stage A/B/D implementation paths in Python with QC figure output and `.xls` ROI exports.
- Dataset-backed DCE parity fixtures and tests:
  - downsampled `BBB data p19` committed fixture
  - Tofts `Ktrans` and `ve` parity metrics (corr/MSE) against MATLAB baselines
- CI Python checks (unit + contract parity + downsample DCE pipeline parity).
- Tiny fast fixture + settings matrix tests for fit constraints, initial guesses, and blood T1 overrides.
- MATLAB `dce_preferences.txt` bridge in Python defaults with override precedence and CLI exposure.
- Flattened Python module layout (`python/rocketship/*` moved to `python/*`).
- Added Python DCE GUI v1 (PySide6) as optional frontend over CLI.
- Added CLI stdout progress events + JSONL event log output.

## Scope guardrails for Python CLI port
- Primary target: DCE parts `A`, `B`, and `D` as CLI workflow.
- Default runtime shape: single-process, end-to-end in-memory pipeline.
- Explicitly excluded/deprecated:
  - `neuroecon` execution path.
  - GUI batch queue/prep flow for part D.
  - Email completion notification flow.
  - Manual click-based AIF tools.
  - ImageJ ROI input support (`.roi`).
  - Legacy MATLAB GUI helper flows tied to queue/prep behavior.
  - MATLAB-specific batch helper scripts.
- Explicitly retained:
  - ROI spreadsheet outputs (`.xls`).
  - QC figure saving.

## Near-term next steps
1. Complete MATLAB option parity audit at script level:
   - map `script_preferences.txt` keys to Python config/stage overrides
   - mark each key as supported, intentionally dropped, or pending
   - add targeted tests for newly-wired option families
2. Broaden dataset-backed DCE regression beyond current Tofts map checks:
   - ROI `.xls` parity checks
   - additional map/model parity (for example `patlak`, `ex_tofts`, `tissue_uptake`)
3. Expand tiny edge-case fixture variants:
   - low-SNR variant
   - non-uniform timer variant
   - harsh-bounds/low-iteration stress variant
4. Decide whether to port currently unsupported DCE model branches:
   - `nested`
   - `FXL_rr`
5. Close next DSC parity gap:
   - add baseline + contracts for `DSC_convolution_oSVD`
   - port `DSC_convolution_oSVD` and compare against baseline
6. Performance pass after parity lock:
   - investigate Python-vs-MATLAB runtime gap (`~2x-4x` on full DCE runs)
   - profile and optimize Python Stage D hot spots
   - evaluate GPUfit CPU backend options as an additional fast path
