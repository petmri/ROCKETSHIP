# Test Roadmap

## Completed in this initial pass
- Core MATLAB unit test scaffold for DCE/DSC/parametric algorithms.
- Parity contract definitions and tolerance profiles.
- MATLAB parity baseline export script for deterministic synthetic fixtures.
- CI wiring for unit/integration plus PR synthetic smoke checks.
- Python parity runner and baseline comparison tooling.
- Initial Python ports for DCE/DSC helper/parametric core models.
- MATLAB baseline + Python parity coverage for DSC `ssvd_deconvolution`.
- Expanded DCE forward parity coverage (`model_vp_cfit`, `model_tissue_uptake_cfit`, `model_2cxm_cfit`).
- Expanded DCE inverse parity coverage (`model_vp`, `model_tissue_uptake`, `model_2cxm`).
- Expanded DCE FXR parity coverage (`model_fxr_cfit`, `model_fxr`).
- Added Python in-memory DCE CLI scaffold (`A -> B -> D`) with optional stage checkpoints.
- Added initial real Stage-A implementation path with QC figure generation in Python CLI pipeline.
- Added initial real Stage-B implementation path with non-interactive AIF modes and QC figure generation.
- Added initial real Stage-D implementation path with voxel/ROI model fitting, map outputs, and `.xls` ROI tables.

## Scope guardrails for Python CLI port
- Primary target: DCE parts `A`, `B`, and `D` as CLI workflow.
- Default runtime shape: single-process, end-to-end in-memory pipeline.
- Explicitly excluded/deprecated:
  - `neuroecon` execution path.
  - GUI batch queue/prep flow for part D.
  - Email completion notification flow.
  - Manual click-based AIF tools.
  - ImageJ ROI input support (`.roi`).
  - GUI entrypoints/UI helpers.
  - MATLAB-specific batch helper scripts.
- Explicitly retained:
  - ROI spreadsheet outputs (`.xls`).

## Implementation notes for parity-safe CLI
- Keep A/B/D as logical stages but pass stage outputs in memory.
- Add optional checkpoint export (A_out/B_out/D_out) only for parity/debug workflows.
- Keep CPU fitting as the canonical baseline path.
- Add GPUfit as optional backend once installed; compare against CPU with backend-aware tolerances.

## Next expansion steps
1. Add fixture generators for edge cases:
   - low-SNR synthetic curves
   - non-uniform time vectors
   - boundary parameter values (near lower/upper fit limits)
2. Add ROI/map regression tests:
   - compare summary metrics (mean/median/p95) on `test_data/BBB data p19`
   - compare voxel-map hashes with robust tolerances
3. Add pipeline integration tests:
   - scripted A->B->D DCE checks with assertions on numerical outputs
   - non-interactive DSC workflow checks for both sSVD and oSVD paths
4. Add dataset-backed DCE regression:
   - map summary/regression checks on `test_data`
   - tighter expected tolerances for end-to-end A->B->D outputs
5. Decide whether to port currently unsupported DCE model branches:
   - `nested`
   - `FXL_rr`
6. Close next DSC parity gap:
   - add baseline + contracts for `DSC_convolution_oSVD`
   - port `DSC_convolution_oSVD` and compare against baseline
7. Grow real datasets as needed:
   - include at least one additional acquisition profile (different TR/FA/time resolution)
   - include one dataset with known challenging AIF behavior
