# Test Roadmap

## Completed in this initial pass
- Core MATLAB unit test scaffold for DCE/DSC/parametric algorithms.
- Parity contract definitions and tolerance profiles.
- MATLAB parity baseline export script for deterministic synthetic fixtures.
- CI hook to run algorithm unit tests.

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
4. Define Python parity runner:
   - consume `tests/contracts/*.json`
   - compare Python outputs against `tests/contracts/baselines/matlab_reference_v1.*`
5. Grow real datasets as needed:
   - include at least one additional acquisition profile (different TR/FA/time resolution)
   - include one dataset with known challenging AIF behavior
