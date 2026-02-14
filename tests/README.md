# ROCKETSHIP Test Suite (Algorithm-Focused)

This test suite is scoped to **core algorithms** and intentionally avoids GUI behavior.

## Goals
- Verify numerical correctness of core MATLAB algorithms.
- Build reusable parity contracts so future Python ports can be validated against MATLAB reference behavior.
- Support both synthetic fixtures and curated real fixtures from `test_data`.

## Layout
- `tests/matlab/unit/`: fast deterministic unit tests for core DCE/DSC/parametric algorithms.
- `tests/matlab/integration/`: fixture integrity and heavier workflow checks.
- `tests/matlab/helpers/`: shared MATLAB helpers for path setup, fixtures, and assertions.
- `tests/contracts/`: cross-language parity contracts and tolerance profiles.
- `tests/contracts/baselines/`: generated MATLAB baseline outputs used by future Python parity checks.

## Running MATLAB tests
From MATLAB:

```matlab
results = run_unit_tests();
```

or

```matlab
results = run_all_tests('suite', 'all', 'includeIntegration', true);
```

## Generating parity baselines
Generate canonical MATLAB outputs for synthetic fixtures:

```matlab
baseline = export_parity_baseline();
```

This writes:
- `tests/contracts/baselines/matlab_reference_v1.mat`
- `tests/contracts/baselines/matlab_reference_v1.json`

These files are intended for direct numerical comparison when the Python implementation is introduced.

## Generating synthetic datasets
Create deterministic synthetic BIDS-like fixtures derived from `test_data/BIDS_test`:

```matlab
manifest = generate_synthetic_datasets();
```

Default output:
- `test_data/synthetic/generated/noisy_low`
- `test_data/synthetic/generated/noisy_high`
- `test_data/synthetic/generated/downsample_x2`
- `test_data/synthetic/generated/bolus_delay`

You can also generate into a temp directory:

```matlab
manifest = generate_synthetic_datasets('outputRoot', fullfile(tempdir, 'rocketship_synth'));
```
