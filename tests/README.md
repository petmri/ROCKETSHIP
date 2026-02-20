# ROCKETSHIP Test Suite (Algorithm-Focused)

This test suite is scoped to **core algorithms** and intentionally avoids GUI behavior.

## Goals
- Verify numerical correctness of core MATLAB algorithms.
- Build reusable parity contracts so future Python ports can be validated against MATLAB reference behavior.
- Support both synthetic fixtures and curated real fixtures from `tests/data`.

## Layout
- `tests/matlab/unit/`: fast deterministic unit tests for core DCE/DSC/parametric algorithms.
- `tests/matlab/integration/`: fixture integrity and heavier workflow checks.
- `tests/matlab/helpers/`: shared MATLAB helpers for path setup, fixtures, and assertions.
- `tests/contracts/`: cross-language parity contracts and tolerance profiles.
- `tests/contracts/baselines/`: generated MATLAB baseline outputs used by future Python parity checks.
- `tests/data/osipi/`: imported OSIPI reference datasets, provenance docs, and peer-result tolerance summaries used by OSIPI-labeled Python tests.

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
Create deterministic synthetic BIDS-like fixtures derived from `tests/data/BIDS_test`:

```matlab
manifest = generate_synthetic_datasets();
```

Default output:
- `tests/data/synthetic/generated/noisy_low`
- `tests/data/synthetic/generated/noisy_high`
- `tests/data/synthetic/generated/downsample_x2`
- `tests/data/synthetic/generated/bolus_delay`

You can also generate into a temp directory:

```matlab
manifest = generate_synthetic_datasets('outputRoot', fullfile(tempdir, 'rocketship_synth'));
```

Generate a fast, nearest-neighbor downsampled `BBB data p19` fixture (`x3,y3`) for Python-vs-MATLAB DCE map parity checks:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python tests/data/scripts/generate_bbb_p19_downsample.py --clean --factor-x 3 --factor-y 3
```

CI uses committed lightweight fixtures (no per-run generation required):
- `tests/data/ci_fixtures/dce/downsample_x2_bids` (MATLAB PR smoke DCE run)
- `tests/data/ci_fixtures/dce/bbb_p19_downsample_x3y3` (Python DCE pipeline parity)

Generate MATLAB Tofts Ktrans parity baselines (`processed/results_matlab`) for both
downsampled and full-volume BBB datasets:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
matlab -batch "cd('/Users/samuelbarnes/code/ROCKETSHIP'); addpath('tests/matlab'); generate_dce_tofts_parity_map('subjectRoot','/Users/samuelbarnes/code/ROCKETSHIP/tests/data/synthetic/generated/bbb_p19_downsample_x3y3')"
matlab -batch "cd('/Users/samuelbarnes/code/ROCKETSHIP'); addpath('tests/matlab'); generate_dce_tofts_parity_map('subjectRoot','/Users/samuelbarnes/code/ROCKETSHIP/tests/data/BBB data p19')"
```

## MATLAB-vs-Python parity runner
Use the lightweight Python comparator in `/Users/samuelbarnes/code/ROCKETSHIP/tests/contracts/`:

```bash
python3 tests/contracts/compare_with_matlab_baseline.py \
  --write-template tests/contracts/python_results_template.json
```

Then compare Python outputs by contract:

```bash
python3 tests/contracts/compare_with_matlab_baseline.py \
  --python-results tests/contracts/python_results_template.json
```

Current Python ports are under `/Users/samuelbarnes/code/ROCKETSHIP/python/`.
