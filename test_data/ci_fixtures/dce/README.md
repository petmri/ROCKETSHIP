# DCE CI Fixtures

This folder contains committed lightweight fixtures used by CI to avoid
regenerating synthetic data on every run.

## Fixtures

- `downsample_x2_bids`
  - Source: `test_data/synthetic/generated/downsample_x2`
  - Used by MATLAB PR smoke run in `.github/workflows/run_DCE.yml`
  - Runtime path:
    - `rawdata/sub-01/ses-01/`
    - `derivatives/sub-01/ses-01/`

- `bbb_p19_downsample_x3y3`
  - Source: nearest-neighbor downsample of `test_data/BBB data p19`
  - Used by Python DCE pipeline parity test
  - Contains minimal files required by `tests/python/test_dce_pipeline_parity_metrics.py`:
    - `Dynamic_t1w.nii`
    - `processed/T1_AIF_roi.nii`
    - `processed/T1_brain_roi.nii`
    - `processed/T1_map_t1_fa_fit_fa10.nii`
    - `processed/T1_noise_roi.nii`
    - `processed/results_matlab/Dyn-1_tofts_fit_Ktrans.nii`

## Regeneration notes

- Rebuild `downsample_x2` source using MATLAB synthetic generator:
  - `generate_synthetic_datasets('outputRoot','test_data/synthetic/generated','clean',true);`
- Rebuild BBB downsample source using Python script:
  - `.venv/bin/python tests/python/generate_bbb_p19_downsample.py --clean --factor-x 3 --factor-y 3`
- Rebuild MATLAB Tofts baseline map:
  - `matlab -batch "cd('/Users/samuelbarnes/code/ROCKETSHIP'); addpath('tests/matlab'); generate_dce_tofts_parity_map('subjectRoot','/Users/samuelbarnes/code/ROCKETSHIP/test_data/synthetic/generated/bbb_p19_downsample_x3y3')"`
