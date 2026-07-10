# ROCKETSHIP Test Suite (Algorithm-Focused)

Scoped to **core algorithms** (MATLAB and the Python port); GUI behavior is intentionally out of scope.
All commands below assume you are at the repo root and using the project venv (`.venv/bin/python`),
shown here as `pytest` for brevity.

## How do I run …?

| Goal | Command |
|---|---|
| Default Python suite (incl. gated DCE parity) | `pytest tests/python` |
| DCE parity, all models (reported extras) | `pytest tests/python -m parity --parity-suite=allmodels -s` |
| Runtime parity vs MATLAB (needs MATLAB) | `pytest tests/python/test_runtime_parity.py --run-runtime-parity` |
| OSIPI reliability | `pytest tests/python -m osipi -v` (add `--osipi-slow` for long fits) |
| BIDS qualification | `pytest tests/python --run-qualification` |
| MATLAB unit tests | `run_unit_tests()` in MATLAB |
| Coverage | `pytest tests/python -q --cov=python --cov-report=term-missing --cov-fail-under=60` |

## Layout
- `tests/matlab/{unit,integration,helpers}/`: MATLAB algorithm tests, fixtures, shared helpers.
- `tests/contracts/`, `tests/contracts/baselines/`: cross-language parity contracts and generated MATLAB baselines.
- `tests/python/`: Python pytest suite (pipeline, parity, OSIPI, qualification).
- `tests/data/`: fixtures. `ci_fixtures/` are committed lightweight fixtures used by CI (no per-run generation).
- `tests/data/osipi/`: imported OSIPI datasets + provenance + peer-result tolerances (see that dir's `README.md`).

## DCE Python↔MATLAB parity

The parity suite compares the Python pipeline against committed MATLAB baseline maps. It is organized by a
single **`--parity-suite`** selector and split into **gated** vs **reported-only** checks.

- **`--parity-suite=standard`** (default; runs on a plain `pytest`): gates **Tofts & Patlak, Ktrans only**,
  Python-vs-MATLAB (cpu & auto), on **RMSE and Corr**.
- **`--parity-suite=allmodels`**: additionally runs **ex_tofts, tissue_uptake, 2cxm** as **reported-only**
  diagnostics (never gated — they are not identifiable on this fixture).
- **`--parity-suite=all`**: union of the above.

**Regions and the gated set.** Each model/param is evaluated over three ROIs — whole **brain** (sparse),
**GM**, and **WM**. Patlak Ktrans gates on all three. Tofts Ktrans gates on **brain + WM only**; **tofts-GM
is reported-only** because Tofts Ktrans is non-identifiable in that GM patch (flat objective along Ktrans;
Python's fit is equal-or-better than MATLAB's by SSE — see `docs/parity-testing-improvement-plan.md`).
Non-Ktrans params (ve/vp/fp) and backend-consistency (auto-vs-cpu) are always reported, never gated.

**Reported metrics.** Every check logs `corr` and `rmse`; each Python-vs-MATLAB parameter check also
logs CI-aware diagnostics — **`ci_norm_absdiff_p95`** (p95 of `|py−matlab| / CI-width`, both sides are
95% CI) and **proportion outside the CI**. A full summary JSON is written to `--parity-summary-dir`.

`test_bbb_p19_region_parity` replaced the former per-scenario voxelwise parity tests
(`*_tofts_ktrans`, `*_primary_models_ktrans_cpu`). Gated checks whose masks collapse to `<2` valid
voxels **fail** (a collapse is a silent hole, not a pass), and the suite asserts at least one gated
check compared real data.

**ROI-summary `.xls` parity** is a separate opt-in check, `test_bbb_p19_roi_xls_parity`:

```bash
pytest tests/python/test_dce_pipeline_parity_metrics.py::test_bbb_p19_roi_xls_parity --run-parity
```

MATLAB averages each parameter's concentration curve over the whole-brain ROI and fits once
(average-then-fit). Python reproduces this exactly via the pipeline's **ROI-only mode**
(`stage_overrides.fit_voxels=0`), which skips the per-voxel fit — so the check runs in a few seconds
and matches MATLAB's tables within tolerance. See `docs/dce_options.md` for `fit_voxels`.

### Thresholds

Gate thresholds default to `tests/python/parity_thresholds_default.json`. Override with a copy:

```bash
pytest tests/python -m parity --parity-thresholds path/to/my_thresholds.json
```

Only the keys you include are overlaid. The standard gate uses `model_ktrans_corr_min` and
`model_ktrans_mse_max` (RMSE gate = `sqrt(model_ktrans_mse_max)`). The individual `--parity-*-corr-min` /
`--parity-*-mse-max` CLI flags still work but are secondary to the JSON.

### Deprecated flags

`--run-multi-model-backend-parity` / `--mm-parity`, `--parity-required-models`, `--parity-cpu-optional-models`,
and `--parity-require-all-models` / `--all-models` are superseded by `--parity-suite` (the gated/reported
split is now fixed in code). They still work as aliases; migration of CI to `--parity-suite` is tracked in
`docs/project-management/PORTING_STATUS.md`.

## Other Python test groups

**Noisy-data parity (function level, default-on):** compares the Python fit of a stored noisy curve against
MATLAB's fit of the same curve, gated per-parameter on identifiability.

```bash
pytest tests/python/test_dce_noisy_parity.py -v
```

**End-to-end T1 map parity:** generate the MATLAB reference then compare (identifiable voxels only).

```bash
matlab -batch "addpath('tests/matlab'); addpath('tests/matlab/helpers'); \
  generate_t1_parity_map('vfaFiles', {'tests/data/ci_fixtures/t1/vfa_small/flip-02deg_VFA.nii.gz', \
   'tests/data/ci_fixtures/t1/vfa_small/flip-05deg_VFA.nii.gz', \
   'tests/data/ci_fixtures/t1/vfa_small/flip-10deg_VFA.nii.gz'}, 'flipAngles', [2 5 10], \
  'trMs', 8.012, 'fitType', 't1_fa_fit', \
  'outputPath', 'tests/data/ci_fixtures/t1/vfa_small/results_matlab/T1_map_t1_fa_fit.nii', 'rsquaredThreshold', 0);"
pytest tests/python/test_t1_map_parity.py --run-parity
```

**OSIPI reliability** (ground-truth correctness against published peer tolerances):

```bash
pytest tests/python -m osipi -v                 # all; add --osipi-slow for long fits
pytest tests/python/test_osipi_backend_consistency.py -v   # cpu vs cpufit/gpufit
pytest tests/python/test_osipi_pycpufit.py tests/python/test_osipi_pygpufit.py -m fast -v
python tests/python/run_osipi_reliability.py --suite all --summary-json /tmp/osipi_summary.json
```

**BIDS discovery and qualification:**

```bash
python run_bids_discovery.py --bids-root tests/data/BIDS_test --output-json out/bids_manifest.json --print-json
python run_python_qualification.py --bids-root tests/data/BIDS_test \
  --output-root out/python_qualification_bids_test --backend cpu --print-summary-json
```

**Synthetic phantom GT reliability (diagnostic, not a merge gate yet):**

```bash
python tests/python/run_phantom_gt_reliability.py --backend auto [--subject sub-08phantom]
```

The phantom tolerance profile (`tests/data/BIDS_test/phantom_gt_mae_tolerances.json`) is provisional;
`test_phantom_gt_reliability.py` is qualification-gated and `xfail`s when `gate_ready=false`. See
`docs/project-management/projects/phantom-gt/PHANTOM_GT_QUALIFICATION_STATUS.md`.

## Helpers

```bash
python tests/python/run_dce_parity.py --suite multi-model      # prints parity summary metrics
python tests/python/run_dce_benchmark.py                       # benchmarks
python tests/python/run_dce_postfit_analysis.py --analysis ftest --region roi \
  --result lower_model_fit_postfit_arrays.npz --result higher_model_fit_postfit_arrays.npz \
  --output-dir /tmp/dce_postfit_ftest --print-summary-json
```

Generate Part E NPZ inputs from Stage D with `stage_overrides.write_postfit_arrays=true`.

## MATLAB tests and baselines

```matlab
results = run_unit_tests();
results = run_all_tests('suite', 'all', 'includeIntegration', true);
baseline = export_parity_baseline();          % writes tests/contracts/baselines/matlab_reference_v1.{mat,json}
manifest = generate_synthetic_datasets();     % deterministic synthetic BIDS-like fixtures
```

Regenerate the downsampled BBB p19 DCE parity fixture and its MATLAB baseline maps:

```bash
python tests/data/scripts/generate_bbb_p19_downsample.py --clean --factor-x 3 --factor-y 3
matlab -batch "addpath('tests/matlab'); generate_dce_tofts_parity_map('subjectRoot', \
  'tests/data/ci_fixtures/dce/bbb_p19_downsample_x3y3', 'models', {'tofts', 'patlak'});"
```
