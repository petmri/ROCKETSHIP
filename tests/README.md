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
| OSIPI reliability | `pytest tests/python -m osipi -v` (runs the full 2CXM/2CUM sweeps by default) |
| BIDS qualification | `pytest tests/python --run-qualification` |
| MATLAB unit tests | `run_unit_tests()` in MATLAB |
| Coverage | `pytest tests/python -q --cov=python --cov-report=term-missing --cov-fail-under=60` |

## Layout
- `tests/matlab/{unit,integration,helpers}/`: MATLAB algorithm tests, fixtures, shared helpers.
- `tests/contracts/`, `tests/contracts/baselines/`: cross-language parity contracts and generated MATLAB baselines.
- `tests/python/`: Python pytest suite (pipeline, parity, OSIPI, qualification).
- `tests/data/`: fixtures. `BIDS_test/` holds the committed lightweight fixtures used by CI (no per-run generation), including the `sub-10bbbdownsample` / `sub-11tiny` fit-parity subjects.
- `tests/data/osipi/`: imported OSIPI datasets + provenance + peer-result tolerances (see that dir's `README.md`).

## DCE Python↔MATLAB parity

The parity suite compares the Python pipeline against committed MATLAB baseline maps. It is organized by a
single **`--parity-suite`** selector and split into **gated** vs **reported-only** checks.

- **`--parity-suite=standard`** (default; runs on a plain `pytest`): gates **Tofts & Patlak, Ktrans only**,
  Python-vs-MATLAB (cpu & auto), on **RMSE and Spearman (rank) correlation**.
- **`--parity-suite=allmodels`**: additionally runs **ex_tofts, tissue_uptake, 2cxm** as **reported-only**
  diagnostics (never gated — they are not identifiable on this fixture).
- **`--parity-suite=all`**: union of the above.

**Regions and the gated set.** Each model/param is evaluated over three ROIs — whole **brain** (sparse),
**GM**, and **WM**. Both gated models (tofts, patlak) gate Ktrans on all three regions: **12 gated checks,
no exceptions.** The former tofts-GM exception (reported-only, on the grounds that Tofts Ktrans was
non-identifiable in that GM patch) was retired 2026-07-28 — the Stage-B AIF fix lifted tofts-GM to
`corr` 0.980 (cpu) / 0.992 (auto) against a 0.95 floor. Non-Ktrans params (ve/vp/fp) and
backend-consistency (auto-vs-cpu) are always reported, never gated.

**Reported metrics.** Every check logs `corr` (Spearman rank correlation, not Pearson — robust to the
single high-leverage/non-identifiable voxel that can otherwise dominate a sum-of-products statistic;
see `docs/project-management/projects/archived/batch-parity/batch_parity.md`) and `rmse`; each Python-vs-MATLAB
parameter check also logs CI-aware diagnostics — **`ci_norm_absdiff_p95`** (p95 of `|py−matlab| /
CI-width`, both sides are 95% CI) and **proportion outside the CI**. These CI-aware fields are
reported-only, never gated. They were non-functional until 2026-07-23 (the MATLAB reference had been
regenerated on GPU, and gpufit zero-pads CI columns); the reference is CPU-generated again and the
MATLAB-side fields are live. The Python-side `prop_matlab_outside_py_ci` is still `NaN` on `auto`
runs for the same underlying reason — gpufit produces no CIs — and now reports that honestly via
`n_zero_py_ci_width` rather than as a spurious 100% disagreement. See `_ci_metrics()`.
A full summary JSON is written to `--parity-summary-dir`.

`test_bbb_p19_region_parity` replaced the former per-scenario voxelwise parity tests
(`*_tofts_ktrans`, `*_primary_models_ktrans_cpu`). Gated checks whose masks collapse to `<2` valid
voxels **fail** (a collapse is a silent hole, not a pass), and the suite asserts at least one gated
check compared real data.

**ROI-summary `.xls` parity** is a separate check, `test_bbb_p19_roi_xls_parity` (default-on):

```bash
pytest tests/python/test_dce_pipeline_parity_metrics.py::test_bbb_p19_roi_xls_parity
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

**End-to-end T1 map parity** (default-on; compares against the committed MATLAB reference over
identifiable voxels). Regenerate the reference only when the MATLAB T1 algorithm changes:

```bash
matlab -batch "addpath('tests/matlab'); addpath('tests/matlab/helpers'); \
  generate_t1_parity_map('vfaFiles', {'tests/data/BIDS_test/rawdata/sub-11tiny/ses-01/anat/sub-11tiny_ses-01_flip-01_VFA.nii.gz', \
   'tests/data/BIDS_test/rawdata/sub-11tiny/ses-01/anat/sub-11tiny_ses-01_flip-02_VFA.nii.gz', \
   'tests/data/BIDS_test/rawdata/sub-11tiny/ses-01/anat/sub-11tiny_ses-01_flip-03_VFA.nii.gz'}, 'flipAngles', [2 5 10], \
  'trMs', 8.012, 'fitType', 't1_fa_fit', \
  'outputPath', 'tests/data/BIDS_test/derivatives/matlabref/sub-11tiny/ses-01/anat/sub-11tiny_ses-01_desc-t1fafit_T1map.nii', 'rsquaredThreshold', 0);"
pytest tests/python/test_t1_map_parity.py
```

**OSIPI reliability** (ground-truth correctness against published peer tolerances):

```bash
pytest tests/python -m osipi -v                 # all OSIPI checks (incl. full 2CXM/2CUM sweeps)
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
python tests/python/run_baseline_end_reliability.py \
  --derivatives-root tests/data/BIDS_test/derivatives --raw-root tests/data/BIDS_test/rawdata \
  --output-dir out/baseline_end_reliability   # end-baseline detector accuracy vs AIFArtist-rated GT
```

`run_baseline_end_reliability.py` compares five detectors: `piecewise_constant`, `legacy_sobel`,
`glr`, `tv`, and `biexp_fit` (the production default). `biexp_fit` reads its `aif_*` settings from
`--config-template` (default `python/dce_default.json`) so the harness measures the configuration
that actually ships; the other four are pure functions of the signal curve and ignore it. Because
it is a model fit rather than a shape heuristic it also reports a fractional injection end
(`t0_exp`) and a fitted curve — both drawn on the per-session figures, with the extra diagnostics
in `per_session_details.csv`. It is seeded from `tv` and falls back to it, so check the
`biexp_fit outcome breakdown` in the summary before reading its accuracy row as independent.

For a dataset that was never rated in AIFArtist there is no `SteadyStateEndTimeIndex` to score
against, so pass `--no-ground-truth` to discover AIF masks by filename instead. Everything still
runs; the accuracy table is replaced by a cross-detector agreement table. When `--raw-root` points
at a derivatives tree rather than a raw one, also pass `--dynamic-pattern` so the detectors see the
series production fits — otherwise the one-dynamic-per-session heuristic picks alphabetically among
`desc-bfc`, `desc-hmc`, `desc-biases`, and friends:

```bash
python tests/python/run_baseline_end_reliability.py --no-ground-truth \
  --derivatives-root RUNNER_DATA/derivatives/dceprep-python \
  --raw-root RUNNER_DATA/derivatives/dceprep-python \
  --dynamic-pattern '*desc-bfcz_DCE.nii*' \
  --output-dir out/baseline_end_reliability_runner
```

Generate Part E NPZ inputs from Stage D with `stage_overrides.write_postfit_arrays=true`.

## MATLAB tests and baselines

```matlab
results = run_unit_tests();
results = run_all_tests('suite', 'all', 'includeIntegration', true);
baseline = export_parity_baseline();          % writes tests/contracts/baselines/matlab_reference_v1.{mat,json}
manifest = generate_synthetic_datasets();     % deterministic synthetic BIDS-like fixtures
```

Regenerate the downsampled BBB p19 DCE parity fixture and its MATLAB baseline maps. Include
`roiList` (whole-brain mask) or the generator's `roi_list` stays empty and it silently skips
writing `Dyn-1_*_fit_rois.xls` — the ROI-xls baselines then go stale relative to the `.nii`
maps the very next time only this command (without `roiList`) is rerun after an algorithm
change (hit in practice: `find_end_ss` -> `find_end_ss_tv` migration regenerated the `.nii`
maps but left `.xls` on the old detector, breaking `test_bbb_p19_roi_xls_parity` for `tofts`/
`tissue_uptake` until backfilled).

When regenerating after a Stage-A/B timing change, also check that
`_make_config`'s `steady_state_auto_method` in `tests/python/test_dce_pipeline_parity_metrics.py`
still names the same detector `A_make_R1maps_func.m` calls (currently `find_end_ss_tv` /
`"tv"`). A mismatch there compares two different baseline-end algorithms and reads as
a model disagreement, the same trap the `startInjectionMin` note below describes. This box's `gpufit` mex is compiled and a GPU is present, so
also set `force_cpu = 1` in `dce/dce_preferences.txt` before running (revert to `0` after) to
match the CPU-path reference `test_bbb_p19_roi_xls_parity`/`test_bbb_p19_region_parity` gate
against — otherwise gpufit's CI zero-padding contaminates the regenerated maps (see `60c43da`).
Pass every model the parity suite reads (including `2cxm`) for the same reason the `roiList`
note exists: a partial regeneration leaves the omitted models' maps on the old settings.

Leave `startInjectionMin`/`endInjectionMin` at their `-1` (auto) defaults. The whole point of
the suite is that both pipelines run the *same* settings, and the Python fixture auto-detects
its injection window from Stage-A's steady state. Pinning the MATLAB side to fixed minutes
decouples the two Stage-B inputs — the fitted AIF's onset lands on a different frame — while
every other setting still matches, which reads as a model disagreement rather than a
configuration one:

```bash
python tests/data/scripts/generate_bbb_p19_downsample.py --clean --factor-x 3 --factor-y 3
S=tests/data/BIDS_test; sub=sub-10bbbdownsample; ses=ses-01
matlab -batch "addpath('tests/matlab'); generate_dce_tofts_parity_map( \
  'dynamicPath', '$S/rawdata/$sub/$ses/dce/${sub}_${ses}_DCE.nii', \
  'aifRoiPath', '$S/derivatives/$sub/$ses/dce/${sub}_${ses}_desc-AIFroi_mask.nii', \
  'brainRoiPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_desc-brain_mask.nii', \
  't1MapPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_space-DCEref_T1map.nii', \
  'noiseRoiPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_desc-noise_mask.nii', \
  'outputRoot', '$S/derivatives/matlabref/$sub/$ses/dce', \
  'roiList', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_desc-brain_mask.nii', \
  'models', {'tofts', 'ex_tofts', 'patlak', 'tissue_uptake', '2cxm'});"
```
