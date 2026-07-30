# ROCKETSHIP Test Suite (Algorithm-Focused)

Scoped to **core algorithms** (MATLAB and the Python port); GUI behavior is intentionally out of scope.
All commands below assume you are at the repo root and using the project venv (`.venv/bin/python`),
shown here as `pytest` for brevity.

## How do I run …?

| Goal | Command |
|---|---|
| Default Python suite (incl. gated DCE parity) | `pytest tests/python` |
| DCE parity, incl. reported-only extras | `pytest tests/python -m parity --parity-suite=allmodels -s` |
| Runtime parity vs MATLAB (needs MATLAB) | `pytest tests/python/test_runtime_parity.py --run-runtime-parity -s` |
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

- **`--parity-suite=standard`** (default; runs on a plain `pytest`): gates **Tofts, Patlak and ex-Tofts**,
  **every fitted parameter**, Python-vs-MATLAB (cpu & auto), on **RMSE and Spearman (rank) correlation**.
- **`--parity-suite=allmodels`**: additionally runs **tissue_uptake and 2cxm** as **reported-only**
  diagnostics.
- **`--parity-suite=all`**: union of the above.

**The gate policy is one rule with no exceptions** (reviewed 2026-07-29). Every parameter of every gated
model is gated, over all three ROIs — whole **brain** (sparse), **GM**, **WM** — against **one threshold
pair**, after **one identifiability filter**. That is **42 gated checks**: tofts (Ktrans, ve) + patlak
(Ktrans, vp) + ex_tofts (Ktrans, ve, vp) × 3 regions × {cpu, auto} vs MATLAB. There are no per-model,
per-parameter or per-region carve-outs.

**The identifiability filter.** A voxel counts only if **neither side left any of that model's compared
parameters sitting on a bound**. Against a bound the objective is flat, so two optimizers stop at
different points of one plateau and the disagreement measures the constraint rather than the port —
the same mechanism, and the same rule, as `test_backend_equivalence.py`. It replaced two hand-rolled
masks that each covered one corner of it (`ktrans_upper_exclude`, Ktrans near its 2.0 ceiling and only
for ex_tofts/2cxm; `ve_ktrans_min`, Ktrans at its 1e-7 floor and only for ve). Two measured consequences:

- **ex_tofts became gateable.** Its worst check went `0.807 → 0.9998` (Ktrans/GM/cpu) and `0.765 →
  0.9998` (auto). The partial masks, not the model, were why it read as "not identifiable on this fixture."
- **It removes padding, and can lower a number.** 60 of tofts' 229 brain voxels sat on `ve`'s 0.02 floor
  and agreed trivially (`corr` 0.9952), inflating tofts Ktrans/brain to 0.9751; the honest value on
  determined voxels is **0.9616**, which is now the tightest gated check.

A gated check whose identifiable subset drops below **25%** of QoF-passing voxels **fails** rather than
passing on a handful of voxels (observed minimum is 0.578, ex_tofts/brain).

**Verified by breaking it** (2026-07-29), since a gate is only worth its failures. Each of these was
injected into the Python maps and confirmed to fail the checks it should: a 40% patlak Ktrans scale
error (`corr` stays at exactly 1.000000 — the case only the normalized bound catches, and one the old
absolute bound passed by 30×); a shuffle of tofts Ktrans within the mask (`corr` 0.01, `nrmse` 1.16);
ex_tofts `ve` driven entirely onto its bound (mask collapse, 0 valid voxels); and 90% of patlak `vp`
pinned (identifiable fraction 0.091 < 0.25, the partial-collapse guard).

**Reported-only, with the measurement that puts them there.** `tissue_uptake` — `Fp` is not identifiable
at this fixture's 15.84 s frames (filtered `corr` 0.08 WM / 0.18 GM), and since the model fits
`E = Ktrans/Fp`, `Ktrans` and `vp` inherit it; this is the same root cause as the ROI-xls `Fp` exclusion
below. `2cxm` — the identifiable subset collapses (0/57 GM, 9/119 WM, 26/222 brain), so there is nothing
determined left to compare. Both reasons are re-measured **with** the filter applied, so neither is a
bound-pinning artifact.

**Backend consistency (auto-vs-cpu) is reported, never gated** — CI installs no accelerator, so `auto`
resolves to the same code path as `cpu` there and gating it would assert that a function equals itself.
`test_backend_equivalence.py` is the real cpu-vs-cpufit/gpufit gate, and CI runs it with the backends
installed.

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

**One limit for every model and column**: `ROI_XLS_MAX_ABS_ERR = 0.01`. This replaced four hand-tuned
per-model limits (tofts 0.03, ex_tofts/patlak 0.01, tissue_uptake 0.05) that had drifted into 20–770×
headroom as the Stage-B AIF work landed; measured worst column per model on 2026-07-29 is tofts
0.001440, ex_tofts 0.000013, patlak 0.000013, tissue_uptake 0.002235, so the single value is ~7× the
worst case and tightens three of the four.

**The suite's one remaining hand-curated exclusion** is `tissue_uptake`'s `Fp` and its two CI columns
(`ROI_XLS_EXCLUDED_COLUMNS`) — and it is the *same* finding that keeps `tissue_uptake` out of the
voxelwise gated set above, not a second independent exception. Re-measured 2026-07-29 and still
precisely scoped: `Fp` (0.073345) and its CI (0.055581, 0.091109) are the only columns over the limit;
the worst of all the others is `Vp 95% high` at 0.002235.

### Stage-B AIF contract

Everything above compares **final maps**, which is how a structurally different Stage-B AIF fit
survived for months behind passing map checks (issue #2 / `aif_fitting_parity.md`).
`test_stage_b_aif_parity.py` closes that: it gates Stage-B's own outputs — the fitted `Cp_use`,
`CpROI`, `Stlv_use`, `timer`, the injection `step` window, `start_time`/`end_time`/`max_index`, and
the AIF fit coefficients `[A B c d t_base_end t0_exp]` — against a committed MATLAB payload.

```bash
pytest tests/python/test_stage_b_aif_parity.py
```

No MATLAB needed: the reference is
`derivatives/matlabref/sub-10bbbdownsample/ses-01/dce/Dyn-1_stage_b_aif.json`, written by
`tests/matlab/helpers/write_stage_b_aif_contract.m` whenever the map generator runs. Python and
MATLAB currently agree to ~1e-8 relative on every curve; the gates sit at `rel_mae ≤ 1e-5`,
`rel_max_abs ≤ 1e-4`, `corr ≥ 0.99999`, timing to 1e-6 min, and **exact** integer equality on the
indices. Correlation alone would not catch this bug class — a uniformly rescaled `Cp_use` still
correlates at 1.0 — so the absolute-error gates are the load-bearing ones.

Regenerate the payload without paying for a voxelwise Stage-D run:

```bash
matlab -batch "addpath('tests/matlab'); generate_dce_tofts_parity_map( \
  'outputRoot', 'tests/data/BIDS_test/derivatives/matlabref/sub-10bbbdownsample/ses-01/dce', \
  ... , 'stageBOnly', true);"
```

MATLAB-side drift (a stale committed payload) is caught by
`tests/contracts/check_matlabref_map_drift.py`, which now diffs this JSON alongside the NIfTI maps.
The `force_cpu` rule below applies to the maps, not to Stage-B — the AIF fit never reaches gpufit —
but the generator's guard is shared, so the same recipe applies.

### Backend equivalence (cpu vs cpufit/gpufit)

Everything above compares Python to MATLAB. `test_backend_equivalence.py` compares Python's
Stage-D backends to *each other* on identical inputs — 2000 voxels of Stage-B arrays frozen from
`RUNNER_DATA/sub-1101743/ses-01` into `tests/data/stage_b_frozen/` (332 KB), so no network drive
is needed to run it.

```bash
pytest tests/python/test_backend_equivalence.py
```

**Skips with a reason** when `pycpufit`/`pygpufit` are absent. CI runs it in a dedicated
`backend_equivalence` job that installs the backends with
`python install_python_acceleration.py --no-matlab --no-gui` (prebuilt wheels from the
`ironictoo/Gpufit` release). GitHub runners have no CUDA, so the `cpufit` half gates and the
`gpufit` half skips itself; both halves gate on a CUDA workstation.

**Why it is not a plain correlation gate.** This is a low-enhancement BBB dataset where a large
minority of voxels pin a parameter at a bound; on that flat objective two optimizers stop at
different points of an equally good plateau. Over all voxels ex_tofts Ktrans reads `corr` 0.23 —
but **0.9998** once bound-pinned voxels are excluded, with the two backends' SSE agreeing to ~1e-6
relative. A whole-sample correlation gate would therefore have to be set so loose it caught
nothing. The three parts instead are:

1. **Identifiable subset** — voxels where neither backend pinned a core parameter. Gated on
   `corr ≥ 0.98`, RMS-relative scatter (per parameter: Ktrans 0.06, ve 0.15, vp 0.02) and median
   relative bias within ±0.02. Scatter is normalized by the reference **RMS, not its max** —
   these parameters are strongly right-skewed, and max-normalization let a 20% Ktrans bias read
   as 0.02.
2. **Objective agreement** — SSE over *all* voxels: `corr ≥ 0.99`, median relative difference
   ≤ 1e-3. Backends may stop at different plateau points but must not find worse optima.
3. **Bound-hit symmetry** — each backend must pin core parameters at rates within 0.05. This is
   the only check that would catch a bound-handling difference between backends.

Regenerate the fixture (needs the network drive) with
`python tests/python/freeze_stage_b_backend_fixture.py`.

### Thresholds

Gate thresholds default to `tests/python/parity_thresholds_default.json`. Override with a copy:

```bash
pytest tests/python -m parity --parity-thresholds path/to/my_thresholds.json
```

Only the keys you include are overlaid. There are exactly two, applied to every gated
model/parameter/region alike: **`gate_corr_min`** (0.95; measured worst case 0.9616) and
**`gate_nrmse_max`** (0.25; measured worst case 0.1423). The equivalent CLI flags are
`--parity-gate-corr-min` / `--parity-gate-nrmse-max`.

**Scatter is gated on `nrmse` — RMSE over the reference RMS — not on absolute RMSE.** These
parameters differ ~50× in scale across models (reference RMS runs 0.0012 for patlak Ktrans to
0.058 for ex_tofts ve), so no single absolute bound is honest for all of them: an `rmse_max` of
0.02, the tightest value the tofts numbers would have allowed, still admits a **1737%** error on
patlak Ktrans. Normalizing makes a proportional error read as itself, and matches how
`test_backend_equivalence.py` measures the same kind of scatter. A degenerate reference (zero RMS
over the mask) yields a non-finite `nrmse` and **fails** — there is nothing to normalize against,
so nothing was verified.

The two metrics are deliberately orthogonal: correlation cannot see a pure scale error at all
(the 40% patlak injection below reports `corr=1.000000`), and rank correlation catches a
structural change that leaves the value distribution intact.

The former per-scope knobs (`model_ktrans_*`, `model_param_*`, `ve_ktrans_min`, `ktrans_upper_exclude`,
and the never-consumed `downsample_*` / `full_*` / `cpu_auto_*` / `ex_tofts_ktrans_corr_min`) were
removed on 2026-07-29 — the first four are subsumed by the single pair plus the identifiability filter,
and the rest had no reader in the suite at all. Unknown keys in an override JSON are still accepted and
ignored, so an old file will not error.

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
  'aifRoiPath', '$S/derivatives/$sub/$ses/dce/${sub}_${ses}_label-AIF_mask.nii', \
  'brainRoiPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_label-brain_mask.nii', \
  't1MapPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_space-DCEref_T1map.nii', \
  'noiseRoiPath', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_label-noise_mask.nii', \
  'outputRoot', '$S/derivatives/matlabref/$sub/$ses/dce', \
  'roiList', '$S/derivatives/$sub/$ses/anat/${sub}_${ses}_label-brain_mask.nii', \
  'models', {'tofts', 'ex_tofts', 'patlak', 'tissue_uptake', '2cxm'});"
```
