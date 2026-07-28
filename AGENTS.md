# ROCKETSHIP Agent Guidance

## What this is

ROCKETSHIP is a toolbox for processing and analyzing parametric MRI and DCE-MRI
(dynamic contrast-enhanced) data. It has two parallel implementations:

- **MATLAB** (`dce/`, `parametric_scripts/`, `external_programs/`, `dsc/`) — the original,
  still-maintained implementation. GUIs (`dce.m`, `fitting_gui.m`) and CLI entry points
  (`run_dce.m`, `run_parametric.m`, `run_dce_cli.m`).
- **Python** (`python/`) — the actively-developed port and the recommended path for new
  work. No production users yet, so prioritize correctness and clean architecture over
  preserving legacy MATLAB behavior that isn't required for parity.

Because the Python port must reproduce MATLAB's numeric behavior, most non-trivial
changes touch both sides and get validated by the MATLAB-vs-Python parity test suite
(see Testing below). When MATLAB and Python disagree, prefer fixing whichever one
deviates from the *intended* algorithm rather than tuning tolerances to paper over it.

## Setup

```bash
python3 install_python_acceleration.py   # creates .venv, installs deps + pyGpufit/pyCpufit
```

Manual alternative: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
(add `-r requirements_gui.txt` for the PySide6 GUI). MATLAB toolboxes required: Curve
Fitting, Parallel Computing, Statistics and Machine Learning, Image Processing.

## Running the pipelines

```bash
# DCE (Python), built-in tiny fixture:
.venv/bin/python run_dce_python_cli.py
# DCE with an explicit config + overrides:
.venv/bin/python run_dce_python_cli.py --config tests/python/dce_cli_config.example.json --set voxel_MaxFunEvals=100

# Parametric T1 (Python):
.venv/bin/python run_parametric_python_cli.py

# BIDS batch processing across a dataset:
.venv/bin/python run_dce_bids_batch.py --bids-root <path> --pipeline-folder dceprep --backend gpufit
.venv/bin/python run_parametric_bids_batch.py --bids-root <path> --pipeline-folder t1prep

# MATLAB (from the MATLAB command line, repo root on path):
run_dce_cli('rawdata/sub-01/ses-01/', 'derivatives/sub-01/ses-01/')
```

Full CLI reference, config precedence, and output formats: `python/README.md` and
`docs/dce_options.md`.

## Testing

```bash
# Fast Python unit/integration suite (the default; includes gated DCE parity):
.venv/bin/python -m pytest tests/python

# With coverage (matches CI's python_checks job, --cov-fail-under=60):
.venv/bin/python -m pytest tests/python -q --cov=python --cov-report=term-missing --cov-fail-under=60

# Single test:
.venv/bin/python -m pytest tests/python/test_dce_pipeline.py::TestDcePipeline::test_resolve_baseline_window_accepts_glr_alias -v

# DCE parity vs MATLAB (Tofts/Patlak Ktrans, gated on corr+RMSE over brain/GM/WM):
.venv/bin/python -m pytest tests/python -m parity
.venv/bin/python -m pytest tests/python -m parity --parity-suite=allmodels -s   # + reported-only extras

# ROI-summary .xls parity (separate, default-on, few seconds):
.venv/bin/python -m pytest tests/python/test_dce_pipeline_parity_metrics.py::test_bbb_p19_roi_xls_parity

# Runtime parity vs a live MATLAB run (needs MATLAB on PATH):
.venv/bin/python -m pytest tests/python/test_runtime_parity.py --run-runtime-parity

# OSIPI reliability (ground truth vs published peer tolerances):
.venv/bin/python -m pytest tests/python -m osipi -v

# BIDS-level qualification:
.venv/bin/python -m pytest tests/python --run-qualification

# MATLAB tests (from MATLAB):
results = run_unit_tests();
results = run_all_tests('suite', 'all', 'includeIntegration', true);
```

Full test-suite docs (regions, gated-vs-reported split, thresholds, fixture regeneration
commands): `tests/README.md`. Pytest markers are declared in `pytest.ini`
(`unit`, `integration`, `parity`, `slow`, `portability`, `osipi`, `qualification`, `fast`).

CI (`.github/workflows/run_DCE.yml`) runs, per push/PR to `master`/`dev`: `python_checks`
(unit tests + coverage + OSIPI summary), `parity_checks` (MATLAB contract/baseline drift
guards + Python-vs-MATLAB dataset parity), `python_portability` (Windows/macOS, non-parity),
and `matlab_checks` (a MATLAB release × OS matrix: unit/integration tests + a full DCE CLI
run). `matlab_checks` runners have no GPU — `backend="auto"` always resolves to plain CPU
there, so accelerated-backend-only issues won't surface in CI.

## Architecture

### DCE pipeline stages (both languages implement the same A → B → D shape)

- **Stage A** — signal-to-concentration conversion from dynamic images + T1 maps + AIF/ROI
  masks, and steady-state baseline window resolution. MATLAB: `A_make_R1maps_func.m`.
  Python: `run_dce_pipeline`'s Stage A path in `python/dce_pipeline.py`.
- **Stage B** — AIF fitting/timing (biexponential fit or reference-region). MATLAB:
  `B_AIF_fitting_func.m`. Python: same file, `_fit_aif_biexp` and friends.
- **Stage D** — per-voxel or per-ROI kinetic model fitting, producing parameter maps.
  MATLAB: `D_fit_voxels_func.m` + `FXLfit_generic.m` + per-model `model_*.m`/`model_*_cfit.m`.
  Python: same file, dispatching to `python/dce_models.py` (CPU/scipy) or the accelerated
  path (see below).
- **Part E** (post-fit statistical comparison, f-test/AIC) is Python-only so far:
  `python/dce_postfit_analysis.py`.

`python/dce_pipeline.py` is the core (~4500 lines) — nearly everything for Stage A/B/D
config resolution, metadata/sidecar discovery, and backend dispatch lives there.
`python/dce_models.py` holds the CPU/scipy model implementations (ports of the MATLAB
`model_*.m` math). `python/dce_cli.py` is a thin CLI wrapper; `python/parametric_pipeline.py`
+ `python/parametric_models.py` are the equivalent stack for VFA T1 mapping.

### Config resolution

Python config precedence (highest to lowest): CLI `--set` overrides → `stage_overrides` in
the JSON config → `dce_default.json`/`dceprep_default.json` base values → built-in
fallback defaults. Scan parameters (TR/FA/time-resolution) are resolved strictly from a
DCE metadata JSON sidecar when present; partial manual override alongside a sidecar is
rejected (all three or none — no silent per-field fallback).

The steady-state/baseline window follows its own precedence in `_resolve_baseline_window`
(`python/dce_pipeline.py`): explicit `stage_overrides.steady_state_end` → a
`SteadyStateEndTimeIndex` field in the AIF file's JSON sidecar (the documented mechanism
for a fixed/reproducible run — same discovery convention as the metadata sidecar, `.nii`/
`.nii.gz` swapped for `.json`) → auto-detect via `stage_overrides.steady_state_auto_method`.
`tv` is the default (MATLAB `dce/find_end_ss_tv.m`): a total-variation denoise followed by the
first significant upward jump. `biexp_fit` (MATLAB `dce/find_end_ss_biexp.m`) fits a 6-parameter
biexponential to the mean AIF *signal* curve and, unlike the shape heuristics, also reports where
the injection ends — but on 280 human-rated sessions it is right 74.6% of the time against `tv`'s
95.0%, always erring one frame late, so it is selectable rather than default (see S11 in
`docs/project-management/projects/batch-parity/aif_fitting_parity.md`). `piecewise_constant` ports
MATLAB `find_end_ss`; `legacy_sobel` ports the different `dce_auto_aif.m` heuristic; `glr` is an
additional ported detector.

Stage B's AIF fit (`_fit_aif_biexp`, `dce/AIFbiexpfithelp.m`) always holds `t_base_end` at the
resolved baseline end and always fits the upslope duration as `t0_exp = t_base_end + delta`,
with `delta` floored at one frame. There is no `start_injection_min` option: the injection start
*is* the baseline end. Background and rationale:
`docs/project-management/projects/batch-parity/aif_fitting_parity.md`.

### Backend selection (Stage D acceleration)

`backend` is `auto` | `cpu` | `gpufit`. `auto` tries `pygpufit`+CUDA, then `pycpufit`
(CPU), then falls back to the pure Python/scipy path — see `probe_acceleration_backend`/
`_resolve_backend_selection` in `dce_pipeline.py`. Accelerated models (`tofts`, `ex_tofts`,
`patlak`, `tissue_uptake`, `2cxm`) all fit through the shared `python/dce_fit_backends.py`
multi-start machinery (`FitInputs`, per-model `assemble_*_candidates`, `fit_with_multistart`),
so every backend — cpufit/gpufit or plain Python — sees the same candidate starting points
and the same bounds clamp; `tissue_uptake`/`2cxm` are fit in E-space (`E = Ktrans/Fp`) and
converted back on output. `pygpufit`/`pycpufit` are not in `requirements.txt` (installed
separately via `install_python_acceleration.py`), so CI's `auto` always resolves to pure CPU.

### Data layout and fixtures

BIDS-style `rawdata/` (raw images, scan-parameter sidecars) + `derivatives/` (masks, T1
maps, pipeline outputs) trees, discovered via `python/bids_discovery.py`. Test fixtures
live under `tests/data/BIDS_test/`, committed and lightweight (no per-run regeneration in
CI) — key subjects: `sub-10bbbdownsample` (DCE Tofts/Patlak parity), `sub-11tiny` (T1/DCE
settings matrix), `sub-0Xphantom` (synthetic ground-truth reliability, diagnostic only).
MATLAB reference maps live under `derivatives/matlabref/...`; regenerate them only when
the MATLAB algorithm actually changes (commands in `tests/README.md`), and expect to
update the committed maps in the same change — a stale committed baseline vs. a
freshly-regenerated one is a real, previously-hit failure mode
(`tests/contracts/check_matlabref_map_drift.py` guards against it in CI).

### Cross-language parity contracts

`tests/contracts/` holds the MATLAB↔Python numeric contract: `export_parity_baseline.m`
(MATLAB) writes `tests/contracts/baselines/matlab_reference_v1.json` from synthetic
curves fed straight to the model math (no imaging pipeline involved);
`generate_python_results.py` + `compare_with_matlab_baseline.py` check Python against it.
`check_baseline_drift.py` catches MATLAB algorithm drift at that (synthetic-curve) layer;
`check_matlabref_map_drift.py` catches drift at the full-pipeline NIfTI-map layer — these
are deliberately separate because the synthetic-curve contract never exercises
`A_make_R1maps_func`/`find_end_ss` (steady-state detection, AIF extraction, etc.).

## Documentation Discipline

Canonical planning and status docs:
- `docs/project-management/ROADMAP.md` — strategy/sequencing, merge-readiness criteria.
- `docs/project-management/TODO.md` — active open tasks/blockers only.
- `docs/project-management/PORTING_STATUS.md` — current measurable port state.
- `docs/project-management/COMPLETED.md` — historical completion log.

Keep planning docs non-overlapping. Do not update all planning docs by default.

Document roles:
- `ROADMAP.md`: strategy and sequencing only.
  - Includes merge-readiness criteria, long-horizon workstreams, and delivery order.
  - Excludes day-to-day checklists and historical changelog entries.
- `TODO.md`: active actionable tasks only.
  - Includes open blockers, open follow-ups, and open external handoff items.
  - Excludes completed history and broad strategic narrative.
- `PORTING_STATUS.md`: current measurable state only.
  - Includes latest test/qualification snapshot, current blockers, and active risks.
  - Excludes long task inventories and archived progress history.
- `COMPLETED.md`: historical completion log only.
  - Includes resolved milestones, completed work packages, and retired status notes.
  - Excludes open tasks.

Update decision rule (apply smallest necessary set):
- Strategy changed: update `ROADMAP.md`.
- Open work changed: update `TODO.md`.
- Current test/qualification state changed: update `PORTING_STATUS.md`.
- Work finished or historical status archived: update `COMPLETED.md`.

Do not leave important caveats only in commit messages or chat; record them in the single appropriate document above.

When you discover a problem that cannot be fixed immediately, document it before moving on:
- Write it up under `docs/project-management/projects/<initiative>/` -- integrate it into
  an existing initiative folder if the problem clearly belongs to one (e.g. a Stage-D
  backend divergence found while working on batch parity goes in `projects/batch-parity/`),
  or create a new `projects/<slug>/` folder if it doesn't fit any existing initiative.
  Include what's confirmed (root cause, evidence), what's still open, and any agreed
  near-term mitigation, so the next person (or your future self) doesn't have to
  re-derive it from scratch.
- Still add a one-line pointer in `TODO.md` per the update rule above, since undocumented
  open work is easy to lose track of and `TODO.md` is where open work is expected to be
  discoverable; keep the actual detail in the project folder, not duplicated in `TODO.md`.

When asked to plan or estimate a new initiative, also write it up under `docs/project-management/projects/<initiative>/`. Upon completion, move the initiative folder to `docs/project-management/projects/archived/` and add a one-line pointer in `COMPLETED.md` per the update rule above.

Other reference docs:
- `docs/dce_options.md` — full `stage_overrides` field reference (shared by CLI + GUI).
- `python/README.md` — Python usage guide (CLIs, batch processing, GUI, output formats).
- `tests/README.md` — full test-suite reference.

OSIPI reference repos available locally for verification work:
- `~/code/DCE-DSC-MRI_CodeCollection`
- `~/code/DCE-DSC-MRI_TestResults`

## Engineering Priorities
- The Python code path is still in development and has no production users yet.
- Prioritize correctness, numerical reliability, and clean architecture over preserving legacy behavior that is not needed.
- Do not keep unnecessary Python files/functions just for historical symmetry; remove dead code aggressively once replacements are validated.
- Keep MATLAB and Python comparisons explicit when parity is required, but do not preserve obsolete MATLAB UX patterns that add maintenance overhead.

## Academic Software Standards
This is an academic software project. Favor:
- readable code with clear naming and unit-aware variables
- modular design that supports future method updates and extensions
- transparent algorithm choices and testable interfaces
- documentation that makes assumptions and limitations explicit

## Porting Focus
Primary focus:
- Parametric maps, especially T1 fitting (with GUI support)
- DCE model reliability for Patlak, Tofts, and Extended Tofts
- DCE Part E post-fitting analysis workflow

Secondary focus:
- further refinement of 2CXM and tissue uptake
- DSC workflow expansion

## OSIPI Verification Expectations
- Keep OSIPI-labeled tests additive; do not replace existing ROCKETSHIP tests.
- Maintain explicit provenance for imported OSIPI data and peer-result CSVs in `tests/data/osipi/README.md`.
- During T1 workflow work, prioritize both:
  - T1 mapping reliability checks against OSIPI reference datasets/peer result summaries.
  - Signal-intensity to concentration conversion checks against OSIPI SI2Conc datasets/peer result summaries.

Explicitly not targeted for Python parity unless scope changes:
- legacy neuroecon execution path
- legacy email notification flow
- manual click-based MATLAB AIF tools and ImageJ `.roi` compatibility paths
