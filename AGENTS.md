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
python3 install.py   # creates .venv, installs deps + pyGpufit/pyCpufit, writes rocketship.sh
```

MATLAB is optional here: without it on `PATH` the installer warns that the MATLAB MEX files
could not be verified and still exits 0. `rocketship_dce.sh` and `rocketship_parametric.sh`
(`.bat` on Windows) are generated, machine-local, gitignored wrappers that activate the venv
and launch one GUI each; `install.py` writes both from one template (`LAUNCHERS`) and deletes
the superseded single `rocketship.sh`. One launcher per GUI rather than one taking
`[dce|parametric]`, because the subcommand form left the parametric GUI unmentioned in the
installer's own next-steps output and nobody found it.

Manual alternative: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
(add `-r requirements_gui.txt` for the PySide6 GUI). MATLAB toolboxes required: Curve
Fitting, Parallel Computing, Statistics and Machine Learning, Image Processing.

## Running the pipelines

```bash
# DCE (Python), shipped example config (sub-02downsample, ~11 s):
.venv/bin/python run_dce_python_cli.py
# DCE smoke test on the tiny fixture (~0.5 s):
.venv/bin/python run_dce_python_cli.py --config tests/python/dce_run_tiny.json
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

# MATLAB tests (from MATLAB):
results = run_unit_tests();
results = run_all_tests('suite', 'all', 'includeIntegration', true);
```

Full test-suite docs (regions, gated-vs-reported split, thresholds, fixture regeneration
commands): `tests/README.md`. Pytest markers are declared in `pytest.ini`
(`unit`, `integration`, `parity`, `slow`, `portability`, `osipi`, `fast`).

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

Every DCE default, limit and preference lives in **`python/dce_defaults.json`**. Source
code carries no fallback values: precedence is CLI `--set` → `stage_overrides` in the run
config → `dce_defaults.json` → **error**. A key missing everywhere raises `DceConfigError`
naming the key and the file; a key in the run config that the defaults file does not
declare is rejected as a typo. `python/dce_config.py` is the resolver.

Parametric T1 follows the same rule with its own pair: **`python/parametric_defaults.json`**
holds the defaults and `python/parametric_config.py` resolves against it, with the same
`required`/`optional`/`defaults` sections and the same typo guard. Its config is flat -- there
is no `stage_overrides` block -- so a run config's own top-level keys are the overrides. Input
paths (`output_dir`, `vfa_files`, `mask_file`, `b1_map_file`) describe one study rather than
how the software behaves, so they belong in a run config and never in the defaults file;
`python/parametric_run_example.json` is the runnable example.

`relaxivity` and `hematocrit` are per-scan values, so for those two the DCE image's JSON
sidecar wins over the run config: sidecar → run config → defaults file → error.
`relaxivity` deliberately has **no default** — the right value depends on the contrast
agent, so a run without one stops rather than guessing. (MATLAB keeps its
`script_preferences.txt` fallback; that divergence is intentional and documented in
`docs/dce_options.md`.) Scan parameters (TR/FA/time-resolution) are resolved strictly from
the metadata sidecar when present; partial manual override alongside a sidecar is rejected
(all three or none — no silent per-field fallback).

A run config names its inputs in `dynamic_files`/`aif_files`/`roi_files`/`t1map_files`, but
any of those left empty is filled from the dceprep naming convention under `subject_tp_path`
when one is set (`dce_file_discovery.DISCOVERABLE_FILE_LISTS`, applied in
`DcePipelineConfig.from_dict`, reported as "Found by BIDS convention"). Naming a
file wins; only empty lists are filled, and `drift_files` is never discovered. The batch
driver and the GUI's auto-find use the same discovery, so all three interfaces select the
same files. `subject_source_path`/`subject_tp_path` are optional -- omitting them is the
non-BIDS case, where every input must be named and TR/FA/time-resolution/relaxivity stated.

Relative paths in a run config are anchored to **the directory holding that config file**,
not the process cwd, so a config runs the same from anywhere. Paths given on the command
line (`--output-dir`, `--set dce_metadata_path=...`) are anchored to the cwd, where they
were typed. Entry points pass the anchor as `from_dict(..., base_dir=...)`; the path-valued
preference keys are listed in `dce_config.PATH_VALUED_KEYS` and resolved by
`dce_config.resolve_override_paths`. The parametric side follows the same rule.

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
`docs/project-management/projects/archived/batch-parity/aif_fitting_parity.md`). `piecewise_constant` ports
MATLAB `find_end_ss`; `legacy_sobel` ports the different `dce_auto_aif.m` heuristic; `glr` is an
additional ported detector.

Stage B's AIF fit (`_fit_aif_biexp`, `dce/AIFbiexpfithelp.m`) always holds `t_base_end` at the
resolved baseline end and always fits the upslope duration as `t0_exp = t_base_end + delta`,
with `delta` floored at one frame. There is no `start_injection_min` option: the injection start
*is* the baseline end. Background and rationale:
`docs/project-management/projects/archived/batch-parity/aif_fitting_parity.md`.

### Run output

The pipelines emit a structured event stream (`_emit_progress` in `dce_pipeline.py`,
`_emit_event` in `parametric_pipeline.py`). That stream is the machine record and always
lands in full in `<output_dir>/*_events.jsonl`. `python/run_reporting.py` is the only place
that turns it into text for people, and the CLIs, both batch drivers and both GUI log views
all render through it, so every interface describes a run the same way.

Verbosity (`--verbosity quiet|normal|detailed|debug`, `-v`/`-vv`/`-q`) selects how much of
that stream is rendered, never how much is recorded. `normal` is the default; the GUI log
renders at `detailed` (`run_reporting.GUI_VERBOSITY`). `--events on` puts the raw JSON
stream on stdout *instead of* human progress -- stdout carries one audience at a time --
and is how the GUI drives its progress bar. Below `debug`, a failed run reports the error
and exits non-zero without a traceback.

Every run records which ROCKETSHIP produced it. `python/version.py` is the single source:
`__version__` plus `git_revision()`, the short commit with `-dirty` when the tree has
uncommitted changes, or `None` outside a checkout. `version.build_identity()` returns both
as one dict, and that dict goes into the `cli_config` and `run_start` events and into the
run summary JSON. The rendered header shows the version at every level and the revision at
`detailed` and above. Readers take it from the *event*, never by importing `version`
themselves, so a GUI rendering a subprocess or a replay of an old log reports the build that
produced the run rather than the one doing the reading.

A few messages come from call sites with no event callback in scope (config-time BIDS
discovery, per-scan value provenance, Stage-D backend choice). They call
`run_reporting.notice(text, level)`, which the entry point routes to a reporter or into the
event stream; with no sink installed it prints, so the pipeline stays usable as a library.
Do not add bare `print` calls to the pipelines -- they bypass verbosity and reach every
interface.

### GUI structure

Both GUIs are one interface over two pipelines. `python/gui_common.py` holds everything that
is the same by nature -- the window palette and `WINDOW_QSS`, path resolution against the
config that holds a path, browse dialogs, the collapsible section, the log view and its
reporter, the run bar, the figures panel, and the `QProcess` lifecycle including the event
demux -- exposed as `GuiCommonMixin` plus a few builders. Each window keeps only what differs:
which settings exist, how a config payload is assembled, and what an event means for the
progress bar.

Do not re-add a local copy of a mixin method to a window. Two copies of "how a GUI drives a
CLI" is what produced four different `--set` parsers; `tests/python/test_gui_common.py` fails
if one comes back. Both windows show the same four tabs (Inputs, CLI Output, QC Figures,
Results) and share `dce_volume_viewer.py`, which is pipeline-agnostic despite its name.

### Backend selection (Stage D acceleration)

`backend` is `auto` | `cpu` | `gpufit`. `auto` tries `pygpufit`+CUDA, then `pycpufit`
(CPU), then falls back to the pure Python/scipy path — see `probe_acceleration_backend`/
`_resolve_backend_selection` in `dce_pipeline.py`. Accelerated models (`tofts`, `ex_tofts`,
`patlak`, `tissue_uptake`, `2cxm`) all fit through the shared `python/dce_fit_backends.py`
multi-start machinery (`FitInputs`, per-model `assemble_*_candidates`, `fit_with_multistart`),
so every backend — cpufit/gpufit or plain Python — sees the same candidate starting points
and the same bounds clamp; `tissue_uptake`/`2cxm` are fit in E-space (`E = Ktrans/Fp`) and
converted back on output. `pygpufit`/`pycpufit` are not in `requirements.txt` (installed
separately via `install.py`), so CI's `auto` always resolves to pure CPU.

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
- `docs/project-management/COMPLETED.md` — historical completion log.

Current measurable state is the test suite itself, not a document. Run it rather than
consulting a snapshot; a checked-in status file went stale between reads and was removed.

Keep planning docs non-overlapping. Do not update all planning docs by default.

Document roles:
- `ROADMAP.md`: strategy and sequencing only.
  - Includes merge-readiness criteria, long-horizon workstreams, and delivery order.
  - Excludes day-to-day checklists and historical changelog entries.
- `TODO.md`: active actionable tasks only.
  - Includes open blockers, open follow-ups, and open external handoff items.
  - Excludes completed history and broad strategic narrative.
- `COMPLETED.md`: historical completion log only.
  - Includes resolved milestones, completed work packages, and retired status notes.
  - Excludes open tasks.

Update decision rule (apply smallest necessary set):
- Strategy changed: update `ROADMAP.md`.
- Open work changed: update `TODO.md`.
- Work finished or historical status archived: update `COMPLETED.md`.

Do not leave important caveats only in commit messages or chat; record them in the single appropriate document above.

When you discover a problem that cannot be fixed immediately, document it before moving on:
- Write it up under `docs/project-management/projects/<initiative>/` -- integrate it into
  an existing initiative folder if the problem clearly belongs to one (e.g. a Stage-D
  backend divergence found while working on batch parity goes in `projects/archived/batch-parity/`),
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
- `docs/parametric_options.md` — parametric T1 field reference (shared by CLI + GUI + batch).
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

### Comment style
Keep code comments brief — one or two lines is the norm, and prefer none over restating
what the code already says. Explain *why*, not *what*.

Multi-line comment blocks need to earn their length. A short pointer to the durable record
beats an inline retelling: cite the commit, the issue, or the file in
`docs/project-management/` and let that carry the detail. Narrative belongs in those docs
and in commit messages, not in the source.

The exception is a non-obvious constraint that will be silently broken by someone editing
nearby — a numerical-stability requirement, a MATLAB-parity contract, a guard whose
condition is subtler than it looks. State it in a sentence or two, then link out.

### Commit messages
Keep them shorter than the instinct to be thorough suggests. A subject line that states the
change, then a body of a few short paragraphs — roughly 150-250 words — covering *why* the
change was made and anything a reader could not infer from the diff. One paragraph per
substantive change, not per file, per defect and per detail.

The diff already says what moved. The durable record for reasoning, evidence and open
caveats is `docs/project-management/` (see Documentation Discipline above) — cite it and let
it carry the detail rather than restating it here. A message that runs past a screen is
usually a document that was written in the wrong place.

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
