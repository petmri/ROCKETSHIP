# Pytest Migration Plan (Scorched Earth)

## Mission
Replace the Python test stack’s `unittest` execution and style with idiomatic `pytest` across local workflows, CI, and documentation, while preserving existing behavioral coverage and parity checks.

## Scope
- In scope: Python test execution, fixtures, markers, parity controls, coverage reporting, CI Python jobs, docs, and helper utilities under `tests/python`.
- Out of scope: MATLAB test framework and MATLAB CI jobs (unchanged).

## Migration Rules
- New branch only: `refactor/pytest-scorched-earth`
- Freeze: no new Python `unittest` tests during migration.
- Keep changes sequenced by phase; each phase must be green before moving on.
- Prefer explicit pytest options/markers over hidden env-variable behavior.

## Marker Taxonomy (target)
- `unit`: Pure logic tests, no large fixture IO.
- `integration`: Pipeline and filesystem interactions.
- `parity`: Python-vs-MATLAB parity checks.
- `slow`: Expensive tests (large fixture or long runtime).
- `portability`: Tests expected to run on macOS/Windows/Linux portability jobs.

## Acceptance Criteria (Definition of Done)
- [x] Python tests run with `pytest` only (no required `unittest` command paths).
- [x] Parity controls are available via pytest options/markers (not env-vars required for basic use).
- [x] CI Python jobs in `.github/workflows/run_DCE.yml` use pytest commands.
- [x] Coverage summary appears in CI logs, plus machine-readable report artifact.
- [x] `tests/python/README.md` and `python/README.md` use pytest-first commands.
- [x] Failure output provides model/param/threshold context for parity failures.
- [x] Final sweep removes stale `unittest`-specific docs/helpers from Python path.
- [x] Snapshot-style parity summaries are emitted for parity runs.

## Phase Checklist

### Phase 0 — Branch + Bootstrap
- [x] Create migration branch `refactor/pytest-scorched-earth`.
- [x] Add this migration plan (`TODO_pytest.md`).
- [x] Add pytest dependencies (`pytest`, `pytest-cov`, optional `pytest-timeout`, optional `pytest-xdist`).
- [x] Add `pytest.ini` with strict markers and default options.
- [x] Add root `tests/python/conftest.py` with common fixtures and CLI options.

### Phase 1 — Infrastructure Conversion
- [x] Add pytest CLI options:
  - [x] `--run-parity`
  - [x] `--run-full-parity`
  - [x] `--dataset-root`
  - [x] `--full-root`
  - [x] `--roi-stride`
- [x] Convert parity skip gates from env checks to marker+option checks.
- [x] Centralize warning behavior in pytest config/options.
- [x] Centralize temp output and fixture path handling.
- [x] Replace core parity env toggles with pytest options.

### Phase 2 — Mechanical Test Conversion
- [x] Convert `unittest.TestCase` classes to pytest test functions where practical.
- [x] Replace `self.assert*` with plain `assert` and helper assertions.
- [x] Replace `setUp`/`tearDown` with fixtures (`function`, `module`, `session`).
- [x] Preserve test IDs/intent while modernizing structure.

### Phase 3 — Idiomatic Refactor
- [x] Parametrize repetitive model/parameter/back-end matrices.
- [x] Split oversized files where useful (especially parity modules).
- [x] Improve assertion messages for parity diagnostics.
- [x] Standardize CLI test invocation helpers.
- [x] Add snapshot-style parity summary outputs.

### Phase 4 — Coverage + CI Hard Cut
- [x] Update GitHub Actions Python jobs to run pytest.
- [x] Add coverage terminal summary (`term-missing`).
- [x] Emit coverage XML (and/or HTML artifact) in CI.
- [x] Keep parity in dedicated marker-invoked CI step.

### Phase 5 — Docs + Cleanup
- [x] Rewrite `tests/python/README.md` with pytest workflows.
- [x] Rewrite `python/README.md` Python test commands to pytest.
- [x] Remove obsolete `unittest` command examples.
- [x] Remove Python-side compatibility leftovers no longer needed.

### Phase 6 — Stabilization + Signoff
- [x] Run fast suite (`-m "not parity and not slow"`).
- [ ] Run parity suite (`-m parity` with required options).
  - Current status: downsample + multi-model parity pass; full-volume parity is blocked until `test_data/BBB data p19/processed/results_matlab/*tofts*.nii` baselines are generated.
- [ ] Run portability-targeted suite (`-m portability`).
  - Current status: no tests currently marked `portability` (0 selected).
- [x] Resolve flakes and finalize thresholds.
- [ ] Merge into `dev` once all gates pass.

## Suggested Command Targets (end state)
- Fast local loop:
  - `pytest -m "not parity and not slow" -q`
- Full Python suite:
  - `pytest tests/python -v`
- Downsample parity:
  - `pytest -m parity --run-parity --dataset-root test_data/ci_fixtures/dce/bbb_p19_downsample_x3y3 -v`
- Coverage run:
  - `pytest tests/python --cov=python --cov-report=term-missing --cov-report=xml`

## Risks and Mitigations
- Risk: parity runtime explosion
  - Mitigation: enforce marker split (`parity`, `slow`) and default fast run path.
- Risk: hidden env coupling remains
  - Mitigation: fail-fast fixtures for required options and explicit skip reasons.
- Risk: CI migration churn
  - Mitigation: phase-gated rollout and clear diff per workflow step.

## Working Notes
- Prefer preserving expected numerical tolerances before changing thresholds.
- Any threshold change requires rationale in PR notes.
- Keep helper utilities minimal; avoid creating a second custom test framework.
