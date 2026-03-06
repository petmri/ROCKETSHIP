# ROCKETSHIP Agent Guidance

## Project Direction
ROCKETSHIP is actively transitioning core workflows from MATLAB to Python.

Canonical planning and status docs:
- `~/code/ROCKETSHIP/docs/project-management/ROADMAP.md`
- `~/code/ROCKETSHIP/docs/project-management/PORTING_STATUS.md`
- `~/code/ROCKETSHIP/docs/project-management/TODO.md`
- `~/code/ROCKETSHIP/docs/project-management/COMPLETED.md`

## Documentation Discipline
- Keep planning docs non-overlapping. Do not update all planning docs by default.

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
