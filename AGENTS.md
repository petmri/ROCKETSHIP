# ROCKETSHIP Agent Guidance

## Project Direction
ROCKETSHIP is actively transitioning core workflows from MATLAB to Python.

Canonical planning and status docs:
- `/Users/samuelbarnes/code/ROCKETSHIP/python/ROADMAP.md`
- `/Users/samuelbarnes/code/ROCKETSHIP/python/PORTING_STATUS.md`
- `/Users/samuelbarnes/code/ROCKETSHIP/TODO.md`

## Documentation Discipline
- When implementation or test behavior changes, update the planning/status docs in the same workstream:
  - `/Users/samuelbarnes/code/ROCKETSHIP/TODO.md`
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/ROADMAP.md`
  - `/Users/samuelbarnes/code/ROCKETSHIP/python/PORTING_STATUS.md`
- Do not leave important caveats only in commit messages or chat; record them in the docs above.

OSIPI reference repos available locally for verification work:
- `/Users/samuelbarnes/code/DCE-DSC-MRI_CodeCollection`
- `/Users/samuelbarnes/code/DCE-DSC-MRI_TestResults`

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
