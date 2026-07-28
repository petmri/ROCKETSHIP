# Project Management Docs

This folder contains canonical planning and status documents for the Python transition, plus project-specific status/report material.

## Top-Level Canonical Docs

- `TODO.md`: active open tasks only (blockers, follow-ups, open handoffs).
- `COMPLETED.md`: historical completed log only.
- `ROADMAP.md`: strategy, merge criteria, and long-horizon sequencing only.
- `PORTING_STATUS.md`: current measurable state only (latest test/qualification snapshot, current blockers, active risks).

## Project Folders

- `projects/osipi-verification/`
  - `osipi_summary.md`: OSIPI accuracy summary and peer-comparison snapshot.
- `projects/qualification/`
  - `QUALIFICATION_MERGE_PACKET.md`: qualification run packet and blocker classification.
- `projects/phantom-gt/`
  - `PHANTOM_GT_QUALIFICATION_STATUS.md`: synthetic phantom GT qualification status.
- `projects/large-data-distribution/`
  - `large_data_distribution.md`: options for distributing large test/parity data outside the main git repo.

## Archived Projects

Historical snapshots under `projects/archived/`. Each carries a status header saying what was
completed and where any residual open work went. **They are not live tracking docs** — read them
for how something was built and why, not for what is outstanding.

- `projects/archived/batch-parity/` — MATLAB vs Python DCE parity (archived 2026-07-28, complete:
  12/12 gated voxelwise + 4/4 ROI-xls, no hand-curated exceptions). Contains `batch_parity.md`
  (the parent), `aif_fitting_parity.md` (Stage-B AIF algorithm unification), `quality_of_fit.md`
  (per-voxel reduced-χ² reliability) and `sigma_estimators.md` (noise-σ estimation). Residuals
  live in `TODO.md`: parity testing gaps A/B, QoF-aware ROI stats, `shrink_sigma` default.
- `projects/archived/steady-state-tv-default/` — choosing and rolling out the `tv` baseline-end
  detector.
- `projects/archived/stage-d-fit-consolidation/` — Stage-D fit-backend consolidation across all
  five models.

## Update Policy

Use the smallest necessary update set:

1. Strategy/sequence changed: update `ROADMAP.md`.
2. Open work changed: update `TODO.md`.
3. Current test/qualification state changed: update `PORTING_STATUS.md`.
4. Work completed or status archived: update `COMPLETED.md`.
5. Project-specific diagnostics changed: update the relevant file under `projects/`.

Keep caveats and diagnostic notes in these docs, not only in commits or chat.
