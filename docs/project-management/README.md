# Project Management Docs

This folder contains canonical planning and status documents for the Python transition, plus project-specific status/report material.

## Top-Level Canonical Docs

- `TODO.md`: active open tasks only (blockers, follow-ups, open handoffs).
- `COMPLETED.md`: historical completed log only.
- `ROADMAP.md`: strategy, merge criteria, and long-horizon sequencing only.
- `PORTING_STATUS.md`: current measurable state only (latest test/qualification snapshot, current blockers, active risks).

## Project Folders

- `projects/batch-parity/`
  - `batch_parity.md`: MATLAB vs Python batch parity tracking and diagnostics.
- `projects/osipi-verification/`
  - `osipi_summary.md`: OSIPI accuracy summary and peer-comparison snapshot.
- `projects/qualification/`
  - `QUALIFICATION_MERGE_PACKET.md`: qualification run packet and blocker classification.
- `projects/phantom-gt/`
  - `PHANTOM_GT_QUALIFICATION_STATUS.md`: synthetic phantom GT qualification status.

## Update Policy

Use the smallest necessary update set:

1. Strategy/sequence changed: update `ROADMAP.md`.
2. Open work changed: update `TODO.md`.
3. Current test/qualification state changed: update `PORTING_STATUS.md`.
4. Work completed or status archived: update `COMPLETED.md`.
5. Project-specific diagnostics changed: update the relevant file under `projects/`.

Keep caveats and diagnostic notes in these docs, not only in commits or chat.
