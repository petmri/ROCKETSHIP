# CPUfit Handoff: `TOFTS_EXTENDED` Short-Timer Repro

This package is for upstream `pycpufit`/CPUfit troubleshooting of a data-regime-specific
`TOFTS_EXTENDED` fit-state failure.

## Problem Summary
- On ROCKETSHIP BIDS qualification curves (`tests/data/BIDS_test`, Stage-B outputs), CPUfit
  `TOFTS_EXTENDED` returns state `2` at iteration `0` for all tested voxels.
- ROCKETSHIP pure CPU fitting on the same curves returns finite `ex_tofts` parameters.
- OSIPI single-curve `extended_tofts` control still passes through CPUfit.

ROCKETSHIP now guards this by falling back from acceleration to CPU when accelerated output
has no usable finite primary parameters, but the upstream CPUfit behavior is still worth fixing.

## Files
- `bids_short_timer_repro.npz`: minimal failing payload (single Stage-B voxel).
- `osipi_control_repro.npz`: minimal control payload (single OSIPI DRO row).
- `run_cpufit_tofts_extended_repro.py`: reproducible runner comparing CPUfit vs CPU reference.

## Run
```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python tests/contracts/handoffs/cpufit_tofts_extended/run_cpufit_tofts_extended_repro.py --json
```

Optional timing-scale probe (to test sensitivity to very small timer units):
```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python tests/contracts/handoffs/cpufit_tofts_extended/run_cpufit_tofts_extended_repro.py \
  --json \
  --time-scale 600
```

## Expected
- `bids_short_timer_repro.npz`:
  - CPUfit state is expected to be non-zero (observed `2`) with `iterations=0`.
  - CPU reference (`python/dce_models.py:model_extended_tofts_fit`) remains finite.
- `osipi_control_repro.npz`:
  - CPUfit state is expected to be `0` (successful fit).

## Payload Provenance
- `bids_short_timer_repro.npz` extracted from:
  - `out/python_qualification_bids_test_auto_gated/sub-01original_ses-01/dce/checkpoints/b_out_arrays.npz`
  - first tumor voxel (`Ct[:,0]`) with corresponding `Cp_use` and `timer`.
- `osipi_control_repro.npz` extracted from:
  - `tests/data/osipi/dce_models/dce_DRO_data_extended_tofts.csv` (row 1)
