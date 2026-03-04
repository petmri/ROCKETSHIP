# CPUfit PATLAK Batch Repro (RUNNER_DATA clean reference)

This handoff package reproduces accelerated PATLAK drift observed in ROCKETSHIP batch parity diagnostics on real data.

## Payloads
- `ses01_control_repro.npz`
  - sampled from `RUNNER_DATA/derivatives/dceprep-python-batch-cleanref-aifw2/sub-1101743/ses-01`
- `ses02_drift_repro.npz`
  - sampled from `RUNNER_DATA/derivatives/dceprep-python-batch-cleanref-aifw2/sub-1101743/ses-02`

Each payload includes:
- `ct` (`n_points x n_curves`)
- `cp_use` and `timer_min`
- MATLAB reference `matlab_ktrans` from `dce_patlak_fit_Ktrans.nii`
- fit prefs/bounds/tolerances used in ROCKETSHIP

## Repro command
```bash
.venv/bin/python tests/contracts/handoffs/cpufit_patlak_batch/run_cpufit_patlak_repro.py \
  --json out/cpufit_patlak_batch_repro_20260303.json
```

## Observed output (local)
- `ses-01` (4096 curves):
  - CPU vs MATLAB: `corr=1.000000`, `slope=1.000008`
  - CPUfit vs MATLAB: `corr=0.815282`, `slope=0.676330`
  - CPUfit fit states: `CONVERGED=4096`
- `ses-02` (4096 curves):
  - CPU vs MATLAB: `corr=1.000000`, `slope=1.000006`
  - CPUfit vs MATLAB: `corr=0.367318`, `slope=0.135836`
  - CPUfit fit states: `CONVERGED=4096`

Interpretation:
- Python CPU PATLAK is MATLAB-aligned on the same Stage-B arrays.
- Accelerated CPUfit PATLAK can return strongly divergent Ktrans values while still reporting all fits as converged.
