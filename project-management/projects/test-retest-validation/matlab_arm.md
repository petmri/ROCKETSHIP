# MATLAB arm: how to run it on another machine

## Status (2026-11-16)
MATLAB is not installed on the analysis workstation (no `matlab` on `PATH`, nothing under
`/usr/local/MATLAB` or `/opt`, no licence file; Docker absent). The MATLAB arm is therefore
set up to run elsewhere: a job list exported from the Python fits, and a driver that runs
it. **The driver has not been executed.** It was written against the MATLAB sources of the
validation branch without a MATLAB to run it; expect to fix small things on first contact.

Nothing on the MATLAB side has been run. The July run (`dceprep-testretest`) stays the only
MATLAB output, and it is not matched to the current inputs.

## What the other machine needs
1. MATLAB with a licence and the toolboxes in AGENTS.md (Curve Fitting, Parallel Computing,
   Statistics and Machine Learning, Image Processing). `jsondecode` needs R2016b or later.
2. The DCE BIDS share mounted (any path), with write access to
   `derivatives/rocketship-testretest/`.
3. This ROCKETSHIP branch checked out, and on the MATLAB path. For GPU fits, the Gpufit
   MATLAB bindings on the path as well; otherwise MATLAB fits on the CPU pool.

## Running it
On the analysis workstation, once per rater (already done for the fits that exist; the
job lists are in `derivatives/rocketship-testretest/`):

```bash
run_testretest_export_matlab_jobs.py --rater AC \
    --fit-root  <DCE_BIDS>/derivatives/rocketship-testretest/python_AC \
    --bids-root <DCE_BIDS> \
    --out       <DCE_BIDS>/derivatives/rocketship-testretest/matlab_jobs_AC.json
```

Each job names the four input files (relative to the share) and the values the Python
fit's Stage A resolved for that session, so the MATLAB arm is matched by construction
rather than by a second reading of the sidecars.

On the MATLAB machine:

```matlab
addpath('<ROCKETSHIP>');
addpath('<ROCKETSHIP>/project-management/projects/test-retest-validation/matlab');
bids = '<DCE_BIDS>';
jobs = fullfile(bids, 'derivatives/rocketship-testretest/matlab_jobs_AC.json');
out  = fullfile(bids, 'derivatives/rocketship-testretest/matlab_AC');

% one session first
run_testretest_matlab(jobs, bids, out, 'python', 'subject', 'sub-<one subject>');
% then the cohort (finished sessions are skipped)
run_testretest_matlab(jobs, bids, out, 'python');
```

Then check the first session before the rest: its folder should hold
`<rootname>_patlak_fit_Ktrans.nii` and `matlab_fit.log`, and
`run_testretest_analysis.py collect --arm matlab:AC:<out>:<dceprep run>` should report no
grid problems. Comparing that one map with the Python map of the same session is a parity
check, not a repeatability result, and is the right first test of the driver.

The driver stages each session's inputs next to its outputs (unzipped copies, deleted
after the fit unless `'keep_inputs', true`), so the rater runs stay read-only. It writes
`matlab_fit_manifest.tsv` and `matlab_provenance.json` (MATLAB version, ROCKETSHIP commit,
the preference values it read, whether Gpufit was found).

## Decisions the run depends on
- **`baseline` argument** (`'python'` or `'auto'`, no default). Where the rater marked the
  baseline end, that frame is passed in both modes. Where the Python fit auto-detected it
  (all of rater AC), `'python'` passes the frame Python chose, which isolates the fit from
  the detector; `'auto'` lets MATLAB's `find_end_ss_tv` choose, which makes the detector
  part of the comparison. This is part of what "matched" means and belongs in the
  pre-registration.
- **Which code is "MATLAB 1.x"**: the `1.2` tag or the MATLAB side of the validation
  commit. The driver targets the latter. A comment in `A_make_R1maps_func.m` says the
  explicit-baseline path used to leave the injection times undefined, so on `1.2` the
  per-session baseline the driver passes may not work, and `find_end_ss_tv` does not exist
  there.
- **Voxel-fit options** come from `dce/dce_preferences.txt` on the MATLAB side and from
  `python/dce_defaults.json` on the Python side. The repository's parity contract keeps
  them aligned; the driver does not check it.

## Settings the MATLAB arm must match
The Python arm's settings are in `testretest_run_config.json`; each session's literal
config is `<fits>/python_<RATER>/sub-*/ses-*/run_config.json`.

| Setting | Python arm | MATLAB (`script_preferences.txt` / argument) |
|---|---|---|
| Inputs | `dce/*_desc-bfcz_DCE`, `dce/*_label-AIF_T1map`, `anat/*_space-DCEref_label-brain_mask`, `anat/*_space-DCEref_T1map` of the rater run | same globs (`dynamic_files`, `aif_files`, `roi_files`, `t1map_files`) |
| AIF curve | `aif_mode = raw` | `aif_type = 2` |
| Blood T1 | `blood_t1_ms` 1441 (RIDER) / 1664 (QIN-GBM) | `aif_rr_type = aif_roi_static`, `blood_t1` 1.441 / 1.664 (seconds), per cohort |
| Baseline end | rater's `SteadyStateEndTimeIndex` sidecar where present, else `tv` auto-detect | the job's `steady_state_end` as the `steady_state_time` argument, or `-2` (auto) in `'auto'` mode |
| First/last frame | `start_t` 1, `end_t` last | `start_t = 1`, `end_t` empty (the shipped file has `start_t = 3`) |
| Relaxivity | staged DCE sidecar: 3.8 (3 T), 4.3 (1.5 T) | the job's `relaxivity`, passed to Stage A |
| Hematocrit | 0.45 | `hematocrit = 0.45` |
| TR / FA / frame spacing | `RepetitionTimeExcitation`, `FlipAngle`, `RepetitionTime` of the staged sidecar | same fields, same precedence in `run_dce_cli` |
| Models | Patlak, extended Tofts | `patlak = 1`, `ex_tofts = 1` |
| Voxel fits | all voxels of the brain mask | `fit_voxels = 1` |
| Smoothing | none | `time_smoothing = none`, `xy_smooth_size = 0` |
| Backend | `gpufit` (CUDA, Gpufit 1.5.1) | record what MATLAB used (GPU or CPU) |

## Why `run_dce_cli` cannot be used unchanged
Read from `run_dce_cli.m` at the validation commit:

- **Relaxivity is overwritten.** For any site whose sidecar is not "USC" it sets
  `relaxivity = 3.4`, whatever the sidecar or preferences say. The staged data carry 3.8
  and 4.3. With a fixed blood T1 the same relaxivity scales tissue and blood
  concentration alike, so Patlak Ktrans and vp should not move, but that is an argument,
  not a matched setting. This is the divergence the plan lists under known risks.
- **No sidecar baseline.** MATLAB does not read `SteadyStateEndTimeIndex`. WN's hand-set
  baseline frames have to be passed per session through `injection_time`.
- **Retries change `start_t`.** On some Stage A/B errors the CLI lowers `start_t` and
  tries again, so the frame window can differ from the one asked for. Log the window used.
- **Outputs land in the inputs.** Results are written under the session's own `dce/`
  folder, which here is the rater run. The driver has to stage each session's inputs in a
  separate tree (links are enough) so the rater runs stay read-only.

`matlab/run_testretest_matlab.m` is that driver: it loops over the job list, stages the
inputs, and calls `A_make_R1maps_func`, `B_AIF_fitting_func` and `D_fit_voxels_func` with
the job's values, in the argument layout of `run_dce_cli`. It does not retry with a
different `start_t`; a session that fails is recorded as failed.

## After the MATLAB fits exist
MATLAB writes `<rootname>_<model>_fit_<parameter>.nii`, the same names the Python arm
writes, so the collector needs nothing new:

```bash
run_testretest_analysis.py collect --out <tables> \
    --arm python:AC:<fits>/python_AC:<derivatives>/dceprep-testretest_AC \
    --arm matlab:AC:<fits>/matlab_AC:<derivatives>/dceprep-testretest_AC
```

Check the grids first: `collect` reports a map whose grid differs from the masks instead
of resampling it. The MATLAB arm has no QoF maps, so the QoF exclusion rule would be
driven by the Python fits alone; it removes the pair from both pipelines either way.
