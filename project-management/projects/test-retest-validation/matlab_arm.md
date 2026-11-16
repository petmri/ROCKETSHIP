# MATLAB arm: status and what it needs

## Status (2026-11-16): blocked, MATLAB is not installed on the analysis workstation
Checked on the workstation that holds the Python fits: no `matlab` on `PATH`, nothing under
`/usr/local/MATLAB` or `/opt`, no MATLAB binary or licence file anywhere on the local disks.
What exists is only residue: `~/.matlab/R2019b` (an empty preferences folder from 2023),
`~/.matlab/MWI` (matlab-proxy host cache), and FreeSurfer's `~/matlab/startup.m`. Docker is
not installed. Singularity is, so the DCEPrep image (`lsaca05/dce:R2022a-dev`, MATLAB
R2022a) could be converted and run, but it still needs a licence.

Nothing on the MATLAB side has been run. The July run (`dceprep-testretest`) stays the only
MATLAB output, and it is not matched to the current inputs.

## What is needed to unblock
1. **MATLAB with a licence** on a machine that mounts the DCE BIDS share. Toolboxes
   (AGENTS.md): Curve Fitting, Parallel Computing, Statistics and Machine Learning, Image
   Processing. Either a local install or the DCEPrep container plus a licence server/file.
2. **Which code is "MATLAB 1.x"** (a decision, see the plan's open questions): the `1.2`
   tag, or the MATLAB side of the commit the Python fits ran at. They differ: the current
   MATLAB side has `find_end_ss_tv` as its default baseline detector and reads
   `VolumeTiming`; `1.2` predates both.
3. **A matched driver.** `run_dce_cli` cannot be used unchanged; see below.

## Settings the MATLAB arm must match
The Python arm's settings are in `testretest_run_config.json`; each session's literal
config is `<fits>/python_<RATER>/sub-*/ses-*/run_config.json`.

| Setting | Python arm | MATLAB (`script_preferences.txt` / argument) |
|---|---|---|
| Inputs | `dce/*_desc-bfcz_DCE`, `dce/*_label-AIF_T1map`, `anat/*_space-DCEref_label-brain_mask`, `anat/*_space-DCEref_T1map` of the rater run | same globs (`dynamic_files`, `aif_files`, `roi_files`, `t1map_files`) |
| AIF curve | `aif_mode = raw` | `aif_type = 2` |
| Blood T1 | `blood_t1_ms` 1441 (RIDER) / 1664 (QIN-GBM) | `aif_rr_type = aif_roi_static`, `blood_t1` 1.441 / 1.664 (seconds), per cohort |
| Baseline end | rater's `SteadyStateEndTimeIndex` sidecar where present, else `tv` auto-detect | `injection_time` = that frame per session, else `-2` (auto) |
| First/last frame | `start_t` 1, `end_t` last | `start_t = 1`, `end_t` empty (the shipped file has `start_t = 3`) |
| Relaxivity | staged DCE sidecar: 3.8 (3 T), 4.3 (1.5 T) | must be passed per session (see below) |
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

So the MATLAB arm needs a small driver that loops over sessions, stages inputs, and calls
`A_make_R1maps_func`, `B_AIF_fitting_func` and `D_fit_voxels_func` with the values above.
It is not written yet: it could not be run or tested here, and untested MATLAB should not
be the reference arm of the primary comparison.

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
