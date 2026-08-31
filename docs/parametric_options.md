# Parametric T1 Options Reference

This is the complete reference for the options accepted by the Python parametric T1
pipeline, whether run from the command line (`run_parametric_python_cli.py`), the graphical
interface (`run_parametric_python_gui.py`) or the batch driver
(`run_parametric_bids_batch.py`).

The pipeline fits \(T_1\) from a variable flip angle (VFA) series: several spoiled gradient
echo images of the same anatomy acquired at different flip angles, from which \(T_1\) and the
equilibrium magnetization \(\rho\) are recovered voxel by voxel.

## Configuration files

Two files govern a run, with distinct roles.

**`python/parametric_defaults.json`** holds every default value and preference. This is the
file to edit to change how the software behaves across all runs. It is intended to be
user-editable; changing a default should never require editing source code.

**A run configuration** specifies which data to process, together with only those settings
the run overrides. Keys that match the defaults file are omitted, which keeps run
configurations short and makes the differences between them visible.

`python/parametric_run_example.json` is a worked example that runs as it stands against the
test data. It is what the command line interface loads when invoked without `--config`, and
what the graphical interface opens at startup and returns to under *Reset Defaults*.

!!! note "These were one file until 2.0"
    `parametric_default.json` was both the defaults file and a runnable example, so editing a
    default also edited an example run. Splitting them is what makes the defaults file
    genuinely editable. A configuration written against the old file still loads.

### File paths

Relative paths in a run configuration resolve against **the directory holding that
configuration file**, not the working directory the command was issued from. A configuration
and its data can therefore be moved together, and run from anywhere.

Paths given on the command line (`--output-dir`) resolve against the working directory
instead, since that is where they were typed.

The path-valued keys are `output_dir`, `vfa_files`, `mask_file` and `b1_map_file`. These
describe one study rather than how the software behaves, so they belong in a run
configuration and are not present in the defaults file.

## How much a run prints

Every CLI and batch driver takes `--verbosity quiet|normal|detailed|debug` (`-q`, `-v`,
`-vv`). The default, `normal`, reports what the run is using, each stage as it finishes, and
a closing summary. `detailed` adds the resolved flip angles and repetition time with their
source, the backend choice, and every file written -- it is what the GUI log shows.

Verbosity selects what is rendered, not what is recorded:
`<output_dir>/parametric_t1_events.jsonl` holds every event at every level.

## Precedence

Values are resolved in this order:

1. An explicit value in the run configuration, or a `--set KEY=VALUE` argument
2. The value in `python/parametric_defaults.json`
3. Otherwise the run stops with an error

There is no fallback value embedded in source code. A key supplied by neither the run
configuration nor the defaults file raises an error naming both the key and the file, rather
than proceeding on an assumed value. A key present in the run configuration but not declared
in the defaults file is rejected as a probable typographical error rather than being ignored.

A small number of keys are declared **optional** rather than defaulted, because their absence
is meaningful and hands the decision to automatic detection. `tr_ms` and `flip_angles_deg`
are the clearest examples: left unset, both are read from the images' JSON sidecars.

The graphical interface's *Resolved Settings* view reports this directly, showing for every
key both the value the run will use and whether it came from the run configuration, the
defaults file, or an edit made in the form.

### Acquisition metadata

`tr_ms` and `flip_angles_deg` describe the acquisition rather than the analysis, and are read
from the JSON sidecar beside each VFA image when not supplied. `dce2bids` writes them.

!!! note "An unset `b1_map_file` is not the same as no B1 correction"
    With no `b1_map_file`, the pipeline looks beside each VFA image for
    `B1_scaled_FAreg.nii` or `B1_scaled_FAreg.nii.gz` -- the MATLAB naming convention -- and
    uses the first it finds. Only if none is present are the nominal flip angles used. The
    run summary and the `detailed` log both report which of the three happened
    (`explicit`, `auto` or `none`), so check there rather than assuming.

!!! warning "Flip angles must match the image being fitted"
    `flip_angles_deg` needs one entry per flip frame of the image actually selected. A
    preprocessed VFA image combining fewer frames than there are sidecars produces a count
    mismatch, which the pipeline rejects rather than silently mispairing. If you supply the
    angles yourself, supply the ones belonging to the frames present.

## Keys

### Inputs

| Key | Type | Meaning |
| --- | --- | --- |
| `output_dir` | path | **Required.** Where maps, the run summary and QC figures are written. |
| `vfa_files` | list of paths | The VFA series, one file per flip angle. A single 4-D file with one frame per angle is also accepted, as is a single 3-D file when `flip_angles_deg` matches its last axis. |
| `flip_angles_deg` | list of numbers | One angle per flip frame, in degrees. Optional; read from the sidecars when absent. |
| `tr_ms` | number | Repetition time in milliseconds. Optional; read from the sidecars when absent. |
| `mask_file` | path | Restricts fitting to voxels where the mask is greater than zero. Optional; every voxel is fitted when absent. |
| `b1_map_file` | path | \(B_1\) correction map, scaling the nominal flip angle per voxel. Optional -- but see below, absent does not mean unused. |

### Fitting

| Key | Default | Meaning |
| --- | --- | --- |
| `fit_type` | `t1_fa_fit` | Which estimator to use. See below. |
| `backend` | `auto` | `auto` tries GPU, then CPU acceleration, then the plain Python path. `cpu` and `gpufit` select one outright. |
| `rsquared_threshold` | `0.6` | Voxels whose fit scores below this are replaced by `invalid_fill_value`. Set to `0` to keep every fit. |
| `invalid_fill_value` | `-1.0` | Written wherever a fit was rejected. |
| `odd_echoes` | `false` | Fit only the odd-numbered frames, for interleaved acquisitions. |
| `xy_smooth_sigma` | `0.0` | Gaussian smoothing sigma in voxels, applied in-plane before fitting. `0` disables it. |

The three estimators:

| `fit_type` | Method | When to use it |
| --- | --- | --- |
| `t1_fa_fit` | Non-linear least squares over all flip angles | The default. Most accurate, and the only one that uses every angle properly. |
| `t1_fa_linear_fit` | Linearized form, solved directly | Faster, but noise on the signal biases the result because the linearization is not noise-preserving. |
| `t1_fa_two_point_fit` | Closed form from two flip angles | Only where two angles were acquired, or for a fast approximation. |

### Outputs

| Key | Default | Meaning |
| --- | --- | --- |
| `output_basename` | `T1_map` | Leading part of the map filenames. |
| `output_label` | `""` | Trailing label distinguishing runs. Derived from the input when empty. |
| `write_r_squared` | `true` | Write the goodness-of-fit map beside the \(T_1\) map. |
| `write_rho_map` | `false` | Write the equilibrium magnetization map. |
| `write_qc_figures` | `true` | Write the QC figures described below. |

A run writes `<output_basename>_<fit_type>_<output_label>.nii.gz` for \(T_1\) in
milliseconds, plus `Rsquared_...` and `<output_basename>_rho_...` when enabled, a
`parametric_t1_run.json` summary, and `parametric_t1_events.jsonl`.

### QC figures

Three figures, written when `write_qc_figures` is set, answering the three questions worth
asking of a finished fit:

| Figure | Shows |
| --- | --- |
| `qc_t1_histogram_*.png` | The distribution of fitted \(T_1\), median marked, clipped at the 99th percentile so a few runaway voxels do not flatten the tissue peak. |
| `qc_r2_histogram_*.png` | Goodness of fit against `rsquared_threshold`, annotated with how many voxels that threshold discards. \(R^2\) below zero means the fit is worse than a flat line; those are clipped into the first bin and counted in the axis label. |
| `qc_t1_montage_*.png` | Slices through the \(T_1\) map on one shared window, so a slice reads bright because its \(T_1\) is higher rather than because it was scaled alone. |

Figures are decorative. A run that produced maps is a successful run even when matplotlib is
absent or a plot fails, so a missing figure never fails a run.

## Notes

**Units.** \(T_1\) is written in milliseconds, matching `tr_ms`. The DCE pipeline accepts a
\(T_1\) map in either milliseconds or seconds and infers which from the magnitude, so a map
produced here can be handed to it directly.

**Older key spellings.** `file_list`, `parameters`, `tr` and `xy_smooth_size` are still
accepted as synonyms for `vfa_files`, `flip_angles_deg`, `tr_ms` and `xy_smooth_sigma`.
