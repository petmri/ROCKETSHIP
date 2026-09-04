# DCE Options Reference

This is the complete reference for the options accepted by the Python DCE pipeline, whether
run from the command line (`run_dce_python_cli.py`) or the graphical interface
(`run_dce_python_gui.py`).

## Configuration files

Two files govern a run, with distinct roles.

**`python/dce_defaults.json`** holds every default value, limit and preference. This is the
file to edit to change how the software behaves across all runs. It is intended to be
user-editable; changing a default should never require editing source code.

**A run configuration** specifies which data to process, together with only those settings the
run overrides. Keys that match the defaults file are omitted, which keeps run configurations
short and makes the differences between them visible.

Two worked examples ship, one per layout. Both run as they stand against the test data:

| File | Data | Use it when |
| --- | --- | --- |
| `python/dce_run_example_bids.json` | `tests/data/BIDS_test`, subject `sub-02downsample` | Your data is in BIDS layout. Names two folders and no files at all. Loaded by the graphical interface, and by the command line interface when invoked without arguments. |
| `python/dce_run_example_nonbids.json` | `tests/data/BBB data p19`, a flat folder of NIfTIs | Your data is anything else. |

`python/dceprep_run_example.json` is a third form: the BIDS example with glob patterns
instead of literal file lists, which is what `run_dce_bids_batch.py --config-template`
expects for sweeping many sessions.

### BIDS and non-BIDS data

The pipeline does not require BIDS. Every input is named outright by `dynamic_files`,
`aif_files`, `roi_files` and `t1map_files`, so the layout on disk is irrelevant and the
folders may be named anything.

BIDS layout buys two conveniences, both driven by the session folders.

**Input files.** With `subject_tp_path` set, any file list left empty is filled from the
dceprep naming convention under that folder:

| Config key | Found at |
| --- | --- |
| `dynamic_files` | `dce/*desc-bfcz_DCE.nii*`, or `dce/*DCE.nii*` |
| `aif_files` | `dce/*label-AIF_T1map.nii*` |
| `roi_files` | `anat/*space-DCEref_label-brain_mask.nii*` |
| `t1map_files` | `anat/*space-DCEref_T1map.nii*` |
| `noise_files` | `anat/*label-noise_mask.nii*` (optional) |

Each file found is named in the run log, so a config that lists nothing still leaves a record
of what it ran on. Naming a file always wins: only empty lists are filled, so a config can
take the convention for most inputs and override one. `drift_files` is never discovered, as
it has no naming convention. The same discovery backs `run_dce_bids_batch.py` and the
graphical interface's *Auto find BIDS files*, so all three select identically.

**Acquisition metadata.** With `subject_source_path` set, Stage A finds the sidecar at
`<subject_source_path>/dce/*DCE.json` and reads the repetition time, flip angle, temporal
resolution and relaxivity from it.

Without the session folders, the same information has to reach the run another way. Inputs
must be named outright in the file lists. For the metadata, in order of preference:

1. Place the JSON beside the dynamic image with a matching name (`dyn.nii.gz` and
   `dyn.json`), which is found automatically.
2. Name the JSON with `stage_overrides.dce_metadata_path`.
3. State the values directly as `stage_overrides.tr_ms`, `fa_deg` and `time_resolution_sec`,
   as `dce_run_example_nonbids.json` does.

`relaxivity` is subject to the same routes and has no default anywhere, so a run that
supplies it by none of them stops rather than guessing.

### File paths

A path in a run configuration may be written relative. It is resolved against **the directory
containing the configuration file**, not the directory the command happens to be run from, so
a configuration stored beside its data keeps working wherever it is launched. This applies to
the image and mask lists, to `output_dir` and `checkpoint_dir`, and to the path-valued
settings inside `stage_overrides` (`dce_metadata_path`, `import_aif_path`,
`time_vector_path`).

Paths given on the command line are resolved against the current working directory instead,
which is where they were typed. This covers `--output-dir`, `--checkpoint-dir` and a
`--set` that names a file.

Absolute paths are used exactly as given. The parametric T1 interface follows the same rule.

## How much a run prints

Every CLI and batch driver takes `--verbosity quiet|normal|detailed|debug` (`-q`, `-v`, `-vv`).
The default, `normal`, reports what the run is using, each stage as it finishes, and a closing
summary. `detailed` adds the settings this config overrides, where each per-scan value came
from, the Stage-D backend choice, and every file written -- it is what the GUI log shows.

Verbosity selects what is rendered, not what is recorded: `<output_dir>/dce_pipeline_events.jsonl`
holds every event at every level. `--events on` puts that machine-readable stream on stdout in
place of human progress, which is how the GUI drives its progress bar.

## Precedence

For options within `stage_overrides`, values are resolved in this order:

1. An explicit value in the run configuration, or a `--set KEY=VALUE` argument
2. The value in `python/dce_defaults.json`
3. Otherwise the run stops with an error

There is no fallback value embedded in source code. A key supplied by neither the run
configuration nor the defaults file raises an error naming both the key and the file, rather
than proceeding on an assumed value. A key present in the run configuration but not declared
in the defaults file is rejected as a probable typographical error rather than being ignored.

A small number of keys are declared optional rather than defaulted, because their absence is
meaningful and hands the decision to automatic detection. `steady_state_end` is the clearest
example. These resolve to an unset state rather than raising an error.

### Per-scan values

`relaxivity` and `hematocrit` describe the scan rather than the analysis, and may legitimately
differ between participants. Their precedence is therefore inverted relative to every other
option:

1. The JSON sidecar accompanying the DCE image
2. The run configuration, or `--set`
3. `python/dce_defaults.json`
4. Otherwise the run stops with an error

!!! danger "Relaxivity has no default"
    The correct relaxivity depends on the contrast agent, so there is no safe value to fall
    back on. A run that does not supply one stops. An incorrect value rescales every
    concentration, and therefore every \(K^{trans}\), producing results that appear entirely
    plausible but are wrong by a constant factor.

    Haematocrit does have a default of 0.45, on the basis that it is normally a single
    study-wide value.

## Top-level keys

| Key | Description |
| --- | --- |
| `subject_source_path` | Optional. BIDS session rawdata folder; enables metadata sidecar discovery at `<path>/dce/*DCE.json`. Omit for non-BIDS data |
| `subject_tp_path` | Optional. BIDS session derivatives folder; empty file lists are discovered beneath it. Omit for non-BIDS data |
| `output_dir` | Destination for maps, logs, figures and summaries. The only required key |
| `checkpoint_dir` | Optional folder for stage checkpoints |
| `backend` | `auto`, `cpu` or `gpufit`; see [GPU and CPU Acceleration](wiki/enable-gpu-acceleration.md) |
| `write_xls` | Write region of interest results to a spreadsheet |
| `aif_mode` | `fitted`, `raw` or `imported`; see [Arterial Input Function](reference/models/aif.md) |
| `imported_aif_path` | Curve to load when `aif_mode` is `imported` |
| `dynamic_files` | Dynamic DCE NIfTI files |
| `aif_files` | Arterial region mask files |
| `roi_files` | Tissue region of interest mask files |
| `t1map_files` | Pre-contrast \(T_1\) map files |
| `noise_files` | Optional noise mask files |
| `drift_files` | Optional drift correction files |
| `model_flags` | Models to fit |
| `stage_overrides` | Advanced settings and fit controls |

### `model_flags`

Set each entry to `1` to enable the model or `0` to disable it. Any combination may be
enabled; each produces its own outputs.

`tofts`, `ex_tofts`, `patlak`, `tissue_uptake`, `two_cxm`, `fxr`, `auc`, `nested`, `FXL_rr`

Equations and selection guidance for each are in the
[pharmacokinetic models reference](reference/models/index.md).

---

## Stage overrides

### Runtime and staging

| Option | Description |
| --- | --- |
| `stage_a_mode` | `real` or `scaffold` |
| `stage_b_mode` | `real`, `scaffold` or `auto` |
| `stage_d_mode` | `real`, `scaffold` or `auto` |
| `rootname` | Prefix for output filenames |
| `write_param_maps` | Write per-voxel parameter map NIfTIs. Default `true` |
| `write_postfit_arrays` | Export post-fit arrays as `*_postfit_arrays.npz` |

### Backend

| Option | Description |
| --- | --- |
| `force_cpu` | When `backend` is `auto`, force the standard CPU path if non-zero |

### Acquisition and timing

| Option | Description |
| --- | --- |
| `dce_metadata_path` | Explicit path to a metadata JSON file |
| `tr_ms` | Repetition time, in milliseconds |
| `fa_deg` | Flip angle, in degrees |
| `time_resolution_sec` | Temporal resolution, in seconds |
| `time_vector_path` | Path to an explicit time vector, for unequally spaced acquisitions |

Repetition time, flip angle and temporal resolution are preferably resolved from the JSON
sidecar accompanying the DCE image, or from an explicit `dce_metadata_path`. Where no metadata
JSON is available, all three must be set manually. Setting only some of them while a metadata
JSON is present is rejected: supply all three, or none.

### Trimming leading frames

| Option | Description |
| --- | --- |
| `start_t` | One-based first dynamic frame to analyse. Absent, with no automatic detection: 1 |
| `end_t` | One-based last dynamic frame to analyse. Absent: the last frame |
| `start_t_auto_method` | `none`, the default, or `transient` to detect the leading frames |

The first frames of a dynamic series are often acquired before the magnetisation reaches
steady state. They carry transient high signal and slice-to-slice banding, and they corrupt
anything downstream that assumes a flat pre-contrast baseline. Two ways to remove them:

1. **A fixed count.** Set `start_t` to the first good frame — `start_t: 3` discards two.
2. **Automatic detection.** Leave `start_t` unset and set `start_t_auto_method: transient`.

`start_t` wins where both are set, so naming the frame remains the way to pin a run
reproducibly.

Note that "steady state" here means the *magnetisation* steady state, and is unrelated to the
`steady_state_start`/`steady_state_end` keys below, which bound the pre-contrast baseline
window. This trim happens before contrast arrival, which is what those keys locate.

**The `transient` detector** weighs two kinds of evidence, because on 3D acquisitions the mean
signal alone misses most affected frames. Slice-to-slice banding decides *whether* the first
frame is transient: a frame acquired off steady state weights k-space partitions unevenly, and
the resulting oscillation along the slice axis largely cancels in the volume mean. The mean
elevation then decides *how far* the transient extends, since a decaying transient leaves the
following frame elevated but no longer banded.

| Option | Description |
| --- | --- |
| `start_t_auto_osc_z` | Banding ratio above which a frame is transient. Default 5.0 |
| `start_t_auto_z` | Mean deviation, in noise σ, to flag the first frame. Default 4.0 |
| `start_t_auto_z_ext` | Mean deviation, in σ, to extend an established transient. Default 2.0 |
| `start_t_auto_max_chop` | Never trim more than this many frames. Default 3 |
| `start_t_auto_max_baseline` | An arrival past this frame is not believed. Default 8 |

Contrast arrival comes from the detector named by `steady_state_auto_method`, run on the
untrimmed series. Both kinds of evidence measure the first frame against a plateau taken from
the remaining pre-arrival frames, so both depend on that arrival being right. Where it lands
beyond `start_t_auto_max_baseline`, or is not found at all, the plateau would be post-contrast
signal against which the true baseline itself reads as transient — so the detector declines
instead, the run proceeds with `start_t` at 1, and the reason is recorded in the run summary
under `stages.A.timepoint_window.start_t_auto`. Volumes with fewer than seven usable slices
cannot show banding and fall back to the mean-signal evidence alone.

### Baseline and injection timing

The end of the pre-contrast baseline is resolved with the following precedence:

1. `steady_state_end`, an explicit manual setting
2. `SteadyStateEndTimeIndex` in the arterial mask's JSON sidecar
3. `steady_state_auto_method`, automatic detection

Where none is set, automatic detection uses the total variation method.

**`steady_state_start`, `steady_state_end`** pin the baseline window explicitly. These are
applied to the analyzed series, after any trimming by `start_t` and `end_t`.

**AIF JSON sidecar.** A `.json` sidecar placed alongside `aif_files[0]`, following the same
naming convention as the DCE metadata sidecar, may carry a one-based
`SteadyStateEndTimeIndex` field, for example `{"SteadyStateEndTimeIndex": 3}`. This is the
recommended way to fix a reproducible baseline for a particular dataset, since the setting
travels with the data rather than with the run configuration.

The index counts acquired frames, before any trimming, so the pinned frame remains the same
physical timepoint however the analysis window is set; the pipeline accounts for the trim
internally. Where `start_t` advances beyond the pinned frame the run stops, since the window
has removed the entire baseline.

**`steady_state_auto_method`** selects the automatic detector, used only when neither of the
above is set.

| Method | Description |
| --- | --- |
| `tv` | Total variation denoising followed by detection of the first significant upward step. The default |
| `legacy_sobel` | Sobel edge and line-fit heuristic applied to the global signal |
| `piecewise_constant` | Brute-force two-constant split with local minimum backtracking |
| `glr` | One-sided generalized likelihood ratio change-in-mean detector |
| `biexp_fit` | Six-parameter biexponential fit to the mean arterial signal curve, seeded by `tv` |

The `biexp_fit` method differs from the shape heuristics in also reporting where the upslope
ends, which then defines the end of the injection and the starting point for the Stage B fit.
It falls back to its `tv` seed where the fit does not converge. On evaluation against expert
ratings it is less accurate than `tv`, and is not the default.

The following aliases are also accepted: `legacy`, `dce_auto_aif`, `sobel`, `piecewise`,
`find_end_ss`, `edge`, `find_end_ss_edge`, `find_end_ss_tv`, `biexp`, `find_end_ss_biexp`.

**`end_injection_min`** sets the end of the injection, in minutes. There is no corresponding
`start_injection_min`, because the start of the injection is defined as the resolved end of
the baseline; change it through the baseline settings above. Supplying `start_injection_min`
or `start_injection` raises an error directing you to those settings.

### Analysis window

| Option | Description |
| --- | --- |
| `restrict_fit_start_min` | Start of the fitted window, in minutes on the time axis |
| `restrict_fit_end_min` | End of the fitted window, in minutes on the time axis |

Unlike `start_t` and `end_t`, these discard no data and leave the baseline intact, which makes
them the appropriate way to fit only a late phase of the curve.

### Concentration conversion

| Option | Description |
| --- | --- |
| `relaxivity` | Contrast agent relaxivity. Per-scan; no default; required |
| `hematocrit` | Haematocrit. Per-scan; default 0.45 |
| `blood_t1_ms` | Fixed pre-contrast arterial \(T_1\), in milliseconds |
| `noise_pixsize` | Size of the corner square used for noise estimation |
| `snr_filter` | Minimum signal to noise ratio for arterial voxels |

`blood_t1_ms` is read in milliseconds and range checked between 50 and 20000. It is not
rescaled according to magnitude, so a value supplied in seconds raises an error rather than
being silently converted.

The conversion these options control is documented in
[Signal to Concentration](reference/signal-to-concentration.md).

### Arterial input function fit

The curve mode is the top-level `aif_mode` field rather than a stage override.

| Option | Description |
| --- | --- |
| `import_aif_path` | Override for the imported curve path |
| `aif_initial_values` | Initial values, as `[A, B, c, d]` |
| `aif_lower_limits` | Lower bounds, as `[A, B, c, d]` |
| `aif_upper_limits` | Upper bounds, as `[A, B, c, d]` |
| `aif_TolFun` | Function tolerance |
| `aif_TolX` | Parameter tolerance |
| `aif_MaxIter` | Maximum iterations |
| `aif_MaxFunEvals` | Maximum function evaluations |
| `aif_Robust` | `off`, `Bisquare` or `LAR` |
| `aif_Robust_timing` | As `aif_Robust`, applied to the Stage A timing pass. Defaults to `aif_Robust` |
| `aif_peak_weight_exponent` | Exponent for shape-based de-weighting of the peak sample. Default 2; 0 disables |
| `save_aif_figure` | Write the fit figure `dceAIF_fitting.png`. Default `true` |

`aif_Robust` selects a residual-based robust estimator: `off` is ordinary least squares,
`Bisquare` is Tukey biweight iteratively reweighted least squares, and `LAR` is a soft L1
loss. The default is `off`.

`aif_Robust_timing` is separated from `aif_Robust` because the timing pass estimates the
position of the peak, and rejecting the peak as an outlier removes the pass's primary
evidence.

`aif_peak_weight_exponent` de-weights the peak sample using the shape of the curve rather than
its residuals, because the peak has full leverage in this model and a residual-based estimator
cannot detect an inflated value there.

The fitted form and these mechanisms are described in the
[Arterial Input Function reference](reference/models/aif.md).

### Model fitting

| Option | Description |
| --- | --- |
| `time_smoothing` | Temporal smoothing mode |
| `time_smoothing_window` | Temporal smoothing window length |
| `fxr_fw` | Tissue water fraction for the FXR model. Default 0.8 |
| `fit_voxels` | Fit every voxel. Default `true` |
| `time_unit` / `timer_unit` | `minutes` or `seconds`, an optional hint for the direct fit path |

Setting `fit_voxels` to `false` selects region-only mode: the per-voxel fit is skipped and
only each region's averaged concentration curve is fitted. This is considerably faster, and
for nonlinear models the averaging before fitting reduces noise. It requires `roi_files`, and
parameter maps are not written. Each region is averaged over its intersection with the primary
fit region, so `roi_files[0]` should be the encompassing region, such as a whole-brain mask.

Fitting for the tissue uptake and two-compartment exchange models is always carried out
internally in minutes with rate constants per minute. Bounds and initial values are
interpreted in the same units as the supplied time vector and converted internally, and
returned rate parameters are converted back to match.

### Parameter bounds and initial values

Each fitted parameter has a lower limit, an upper limit and an initial value:

```
voxel_lower_limit_<param>
voxel_upper_limit_<param>
voxel_initial_value_<param>
```

where `<param>` is one of `ktrans`, `ve`, `vp`, `fp`, `tp`, `tau` or `ktrans_RR`. The
reference region model additionally uses `voxel_value_ve_RR`, a fixed rather than fitted
value.

Several models have their own settings, distinguished by a suffix, for example
`voxel_lower_limit_ve_2cxm` or `voxel_initial_value_fp_tissue_uptake`. Where a model-specific
setting exists it takes precedence over the general one for that model. Current defaults for
each parameter and model are listed on the individual
[model reference pages](reference/models/index.md).

Convergence is controlled by `voxel_TolFun`, `voxel_TolX`, `voxel_MaxIter`,
`voxel_MaxFunEvals` and `voxel_Robust`, which also accept model-specific suffixes.

### Acceleration

| Option | Description |
| --- | --- |
| `gpu_tolerance` | Solver tolerance for the accelerated path. Default 10⁻⁶ |
| `gpu_max_n_iterations` | Maximum solver iterations per voxel |
| `gpu_initial_value_ktrans` | Initial \(K^{trans}\) for the accelerated path |
| `gpu_initial_value_ve` | Initial \(v_e\) for the accelerated path |
| `gpu_initial_value_vp` | Initial \(v_p\) for the accelerated path |
| `gpu_initial_value_fp` | Initial \(F_p\) for the accelerated path |

Accelerated fitting is available for the Tofts, extended Tofts, Patlak, tissue uptake and
two-compartment exchange models. `gpu_tolerance` applies to both the GPU and CPU acceleration
paths.

!!! warning "Tightening the tolerance reduces voxel yield"
    A tighter `gpu_tolerance` does not improve accuracy. Below approximately 10⁻¹⁰ the
    accelerated solvers begin marking voxels as non-converged and the pipeline excludes them.
    Leave this at its default unless you have a specific reason to change it. Further detail
    is in [GPU and CPU Acceleration](wiki/enable-gpu-acceleration.md).

The Part D stage summary records `selected_backend`, `acceleration_backend`, `backend_reason`
and `backend_used`.

---

## Notes

- MATLAB-style numeric expressions such as `10^-7` are accepted in string values within
  `python/dce_defaults.json`.
- `dce/dce_preferences.txt` and `script_preferences.txt` configure the MATLAB pipeline only.
  The Python implementation does not read them.
- The graphical interface provides file browsers for all path inputs shown on the form.
  `import_aif_path` has no dedicated field and should be set through the run configuration
  when using imported arterial input function mode.
