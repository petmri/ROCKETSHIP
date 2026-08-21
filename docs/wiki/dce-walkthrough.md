# MATLAB DCE Walkthrough

This guide covers DCE-MRI analysis using the MATLAB interface. For new work, the
[Python implementation](python-walkthrough.md) is recommended.

## Launching

Add the main ROCKETSHIP folder to your MATLAB path, then start the DCE interface in one of
three ways:

| Command | Effect |
| --- | --- |
| `rocketship` | Launch the main interface, from which DCE analysis can be selected |
| `run_dce` | Add the required subfolders to the path, then launch DCE analysis |
| `dce` | Launch DCE analysis directly, assuming the subfolders are already on the path |

## Analysis stages

DCE analysis proceeds in four stages, named A, B, D and E. There is no Part C. Each stage
takes the saved output of the previous one, so a stage can be repeated with different settings
without recomputing those before it.

| Stage | Purpose |
| --- | --- |
| A | Load images and regions of interest, convert signal to concentration |
| B | Define analysis timing and derive the arterial input function |
| D | Fit the pharmacokinetic models and produce parameter maps |
| E | Analyse the fitted curves and compare models |

---

## Part A: Image loading and concentration conversion

Loads the input images and region of interest files, then calculates concentration versus time
curves. The mathematics of this conversion is documented in
[Signal to Concentration](../reference/signal-to-concentration.md).

### Input dynamic datasets

Load the dynamic series, normally a set of \(T_1\)-weighted images. DICOM and NIfTI formats
are both supported. The **File Order** setting specifies the arrangement of the slice and time
dimensions within the data.

### Region of interest and \(T_1\) files

**Select AIF/RR** defines the region from which the arterial input function or reference region
is taken. This may be either a binary mask, with ones inside the region and zeros elsewhere,
or a \(T_1\) map in which the region carries valid \(T_1\) values in milliseconds and all other
voxels are zero.

**Select ROI** defines the region over which analysis is performed, excluding background and
anatomy of no interest to reduce processing time. It accepts the same two forms.

**Select T1 map** supplies \(T_1\) values for tissue and for the arterial region. It is
required whenever a binary mask was chosen for either of the selections above. Values are
expected in milliseconds; values supplied in seconds are detected and converted.

**Select Drift ROI** is optional and defines a region used to correct for scanner signal
drift. The region must have constant signal intensity throughout the acquisition, so an
external reference such as a vial of contrast-doped water is required. Tissue cannot be used,
because tissue always takes up some contrast agent. All signal intensities are then normalised
in time against this reference.

**Global drift correction** calculates a single correction factor for the whole image. When
disabled, a separate factor is calculated for each slice; slices in which the reference is not
visible use the factor from the nearest slice in which it is.

### Noise handling

Noise is estimated to allow low signal to noise ratio voxels to be excluded from the arterial
input function. Choose one of:

- **Pick noise file** — a binary mask selecting a region of air containing only noise.
- **Derive noise from corner square** — a square of the given size in the corner of the image.

### Image parameters

| Parameter | Description |
| --- | --- |
| TR | Repetition time of the dynamic series, in milliseconds |
| FA | Flip angle of the dynamic series, in degrees |
| Hematocrit | Subject haematocrit, used to convert whole blood to plasma concentration |
| SNR for AIF filter | Arterial voxels below this signal to noise ratio are excluded |
| End Baseline, image num | Index of the last baseline image before contrast arrival. Use −1 to select it interactively from the arterial curve, or −2 to determine it automatically |
| Contrast agent r1 | Longitudinal relaxivity of the contrast agent |
| Injection Duration | Duration of the injection in number of images, used for automatic arterial selection |

Published relaxivity values for common contrast agents are tabulated in
[Shen et al. (2015)](https://pubmed.ncbi.nlm.nih.gov/25658049/).

### Arterial input function selection

| Option | Behaviour |
| --- | --- |
| ROI defined | Region supplied by the user; \(T_1\) taken from the \(T_1\) map or the region file |
| ROI w/ Static T1 | Region supplied by the user; \(T_1\) fixed to the value given here |
| Auto | Voxels selected automatically by their resemblance to a typical arterial curve, that is a fast rise followed by a slow decay; \(T_1\) taken from the \(T_1\) map |
| Auto w/ Static T1 | Automatic selection as above, with \(T_1\) fixed to the value given here |

**Blood T1** sets the fixed pre-contrast arterial \(T_1\), in milliseconds, for the static
options. A fixed value is often more stable than a measured one, because arterial voxels are
susceptible to inflow and partial volume effects.

---

## Part B: Timing and the arterial input function

Defines the timing parameters for the analysis and derives the arterial input function curve.
Requires the output of Part A.

### Results of A

Select the `.mat` file saved by Part A.

### Input function

| Option | Behaviour |
| --- | --- |
| Raw AIF | Use the measured samples unmodified |
| Fitted AIF | Fit the measured samples to a linear upslope and biexponential decay, and use the fitted curve. Reduces noise; may reduce accuracy |
| Import AIF | Load a curve saved from a previous Part B run, or supplied manually |
| Create average AIF | Average curves from several saved runs, typically across subjects, into a single population input function |

The fitted form and its parameters are documented in the
[Arterial Input Function reference](../reference/models/aif.md).

A manually supplied import file must contain the following variables:

| Variable | Contents |
| --- | --- |
| `Cp_use` | Arterial concentration curve, in mM |
| `Stlv_use` | Arterial signal intensity curve, in arbitrary units, used only for area under the curve calculations |
| `import_timer` | Time of each data point, in minutes |
| `import_start` | End of the baseline and start of bolus arrival, in minutes |

### Timing parameters

| Parameter | Description |
| --- | --- |
| Analysis Interval | Restrict the analysis to a time interval, in minutes. Use 0 for both fields to apply no restriction |
| Injection Duration | Start and end of bolus arrival, in minutes. Use −1 in either field to determine it automatically. Used for the input function fit only |
| Time resolution | Temporal resolution of the dynamic series, in seconds |
| Manual time vector | Load a time vector from a `.mat` file containing the variable `timer`, in minutes. Required for series with unequally spaced timepoints |

---

## Part D: Model fitting

Calculates the parameter maps. Requires the output of Part B.

### Results of B

Select the `.mat` file saved by Part B.

### Model selection

Any number of models may be selected; a separate output file is produced for each. Full
equations, parameters and selection guidance are in the
[pharmacokinetic models reference](../reference/models/index.md).

| Model | Parameters |
| --- | --- |
| [Tofts](../reference/models/tofts.md) | \(K^{trans}\), \(v_e\) |
| [Tofts w/ Vp](../reference/models/extended-tofts.md) | \(K^{trans}\), \(v_e\), \(v_p\) |
| [Patlak](../reference/models/patlak.md) | \(K^{trans}\), \(v_p\) |
| [Tissue Uptake](../reference/models/tissue-uptake.md) | \(K^{trans}\), \(F_p\), \(T_p\) |
| [2CXM](../reference/models/two-compartment-exchange.md) | \(K^{trans}\), \(v_e\), \(v_p\), \(F_p\) |
| [FXR](../reference/models/fxr.md) | \(K^{trans}\), \(v_e\), \(\tau_i\) |
| [Area under curve](../reference/models/auc.md) | AUC, normalised AUC |
| Nested Model | Variable |

### Smoothing

Smoothing may be applied in time or in the imaging plane.

- **Time smoothing** is not generally recommended, since model fitting already smooths
  effectively in the time dimension. Robust local regression is useful for suppressing
  isolated outlying timepoints arising from motion or artefact, preventing them from
  influencing the fit.
- **XY smooth size** sets the standard deviation, in voxels, of a Gaussian kernel applied in
  the imaging plane.

### Fitting

**ROIs to fit** performs a single fit per region. All voxels within the region are averaged at
each timepoint, and one fit is performed on the resulting curve. Regions may be supplied as
ImageJ region files (`.roi`) or as NIfTI binary masks. Averaging before fitting improves the
signal to noise ratio of the curve substantially, at the cost of any within-region detail.

**Fit all voxels** performs an independent fit at every voxel, producing full parameter maps.
This is considerably more time consuming; see
[GPU and CPU Acceleration](enable-gpu-acceleration.md).

### Number of CPUs

Sets the number of MATLAB workers used for fitting. Use 0 for the number of available cores,
or −1 for one fewer than that. The latter leaves a core free, which keeps the machine usable
during a long run.

---

## Part E: Curve analysis and model comparison

Runs `fitting_analysis.m` to examine the fitted curves and compare models.

### Fitting results

- **Models to Analyze** — the `.mat` files saved by Part D.
- **Voxel Selection Image** — the image displayed for interactive voxel selection, usually a
  parameter map such as \(K^{trans}\), or the \(T_1\) map.

### Fitting analysis

- **ROI List** — regions processed in Part D. Selecting one opens the results for that region.
  Empty if no regions were processed.
- **Show original unsmoothed data** — overlay the unsmoothed curve where time smoothing was
  applied.
- **Show 95% confidence interval curves** — overlay the confidence bounds on the fitted curve.
- **Run Voxel Analysis** — open the interactive voxel analysis tool.

### Statistical model comparison

Where several models have been fitted to the same data, these tests indicate which is best
supported.

| Test | Applies |
| --- | --- |
| Akaike | Akaike information criterion, penalising additional parameters ([Glatting et al.](https://pubmed.ncbi.nlm.nih.gov/18072493/)) |
| F Test | F test between nested models ([Glatting et al.](https://pubmed.ncbi.nlm.nih.gov/18072493/)) |
| FMI/FRI | Fit micro and macro indices ([Balvay et al.](https://pubmed.ncbi.nlm.nih.gov/16155897/)) |

**Perform ROI Comparison** and **Perform Voxel Comparison** apply the selected tests to region
and voxel results respectively.
