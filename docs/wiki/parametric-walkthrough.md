# MATLAB Parametric Walkthrough

Parametric fitting produces \(T_1\), \(T_2\), \(T_2^{*}\) and apparent diffusion coefficient
(ADC) maps from multi-parameter acquisitions. A \(T_1\) map is required input for DCE
analysis, so this is normally the first step in a DCE workflow.

For new work, the Python parametric interface described in the
[Python walkthrough](python-walkthrough.md) is recommended.

## Launching

`run_parametric.m` adds the required folders to the MATLAB path and starts `fitting_gui.m`,
the main parametric fitting interface.

The parametric fitting code is part of the main ROCKETSHIP repository, in the
`parametric_scripts` folder, and requires no separate installation step.

## Datasets to process

Multiple fitting jobs can be queued and run in sequence. **Add Datasets** appends a further
job to run after the current one completes. Each dataset is an independent job and is
configured separately.

## Notification and parallelism

- **Email to alert** — address notified when all jobs complete. This requires a file named
  `email_preferences.txt` in the `parametric_scripts` folder, specifying the account the
  notifications are sent from. The supplied `email_preferences_example.txt` shows the expected
  format.
- **Number CPUs** — number of MATLAB workers used. Setting this to the number of available
  cores is recommended.

## Job information logging

Controls which data structures and logs are written.

| Option | Effect |
| --- | --- |
| Save text log | Write a `.txt` log for each dataset job |
| Email log | Send the log files to the address given above |
| Save data structures log | Write a `.mat` file containing all relevant variables after processing, for further analysis |
| Save batch log | Write a single log covering the entire batch |
| Select old data structure | Load a previously configured dataset to run again |

## Input files

Select the files to process. Only NIfTI format is supported. The **Up** and **Down** buttons
adjust file order where several files form a single fit, for example one file per flip angle.

Add multiple files here only when they belong to one fitting operation. To run several
independent fits, use **Add Datasets** instead.

## Data order and acquisition parameters

| Parameter | Description |
| --- | --- |
| Data Order | Arrangement of the slice dimension and the parameter dimension in 4D datasets. Select **X, Y, Z / File** where each parameter value is in a separate file |
| TE/TR/FA/TI/B Value list | Parameter values in the order they appear in the dataset. Units are milliseconds for TE, TR and TI, degrees for flip angle, and mm²/s for b value |
| TR | Repetition time in milliseconds, for multi flip angle and multi inversion time \(T_1\) fitting only |
| Only Odd Echoes | For \(T_2\) and \(T_2^{*}\) fitting, use only the odd echoes. Corrects for eddy current effects where echoes were acquired on alternating gradient polarities |
| Fit all voxels | Fit every voxel individually, producing full parameter maps |
| Output file basename | Prefix for the output files |

## Fit types

All fits report the fitted parameter, and unless otherwise stated the sum of squared errors,
the coefficient of determination, and ninety-five percent confidence intervals.

### \(T_1\)

**T1 Multi TR** fits the saturation recovery curve

\[
S(\mathrm{TR}) = A\left(1 - e^{-\mathrm{TR}/T_1}\right)
\]

by nonlinear least squares, with \(T_1\) restricted to 0–10000 ms.

**T1 Multi FA Exponential** fits the spoiled gradient echo signal as a function of flip angle

\[
S(\alpha) = A\,\frac{\left(1 - e^{-\mathrm{TR}/T_1}\right)\sin\alpha}
{1 - e^{-\mathrm{TR}/T_1}\cos\alpha}
\]

by nonlinear least squares, with \(T_1\) restricted to 0–10000 ms. This is the variable flip
angle method, and is the usual route to the \(T_1\) map required for DCE analysis.

**T1 Multi FA Linear** linearises the same relationship. Plotting \(S/\sin\alpha\) against
\(S/\tan\alpha\) gives a straight line of slope \(e^{-\mathrm{TR}/T_1}\), so that

\[
T_1 = \frac{-\mathrm{TR}}{\ln(\text{slope})}
\]

This is substantially faster than the exponential fit. It reports \(T_1\) and the coefficient
of determination only.

**T1 Multi TI** fits the inversion recovery curve

\[
S(\mathrm{TI}) = \left| A \left(1 - 2e^{-\mathrm{TI}/T_1} - e^{-\mathrm{TR}/T_1}\right) \right|
\]

by nonlinear least squares, with \(T_1\) restricted to 0–10000 ms. No linear form exists for
this model, so the acceleration described under **Estimated r² map** below is unavailable.
Trimming noise regions before fitting is recommended.

### \(T_2\) and \(T_2^{*}\)

**T2 Linear Weighted** takes the logarithm of the signal and performs a linear fit weighted by
the original signal intensity. Faster than the exponential fit and nearly as accurate.

**T2 Exponential** fits \(S = A e^{-\mathrm{TE}/T_2}\) by nonlinear least squares.

**T2 Exp + C** fits \(S = A e^{-\mathrm{TE}/T_2} + C\) by nonlinear least squares, the
constant term accounting for a noise floor. This form is used for liver iron quantification
([Wood et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1895207/)). To improve stability,
\(T_2\) is first estimated from the ratio of the first and last echoes; where that estimate is
below 10 ms the constant term is included, and at or above 10 ms the simple exponential is
used instead.

**T2 Linear Fast** takes the logarithm of the signal and performs an unweighted linear fit.
The fastest option; reports \(T_2\) and the coefficient of determination only.

### Apparent diffusion coefficient

**ADC Linear Weighted** takes the logarithm of the signal and performs a linear fit weighted
by the original signal intensity. Faster than the exponential fit and nearly as accurate.

**ADC Exponential** fits \(S = A e^{-b \cdot \mathrm{ADC}}\) by nonlinear least squares.

**ADC Linear Simple** takes the logarithm of the signal and performs an unweighted linear fit.

**ADC Linear Fast** performs the same fit through a faster implementation, reporting ADC and
the coefficient of determination only.

Values are reported in mm²/s.

## Estimated r² map

Nonlinear fitting is expensive, and in background and noise voxels it produces nothing of
value. To avoid that cost, a fast linear fit is performed first and its coefficient of
determination calculated at every voxel. Voxels falling below a threshold are excluded from
the subsequent nonlinear fit.

This accelerates processing without requiring background regions to be masked out by hand.

| Control | Purpose |
| --- | --- |
| r² threshold | Voxels below this value are not fitted. Not applicable to the **Linear Fast** options. Voxels that will be retained at the current threshold are marked in red |
| Slice | Slice of the r² map currently displayed |
| Run/Update Estimate | Calculate the r² map using the corresponding **Linear Fast** option |

Set the threshold by inspecting the displayed map: it should exclude background while
retaining all anatomy of interest. Setting it too high discards genuine tissue.
