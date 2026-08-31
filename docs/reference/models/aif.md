# Arterial Input Function

Every convolution model requires the plasma concentration curve \(C_p(t)\) delivered to the
tissue. ROCKETSHIP measures this curve from arterial voxels in the dynamic series itself, and
optionally replaces the measured samples with a fitted analytic form before the tissue models
use it.

The arterial input function is the single largest source of systematic error in quantitative
DCE-MRI. Because it appears in every model as the input to a convolution, an error in its
amplitude propagates directly and proportionally into \(K^{trans}\) for every voxel in the
image.

## Curve modes

The `aif_mode` option selects where the curve comes from and whether it is fitted.

| Mode | Behavior |
| --- | --- |
| `fitted` | Take samples from the arterial mask and replace them with a fitted biexponential |
| `raw` | Take samples from the arterial mask and use them unmodified |
| `imported` | Load a previously saved curve from `imported_aif_path` and never fit it |

Fitting suppresses noise in the input curve, at the cost of imposing a shape the data may not
support. The raw mode preserves everything the acquisition measured, including its noise. The
imported mode allows a curve measured elsewhere, or a population average built from several
subjects, to be substituted.

## The fitted model

The analytic form is a linear upslope followed by a biexponential decay, constrained to be
continuous at the junction:

\[
C_p(t) =
\begin{cases}
0 & t < t_{base} \\[2ex]
(A + B)\,\dfrac{t - t_{base}}{t_{0} - t_{base}} & t_{base} \le t < t_{0} \\[2ex]
A\,e^{-c\,(t - t_{0})} + B\,e^{-d\,(t - t_{0})} & t \ge t_{0}
\end{cases}
\]

Here \(t_{base}\) is the end of the pre-contrast baseline, \(t_{0}\) is the end of the
upslope, and the curve peaks at \(A + B\) where the two segments meet. The two decay rates
\(c\) and \(d\) describe first pass washout and the slower recirculation and clearance phase
respectively.

Where the curve being fitted is in arbitrary signal intensity units rather than
concentration, a constant baseline offset is added to all three segments, so that the ramp
climbs from the baseline and the decay relaxes back towards it.

## Fit parameters

| Parameter | Symbol | Role |
| --- | --- | --- |
| First decay amplitude | \(A\) | Amplitude of the fast washout term |
| Second decay amplitude | \(B\) | Amplitude of the slow clearance term |
| First decay rate | \(c\) | Fast washout rate constant |
| Second decay rate | \(d\) | Slow clearance rate constant |

The four values are configured as vectors in \([A, B, c, d]\) order through the
`aif_initial_values`, `aif_lower_limits` and `aif_upper_limits` options. Convergence is
controlled by `aif_TolFun`, `aif_TolX`, `aif_MaxIter` and `aif_MaxFunEvals`.

The two transition times \(t_{base}\) and \(t_{0}\) are not free parameters of this fit. They
are resolved beforehand, as described below.

## Baseline and injection timing

The end of the baseline, \(t_{base}\), determines which frames are averaged to form the
pre-contrast signal, and therefore anchors the entire concentration conversion. It is
resolved with the following precedence:

1. An explicit `steady_state_end` in the run configuration.
2. A `SteadyStateEndTimeIndex` field in the JSON sidecar accompanying the arterial mask.
3. Automatic detection, using the method named by `steady_state_auto_method`.

The sidecar mechanism is the recommended way to pin a fixed, reproducible baseline for a
particular dataset, since it travels with the data rather than with the run configuration.
The index it carries counts acquired frames, before any trimming of the analysis window, so
it identifies the same physical timepoint however the window is set.

Several automatic detectors are available, differing in how they identify the departure from
baseline. The default is a total variation denoising method that locates the first
statistically significant upward step. Alternatives include a Sobel edge heuristic, a
piecewise constant split, a generalized likelihood ratio change detector, and a full
biexponential fit that resolves both transition times simultaneously.

The start of the injection is defined as the resolved end of the baseline; there is no
separate option for it. Move it by changing the baseline resolution above.

## Robust fitting and peak weighting

The bolus peak is frequently the least reliable sample in the curve. It is a single frame,
often affected by inflow, partial volume and \(T_2^{*}\) effects at high concentration, and
in this model it has full leverage: the model's maximum \(A + B\) can be placed exactly on
it. A conventional robust estimator, which identifies outliers by their residuals, therefore
cannot detect an inflated peak, because the fit simply interpolates it.

Two independent mechanisms are available:

- **`aif_Robust`** selects a residual-based robust estimator for the fit: `off` for ordinary
  least squares, `Bisquare` for Tukey biweight iteratively reweighted least squares, or `LAR`
  for a soft L1 loss. The default is `off`.
- **`aif_peak_weight_exponent`** de-weights the peak sample using the shape of the curve
  rather than the residuals. The weight is derived from how far the peak stands above the
  rest of the curve, relative to the next largest sample, raised to this exponent. The
  default is 2; setting it to 0 disables the mechanism.

Leaving the peak at full weight tends to spend one of the two exponentials reaching that
single sample, at the expense of describing the washout, which degrades the fit everywhere
else.

## Averaging across subjects

A population average input function can be built by reading the arterial curves from several
completed runs and averaging them. This is useful where individual arterial measurements are
unreliable, at the cost of removing genuine between-subject variation in cardiac output and
injection dynamics.

## Quality control

With `save_aif_figure` enabled, which is the default, the pipeline writes a figure showing
the measured samples against the fitted curve, with the resolved baseline end and upslope end
marked. Inspect this figure for every run. A poorly resolved baseline, a truncated first
pass, or a fit that misses the peak are all immediately visible here and all invalidate every
downstream parameter.

## Configuration

The relevant options are documented in full in the
[DCE Options reference](../../dce_options.md), under the Stage B AIF fit and acquisition
timing sections.
