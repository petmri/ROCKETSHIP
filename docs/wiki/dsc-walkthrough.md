# MATLAB DSC Walkthrough

This guide covers dynamic susceptibility contrast (DSC) perfusion analysis using the MATLAB
interface.

## Launching

Add the main ROCKETSHIP folder to your MATLAB path, then start the DSC interface in one of
three ways:

| Command | Effect |
| --- | --- |
| `rocketship` | Launch the main interface, from which DSC analysis can be selected |
| `run_dsc` | Add the required subfolders to the path, then launch DSC analysis |
| `dsc` | Launch DSC analysis directly, assuming the subfolders are already on the path |

## Image selection

Select the source DSC images. These must be in NIfTI format.

## Noise handling

Noise is estimated from a region of the image containing only air, and is used to establish
the noise level of the acquisition. The region may be selected automatically from a corner of
the image, or supplied as a binary mask in NIfTI format.

## Arterial input function

The input function may be defined in three ways.

| Option | Source |
| --- | --- |
| User Selected AIF | A binary mask supplied as a NIfTI file |
| Import AIF | A curve stored in a MATLAB `.mat` file |
| Use AIF from Previous Run | The curve derived by the most recent run |

## Fitting function

Select the form fitted to the measured input function samples.

| Option | Description |
| --- | --- |
| Biexponential | Two-exponential decay fitted to the whole curve |
| Biexponential local | Two-exponential decay fitted over a restricted interval |
| Gamma-Variate | Gamma variate function, the conventional choice for first-pass DSC |
| Raw Data | No fitting; the measured samples are used directly |
| Upslope copy biexponential linear adjustment | Biexponential with the upslope replaced by the measured samples and a linear adjustment applied |
| Upslope copy biexponential | Biexponential with the upslope replaced by the measured samples |

The gamma variate function is the standard choice where recirculation must be excluded from
the first pass. Fitting reduces noise in the input function at the cost of imposing a fixed
shape; the raw option preserves the measurement unmodified.

## Input parameters and deconvolution

Specify the acquisition parameters, the tissue parameters, and the deconvolution algorithm
used to recover the residue function from the tissue and arterial curves.

## Bolus injection time

| Option | Behaviour |
| --- | --- |
| Automatic | Bolus arrival is detected from the signal intensity curves |
| User Selected | Bolus arrival is identified interactively |

Bolus arrival timing determines which frames form the pre-contrast baseline, and therefore
affects every derived quantity. Where automatic detection is used, confirm the result before
interpreting the output.
