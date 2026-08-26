# FXR (Shutter Speed) Model

Every other model in ROCKETSHIP assumes that water exchange between tissue compartments is
fast enough that the tissue relaxes with a single, well defined \(R_1\) proportional to
contrast agent concentration. The fast exchange regime (FXR) model, also known as the shutter
speed model, removes that assumption. It treats the intracellular water lifetime as a free
parameter and fits the measured relaxation rate directly.

## Why the assumption matters

Contrast agent is confined to the extravascular extracellular space, so it shortens \(T_1\)
there but not inside cells. Whether the voxel behaves as a single relaxing pool depends on how
the rate of water exchange across the cell membrane compares with the difference in relaxation
rates between the compartments.

When contrast agent concentration is low, that difference is small, exchange keeps pace and
the compartments relax together. As concentration rises the difference grows, and beyond some
point exchange can no longer keep the compartments equilibrated. The relaxation then becomes
visibly non-monoexponential, and treating it as a single pool underestimates the true
concentration, which in turn biases \(K^{trans}\) and \(v_e\) downwards.

The name refers to this crossover: the exchange rate acts as a shutter speed, determining
whether the compartments are resolved or blurred together.

## Equations

The model fits the measured tissue relaxation rate \(R_1(t)\) rather than a concentration
curve. The underlying extravascular concentration follows Tofts kinetics,

\[
C_t(t) = K^{trans} \int_{0}^{t} C_p(\tau)\,
e^{-\frac{K^{trans}}{v_e}(t - \tau)}\,\mathrm{d}\tau
\]

and the observed relaxation rate is the smaller root of the two-site exchange expression:

\[
R_1(t) = \frac{1}{2}\left[\,2 R_{1i} + r_1 C_t(t) + X
- \sqrt{\left(\frac{2}{\tau_i} - r_1 C_t(t) - X\right)^{2}
+ \frac{4\,(1 - p_o)}{\tau_i^{2}\, p_o}}\;\right]
\]

where

\[
X = \frac{R_{1o} - R_{1i} + 1/\tau_i}{p_o},
\qquad
p_o = \frac{v_e}{f_w}
\]

Here \(R_{1o}\) is the pre-contrast tissue relaxation rate, \(R_{1i}\) the intracellular
relaxation rate, \(\tau_i\) the mean intracellular water lifetime, \(r_1\) the contrast agent
relaxivity, and \(f_w\) the volume fraction of tissue that is water. The quantity \(p_o\) is
the mole fraction of tissue water residing in the extravascular extracellular space.

In the limit of fast exchange, \(\tau_i \to 0\), this expression reduces to
\(R_1 = R_{1o} + r_1 C_t\), which is the linear relationship the other models assume.

!!! note "Intracellular relaxation rate"
    \(R_{1i}\) is not fitted. ROCKETSHIP sets it equal to the measured pre-contrast tissue
    relaxation rate \(R_{1o}\) for each voxel, taken from the supplied \(T_1\) map. The two
    are therefore per-voxel quantities rather than global constants, which is why this model
    is fitted voxel by voxel rather than in the batched form used by the other models.

## Parameters

| Parameter | Symbol | Units | Default initial value | Default bounds |
| --- | --- | --- | --- | --- |
| Volume transfer constant | \(K^{trans}\) | min\(^{-1}\) | 2 × 10\(^{-4}\) | 10\(^{-7}\) to 2 |
| Extravascular extracellular volume fraction | \(v_e\) | — | 0.2 | 0.02 to 1 |
| Intracellular water lifetime | \(\tau_i\) | min | 0.01 | 0 to 100 |

The tissue water fraction \(f_w\) is a fixed input rather than a fitted parameter, set with
the `fxr_fw` option. Its default is 0.8.

## Inputs

Unlike the other models, FXR requires the tissue relaxation rate time course and the tissue
\(T_1\) map to be carried forward from earlier pipeline stages, since it fits \(R_1(t)\)
rather than \(C_t(t)\). A run that enables this model without those arrays available will
stop with an error.

The arterial input function is still supplied as a plasma concentration curve, converted as
described in [signal to concentration](../signal-to-concentration.md).

## When to use it

Consider the FXR model where contrast agent concentrations in tissue are high enough to make
the exchange regime questionable, and where the resulting bias in \(K^{trans}\) and \(v_e\)
would affect the conclusion being drawn. In practice this most often arises at high doses,
high field strengths, or in tissue with large cells and correspondingly long intracellular
water lifetimes.

The additional parameter has a cost. \(\tau_i\) is estimated from a subtle departure from
monoexponential behaviour, so it demands good signal to noise ratio, and it is often poorly
determined in individual voxels even where the fit as a whole is sound. Fitting regions of
interest rather than voxels, and inspecting confidence intervals, is advisable.

## Configuration

Enable the model with the `fxr` entry in `model_flags`. Reported outputs are \(K^{trans}\),
\(v_e\), \(\tau_i\), the sum of squared errors, and the ninety-five percent confidence
interval for each parameter. This model runs on the standard CPU fitting path; the
accelerated backends do not implement it.

## References

Yankeelov, T.E., et al. [Evidence for shutter-speed variation in CR bolus-tracking studies of
human pathology](https://pubmed.ncbi.nlm.nih.gov/15282806/). *NMR in Biomedicine*, 18(3),
173-185 (2005).

Landis, C.S., et al. [Determination of the MRI contrast agent concentration time course in
vivo following bolus injection: effect of equilibrium transcytolemmal water
exchange](https://doi.org/10.1002/1522-2594(200011)44:5<563::AID-MRM10>3.0.CO;2-#). *Magnetic
Resonance in Medicine*, 44(4), 563-574 (2000).
