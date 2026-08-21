# Pharmacokinetic Models

ROCKETSHIP fits tracer kinetic models to the contrast agent concentration curves produced by
the [signal to concentration conversion](../signal-to-concentration.md). This section gives
the equations for each available model, the parameters it estimates, and guidance on when it
is the appropriate choice.

Any number of models may be enabled for a single run. Each produces its own set of parameter
maps and its own results file, which allows several models to be compared on the same data.

!!! info "Primary reference"
    For a thorough treatment of these models, their underlying assumptions and the reasoning
    behind choosing between them, refer to:

    Sourbron, S.P. and Buckley, D.L. [Classic models for dynamic contrast-enhanced
    MRI](https://doi.org/10.1002/nbm.2940). *NMR in Biomedicine*, 26(8), 1004-1027 (2013).

    It is the standard reference on the subject and covers the material below in considerably
    more depth than these pages can. Users selecting a model for a new study, or needing to
    justify that choice, should start there.

## Common formulation

With the exception of the [area under the curve](auc.md) summary measure, every model
describes the tissue concentration \(C_t(t)\) as the convolution of the arterial plasma
concentration \(C_p(t)\) with an impulse response function \(h(t)\) characteristic of the
model:

\[
C_t(t) = \bigl(C_p * h\bigr)(t) = \int_{0}^{t} C_p(\tau)\, h(t - \tau)\, \mathrm{d}\tau
\]

Fitting a model means finding the parameters of \(h\) that best reproduce the measured
\(C_t(t)\) in the least squares sense. The models differ in how many compartments they
describe and which physiological processes they treat as rate limiting.

## Parameters

The same symbols recur throughout this section.

| Symbol | Name | Units | Interpretation |
| --- | --- | --- | --- |
| \(K^{trans}\) | Volume transfer constant | min\(^{-1}\) | Rate of contrast agent transfer from plasma to extravascular extracellular space |
| \(v_e\) | Extravascular extracellular volume fraction | — | Fraction of tissue volume accessible to contrast agent outside cells and vessels |
| \(v_p\) | Plasma volume fraction | — | Fraction of tissue volume occupied by plasma |
| \(F_p\) | Plasma flow | min\(^{-1}\) | Plasma perfusion per unit tissue volume |
| \(PS\) | Permeability surface area product | min\(^{-1}\) | Capillary wall permeability times surface area per unit volume |
| \(k_{ep}\) | Efflux rate constant | min\(^{-1}\) | \(K^{trans}/v_e\), the rate of return from tissue to plasma |
| \(T_p\) | Plasma mean transit time | min | Mean residence time in the plasma compartment |
| \(T_e\) | Extravascular mean transit time | min | Mean residence time in the extravascular space |
| \(E\) | Extraction fraction | — | \(K^{trans}/F_p\), the fraction of arriving tracer extracted in one pass |
| \(\tau_i\) | Intracellular water lifetime | min | Mean time a water molecule spends inside a cell |

## Choosing a model

The models form a hierarchy. More parameters allow a closer description of the underlying
physiology, but each additional parameter requires more information from the data to
constrain it. Temporal resolution, signal to noise ratio and the duration of the acquisition
all determine how much a given dataset can support.

The summary below is intended to orient the choice, not to settle it. The assumptions behind
each model, the conditions under which they hold, and the consequences of applying a model
outside them are set out in detail by
[Sourbron and Buckley (2013)](https://doi.org/10.1002/nbm.2940).

| Model | Parameters | Assumes | Typical use |
| --- | --- | --- | --- |
| [Tofts](tofts.md) | \(K^{trans}\), \(v_e\) | Negligible intravascular contribution | Weakly vascularised tissue; the historical standard |
| [Extended Tofts](extended-tofts.md) | \(K^{trans}\), \(v_e\), \(v_p\) | Rapid plasma equilibration | Well vascularised tumours; the most widely used model |
| [Patlak](patlak.md) | \(K^{trans}\), \(v_p\) | Negligible backflux | Low permeability tissue, most used for blood–brain barrier integrity |
| [Tissue Uptake](tissue-uptake.md) | \(K^{trans}\), \(F_p\), \(T_p\) | Negligible backflux, finite flow | Separating flow from permeability over a short acquisition |
| [Two-Compartment Exchange](two-compartment-exchange.md) | \(K^{trans}\), \(v_e\), \(v_p\), \(F_p\) | Two well mixed compartments | High temporal resolution data where flow and permeability are separable |
| [FXR](fxr.md) | \(K^{trans}\), \(v_e\), \(\tau_i\) | Finite transcytolemmal water exchange | Where the fast exchange assumption is questionable |
| [Area Under the Curve](auc.md) | AUC, NAUC | No kinetic model | Model-free summary; robust where fitting is unreliable |

!!! tip "Beware of more complicated models"
    A more complicated model frequently gives *worse* accuracy in the fitted parameters. The
    additional degrees of freedom fit noise rather than signal, which increases the uncertainty
    on every parameter estimated.

    Measurement of low \(K^{trans}\) values at the blood–brain barrier is a clear example. The
    two-parameter [Patlak](patlak.md) model has been shown to give the most accurate results
    there, despite being the simplest of the leakage models:

    - Barnes, S.R., et al. [Optimal acquisition and modeling parameters for accurate assessment
      of low Ktrans blood-brain barrier permeability using dynamic contrast-enhanced
      MRI](https://doi.org/10.1002/mrm.25793). *Magnetic Resonance in Medicine*, 75(5),
      1967-1977 (2016).
    - Cramer, S.P. and Larsson, H.B.W. [Accurate determination of blood-brain barrier
      permeability using dynamic contrast-enhanced T1-weighted MRI: a simulation and in vivo
      study on healthy subjects and multiple sclerosis
      patients](https://doi.org/10.1038/jcbfm.2014.126). *Journal of Cerebral Blood Flow and
      Metabolism*, 34(10), 1655-1665 (2014).

    The most complex model a given dataset will support can be determined empirically. The
    MATLAB analysis stage provides Akaike information criterion, F test and FMI/FRI comparisons
    between fitted models on the same regions of interest; see the
    [DCE walkthrough](../../wiki/dce-walkthrough.md).

## Numerical implementation

Several implementation choices apply across the model set:

- **Convolution.** Each model's convolution integral is evaluated by a recurrence that carries
  the running total forward from one timepoint to the next, instead of re-integrating the whole
  curve at every timepoint. Fitting cost (time) therefore grows in proportion to the number of
  timepoints, rather than with its square, this provides a large speed up. Within each sampling interval the curve is
  integrated by the trapezoid rule.

    ??? note "How the recurrence works"
        Each model needs an integral running from the start of the acquisition up to every
        timepoint \(t_k\), of the form

        \[
        \int_{0}^{t_k} C_p(\tau)\, e^{-\lambda (t_k - \tau)}\,\mathrm{d}\tau
        \]

        A running total cannot be accumulated directly, because the weight
        \(e^{-\lambda(t_k - \tau)}\) depends on the endpoint \(t_k\): moving to the next
        timepoint changes the weight applied to every earlier sample.

        The exponential separates, however:

        \[
        e^{-\lambda(t_k - \tau)} = e^{-\lambda \Delta t}\; e^{-\lambda(t_{k-1} - \tau)},
        \qquad \Delta t = t_k - t_{k-1}
        \]

        so the integral already accumulated up to \(t_{k-1}\) becomes correct for \(t_k\)
        after multiplication by the single factor \(e^{-\lambda \Delta t}\). Each timepoint
        then costs one rescaling of the running total plus the contribution of the newest
        interval, which is a fixed amount of work regardless of how much history precedes it.

        The rescaling is exact. The approximation lies in the trapezoid rule applied within
        each interval, and its error is governed by the temporal resolution of the acquisition.

- **Fitting.** Parameters are estimated by bound-constrained nonlinear least squares. Initial
  values and bounds are configurable per parameter and per model; see the
  [DCE Options reference](../../dce_options.md).
- **Confidence intervals.** Ninety-five percent confidence intervals are derived from the
  covariance of the converged fit and reported alongside each parameter.
- **Acceleration.** The Tofts, extended Tofts, Patlak, tissue uptake and two-compartment
  exchange models have accelerated implementations that run on GPU or multi-core CPU. See
  [GPU and CPU Acceleration](../../wiki/enable-gpu-acceleration.md).

## Units

Rate constants are reported per minute and volume fractions are dimensionless. Where the
supplied time vector is in seconds, fitting is still carried out internally in minutes and
the returned rate parameters are converted back to match the units of the input time vector.

## References

Sourbron, S.P. and Buckley, D.L. [Classic models for dynamic contrast-enhanced
MRI](https://doi.org/10.1002/nbm.2940). *NMR in Biomedicine*, 26(8), 1004-1027 (2013).

Sourbron, S.P. and Buckley, D.L. [Tracer kinetic modelling in MRI: estimating perfusion and
capillary permeability](https://doi.org/10.1088/0031-9155/57/2/R1). *Physics in Medicine and
Biology*, 57(2), R1-R33 (2012).

Sourbron, S.P. and Buckley, D.L. [On the scope and interpretation of the Tofts models for
DCE-MRI](https://doi.org/10.1002/mrm.22861). *Magnetic Resonance in Medicine*, 66(3), 735-745
(2011).

Tofts, P.S., et al. [Estimating kinetic parameters from dynamic contrast-enhanced T1-weighted
MRI of a diffusable tracer: standardized quantities and
symbols](https://doi.org/10.1002/(SICI)1522-2586(199909)10:3<223::AID-JMRI2>3.0.CO;2-S).
*Journal of Magnetic Resonance Imaging*, 10(3), 223-232 (1999).

Barnes, S.R., et al. [Optimal acquisition and modeling parameters for accurate assessment of
low Ktrans blood-brain barrier permeability using dynamic contrast-enhanced
MRI](https://doi.org/10.1002/mrm.25793). *Magnetic Resonance in Medicine*, 75(5), 1967-1977
(2016).

Cramer, S.P. and Larsson, H.B.W. [Accurate determination of blood-brain barrier permeability
using dynamic contrast-enhanced T1-weighted MRI: a simulation and in vivo study on healthy
subjects and multiple sclerosis patients](https://doi.org/10.1038/jcbfm.2014.126). *Journal of
Cerebral Blood Flow and Metabolism*, 34(10), 1655-1665 (2014).

References specific to each model are listed on its own page.
