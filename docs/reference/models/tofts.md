# Tofts Model

The Tofts model describes tissue as a single extravascular extracellular compartment supplied
by plasma, with no explicit intravascular contribution to the measured signal. It is the
longest established model in DCE-MRI and remains the reference against which others are
compared.

## Equations

The impulse response is a single decaying exponential:

\[
h(t) = K^{trans} \, e^{-k_{ep} t}, \qquad k_{ep} = \frac{K^{trans}}{v_e}
\]

giving the tissue concentration

\[
\boxed{\;C_t(t) = K^{trans} \int_{0}^{t} C_p(\tau)\,
e^{-\frac{K^{trans}}{v_e}\,(t - \tau)} \; \mathrm{d}\tau \;}
\]

Equivalently, in differential form, the rate of change of tissue concentration is the
difference between influx from plasma and efflux back to it:

\[
\frac{\mathrm{d}C_t}{\mathrm{d}t} = K^{trans} C_p(t) - k_{ep} C_t(t)
\]

The concentration is zero at the first timepoint by construction, since the baseline
re-anchoring described in the
[signal to concentration conversion](../signal-to-concentration.md) sets the pre-contrast
concentration to zero.

## Parameters

| Parameter | Symbol | Units | Default initial value | Default bounds |
| --- | --- | --- | --- | --- |
| Volume transfer constant | \(K^{trans}\) | min\(^{-1}\) | 2 × 10\(^{-4}\) | 10\(^{-7}\) to 2 |
| Extravascular extracellular volume fraction | \(v_e\) | — | 0.2 | 0.02 to 1 |

The efflux rate constant \(k_{ep} = K^{trans}/v_e\) is not fitted independently; it is fully
determined by the two fitted parameters.

Initial values and bounds are set with the `voxel_initial_value_ktrans`,
`voxel_lower_limit_ktrans`, `voxel_upper_limit_ktrans` options and their \(v_e\) equivalents.

## Interpretation

\(K^{trans}\) is a composite quantity. Its physiological meaning depends on which process
limits contrast agent delivery to the tissue:

- Where permeability is the limiting factor, that is \(PS \ll F_p\), then
  \(K^{trans} \approx PS\).
- Where flow is the limiting factor, that is \(PS \gg F_p\), then \(K^{trans} \approx F_p\).
- Between these regimes \(K^{trans} = E F_p\), where \(E = 1 - e^{-PS/F_p}\) is the
  extraction fraction.

The Tofts model cannot distinguish these cases. Separating flow from permeability requires
the [tissue uptake](tissue-uptake.md) or
[two-compartment exchange](two-compartment-exchange.md) model, and correspondingly higher
temporal resolution.

## When to use it

The Tofts model is appropriate where the intravascular contribution to the measured signal is
genuinely negligible: weakly vascularised tissue, or an acquisition whose first timepoints do
not resolve the vascular peak. It is well conditioned and converges reliably, which makes it
a robust choice for data that cannot support a third parameter.

!!! warning "Bias in well vascularised tissue"
    Where a plasma compartment does contribute measurably, omitting it biases both fitted
    parameters. The plasma signal is absorbed into the extravascular compartment, typically
    inflating \(K^{trans}\) and depressing \(v_e\). In such tissue the
    [extended Tofts model](extended-tofts.md) is the more appropriate choice.

## Configuration

Enable the model with the `tofts` entry in `model_flags`. Reported outputs are
\(K^{trans}\), \(v_e\), the sum of squared errors, and the ninety-five percent confidence
interval for each parameter. An accelerated implementation is available on GPU and multi-core
CPU backends.

## References

Tofts, P.S., et al. [Estimating kinetic parameters from dynamic contrast-enhanced T1-weighted
MRI of a diffusable tracer: standardized quantities and
symbols](https://doi.org/10.1002/(SICI)1522-2586(199909)10:3<223::AID-JMRI2>3.0.CO;2-S).
*Journal of Magnetic Resonance Imaging*, 10(3), 223-232 (1999).

Tofts, P.S. and Kermode, A.G. [Measurement of the blood-brain barrier permeability and leakage
space using dynamic MR imaging](https://doi.org/10.1002/mrm.1910170208). *Magnetic Resonance
in Medicine*, 17(2), 357-367 (1991).
