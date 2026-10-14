# Extended Tofts Model

The extended Tofts model adds an explicit plasma compartment to the
[Tofts model](tofts.md). The measured concentration is the sum of contrast agent that has
leaked into the extravascular extracellular space and contrast agent still resident in the
plasma within the voxel. It is the most widely used model in clinical DCE-MRI.

## Equations

The impulse response combines a decaying exponential with an instantaneous plasma term:

\[
h(t) = K^{trans} e^{-k_{ep} t} + v_p \, \delta(t), \qquad k_{ep} = \frac{K^{trans}}{v_e}
\]

giving

\[
C_t(t) = K^{trans} \int_{0}^{t} C_p(\tau)\,
e^{-\frac{K^{trans}}{v_e}(t - \tau)} \,\mathrm{d}\tau \;+\; v_p \, C_p(t)
\]

The plasma term is not convolved. This is the model's defining approximation: plasma within
the voxel is assumed to equilibrate with the arterial supply instantaneously, so its
contribution tracks \(C_p(t)\) with no delay or dispersion.

## Parameters

| Parameter | Symbol | Units | Default initial value | Default bounds |
| --- | --- | --- | --- | --- |
| Volume transfer constant | \(K^{trans}\) | min\(^{-1}\) | 2 × 10\(^{-4}\) | 10\(^{-7}\) to 2 |
| Extravascular extracellular volume fraction | \(v_e\) | — | 0.2 | 0.02 to 1 |
| Plasma volume fraction | \(v_p\) | — | 0.02 | 0.001 to 1 |

## Interpretation

As in the Tofts model, \(K^{trans}\) is a composite of flow and permeability and cannot be
decomposed into the two without a model that describes plasma transit explicitly. What the
plasma term buys is not a separation of flow from permeability, but protection of
\(K^{trans}\) and \(v_e\) from contamination by the vascular signal.

The instantaneous plasma assumption holds when the plasma mean transit time is short compared
with the temporal resolution of the acquisition. Since plasma transit times are typically one
to two seconds and DCE frame times are frequently longer, the assumption is usually
reasonable. Where the acquisition does resolve plasma transit, the
[two-compartment exchange model](two-compartment-exchange.md) describes it explicitly.

## When to use it

This is the appropriate default for well vascularized tissue, and for tumor imaging in
particular. It requires only that the acquisition capture the arrival of the bolus with
enough fidelity to distinguish the vascular peak from subsequent leakage, and it degrades
gracefully towards the Tofts model as \(v_p\) approaches zero.

!!! note "Parameter identifiability"
    Estimating \(v_p\) requires that the acquisition resolve the first pass of the bolus. If
    the baseline is short or the temporal resolution is coarse, \(v_p\) may be poorly
    determined even where the fit as a whole appears sound. Inspect the reported confidence
    intervals before interpreting \(v_p\) quantitatively.

## Configuration

Enable the model with the `ex_tofts` entry in `model_flags`. Reported outputs are
\(K^{trans}\), \(v_e\), \(v_p\), the sum of squared errors, and the ninety-five percent
confidence interval for each parameter. An accelerated implementation is available on GPU and
multi-core CPU backends.

## References

Tofts, P.S., et al. [Estimating kinetic parameters from dynamic contrast-enhanced T1-weighted
MRI of a diffusable tracer: standardized quantities and
symbols](https://doi.org/10.1002/(SICI)1522-2586(199909)10:3<223::AID-JMRI2>3.0.CO;2-S).
*Journal of Magnetic Resonance Imaging*, 10(3), 223-232 (1999).

Sourbron, S.P. and Buckley, D.L. [On the scope and interpretation of the Tofts
models for DCE-MRI](https://doi.org/10.1002/mrm.22861). *Magnetic Resonance in Medicine*,
66(3), 735-745 (2011).
