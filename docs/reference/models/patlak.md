# Patlak Model

The Patlak model describes irreversible uptake: contrast agent leaks from plasma into the
extravascular extracellular space and, over the duration of the acquisition, does not
return. Removing the backflux term reduces the model to two parameters and makes it linear in
those parameters, which gives it stability that the reversible models cannot match.

## Equations

The impulse response is a constant leakage term plus an instantaneous plasma term:

\[
h(t) = K^{trans} + v_p \, \delta(t)
\]

giving

\[
\boxed{\;C_t(t) = K^{trans} \int_{0}^{t} C_p(\tau)\, \mathrm{d}\tau \;+\; v_p \, C_p(t)\;}
\]

This is the limiting case of the [extended Tofts model](extended-tofts.md) as
\(k_{ep} \to 0\), that is, as \(v_e\) becomes large enough that efflux is negligible over the
observation window. Because \(v_e\) no longer appears, it cannot be estimated.

## Linear form

Dividing through by \(C_p(t)\) makes the model linear:

\[
\frac{C_t(t)}{C_p(t)} = K^{trans}\, \frac{\int_{0}^{t} C_p(\tau)\,\mathrm{d}\tau}{C_p(t)}
\;+\; v_p
\]

Plotting \(C_t/C_p\) against the ratio of the plasma integral to \(C_p\), known as the Patlak
plot, yields a straight line whose slope is \(K^{trans}\) and whose intercept is \(v_p\). A
linear least squares solution is available and is used by default, as it requires no initial
values, cannot fail to converge, and returns the global optimum directly.

!!! note "Departure from linearity"
    Curvature in the Patlak plot indicates that backflux is not in fact negligible. Where the
    plot bends at later timepoints, either restrict the analysis to the early linear portion
    or move to a reversible model. The fitting window can be restricted with the
    `restrict_fit_start_min` and `restrict_fit_end_min` options.

## Parameters

| Parameter | Symbol | Units | Default initial value | Default bounds |
| --- | --- | --- | --- | --- |
| Volume transfer constant | \(K^{trans}\) | min\(^{-1}\) | 2 × 10\(^{-4}\) | 10\(^{-7}\) to 2 |
| Plasma volume fraction | \(v_p\) | — | 0.02 | 0.001 to 1 |

## When to use it

The Patlak model is the standard choice for low permeability tissue, where \(K^{trans}\) is
small enough that backflux does not become measurable within a typical acquisition. Its
principal application is the quantification of blood–brain barrier integrity, where the
transfer constants of interest are two to three orders of magnitude below those seen in
tumours and where the stability of a linear estimator is decisive.

It is also useful as a robust fallback for any tissue in which the nonlinear models converge
unreliably, at the cost of not estimating \(v_e\).

## Configuration

Enable the model with the `patlak` entry in `model_flags`. Reported outputs are
\(K^{trans}\), \(v_p\), the sum of squared errors, and the ninety-five percent confidence
interval for each parameter. An accelerated implementation is available on GPU and multi-core
CPU backends.

## References

Patlak, C.S., et al. [Graphical evaluation of blood-to-brain transfer constants from
multiple-time uptake data](https://doi.org/10.1038/jcbfm.1983.1). *Journal of Cerebral Blood
Flow and Metabolism*, 3(1), 1-7 (1983).

Patlak, C.S. and Blasberg, R.G. [Graphical evaluation of blood-to-brain transfer constants
from multiple-time uptake data. Generalizations](https://doi.org/10.1038/jcbfm.1985.87).
*Journal of Cerebral Blood Flow and Metabolism*, 5(4), 584-590 (1985).
