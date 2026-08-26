# Tissue Uptake Model

The tissue uptake model describes a plasma compartment of finite transit time feeding an
extravascular space from which contrast agent does not return. It sits between the
[Patlak model](patlak.md), which treats plasma transit as instantaneous, and the
[two-compartment exchange model](two-compartment-exchange.md), which allows backflux. By
retaining plasma transit while discarding backflux it separates flow from permeability using
one fewer parameter than the full exchange model.

## Equations

The impulse response is a plasma exponential plus a constant retention term. Writing the
extraction fraction as \(E = K^{trans}/F_p\),

\[
h(t) = F_p \left[ E + (1 - E)\, e^{-t/T_p} \right]
\]

Expanding gives the form ROCKETSHIP evaluates:

\[
C_t(t) = K^{trans} \int_{0}^{t} C_p(\tau)\,\mathrm{d}\tau
\;+\; \left(F_p - K^{trans}\right) \int_{0}^{t} C_p(\tau)\,
e^{-\frac{t - \tau}{T_p}} \,\mathrm{d}\tau
\]

The first term is the irreversibly retained contrast agent, identical to the leakage term of
the Patlak model. The second describes contrast agent passing through the plasma compartment
with mean transit time \(T_p\).

## Parameters

| Parameter | Symbol | Units | Default initial value | Default bounds |
| --- | --- | --- | --- | --- |
| Volume transfer constant | \(K^{trans}\) | min\(^{-1}\) | 2 × 10\(^{-4}\) | 10\(^{-7}\) to 2 |
| Plasma flow | \(F_p\) | min\(^{-1}\) | 0.35 | 10\(^{-4}\) to 20 |
| Plasma mean transit time | \(T_p\) | min | 0.12 | 0 to 1.5 |

Two further quantities follow from the fitted parameters:

\[
v_p = F_p \, T_p, \qquad
PS = \frac{K^{trans} F_p}{F_p - K^{trans}}
\]

The permeability surface area product follows from inverting \(K^{trans} = E F_p\) with
\(E = PS/(F_p + PS)\). It is poorly determined when \(K^{trans}\) approaches \(F_p\), since
the denominator then approaches zero.

!!! note "Model-specific defaults"
    The initial values and bounds above differ from those used by the Tofts family. The
    tissue uptake model has its own settings, named with a `_tissue_uptake` suffix, for
    example `voxel_initial_value_fp_tissue_uptake`. Notice in particular that \(T_p\) is
    bounded above at 1.5 minutes; plasma transit times far above that are not physiological
    and generally indicate the fit has drifted to a degenerate solution.

## When to use it

The tissue uptake model is appropriate where flow and permeability need to be distinguished
but the acquisition is too short, or the tissue too impermeable, for backflux to be
observable. It requires temporal resolution sufficient to resolve the first pass of the
bolus, since \(T_p\) is estimated from the shape of that passage.

Where backflux is measurable, the model will absorb it into the other parameters and bias
them. Comparing against the two-compartment exchange model on the same data is the direct
test of whether the irreversibility assumption is defensible.

## Configuration

Enable the model with the `tissue_uptake` entry in `model_flags`. Reported outputs are
\(K^{trans}\), \(F_p\), \(T_p\), the sum of squared errors, and the ninety-five percent
confidence interval for each parameter. Fitting is performed internally in minutes with rate
constants per minute, and returned parameters are converted to match the units of the
supplied time vector. An accelerated implementation is available on GPU and multi-core CPU
backends.

## References

Sourbron, S.P. and Buckley, D.L. [Tracer kinetic modelling in MRI: estimating perfusion and
capillary permeability](https://doi.org/10.1088/0031-9155/57/2/R1). *Physics in Medicine and
Biology*, 57(2), R1-R33 (2012).

Sourbron, S.P. and Buckley, D.L. [On the scope and interpretation of the Tofts models for
DCE-MRI](https://doi.org/10.1002/mrm.22861). *Magnetic Resonance in Medicine*, 66(3), 735-745
(2011).
