# Two-Compartment Exchange Model

The two-compartment exchange model (2CXM) describes tissue as two well mixed compartments,
plasma and extravascular extracellular space, with bidirectional exchange between them and
finite plasma flow. It is the most complete of the models available in ROCKETSHIP and the
only one that estimates flow, permeability and both volume fractions independently.

## Equations

The impulse response is a biexponential:

\[
h(t) = F_p \left[(1 - A)\, e^{-K_{+} t} + A \, e^{-K_{-} t}\right]
\]

so that

\[
\boxed{\;C_t(t) = F_p \int_{0}^{t} C_p(\tau)
\left[(1 - A)\, e^{-K_{+}(t - \tau)} + A\, e^{-K_{-}(t - \tau)}\right] \mathrm{d}\tau\;}
\]

The rate constants and the mixing coefficient derive from three characteristic times.
Writing the extraction fraction as \(E = K^{trans}/F_p\):

\[
T_p = \frac{v_p (1 - E)}{F_p}, \qquad
T_e = \frac{v_e (1 - E)}{E \, F_p}, \qquad
T_b = \frac{v_p}{F_p}
\]

where \(T_p\) is the plasma mean transit time, \(T_e\) the extravascular mean transit time,
and \(T_b\) the plasma volume divided by flow. The exponential rate constants are the roots

\[
K_{\pm} = \frac{1}{2}\left[
\left(\frac{1}{T_p} + \frac{1}{T_e}\right)
\pm \sqrt{\left(\frac{1}{T_p} + \frac{1}{T_e}\right)^{2} - \frac{4}{T_e T_b}}
\;\right]
\]

and the mixing coefficient is

\[
A = \frac{K_{+} - 1/T_b}{K_{+} - K_{-}}
\]

## Parameters

| Parameter | Symbol | Units | Default initial value | Default bounds |
| --- | --- | --- | --- | --- |
| Volume transfer constant | \(K^{trans}\) | min\(^{-1}\) | 2 × 10\(^{-4}\) | 10\(^{-7}\) to 2 |
| Extravascular extracellular volume fraction | \(v_e\) | — | 0.15 | 0.05 to 1 |
| Plasma volume fraction | \(v_p\) | — | 0.02 | 0.001 to 1 |
| Plasma flow | \(F_p\) | min\(^{-1}\) | 0.35 | 10\(^{-4}\) to 20 |

The permeability surface area product follows from the fitted parameters as

\[
PS = \frac{K^{trans} F_p}{F_p - K^{trans}}
\]

Fitting is carried out in an extraction fraction parametrisation, with \(E = K^{trans}/F_p\)
in place of \(K^{trans}\), and the supplied \(K^{trans}\) bounds are mapped into \(E\) space
accordingly. This parametrisation is better conditioned, because it keeps the fitted variable
bounded in \((0, 1)\) and decouples it from the flow estimate.

!!! note "Model-specific defaults"
    The 2CXM uses its own initial values and bounds, named with a `_2cxm` suffix, for example
    `voxel_lower_limit_ve_2cxm`. It also uses higher iteration limits than the simpler models,
    reflecting the larger parameter space.

## Temporal resolution

The 2CXM is the model most sensitive to temporal resolution. Plasma mean transit times are
routinely one to two seconds, which is shorter than a typical DCE frame interval, so
evaluating the model directly on the acquired timebase substantially under-resolves the
plasma exponential.

ROCKETSHIP therefore evaluates the model on a dense internal grid. Following the OSIPI
convention, the arterial input function is interpolated to a 0.1 second grid, the forward
model is evaluated there, and the resulting curve is sampled back at the acquired timepoints
for comparison with the data. This removes the discretisation error introduced by evaluating
on a coarse grid.

!!! warning "Interpolation is not information"
    Dense evaluation removes discretisation error; it cannot recover detail the acquisition
    never recorded. Where the frame interval is long relative to the plasma transit time, the
    residual error from an unresolved first pass dominates every other error source in this
    model, and \(F_p\) and \(v_p\) should be treated as poorly determined. Confidence
    intervals are computed with degrees of freedom based on the number of acquired samples,
    not the number of points on the interpolation grid.

## When to use it

Use the 2CXM where the scientific question requires flow and permeability separately, and
where the acquisition was designed to support it: high temporal resolution through the first
pass, good signal to noise ratio, and a duration long enough for backflux to become
measurable. Given data that meet those conditions it is the most informative model available.

Given data that do not, its extra parameters will be poorly constrained, and one of the
simpler models will produce more reproducible results.

## Configuration

Enable the model with the `two_cxm` entry in `model_flags`. Reported outputs are
\(K^{trans}\), \(v_e\), \(v_p\), \(F_p\), the sum of squared errors, and the ninety-five
percent confidence interval for each parameter. An accelerated implementation is available on
GPU and multi-core CPU backends.

## References

Sourbron, S.P. and Buckley, D.L. [Tracer kinetic modelling in MRI: estimating perfusion and
capillary permeability](https://doi.org/10.1088/0031-9155/57/2/R1). *Physics in Medicine and
Biology*, 57(2), R1-R33 (2012).

Brix, G., et al. [Microcirculation and microvasculature in breast tumors: pharmacokinetic
analysis of dynamic MR image series](https://doi.org/10.1002/mrm.20005). *Magnetic Resonance
in Medicine*, 52(2), 420-429 (2004).
