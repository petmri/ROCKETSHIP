# Area Under the Curve

The area under the curve (AUC) is a model-free summary of enhancement. It makes no assumption
about compartments, permeability or flow, and requires no fitting, so it is defined for every
voxel regardless of whether a kinetic model would converge there. It is correspondingly less
specific: it aggregates flow, permeability and volume effects into a single number that cannot
be decomposed.

## Equations

Four quantities are reported. Two are integrals of the tissue curves, in concentration and in
signal intensity units respectively:

\[
\mathrm{AUC}_c = \int_{t_0}^{t_{end}} C_t(\tau)\,\mathrm{d}\tau,
\qquad
\mathrm{AUC}_s = \int_{t_0}^{t_{end}} S_t(\tau)\,\mathrm{d}\tau
\]

The other two normalize these by the corresponding integrals of the arterial curves:

\[
\mathrm{NAUC}_c = \frac{\mathrm{AUC}_c}{\displaystyle\int_{t_0}^{t_{end}} C_p(\tau)\,\mathrm{d}\tau},
\qquad
\mathrm{NAUC}_s = \frac{\mathrm{AUC}_s}{\displaystyle\int_{t_0}^{t_{end}} S_p(\tau)\,\mathrm{d}\tau}
\]

Integration is performed by the trapezoidal rule over the acquired timepoints. The lower limit
\(t_0\) is the resolved start of the injection, so the pre-contrast baseline is excluded. The
signal intensity curves have their own baseline subtracted before integration, so that both
the concentration and the signal forms measure enhancement above baseline rather than absolute
level.

## Reported values

| Output | Symbol | Basis |
| --- | --- | --- |
| AUC conc | \(\mathrm{AUC}_c\) | Tissue concentration curve |
| AUC sig | \(\mathrm{AUC}_s\) | Tissue signal intensity curve |
| NAUC conc | \(\mathrm{NAUC}_c\) | Normalized by the arterial concentration integral |
| NAUC sig | \(\mathrm{NAUC}_s\) | Normalized by the arterial signal integral |

No parameters are fitted, so no confidence intervals or sum of squared errors are reported for
this model.

## Interpretation

The concentration form has physical units of mM·min and depends on the accuracy of the
relaxivity, the \(T_1\) map and the conversion described in
[signal to concentration](../signal-to-concentration.md). The signal form is in arbitrary
units and is not comparable between scans, subjects or sessions.

The normalized forms are the more useful of the four for comparison across subjects. Dividing
by the arterial integral removes the dependence on injected dose, cardiac output and injection
rate, which are the largest sources of between-subject variability in the unnormalized
measures.

!!! warning "Dependence on the integration window"
    AUC has no fixed endpoint. Its value depends directly on how long the acquisition ran and
    where the integration window was placed, so values are comparable only between
    acquisitions with the same timing. Where AUC is used as an endpoint, fix the window
    explicitly with the `restrict_fit_start_min` and `restrict_fit_end_min` options rather
    than relying on the acquisitions happening to match.

## When to use it

AUC is appropriate as a robust descriptive measure where model fitting is unreliable: low
signal to noise ratio, coarse temporal resolution, an unreliable arterial input function, or
tissue whose enhancement pattern no available model describes well. It is also useful as a
sanity check alongside a fitted model, since a region with substantial enhancement but a
degenerate \(K^{trans}\) estimate usually indicates a fitting problem rather than a
physiological one.

It is not a substitute for a kinetic parameter where one can be estimated reliably, because it
confounds the processes a kinetic model separates.

## Configuration

Enable the measure with the `auc` entry in `model_flags`. It requires the tissue and arterial
signal intensity curves from the earlier pipeline stages in addition to the concentration
curves.

## References

Evelhoch, J.L. [Key factors in the acquisition of contrast kinetic data for
oncology](https://doi.org/10.1002/(SICI)1522-2586(199909)10:3<254::AID-JMRI5>3.0.CO;2-9).
*Journal of Magnetic Resonance Imaging*, 10(3), 254-259 (1999).

Walker-Samuel, S., et al. [Evaluation of response to treatment using DCE-MRI: the relationship
between initial area under the gadolinium curve (IAUGC) and quantitative pharmacokinetic
analysis](https://doi.org/10.1088/0031-9155/51/14/017). *Physics in Medicine and Biology*,
51(14), 3593-3602 (2006).
