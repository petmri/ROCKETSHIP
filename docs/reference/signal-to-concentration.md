# Signal Intensity to Contrast Agent Concentration

Pharmacokinetic models are defined in terms of contrast agent concentration, but a DCE-MRI
acquisition measures signal intensity in arbitrary units. This page documents the conversion
ROCKETSHIP applies between the two, the assumptions it rests on, and the acquisition
parameters it requires.

The conversion is performed once per run, in Part A of the DCE pipeline, and produces two
quantities used by every downstream model: the tissue concentration curve \(C_t(t)\) and the
plasma concentration curve \(C_p(t)\) that serves as the arterial input function.

## Notation

| Symbol | Quantity | Units |
| --- | --- | --- |
| \(S(t)\) | Measured signal intensity | arbitrary |
| \(\bar{S}_0\) | Mean signal over the pre-contrast baseline | arbitrary |
| \(T_{10}\) | Pre-contrast longitudinal relaxation time | s |
| \(R_1(t)\) | Longitudinal relaxation rate, \(1/T_1(t)\) | s\(^{-1}\) |
| \(\mathrm{TR}\) | Repetition time | s |
| \(\alpha\) | Flip angle | rad |
| \(r_1\) | Contrast agent longitudinal relaxivity | s\(^{-1}\) mM\(^{-1}\) |
| \(\mathrm{Hct}\) | Haematocrit | dimensionless |
| \(C_t(t)\) | Tissue contrast agent concentration | mM |
| \(C_p(t)\) | Plasma contrast agent concentration | mM |

## 1. The spoiled gradient echo signal model

ROCKETSHIP assumes the dynamic series is acquired with a spoiled gradient echo (SPGR, also
called FLASH or SPGR/T1-FFE) sequence at steady state. For that sequence the signal is

\[
S = M_0 \sin\alpha \,\frac{1 - E_1}{1 - \cos\alpha \; E_1},
\qquad E_1 = e^{-\mathrm{TR}\,R_1}
\]

where \(R_1=1/T_1\) and \(M_0\) collects the equilibrium magnetization, receive gain, proton density and all
\(T_2^{*}\) weighting. These factors are unknown, and the purpose of the derivation below is
to eliminate them.

!!! note "Assumptions"
    The derivation assumes spins in the steady state, ideal spoiling, and that
    \(T_2^{*}\) weighting is unchanged by the contrast agent. The first and last assumptions are frequently broken in arteries. Inflow breaks the steady state assumption and at high concentrations, most notably inside large arteries at peak bolus, Gd contrast will alter \(T_2^{*}\). This is one reason why measuring an arterial input function is difficult.

## 2. Eliminating the unknown scaling

Because \(M_0\) does not change during the acquisition, it cancels in the ratio of each
timepoint to the pre-contrast baseline. Define the baseline signal factor from the
pre-contrast relaxation rate \(R_{10} = 1/T_{10}\),

\[
S^{*} = \frac{1 - e^{-\mathrm{TR}/T_{10}}}{1 - \cos\alpha \; e^{-\mathrm{TR}/T_{10}}}
\]

and form the normalized signal \(u(t) = S^{*} \, S(t) / \bar{S}_0\). Substituting into the
SPGR expression and solving for \(E_1\) gives

\[
E_1(t) = \frac{1 - u(t)}{1 - u(t)\cos\alpha}
\]

which inverts directly to the relaxation rate:

\[
R_1(t) = \frac{1}{\mathrm{TR}}\,
\ln\!\left(\frac{1 - u(t)\cos\alpha}{1 - u(t)}\right)
\]

This is the form ROCKETSHIP evaluates. The numerator and denominator are computed separately
so that voxels approaching the singularity at \(u \to 1\) can be identified and excluded
before the logarithm is taken.

### Voxel screening

Two screens are applied at this stage, because the expression above is unstable for voxels
whose signal is dominated by noise:

- Voxels whose mean baseline signal is at or below zero are removed, since \(\bar{S}_0\)
  appears in a denominator.
- Voxels whose ratio \((1 - u\cos\alpha)/(1 - u)\) or resulting \(R_1\) falls outside a
  physically plausible range are removed. The tolerated fraction differs between the arterial
  and tissue paths, because arterial voxels legitimately reach far larger relaxation changes.

Voxels removed here are excluded from all subsequent stages rather than being carried through
as invalid numbers.

## 3. Baseline re-anchoring

The measured \(R_1(t)\) is derived from a signal ratio, so it inherits any noise present in
the baseline average. ROCKETSHIP therefore shifts each voxel's curve so that its pre-contrast
mean equals the relaxation rate implied by the supplied \(T_1\) map:

\[
R_1(t) \;\leftarrow\; R_1(t) + \left(\frac{1}{T_{10}}
- \frac{1}{N_b}\sum_{k \in \text{baseline}} R_1(t_k)\right)
\]

After this step the pre-contrast concentration is zero by construction, which is the condition
every pharmacokinetic model assumes at \(t = 0\).

## 4. Relaxation rate to concentration

Contrast agent shortens \(T_1\) in proportion to its concentration. In the fast-exchange limit
the relationship is linear:

\[
R_1(t) = R_{10} + r_1 \, C(t)
\]

Rearranging gives the tissue concentration directly,

\[
C_t(t) = \frac{R_1(t) - 1/T_{10}}{r_1}
\]

For the arterial input function an additional correction is required. The contrast agent is
confined to the plasma, but the measured arterial voxel contains whole blood. Dividing by the
plasma volume fraction \((1 - \mathrm{Hct})\) converts a whole-blood concentration to the
plasma concentration the models require:

\[
C_p(t) = \frac{R_1(t) - 1/T_{10,\text{blood}}}{r_1 \,(1 - \mathrm{Hct})}
\]

!!! warning "The fast-exchange assumption"
    The linear relationship above assumes water exchange between tissue compartments is fast
    compared with the relaxation rate difference between them. Where that assumption is not
    appropriate, use the [FXR (shutter speed) model](models/fxr.md), which fits the measured
    relaxation rate directly and models the exchange rate as a free parameter.

## 5. Enhancement as an intermediate

Some workflows express the dynamic series as percentage enhancement relative to baseline
rather than as raw signal:

\[
E(t) = 100 \times \frac{S(t) - \bar{S}_0}{\bar{S}_0}
\]

ROCKETSHIP provides a closed-form conversion from enhancement to concentration that follows
the OSIPI reference implementation, including an optional \(B_1\) correction factor
\(\kappa\) that scales the nominal flip angle to the actual flip angle:

\[
C(t) = -\frac{1}{\mathrm{TR}\, r_1}
\ln\!\left(
\frac{e^{\mathrm{TR}/T_{10}}\left(E - 100\cos(\kappa\alpha) - E\,e^{\mathrm{TR}/T_{10}} + 100\right)}
{100\,e^{\mathrm{TR}/T_{10}} + E\cos(\kappa\alpha) - 100\,e^{\mathrm{TR}/T_{10}}\cos(\kappa\alpha) - E\,e^{\mathrm{TR}/T_{10}}\cos(\kappa\alpha)}
\right)
\]

This is algebraically equivalent to the two-step route in sections 2 to 4 when
\(\kappa = 1\). It is provided for interoperability with pipelines that exchange enhancement
curves, and for workflows that have a measured \(B_1\) map available.

## 6. Required inputs

The conversion cannot proceed without the following. Each is documented in the
[DCE Options reference](../dce_options.md).

| Input | Option | Source |
| --- | --- | --- |
| Repetition time | `tr_ms` | Image JSON sidecar, or set manually |
| Flip angle | `fa_deg` | Image JSON sidecar, or set manually |
| Temporal resolution | `time_resolution_sec` | Image JSON sidecar, or set manually |
| Pre-contrast \(T_1\) map | `t1map_files` | Parametric \(T_1\) mapping, in ms |
| Contrast agent relaxivity | `relaxivity` | Image JSON sidecar, or run configuration |
| Haematocrit | `hematocrit` | Image JSON sidecar, or run configuration |
| Baseline extent | `steady_state_end` | Detected automatically, or set manually |

!!! danger "Relaxivity has no default"
    Relaxivity depends on the contrast agent, the field strength and the medium, so no value
    is safe to assume. A run that does not supply one stops with an error rather than
    proceeding. An incorrect relaxivity rescales every concentration, and therefore every
    \(K^{trans}\), producing results that look entirely plausible but are wrong by a constant
    factor. Published values for common agents are tabulated in
    [Shen et al. (2015)](https://pubmed.ncbi.nlm.nih.gov/25658049/).

    Haematocrit does have a default of 0.45, on the basis that it is usually a single
    study-wide value. Supply a measured value where one is available.

## 7. Blood \(T_1\)

The arterial \(T_{10}\) may be taken from the \(T_1\) map at the arterial voxels, or fixed to
a literature value with the `blood_t1_ms` option. A fixed value is often the more stable
choice, because arterial voxels are prone to inflow and partial volume effects that corrupt
a variable flip angle \(T_1\) measurement. The value is read in milliseconds and range
checked; a value supplied in seconds is rejected rather than silently rescaled.
