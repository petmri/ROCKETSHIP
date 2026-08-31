# OSIPI Peer Accuracy Summary (Imported — Informational Only)

Human-readable view of `osipi_peer_error_summary.json`: the error **spread** (p95 / max
of |measured − reference|) of the published OSIPI contributor implementations around
ground truth.

> **This is not the test gate.** The DCE reliability tests gate on the OSIPI *official
> acceptance tolerances* in `osipi_official_tolerances.json` (see
> [`../README.md`](../README.md)). The peer spread below is reported for context only.

## Provenance and reproducibility

Source: the OSIPI DCE-DSC-MRI testing framework, published as **van Houdt et al.,
*Magnetic Resonance in Medicine*, 2023** ([doi:10.1002/mrm.29826](https://doi.org/10.1002/mrm.29826)),
via its result repository [OSIPI DCE-DSC-MRI_TestResults](https://github.com/OSIPI/DCE-DSC-MRI_TestResults)
@ `23d3714797045d8103d5b5fa4f4c016840094dc0`.

- **Fully reproducible.** Every per-contributor result CSV (`*_ref` vs `*_meas`) is
  committed under `dce_models_results/`, `t1_mapping_results/`, `si_to_conc_results/`, and
  `dsc_models_results/`. `generate_peer_error_summary.py` pools them and recomputes this
  JSON exactly (verified with `--check`). Every DCE value was independently confirmed
  against the upstream CSVs to machine precision.
- **Near-circular for LEK-ported models — do not gate.** The pool *includes* the
  LEK/Edinburgh implementation that ROCKETSHIP ports (`2cxm`, `tissue_uptake`); our fit
  reproduces LEK, so `peer max` tracks our own error to ~4 significant figures. This is
  why the reliability tests gate on OSIPI's official tolerances instead.

## DCE peer spread (informational)

- `tofts` — `Ktrans` max = `0.0036875134`, p95 = `0.0024899795`; `ve` max = `0.0042475952`, p95 = `0.0021140881`
- `etofts` — `Ktrans` max = `0.0035122432`; `ve` max = `0.0075108682`; `vp` max = `0.0022194043`
- `patlak` — `ps` max = `0.0004790723`; `vp` max = `0.0019779568`
- `2cxm` — `ve` max = `0.0158680531`; `vp` max = `0.0185702304`; `fp` max = `1.9407365`; `ps` max = `0.0186094960`
- `2cum` — `vp` max = `0.0034000192`; `fp` max = `4.4932619`; `ps` max = `0.0017355830`

## T1 mapping peer spread (reproducible)

- `linear` — `r1` max = `0.4282723349`, p95 = `0.0564628026` (n = 513)

## How this is used

- The generator (`generate_osipi_summary.py`) reports, per parameter, ROCKETSHIP's error
  next to both the OSIPI official tolerance (the gate) and this peer max (as `our/peer`,
  where values near 1.0 flag the near-circular DCE limits).
- `run_osipi_reliability.py` emits the peer spread in its non-gating summary payload.
