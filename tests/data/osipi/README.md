# OSIPI Verification: Reference Data and Reliability Tests

## Why these tests exist

DCE-MRI and T1 mapping rely on nonlinear model fitting, where small implementation
choices — unit conventions, optimizer settings, how a model equation is coded — can
shift the estimated parameters enough to matter clinically. To give confidence that
ROCKETSHIP's fitting routines are free of such bugs, they are verified against an
independent external standard: the **Open Science Initiative for Perfusion Imaging
(OSIPI)**, an ISMRM-led effort that publishes reference data and a community framework
for comparing perfusion software.

OSIPI provides **digital reference objects (DROs)** — synthetic concentration–time data
generated from *known* kinetic parameters — and a testing framework that runs many
independent research-group implementations against those same DROs
(van Houdt et al., *Magnetic Resonance in Medicine*, 2023,
[doi:10.1002/mrm.29826](https://doi.org/10.1002/mrm.29826)).

The tests run ROCKETSHIP's own fitting routines on the OSIPI DROs and check the results
against OSIPI's published acceptance criteria.

## Which fitting routines are checked

ROCKETSHIP can fit DCE data with four backends: **MATLAB**, **python** (pure-CPU SciPy),
**cpufit** (pyCpufit) and **gpufit** (pyGpufit, CUDA). The three non-MATLAB backends are
each verified against OSIPI, where available on the machine running the tests:

- **python** — always checked; it is the reference DCE fit and the only backend for T1
  mapping and signal-to-concentration.
- **cpufit / gpufit** — the accelerated fits for the five DCE models, checked when the
  respective package (and, for gpufit, a CUDA GPU) is available.

## What they show

- **ROCKETSHIP's python fits recover the known ground truth** for the Tofts, extended
  Tofts, Patlak, two-compartment exchange (2CXM) and two-compartment uptake (2CUM) models,
  and for variable-flip-angle T1 mapping and signal-to-concentration conversion — all
  within OSIPI's official pass/fail tolerances. This is evidence the ported code has no
  gross errors, unit mistakes, or model-implementation defects.
- **The accelerated backends agree with python on the simpler models** (Tofts, extended
  Tofts, Patlak) but **diverge on the stiff multi-compartment 2CXM/2CUM fits** — the
  fixed-iteration accelerated solver does not reliably converge there, so those models
  should be fit with the python backend. The per-backend accuracy table and figures make
  this explicit.
- **How ROCKETSHIP compares to the field.** Beyond pass/fail, the accuracy summary places
  ROCKETSHIP's error next to the spread of the published community implementations, so you
  can see where it sits relative to established software.

See the generated report at
`docs/project-management/projects/osipi-verification/osipi_summary.md`. Everything below is
committed so the verification is fully reproducible.

## The reference data

All DRO datasets are **byte-identical (MD5) to the OSIPI source** at commit `23d3714` of
the [DCE-DSC-MRI_TestResults](https://github.com/OSIPI/DCE-DSC-MRI_TestResults) repository
(`test/DCEmodels/data/`). Each row's parameter columns (`vp/ve/fp/ps` or `Ktrans/ve/vp`)
are the *true values used to generate the data*; the pharmacokinetic DROs were generated
by M. Thrippleton ([mjt320/DCE-functions](https://github.com/mjt320/DCE-functions);
Manning et al., MRM 2021, [doi:10.1002/mrm.28833](https://doi.org/10.1002/mrm.28833)).

- DCE model DROs — `tests/data/osipi/dce_models/`
  - `dce_DRO_data_tofts.csv`, `dce_DRO_data_extended_tofts.csv`
  - `patlak_sd_0.02_delay_{0,5}.csv`
  - `2cxm_sd_0.001_delay_{0,5}.csv`
  - `2cum_sd_0.0025_delay_{0,5}.csv`
- T1 mapping data — `tests/data/osipi/t1_mapping/`
  - `t1_brain_data.csv`, `t1_quiba_data.csv`, `t1_prostate_data.csv`
- Signal-to-concentration data — `tests/data/osipi/si_to_conc/`
  - `SI2Conc_data.csv`
- Patlak arterial-delay reference values — `tests/data/osipi/reference/patlak_delay_reference_values.json`
  - Links each Patlak case to its delay-0 and delay-5 reference values (for future
    delay-fitting coverage; ROCKETSHIP does not yet fit arterial delay).

The published per-implementation results of the OSIPI framework (each group's fitted
`*_meas` values next to the `*_ref` ground truth) are mirrored from `test/results/` in the
same upstream repo, and are the source of the peer comparison described below:

- `tests/data/osipi/reference/dce_models_results/` (DCE models)
- `tests/data/osipi/reference/t1_mapping_results/` (T1 mapping)
- `tests/data/osipi/reference/si_to_conc_results/` (signal-to-concentration)
- `tests/data/osipi/reference/dsc_models_results/` (DSC parameter derivation)

## How accuracy is judged: two reference files

**1. OSIPI official acceptance tolerances — the pass/fail gate.**
`reference/osipi_official_tolerances.json` holds OSIPI's own per-parameter tolerances,
transcribed verbatim from the OSIPI test suite (`test/DCEmodels/DCEmodels_data.py`), where
each implementation is checked with
`assert_allclose(measured, reference, atol=a_tol, rtol=r_tol)`. Per the OSIPI paper these
tolerances are deliberately **wide validity checks** — set to catch gross/unit errors, and
"not intended to indicate an acceptable level of accuracy." The ROCKETSHIP reliability and
fast-backend tests gate on these (via `tests/python/osipi_official_tolerances.py`).

**2. Peer-implementation error spread — accuracy context, not a gate.**
`reference/osipi_peer_error_summary.json` (human-readable view:
`reference/peer_accuracy_summary.md`) holds the pooled error spread (mae / p90 / p95 / max
of |measured − reference|) across every published contributor implementation.

- It is reported for context — it shows how ROCKETSHIP's error compares to the range of
  established software, but it is **not** used as a pass/fail bar.
- It is **not** gated on because the comparison is partly self-referential: ROCKETSHIP's
  2CXM and 2CUM fits are reimplementations of the LEK/Edinburgh code that is *also* in the
  peer pool, so for those models ROCKETSHIP reproduces LEK and its error naturally tracks
  the peer maximum to ~4 significant figures.

## Reproducing the verification

**Peer error summary.** `reference/generate_peer_error_summary.py` pools every committed
per-implementation result CSV and recomputes `osipi_peer_error_summary.json`. It reproduces
the committed file to machine precision:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python tests/data/osipi/reference/generate_peer_error_summary.py --check  # verify
.venv/bin/python tests/data/osipi/reference/generate_peer_error_summary.py          # rewrite
```

**Accuracy summary + figures.** `reference/generate_osipi_summary.py` fits every DRO with
the same functions the tests gate on, then writes a plain-markdown report (data provenance,
a table of ROCKETSHIP error vs the OSIPI gate and the peer spread, and per-case
ground-truth-vs-fit tables) plus comparison figures:

```bash
cd /path/to/ROCKETSHIP
.venv/bin/python tests/data/osipi/reference/generate_osipi_summary.py
# -> docs/project-management/projects/osipi-verification/osipi_summary.md
# -> tests/data/osipi/reference/figures/*.png
```

## Running the tests

The tests are labelled `@pytest.mark.osipi`:

- `tests/python/test_osipi_dce_reliability.py` — DCE pharmacokinetic models, **python**
  backend (full sweep of all DRO cases)
- `tests/python/test_osipi_pycpufit.py` — DCE models, **cpufit** backend
- `tests/python/test_osipi_pygpufit.py` — DCE models, **gpufit** backend (skipped without a
  CUDA GPU)
- `tests/python/test_osipi_t1_reliability.py` — T1 mapping (linear, nonlinear, two-FA)
- `tests/python/test_osipi_si_to_conc_reliability.py` — signal-to-concentration
- `tests/python/test_osipi_backend_consistency.py` — python vs cpufit/gpufit agreement for
  the primary DCE models, where an accelerated backend is available
- `tests/python/run_osipi_reliability.py` — command-line runner that prints a reliability
  summary (ROCKETSHIP error vs the OSIPI gate, with the peer spread shown for context)

The cpufit/gpufit tests check representative cases (where the accelerated solver is
reliable); the per-backend accuracy report above characterizes the full sweep, including
where cpufit/gpufit diverge on 2CXM/2CUM.

```bash
cd /path/to/ROCKETSHIP

# OSIPI tests only (includes the full 2CXM / 2CUM sweeps + reliability fits by default)
.venv/bin/python -m pytest tests/python -m osipi -v

# reliability summary to a JSON file
.venv/bin/python tests/python/run_osipi_reliability.py \
  --suite all \
  --summary-json /tmp/osipi_reliability_summary.json
```

## Source and licensing

The reference data and per-implementation results are drawn from the OSIPI project
(Apache-2.0 licensed):

- [OSIPI DCE-DSC-MRI_TestResults](https://github.com/OSIPI/DCE-DSC-MRI_TestResults) @ `23d3714797045d8103d5b5fa4f4c016840094dc0` — DROs and peer results
- [OSIPI DCE-DSC-MRI_CodeCollection](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection) @ `2654dfa80ce60f8b9164736869eb7c2bc6f62930` — the contributed implementations these results come from

Please cite van Houdt et al., MRM 2023 ([doi:10.1002/mrm.29826](https://doi.org/10.1002/mrm.29826))
when referring to the OSIPI framework, and Manning et al., MRM 2021
([doi:10.1002/mrm.28833](https://doi.org/10.1002/mrm.28833)) for the pharmacokinetic DROs.
