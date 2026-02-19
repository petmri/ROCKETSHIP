# OSIPI Reference Data and Reliability Tests

This directory contains **additive** OSIPI-backed assets and tests. Existing ROCKETSHIP tests are unchanged.

## Source Repositories

Reference data and peer result summaries were imported from:

- [OSIPI DCE-DSC-MRI CodeCollection](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection)
  - Commit: `2654dfa80ce60f8b9164736869eb7c2bc6f62930`
  - License: Apache-2.0 (`LICENSE` in the upstream repository)
- [OSIPI DCE-DSC-MRI TestResults](https://github.com/OSIPI/DCE-DSC-MRI_TestResults)
  - Commit: `23d3714797045d8103d5b5fa4f4c016840094dc0`

## Imported Data

- DCE model datasets: `/Users/samuelbarnes/code/ROCKETSHIP/tests/osipi/data/dce_models/`
  - `dce_DRO_data_tofts.csv`
  - `dce_DRO_data_extended_tofts.csv`
  - `patlak_sd_0.02_delay_0.csv`
  - `patlak_sd_0.02_delay_5.csv`
  - `2cxm_sd_0.001_delay_0.csv`
  - `2cxm_sd_0.001_delay_5.csv`
  - `2cum_sd_0.0025_delay_0.csv`
  - `2cum_sd_0.0025_delay_5.csv`
- T1 datasets: `/Users/samuelbarnes/code/ROCKETSHIP/tests/osipi/data/t1_mapping/`
  - `t1_brain_data.csv`
  - `t1_quiba_data.csv`
  - `t1_prostate_data.csv`

## Patlak Delay Values (Imported Now)

Patlak delay reference values are normalized into:

- `/Users/samuelbarnes/code/ROCKETSHIP/tests/osipi/reference/patlak_delay_reference_values.json`

This manifest links each base Patlak case label to the delay-0 and delay-5 reference values and preserves `vp`/`ps` references for future delay-fit model tests.

## Peer Accuracy Summary Used for Tolerance Baselines

Peer error summaries are stored in:

- `/Users/samuelbarnes/code/ROCKETSHIP/tests/osipi/reference/osipi_peer_error_summary.json`
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/osipi/reference/peer_accuracy_summary.md`

These values are computed from OSIPI `TestResults` CSV outputs and used by the OSIPI tests to set comparison thresholds in a reproducible way.

## Test Modules

- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/test_osipi_dce_reliability.py`
- `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/test_osipi_t1_reliability.py`

All tests are labeled with `@pytest.mark.osipi`.

Run only OSIPI tests:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pytest tests/python -m osipi -v
```

Run OSIPI tests including long-running fits:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python -m pytest tests/python -m osipi -v --osipi-slow
```
