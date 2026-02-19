# OSIPI Peer Accuracy Summary (Imported)

Source: [OSIPI DCE-DSC-MRI_TestResults](https://github.com/OSIPI/DCE-DSC-MRI_TestResults) @ `23d3714797045d8103d5b5fa4f4c016840094dc0`.

These values are aggregated from the peer implementation result CSV files and are used to set baseline tolerances for ROCKETSHIP OSIPI tests.

## DCE Models

- `tofts`
  - `Ktrans` absolute error: p95 = `0.0024899795`, max = `0.0036875134`
  - `ve` absolute error: p95 = `0.0021140881`, max = `0.0042475952`
- `etofts`
  - `Ktrans` absolute error: p95 = `0.0015653588`, max = `0.0035122432`
  - `ve` absolute error: p95 = `0.0039657175`, max = `0.0075108682`
  - `vp` absolute error: p95 = `0.0013412044`, max = `0.0022194043`
- `patlak`
  - `ps` absolute error: p95 = `0.0003789129`, max = `0.0004790723`
  - `vp` absolute error: p95 = `0.0017679831`, max = `0.0019779568`
  - `delay` absolute error (for delay-capable implementations): p95 = `0.0203447720`, max = `0.0373983796`

## T1 Mapping

- `linear`
  - `r1` absolute error: p95 = `0.0564628026`, max = `0.4282723349`

## How Current OSIPI Tests Use This

- DCE reliability tests currently gate per-case errors against peer `max` error for each tested parameter.
- T1 linear reliability test gates:
  - `max(error) <= peer max`
  - at least `95%` of cases within `1.2 * peer p95`.
- Patlak delay references are imported and normalized in
  `/Users/samuelbarnes/code/ROCKETSHIP/tests/data/osipi/reference/patlak_delay_reference_values.json`
  for upcoming delay-fit test wiring.
