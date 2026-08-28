# Testing

ROCKETSHIP ships an automated test suite covering the core algorithms of both the MATLAB and
Python implementations. Running it is not necessary for ordinary analysis, but it is worth
doing after installation, after enabling acceleration, or before relying on the software for a
new study.

All commands are run from the repository root using the project virtual environment.

## Test groups

| Group | Command | Covers |
| --- | --- | --- |
| Python suite | `.venv/bin/python -m pytest tests/python` | The Python pipeline end to end. The usual first check after installation |
| MATLAB/Python agreement | `.venv/bin/python -m pytest tests/python -m parity --parity-suite=allmodels -s` | Python parameter maps against committed MATLAB reference maps |
| Backend agreement | included in the parity suite | That the CPU, CPUfit and GPUfit paths produce equivalent results |
| OSIPI reliability | `.venv/bin/python -m pytest tests/python -m osipi -v` | Accuracy against external reference data with known ground truth. See [OSIPI Verification](osipi-verification.md) |
| MATLAB unit tests | `run_unit_tests()` in MATLAB | The MATLAB algorithm implementations |

Full documentation, including the individual test selectors, tolerances and how the reference
baselines are generated, is in the repository:

- [tests/README.md](https://github.com/petmri/ROCKETSHIP/blob/master/tests/README.md) — the
  complete test suite reference
- [tests/contracts/README.md](https://github.com/petmri/ROCKETSHIP/blob/master/tests/contracts/README.md)
  — the cross-language contracts and MATLAB baselines

Graphical interface behavior is intentionally out of scope; the suite targets the algorithms.

## OSIPI verification

Accuracy is verified against external reference data with known ground truth, published by the
Open Science Initiative for Perfusion Imaging. This is the primary evidence that ROCKETSHIP's
fitted parameters are correct rather than merely self-consistent, and it is documented on its
own page:

**[OSIPI Verification](osipi-verification.md)** — the reference data, how accuracy is judged
against OSIPI's acceptance tolerances and the published peer implementations, per-backend
results, and known limitations.

## Performance benchmark

`run_dce_benchmark.py` times a complete DCE run, Stage A through Stage D, once for each
available backend, and prints a comparison table. Use it to find out which backend is fastest
on your hardware and what a real analysis will cost you in wall clock time.

```bash
.venv/bin/python tests/python/run_dce_benchmark.py --dataset-root /path/to/bids_dataset
```

### Configurations

Five configurations are attempted. Any that is unavailable on your system is reported as
`SKIP` rather than failing the run.

| Configuration | Description |
| --- | --- |
| `matlab_cpu` | MATLAB pipeline with acceleration disabled |
| `matlab_gpufit` | MATLAB pipeline using the Gpufit MEX interface |
| `python_cpu` | Python pipeline on the standard fitting path |
| `python_cpufit` | Python pipeline using the Cpufit multi-core CPU backend |
| `python_gpufit` | Python pipeline using Gpufit, on CUDA where available |

Restrict the comparison with `--configs`, for example
`--configs python_cpu,python_gpufit` to measure only what GPU acceleration buys you.

### Useful options

| Option | Purpose |
| --- | --- |
| `--dataset-root` | BIDS dataset to benchmark against |
| `--subject`, `--session` | Which subject and session to run |
| `--models` | Model flags to fit. Defaults to `patlak`; use `all` for every model |
| `--repeats` | Repetitions per configuration, to average out variation |
| `--configs` | Restrict to a subset of the five configurations |
| `--output-json` | Write the detailed results to a file |

### Reading the results

The headline number is `Total(s)`, and it is **whole-process wall clock time**, including
interpreter startup and file input and output. This is deliberate: it is what you actually
wait for, rather than a figure that flatters the software by excluding its overheads.

The table also breaks the total into `A(s)`, `B(s)`, `D(s)` and an `Other(s)` remainder.

!!! warning "Only the total is comparable between languages"
    The per-stage columns come from each pipeline's own internal timers, and the two do not
    bracket the same work. MATLAB's timers cover computation only, so reading the dynamic
    series and writing the output maps and figures land in its `Other(s)`. Python's stage
    timers include that input and output. Comparing `D(s)` between the two is therefore
    meaningless; compare `Total(s)`.

The measured interpreter startup floor is printed for each language, approximately ten seconds
for MATLAB against under one second for Python, so that a small total can be recognized as
mostly startup rather than mistaken for fast fitting.

The number of fitted voxels is reported for each configuration, and a mismatch is flagged.
Configurations that fit different numbers of voxels are not measuring the same workload and
their times should not be compared.

### What is excluded, and why

Two costs are switched off by default because they would distort the comparison rather than
inform it:

- **Stage checkpoints** are off. Checkpointing writes every stage array to disk, roughly 950 MB
  on a full-resolution subject, inside the timed window, for work an ordinary run does not do.
  Enable with `--checkpoints on` if you intend to use checkpointing in production.
- **Spreadsheet output** is off, because with no regions of interest configured the MATLAB
  pipeline writes no table, while Python would perform an additional whole-brain fit with no
  MATLAB counterpart.

Costs a real run genuinely pays are left in on both sides, including the Python quality control
figures and the MATLAB figure export.

### Dataset layout

Raw data is read from `<root>/sourcedata/raw/<subject>/<session>` and derivatives from
`<root>/derivatives/<subdir>/<subject>/<session>`, adjustable with `--raw-subdir` and
`--derivatives-subdir`. Inputs are matched on the BIDS `desc-` naming convention. Where the
named derivatives subdirectory does not contain the subject, a bounded fallback scan probes
sibling directories and reports which one it used.

Full details are in
[tests/README.md](https://github.com/petmri/ROCKETSHIP/blob/master/tests/README.md#performance-benchmark).
