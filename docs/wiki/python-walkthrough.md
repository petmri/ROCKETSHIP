# Python Walkthrough

This guide covers running the Python implementation of ROCKETSHIP, which is the recommended
interface for new work. It describes environment setup, the command line and graphical
interfaces for DCE and parametric analysis, and the outputs each produces.

All commands are run from the repository root.

## Available interfaces

| Interface | Entry point |
| --- | --- |
| DCE command line | `run_dce_python_cli.py` |
| DCE graphical interface | `run_dce_python_gui.py` |
| Parametric \(T_1\) command line | `run_parametric_python_cli.py` |
| Parametric \(T_1\) graphical interface | `run_parametric_python_gui.py` |
| BIDS dataset discovery | `run_bids_discovery.py` |
| BIDS batch processing | `run_dce_bids_batch.py` |
| GUI launchers (written by the installer) | `rocketship_dce.sh`, `rocketship_parametric.sh` (`.bat` on Windows) |

## 1. Environment setup

The installer creates a virtual environment, installs dependencies, and adds the optional
acceleration libraries for your platform:

```bash
python3 install.py
```

It also writes `rocketship_dce.sh` and `rocketship_parametric.sh` (`.bat` on Windows) into the repository root, each a
wrapper that activates the virtual environment and launches a GUI:

```bash
./rocketship_dce.sh          # DCE GUI
./rocketship_parametric.sh   # parametric T1 GUI
```

MATLAB is optional. If it is not on `PATH`, the installer warns that the MATLAB MEX files
could not be verified and finishes successfully; the Python interfaces are unaffected.

See [GPU and CPU Acceleration](enable-gpu-acceleration.md) for platform support and installer
options.

To set up the environment manually, without acceleration:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements_gui.txt
```

The `requirements_gui.txt` file is needed only for the graphical interfaces.

## 2. DCE analysis from the command line

Run with the built-in example configuration:

```bash
.venv/bin/python run_dce_python_cli.py
```

Run with your own configuration file:

```bash
.venv/bin/python run_dce_python_cli.py --config my_study_config.json
```

Individual settings can be overridden at run time without editing the configuration file.
Each `--set` argument takes one `key=value` pair:

```bash
.venv/bin/python run_dce_python_cli.py --config my_study_config.json --set blood_t1_ms=1600
```

Every available option is documented in the [DCE Options reference](../dce_options.md).

### Outputs

| Output | Contents |
| --- | --- |
| `dce_pipeline_run.json` | Complete record of the run, including resolved settings and stage summaries |
| `dce_pipeline_events.jsonl` | Chronological event log |
| Parameter maps | One NIfTI per fitted parameter, per enabled model |
| `dceAIF_fitting.png` | Arterial input function fit, for quality control |
| `dce_timecurves.png` | Relaxation rate and concentration curves, for quality control |
| Spreadsheet output | Region of interest results, when `write_xls` is enabled |

Stage checkpoint files are written when a checkpoint directory is configured, allowing a run
to be resumed or a later stage to be repeated without recomputing the earlier ones.

!!! tip "Inspect the quality control figures"
    Check the arterial input function figure before interpreting any parameter map. A
    misplaced baseline or a poorly fitted input function invalidates every fitted value in the
    run, and both are immediately visible in that figure.

## 3. Parametric \(T_1\) mapping from the command line

```bash
.venv/bin/python run_parametric_python_cli.py
```

Outputs are a run record (`parametric_t1_run.json`), an event log
(`parametric_t1_events.jsonl`), the \(T_1\) map, and a map of the coefficient of determination
for the fit at each voxel.

A \(T_1\) map is required input for DCE analysis, so this step normally comes first.

## 4. Graphical interfaces

Activate the environment, then launch either interface:

```bash
source .venv/bin/activate
python run_dce_python_gui.py
```

```bash
source .venv/bin/activate
python run_parametric_python_gui.py
```

The DCE interface is organised into four tabs:

- **Inputs** — select images, masks and maps, and set acquisition and fitting options.
- **CLI Output** — the run log as it is produced.
- **QC Figures** — the quality control figures described above.
- **Results** — a slice viewer for the resulting parameter maps and dynamic series.

Configurations built in the interface can be saved and reused from the command line, which is
the usual route from exploratory analysis to batch processing.

## 5. Working with BIDS datasets

To enumerate the sessions available in a BIDS dataset and write a manifest:

```bash
.venv/bin/python run_bids_discovery.py \
  --bids-root /path/to/bids_dataset \
  --output-json out/bids_manifest.json \
  --print-json
```

The manifest can then be used to drive batch processing across the dataset with
`run_dce_bids_batch.py`.

## 6. Acquisition metadata

DCE analysis requires the repetition time, flip angle and temporal resolution of the dynamic
series. These are read from the JSON sidecar accompanying the images wherever one is present,
which is the recommended arrangement.

Where no sidecar is available, all three must be supplied manually, through the `tr_ms`,
`fa_deg` and `time_resolution_sec` options. Supplying only some of them alongside a sidecar is
rejected: set all three, or none.

Contrast agent relaxivity and haematocrit follow the opposite precedence, since they describe
the scan rather than the analysis. A value in the image sidecar takes priority over the run
configuration. Relaxivity has no default and must be supplied; see
[Signal to Concentration](../reference/signal-to-concentration.md).

## 7. Running the test suite

```bash
.venv/bin/python -m pytest tests/python -q
```

## Further reading

- [DCE Options reference](../dce_options.md) — every configuration option
- [Pharmacokinetic models](../reference/models/index.md) — model equations and selection
- [Signal to Concentration](../reference/signal-to-concentration.md) — the conversion and its inputs
- [GPU and CPU Acceleration](enable-gpu-acceleration.md) — installation and backend selection
