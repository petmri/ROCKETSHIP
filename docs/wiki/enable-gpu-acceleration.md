# GPU and CPU Acceleration

Voxelwise model fitting is the most computationally demanding part of DCE and parametric
analysis. ROCKETSHIP can offload it to a GPU or to a compiled CPU implementation
through the Gpufit and Cpufit libraries, which typically reduces fitting time by one to two
orders of magnitude for whole-image fits.

Acceleration is optional. Without it, ROCKETSHIP falls back to its standard fitting path and
produces the same results, more slowly.

## Installation

The Python installer sets up the virtual environment and installs the acceleration packages
appropriate for your platform. Run it from the repository root:

```bash
python3 install.py
```

This creates a `.venv` virtual environment, installs the Python dependencies, detects your
platform and CUDA version, and downloads the matching pre-built `pyCpufit` and `pyGpufit`
packages. Where a release bundle includes MATLAB MEX files for your platform, these are
installed as well, and verified when MATLAB is on `PATH`. If MATLAB is not found the
installer reports a warning and still completes successfully, since the Python interfaces do
not need it; rerun the installer after installing MATLAB to get the MEX files verified.

Pre-built packages are published for Linux and Windows with several CUDA versions, and for
macOS on Apple silicon. macOS packages provide CPU acceleration only, as CUDA is not available
on that platform.

### Installer options

| Option | Purpose |
| --- | --- |
| `-e`, `--venv-path` | Install into a virtual environment other than `.venv` |
| `-x`, `--recreate-venv` | Delete and recreate the virtual environment before installing |
| `-t`, `--release-tag` | Install a specific release, for example `-t v1.4.1`, or `-t dev-latest` |
| `-a`, `--asset-id` | Override platform auto-detection, for example `linux-x64-cuda12.8` |
| `-G`, `--no-gui` | Skip the graphical interface dependencies |
| `-M`, `--no-matlab` | Skip installation and verification of the MATLAB MEX files |
| `-m`, `--matlab-cmd` | Use a specific MATLAB executable for post-install verification |
| `-k`, `--no-sha256` | Disable checksum verification of downloaded assets |

Run `python3 install.py --help` for the complete list.

### Verifying the installation

The installer reports which backends it was able to import once it finishes. To check
separately:

```bash
.venv/bin/python -c "import pygpufit.gpufit as gf; print('CUDA available:', gf.cuda_available())"
```

A successful CUDA report means GPU fitting is available. If the import succeeds but CUDA
reports unavailable, the CPU acceleration path will still be used.

## Selecting a backend

The `backend` option controls which fitting path is used. It accepts three values.

| Value | Behaviour |
| --- | --- |
| `auto` | Select the fastest available backend automatically. This is the default. |
| `cpu` | Use the standard fitting path, with no acceleration library. |
| `gpufit` | Require the Gpufit library. Use CUDA where available, otherwise its fallback path. |

Under `auto`, backends are probed in the following order and the first available is used:

1. **`gpufit_cuda`** — Gpufit with an available CUDA device.
2. **`cpufit_cpu`** — the Cpufit multi-core CPU implementation.
3. **`gpufit_cpu_fallback`** — Gpufit's CPU path, where Cpufit is not installed.
4. **`pure_cpu`** — the standard fitting path, where no acceleration library is present.

Setting `force_cpu` to a non-zero value forces the standard path while leaving `backend` at
`auto`.

Each run records which backend was selected, which was actually used, and the reason for the
choice, in the Part D stage summary of the run log. Consult these fields when a run is slower
than expected.

## Supported models

Accelerated fitting is available for the following DCE models:

- [Tofts](../reference/models/tofts.md)
- [Extended Tofts](../reference/models/extended-tofts.md)
- [Patlak](../reference/models/patlak.md)
- [Tissue Uptake](../reference/models/tissue-uptake.md)
- [Two-Compartment Exchange](../reference/models/two-compartment-exchange.md)

The [FXR](../reference/models/fxr.md) model requires a per-voxel baseline relaxation rate that
does not fit the batched form the accelerated backends use, so it runs on the standard path.
[Area under the curve](../reference/models/auc.md) requires no fitting.

Parametric \(T_1\) mapping uses the same backend selection and the same acceleration
libraries.

Where a model or a run cannot use the selected backend, the pipeline falls back through the
remaining options in order and records the reason, rather than failing.

## Tuning

Two options control the accelerated solvers. Both apply to the CUDA and CPU acceleration paths
alike.

| Option | Default | Purpose |
| --- | --- | --- |
| `gpu_tolerance` | 10\(^{-6}\) | Convergence tolerance for the accelerated solver |
| `gpu_max_n_iterations` | — | Maximum solver iterations per voxel |

Initial values for the accelerated path are set separately from the standard path, through
`gpu_initial_value_ktrans`, `gpu_initial_value_ve`, `gpu_initial_value_vp` and
`gpu_initial_value_fp`.

!!! warning "Tightening the tolerance reduces voxel yield"
    A tighter `gpu_tolerance` does not improve accuracy. Below approximately 10\(^{-10}\) the
    accelerated solvers begin marking voxels as non-converged, and the pipeline excludes those
    voxels from the parameter maps. On benchmark data the default of 10\(^{-6}\) returns
    results for every voxel across all models, while 10\(^{-10}\) loses between four and seven
    percent of voxels depending on the model. Leave this at its default unless you have a
    specific reason to change it.

## Numerical agreement

The accelerated and standard paths solve the same problem with different implementations, so
results agree closely but not bit for bit. Differences are at the level of solver convergence
rather than of model formulation, and are far smaller than the measurement uncertainty on any
real acquisition. Where exact reproducibility between runs matters more than speed, fix the
backend explicitly rather than leaving it at `auto`, since `auto` may resolve differently on
different machines.

## MATLAB

The MATLAB pipeline uses the same Gpufit library through its MEX interface. The installer
places the MEX files and verifies that MATLAB can load them, provided a `matlab` executable is
on the path or is named with `--matlab-cmd`. Use `--no-matlab` to skip this step.

To force the MATLAB pipeline onto the CPU, set `force_cpu = 1` in `dce/dce_preferences.txt`.

## Building from source

Pre-built packages cover the supported platforms and are the recommended route. Building from
source is necessary only for an unsupported platform or CUDA version.

The library is a [fork of Gpufit](https://github.com/ironictoo/Gpufit) extended with the
\(T_1\) mapping and DCE model implementations ROCKETSHIP requires. Build it with CMake
following the instructions in that repository, enabling only the models you need. 
Enabling all of the models can cause a crash if they don't fit in the CUDA kernel of your
GPU, but generally all the DCE/T1 related models can be included without problems.

## Troubleshooting

**The installer cannot find a matching release asset.** Platform detection failed or no
package is published for your combination of operating system and CUDA version. Name an asset
explicitly with `--asset-id`, or build from source.

**`pygpufit` imports but reports CUDA unavailable.** The installed package does not match your
CUDA driver version. Check your driver with `nvidia-smi` and reinstall with a matching
`--asset-id`.

**Fitting is no faster than before.** Confirm which backend was actually used by checking the
Part D stage summary in the run log. A backend that failed to load falls back silently by
design, and the recorded reason will say why.

**Results differ slightly from a previous run.** Confirm that both runs used the same backend.
See the note on numerical agreement above.
