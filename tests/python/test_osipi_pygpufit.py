"""OSIPI DCE reliability — gpufit backend (pyGpufit CUDA accelerated Stage-D fit).

Full sweep of every OSIPI DRO case, gated on OSIPI's official acceptance tolerances.
Skipped unless a CUDA gpufit backend is available. Same pattern as cpufit: the accelerated
models fit E=Ktrans/Fp with analytic Jacobians (2CUM/2CXM), using the shared candidate-
assembly/multi-start machinery in dce_fit_backends.py (fixed default + random log-uniform
draws, assembled once and shared by every backend). The CUDA kernels mirror the verified
cpufit math (same reparam + analytic Jacobian) and additionally port the CPU solver's
constrained backtracking line search, confirmed on CUDA hardware. Details:
``project-management/projects/osipi-verification/STATUS.md``.
"""

from __future__ import annotations

import pytest

from osipi_fast_backend_helpers import assert_backend_model_sweep, require_gpufit_backend


@pytest.fixture(scope="module")
def gpufit_backend() -> str:
    return require_gpufit_backend()


@pytest.mark.osipi
def test_osipi_pygpufit_tofts_sweep(gpufit_backend: str) -> None:
    assert_backend_model_sweep("tofts", gpufit_backend)


@pytest.mark.osipi
def test_osipi_pygpufit_extended_tofts_sweep(gpufit_backend: str) -> None:
    assert_backend_model_sweep("ex_tofts", gpufit_backend)


@pytest.mark.osipi
def test_osipi_pygpufit_patlak_sweep(gpufit_backend: str) -> None:
    assert_backend_model_sweep("patlak", gpufit_backend)


@pytest.mark.osipi
def test_osipi_pygpufit_2cxm_sweep(gpufit_backend: str) -> None:
    assert_backend_model_sweep("2cxm", gpufit_backend)


@pytest.mark.osipi
def test_osipi_pygpufit_tissue_uptake_sweep(gpufit_backend: str) -> None:
    # Passes via the backend-agnostic multi-start that rescues the vp<->Fp degenerate minimum.
    assert_backend_model_sweep("tissue_uptake", gpufit_backend)
