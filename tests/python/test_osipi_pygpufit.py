"""OSIPI DCE reliability — gpufit backend (pyGpufit CUDA accelerated Stage-D fit).

Full sweep of every OSIPI DRO case, gated on OSIPI's official acceptance tolerances.
Skipped unless a CUDA gpufit backend is available. Same pattern as cpufit: the accelerated
models fit E=Ktrans/Fp with analytic Jacobians (2CUM/2CXM), using the backend-agnostic
random multi-start (dce_pipeline._accel_multistart_refine). The CUDA kernels mirror the
verified cpufit math (same reparam + analytic Jacobian) but have not been built/run on CUDA
hardware from this environment, so the multi-compartment 2CXM sweep stays xfail(strict=False)
pending on-hardware confirmation. Details:
``docs/project-management/projects/osipi-verification/STATUS.md``.
"""

from __future__ import annotations

import pytest

from osipi_fast_backend_helpers import assert_backend_model_sweep, require_gpufit_backend

_SLOW_XFAIL_REASON = (
    "gpufit accelerated 2CXM (reparam E=Ktrans/Fp + analytic Jacobian) mirrors the verified "
    "cpufit math but the CUDA kernels have not been built/run on CUDA hardware from this "
    "environment; strict=False so it XPASSes once confirmed on hardware. See "
    "docs/project-management/projects/osipi-verification/STATUS.md"
)


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
@pytest.mark.xfail(reason=_SLOW_XFAIL_REASON, strict=False)
def test_osipi_pygpufit_2cxm_sweep(gpufit_backend: str) -> None:
    assert_backend_model_sweep("2cxm", gpufit_backend)


@pytest.mark.osipi
def test_osipi_pygpufit_tissue_uptake_sweep(gpufit_backend: str) -> None:
    # Passes via the backend-agnostic multi-start that rescues the vp<->Fp degenerate minimum.
    assert_backend_model_sweep("tissue_uptake", gpufit_backend)
