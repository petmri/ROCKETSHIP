"""OSIPI DCE reliability — gpufit backend (pyGpufit CUDA accelerated Stage-D fit).

Full sweep of every OSIPI DRO case, gated on OSIPI's official acceptance tolerances.
Skipped unless a CUDA gpufit backend is available. Same pattern as cpufit: Tofts /
extended Tofts / Patlak and tissue-uptake (2CUM) converge and pass, 2CUM via the
backend-agnostic random multi-start (dce_pipeline._accel_multistart_refine). The 2CXM fit
still misses a few low-flow (Fp=5) cases even with multi-start -- weakly-identifiable vp,
not a precision bug (xfail, ``--osipi-slow``; the float64 python backend is the reference).
Details: ``docs/project-management/projects/osipi-verification/gpufit_2cxm_2cum_divergence.md``.
"""

from __future__ import annotations

import pytest

from osipi_fast_backend_helpers import assert_backend_model_sweep, require_gpufit_backend

_SLOW_XFAIL_REASON = (
    "gpufit accelerated 2CXM misses a few low-flow (Fp=5) OSIPI cases even with multi-start "
    "-- weakly-identifiable vp, not a solver/precision bug (the float64 python backend, which "
    "fits E=Ktrans/Fp, is the reference); see "
    "docs/project-management/projects/osipi-verification/gpufit_2cxm_2cum_divergence.md"
)


@pytest.fixture(scope="module")
def gpufit_backend() -> str:
    return require_gpufit_backend()


def _require_slow(run_osipi_slow: bool) -> None:
    if not run_osipi_slow:
        pytest.skip("Use --osipi-slow to run the full accelerated 2CXM/2CUM sweep.")


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
@pytest.mark.osipi_slow
@pytest.mark.slow
@pytest.mark.xfail(reason=_SLOW_XFAIL_REASON, strict=False)
def test_osipi_pygpufit_2cxm_sweep(gpufit_backend: str, run_osipi_slow: bool) -> None:
    _require_slow(run_osipi_slow)
    assert_backend_model_sweep("2cxm", gpufit_backend)


@pytest.mark.osipi
@pytest.mark.osipi_slow
@pytest.mark.slow
def test_osipi_pygpufit_tissue_uptake_sweep(gpufit_backend: str, run_osipi_slow: bool) -> None:
    # Passes via the backend-agnostic multi-start that rescues the vp<->Fp degenerate minimum.
    _require_slow(run_osipi_slow)
    assert_backend_model_sweep("tissue_uptake", gpufit_backend)
