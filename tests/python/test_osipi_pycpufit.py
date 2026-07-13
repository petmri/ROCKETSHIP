"""OSIPI DCE reliability — cpufit backend (pyCpufit accelerated Stage-D fit).

Full sweep of every OSIPI DRO case, gated on OSIPI's official acceptance tolerances.
Tofts / extended Tofts / Patlak and the tissue-uptake (2CUM) model converge and pass;
2CUM relies on the backend-agnostic random multi-start (dce_pipeline._accel_multistart_refine)
to escape the wrong-Fp-basin degenerate minimum. The 2CXM fit still misses a few low-flow
(Fp=5) cases even with multi-start -- weakly-identifiable vp at this noise level, not a
precision bug -- so it is gated behind ``--osipi-slow`` and marked xfail (the float64 python
backend, which fits the extraction fraction E=Ktrans/Fp, is the reference for 2CXM). Details:
``docs/project-management/projects/osipi-verification/gpufit_2cxm_2cum_divergence.md``.
"""

from __future__ import annotations

import pytest

from osipi_fast_backend_helpers import assert_backend_model_sweep, require_cpufit_backend

_SLOW_XFAIL_REASON = (
    "cpufit accelerated 2CXM misses a few low-flow (Fp=5) OSIPI cases even with multi-start "
    "-- weakly-identifiable vp, not a solver/precision bug (the float64 python backend, which "
    "fits E=Ktrans/Fp, is the reference); see "
    "docs/project-management/projects/osipi-verification/gpufit_2cxm_2cum_divergence.md"
)


@pytest.fixture(scope="module")
def cpufit_backend() -> str:
    return require_cpufit_backend()


def _require_slow(run_osipi_slow: bool) -> None:
    if not run_osipi_slow:
        pytest.skip("Use --osipi-slow to run the full accelerated 2CXM/2CUM sweep.")


@pytest.mark.osipi
def test_osipi_pycpufit_tofts_sweep(cpufit_backend: str) -> None:
    assert_backend_model_sweep("tofts", cpufit_backend)


@pytest.mark.osipi
def test_osipi_pycpufit_extended_tofts_sweep(cpufit_backend: str) -> None:
    assert_backend_model_sweep("ex_tofts", cpufit_backend)


@pytest.mark.osipi
def test_osipi_pycpufit_patlak_sweep(cpufit_backend: str) -> None:
    assert_backend_model_sweep("patlak", cpufit_backend)


@pytest.mark.osipi
@pytest.mark.osipi_slow
@pytest.mark.slow
@pytest.mark.xfail(reason=_SLOW_XFAIL_REASON, strict=False)
def test_osipi_pycpufit_2cxm_sweep(cpufit_backend: str, run_osipi_slow: bool) -> None:
    _require_slow(run_osipi_slow)
    assert_backend_model_sweep("2cxm", cpufit_backend)


@pytest.mark.osipi
@pytest.mark.osipi_slow
@pytest.mark.slow
def test_osipi_pycpufit_tissue_uptake_sweep(cpufit_backend: str, run_osipi_slow: bool) -> None:
    # Passes via the backend-agnostic multi-start that rescues the vp<->Fp degenerate minimum.
    _require_slow(run_osipi_slow)
    assert_backend_model_sweep("tissue_uptake", cpufit_backend)
