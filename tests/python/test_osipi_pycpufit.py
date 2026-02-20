"""OSIPI fast backend checks for the pycpufit acceleration path."""

from __future__ import annotations

import pytest

from osipi_fast_backend_helpers import assert_fast_backend_model_case, require_cpufit_backend


@pytest.fixture(scope="module")
def cpufit_backend() -> str:
    return require_cpufit_backend()


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pycpufit_tofts_fast(cpufit_backend: str) -> None:
    assert_fast_backend_model_case("tofts", cpufit_backend)


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pycpufit_extended_tofts_fast(cpufit_backend: str) -> None:
    assert_fast_backend_model_case("ex_tofts", cpufit_backend)


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pycpufit_patlak_fast(cpufit_backend: str) -> None:
    assert_fast_backend_model_case("patlak", cpufit_backend)


@pytest.mark.osipi
@pytest.mark.fast
@pytest.mark.xfail(
    reason="Secondary-goal model: keep visibility but do not block on 2CXM CPUfit reliability yet.",
    strict=False,
)
def test_osipi_pycpufit_2cxm_fast(cpufit_backend: str) -> None:
    assert_fast_backend_model_case("2cxm", cpufit_backend)


@pytest.mark.osipi
@pytest.mark.fast
@pytest.mark.xfail(
    reason="Secondary-goal model: keep visibility but do not block on tissue uptake CPUfit reliability yet.",
    strict=False,
)
def test_osipi_pycpufit_tissue_uptake_fast(cpufit_backend: str) -> None:
    assert_fast_backend_model_case("tissue_uptake", cpufit_backend)
