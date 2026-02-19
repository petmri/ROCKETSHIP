"""OSIPI fast backend checks for the pygpufit acceleration path."""

from __future__ import annotations

import pytest

from osipi_fast_backend_helpers import assert_fast_backend_model_case, require_gpufit_backend


@pytest.fixture(scope="module")
def gpufit_backend() -> str:
    return require_gpufit_backend()


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pygpufit_tofts_fast(gpufit_backend: str) -> None:
    assert_fast_backend_model_case("tofts", gpufit_backend)


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pygpufit_extended_tofts_fast(gpufit_backend: str) -> None:
    assert_fast_backend_model_case("ex_tofts", gpufit_backend)


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pygpufit_patlak_fast(gpufit_backend: str) -> None:
    assert_fast_backend_model_case("patlak", gpufit_backend)


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pygpufit_2cxm_fast(gpufit_backend: str) -> None:
    assert_fast_backend_model_case("2cxm", gpufit_backend)


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pygpufit_tissue_uptake_fast(gpufit_backend: str) -> None:
    assert_fast_backend_model_case("tissue_uptake", gpufit_backend)
