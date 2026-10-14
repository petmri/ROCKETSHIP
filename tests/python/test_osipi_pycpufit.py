"""OSIPI DCE reliability — cpufit backend (pyCpufit accelerated Stage-D fit).

Full sweep of every OSIPI DRO case, gated on OSIPI's official acceptance tolerances.
All accelerated Stage-D models pass: Tofts / extended Tofts / Patlak, plus the multi-
compartment 2CUM (tissue uptake) and 2CXM. 2CUM/2CXM now fit the extraction fraction
E=Ktrans/Fp with an O(N) exponential-recurrence convolution and analytic Jacobians (the
Ktrans=Fp pole becomes the bound E->1), and use the shared candidate-assembly/multi-start
machinery in dce_fit_backends.py (fixed default + random log-uniform draws, assembled
once and shared by every backend) to pick the flow basin. The low-flow (Fp=5) cases
that previously missed now pass once Fp can reach that regime (lower_limit_fp). Details:
``project-management/projects/osipi-verification/STATUS.md``.
"""

from __future__ import annotations

import pytest

from osipi_fast_backend_helpers import assert_backend_model_sweep, require_cpufit_backend


@pytest.fixture(scope="module")
def cpufit_backend() -> str:
    return require_cpufit_backend()


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
def test_osipi_pycpufit_2cxm_sweep(cpufit_backend: str) -> None:
    # Passes the full OSIPI gate after the E=Ktrans/Fp reparam + analytic Jacobian + the
    # lowered Fp floor that lets low-flow (Fp=5) cases be represented.
    assert_backend_model_sweep("2cxm", cpufit_backend)


@pytest.mark.osipi
def test_osipi_pycpufit_tissue_uptake_sweep(cpufit_backend: str) -> None:
    # Passes via the backend-agnostic multi-start that rescues the vp<->Fp degenerate minimum.
    assert_backend_model_sweep("tissue_uptake", cpufit_backend)
