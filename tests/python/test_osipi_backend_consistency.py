"""Primary-model backend consistency checks (CPU vs CPUfit/GPUfit when available)."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

from osipi_fast_backend_helpers import (
    fit_fast_backend_model_case,
    get_fast_backend_case_series,
    require_cpufit_backend,
    require_gpufit_backend,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import model_extended_tofts_fit, model_patlak_fit, model_tofts_fit  # noqa: E402


PRIMARY_MODELS = ("tofts", "ex_tofts", "patlak")


def _cpu_primary_fit(model_name: str) -> dict[str, float]:
    case = get_fast_backend_case_series(model_name)
    ct = case["signal"]
    cp = case["aif"]
    timer = case["timer"]
    if model_name == "tofts":
        fit = model_tofts_fit(ct, cp, timer)
        return {"ktrans_per_sec": float(fit[0]), "ve": float(fit[1])}
    if model_name == "ex_tofts":
        fit = model_extended_tofts_fit(ct, cp, timer)
        return {"ktrans_per_sec": float(fit[0]), "ve": float(fit[1]), "vp": float(fit[2])}
    if model_name == "patlak":
        fit = model_patlak_fit(ct, cp, timer)
        return {"ktrans_per_sec": float(fit[0]), "vp": float(fit[1])}
    raise KeyError(f"Unsupported model '{model_name}'.")


def _assert_model_consistency(model_name: str, backend: str) -> None:
    cpu = _cpu_primary_fit(model_name)
    accel = fit_fast_backend_model_case(model_name, backend)
    for param, cpu_val in cpu.items():
        acc_val = float(accel[param])
        assert math.isfinite(acc_val), f"{model_name} {backend} produced non-finite {param}: {acc_val!r}"
        tol = max(1e-8, abs(cpu_val) * 1e-4)
        diff = abs(acc_val - cpu_val)
        assert diff <= tol, (
            f"{model_name} {backend} drifted from cpu for {param}: "
            f"diff={diff:.8g} tol={tol:.8g} accel={acc_val:.8g} cpu={cpu_val:.8g}"
        )


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_cpufit_primary_models_are_consistent_with_cpu() -> None:
    backend = require_cpufit_backend()
    for model_name in PRIMARY_MODELS:
        _assert_model_consistency(model_name, backend)


@pytest.mark.osipi
@pytest.mark.fast
def test_osipi_pygpufit_primary_models_are_consistent_with_cpu() -> None:
    backend = require_gpufit_backend()
    for model_name in PRIMARY_MODELS:
        _assert_model_consistency(model_name, backend)

