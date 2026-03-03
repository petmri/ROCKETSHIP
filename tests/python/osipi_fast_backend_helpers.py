"""Helpers for OSIPI fast backend pass/fail tests (pycpufit/pygpufit)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _apply_model_specific_prefs,
    _fit_stage_d_model_accelerated,
    _stage_d_fit_prefs,
    probe_acceleration_backend,
)


OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
DCE_DATA_DIR = OSIPI_ROOT / "dce_models"
REFERENCE_DIR = OSIPI_ROOT / "reference"
PEER_ERROR_SUMMARY = json.loads((REFERENCE_DIR / "osipi_peer_error_summary.json").read_text())

FAST_BACKEND_CASES: dict[str, dict[str, str]] = {
    "tofts": {
        "dataset": "dce_DRO_data_tofts.csv",
        "signal_col": "C",
        "aif_col": "ca",
        "time_col": "t",
        "peer_method": "tofts",
    },
    "ex_tofts": {
        "dataset": "dce_DRO_data_extended_tofts.csv",
        "signal_col": "C",
        "aif_col": "ca",
        "time_col": "t",
        "peer_method": "etofts",
    },
    "patlak": {
        "dataset": "patlak_sd_0.02_delay_0.csv",
        "signal_col": "C_t",
        "aif_col": "cp_aif",
        "time_col": "t",
        "peer_method": "patlak",
    },
    "2cxm": {
        "dataset": "2cxm_sd_0.001_delay_0.csv",
        "signal_col": "C_t",
        "aif_col": "cp_aif",
        "time_col": "t",
        "peer_method": "2CXM",
    },
    "tissue_uptake": {
        "dataset": "2cum_sd_0.0025_delay_0.csv",
        "signal_col": "C_t",
        "aif_col": "cp_aif",
        "time_col": "t",
        "peer_method": "2CUM",
    },
}


_BASE_CONFIG = DcePipelineConfig(
    subject_source_path=REPO_ROOT,
    subject_tp_path=REPO_ROOT,
    output_dir=REPO_ROOT,
    backend="cpu",
)
_BASE_PREFS = _stage_d_fit_prefs(_BASE_CONFIG)


def _rows(csv_file: Path) -> list[dict[str, str]]:
    with csv_file.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split()]


def _peer_method_metrics(category: str, method: str) -> dict[str, Any]:
    methods = PEER_ERROR_SUMMARY["metrics"][category]
    if method in methods:
        return methods[method]
    for key, value in methods.items():
        if str(key).lower() == method.lower():
            return value
    raise KeyError(f"Missing peer method '{method}' in category '{category}'")


def _peer_max_abs_error(category: str, method: str, param: str) -> float:
    return float(_peer_method_metrics(category, method)[param]["max_abs_error"])


def _assert_close(actual: float, expected: float, tol: float, label: str, param: str) -> None:
    if not math.isfinite(actual):
        pytest.fail(f"OSIPI {label} {param} produced non-finite value: {actual!r}")
    err = abs(actual - expected)
    assert err <= tol, (
        f"OSIPI {label} {param} abs error {err:.8g} exceeded tolerance {tol:.8g}. "
        f"actual={actual:.8g}, expected={expected:.8g}"
    )


def _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec: float, fp_per_sec: float) -> float:
    if abs(fp_per_sec - ktrans_per_sec) < 1e-12:
        return float("inf")
    return (ktrans_per_sec * fp_per_sec / (fp_per_sec - ktrans_per_sec)) * 60.0


def _stage_d_prefs_for_model(model_name: str) -> dict[str, Any]:
    prefs = dict(_BASE_PREFS)
    if model_name in {"2cxm", "tissue_uptake"}:
        return _apply_model_specific_prefs(prefs, model_name)
    return prefs


def get_fast_backend_case_series(model_name: str) -> dict[str, Any]:
    """Return one representative OSIPI case row and parsed series for a model."""
    if model_name not in FAST_BACKEND_CASES:
        raise KeyError(f"Unsupported fast backend model '{model_name}'.")
    case = FAST_BACKEND_CASES[model_name]
    row = _rows(DCE_DATA_DIR / case["dataset"])[0]
    return {
        "row": row,
        "signal": _series(row[case["signal_col"]]),
        "aif": _series(row[case["aif_col"]]),
        "timer": _series(row[case["time_col"]]),
    }


def _accelerated_fit_row(
    *,
    model_name: str,
    row: dict[str, str],
    signal_col: str,
    aif_col: str,
    time_col: str,
    acceleration_backend: str,
) -> np.ndarray:
    ct = np.asarray(_series(row[signal_col]), dtype=np.float64).reshape(-1, 1)
    cp = np.asarray(_series(row[aif_col]), dtype=np.float64)
    timer = np.asarray(_series(row[time_col]), dtype=np.float64)
    out = _fit_stage_d_model_accelerated(
        model_name=model_name,
        ct=ct,
        cp_use=cp,
        timer=timer,
        prefs=_stage_d_prefs_for_model(model_name),
        acceleration_backend=acceleration_backend,
    )
    assert out is not None, f"Expected accelerated output for {model_name} but got None"
    assert out.shape[0] == 1, f"Expected one accelerated fit row for {model_name}, got {out.shape}"
    return np.asarray(out[0], dtype=np.float64)


def fit_fast_backend_model_case(model_name: str, acceleration_backend: str) -> dict[str, float]:
    """Return accelerated primary-model fit outputs for one representative OSIPI case."""
    case = FAST_BACKEND_CASES.get(model_name)
    if case is None:
        raise KeyError(f"Unsupported fast backend model '{model_name}'.")
    data = get_fast_backend_case_series(model_name)
    row = data["row"]
    fit = _accelerated_fit_row(
        model_name=model_name,
        row=row,
        signal_col=case["signal_col"],
        aif_col=case["aif_col"],
        time_col=case["time_col"],
        acceleration_backend=acceleration_backend,
    )
    if model_name == "tofts":
        return {"ktrans_per_sec": float(fit[0]), "ve": float(fit[1])}
    if model_name == "ex_tofts":
        return {"ktrans_per_sec": float(fit[0]), "ve": float(fit[1]), "vp": float(fit[2])}
    if model_name == "patlak":
        return {"ktrans_per_sec": float(fit[0]), "vp": float(fit[1])}
    if model_name == "2cxm":
        return {
            "ktrans_per_sec": float(fit[0]),
            "ve": float(fit[1]),
            "vp": float(fit[2]),
            "fp_per_sec": float(fit[3]),
        }
    if model_name == "tissue_uptake":
        return {"ktrans_per_sec": float(fit[0]), "fp_per_sec": float(fit[1]), "vp": float(fit[2])}
    raise KeyError(f"Unsupported fast backend model '{model_name}'.")


def require_cpufit_backend() -> str:
    """Return cpufit backend id or skip if unavailable."""
    probe_acceleration_backend.cache_clear()
    probe = probe_acceleration_backend()
    if not bool(probe.get("pycpufit_imported", False)):
        pytest.skip(f"pycpufit unavailable on this platform: {probe.get('pycpufit_error')}")
    return "cpufit_cpu"


def require_gpufit_backend() -> str:
    """Return CUDA gpufit backend id or skip when CUDA gpufit is unavailable."""
    probe_acceleration_backend.cache_clear()
    probe = probe_acceleration_backend()
    if not bool(probe.get("pygpufit_imported", False)):
        pytest.skip(f"pygpufit unavailable on this platform: {probe.get('pygpufit_error')}")
    if str(probe.get("backend", "")) != "gpufit_cuda":
        pytest.skip(
            "pygpufit is importable but CUDA gpufit backend is unavailable; "
            "skip pygpufit reliability checks on non-CUDA platforms."
        )
    return "gpufit_cuda"


def assert_fast_backend_model_case(model_name: str, acceleration_backend: str) -> None:
    """Assert one model's accelerated fit (single representative curve) stays within peer tolerance."""
    if model_name not in FAST_BACKEND_CASES:
        raise KeyError(f"Unsupported fast backend model '{model_name}'.")

    case = FAST_BACKEND_CASES[model_name]
    row = _rows(DCE_DATA_DIR / case["dataset"])[0]
    fit = _accelerated_fit_row(
        model_name=model_name,
        row=row,
        signal_col=case["signal_col"],
        aif_col=case["aif_col"],
        time_col=case["time_col"],
        acceleration_backend=acceleration_backend,
    )
    label = f"{row['label']} ({model_name} {acceleration_backend})"
    method = case["peer_method"]

    if model_name == "tofts":
        _assert_close(
            float(fit[0]) * 60.0,
            float(row["Ktrans"]),
            _peer_max_abs_error("DCEmodels", method, "Ktrans") + 1e-6,
            label,
            "Ktrans",
        )
        _assert_close(
            float(fit[1]),
            float(row["ve"]),
            _peer_max_abs_error("DCEmodels", method, "ve") + 1e-6,
            label,
            "ve",
        )
        return

    if model_name == "ex_tofts":
        _assert_close(
            float(fit[0]) * 60.0,
            float(row["Ktrans"]),
            _peer_max_abs_error("DCEmodels", method, "Ktrans") + 1e-6,
            label,
            "Ktrans",
        )
        _assert_close(
            float(fit[1]),
            float(row["ve"]),
            _peer_max_abs_error("DCEmodels", method, "ve") + 1e-6,
            label,
            "ve",
        )
        _assert_close(
            float(fit[2]),
            float(row["vp"]),
            _peer_max_abs_error("DCEmodels", method, "vp") + 1e-6,
            label,
            "vp",
        )
        return

    if model_name == "patlak":
        _assert_close(
            float(fit[0]) * 60.0,
            float(row["ps"]),
            _peer_max_abs_error("DCEmodels", method, "ps") + 1e-6,
            label,
            "ps",
        )
        _assert_close(
            float(fit[1]),
            float(row["vp"]),
            _peer_max_abs_error("DCEmodels", method, "vp") + 1e-6,
            label,
            "vp",
        )
        return

    if model_name == "2cxm":
        ktrans_per_sec = float(fit[0])
        fp_per_sec = float(fit[3])
        _assert_close(
            float(fit[1]),
            float(row["ve"]),
            _peer_max_abs_error("DCEmodels", method, "ve") + 1e-6,
            label,
            "ve",
        )
        _assert_close(
            float(fit[2]),
            float(row["vp"]),
            _peer_max_abs_error("DCEmodels", method, "vp") + 1e-6,
            label,
            "vp",
        )
        _assert_close(
            fp_per_sec * 60.0 * 100.0,
            float(row["fp"]),
            _peer_max_abs_error("DCEmodels", method, "fp") + 1e-6,
            label,
            "fp",
        )
        _assert_close(
            _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, fp_per_sec),
            float(row["ps"]),
            _peer_max_abs_error("DCEmodels", method, "ps") + 1e-6,
            label,
            "ps",
        )
        return

    if model_name == "tissue_uptake":
        ktrans_per_sec = float(fit[0])
        fp_per_sec = float(fit[1])
        _assert_close(
            float(fit[2]),
            float(row["vp"]),
            _peer_max_abs_error("DCEmodels", method, "vp") + 1e-6,
            label,
            "vp",
        )
        _assert_close(
            fp_per_sec * 60.0 * 100.0,
            float(row["fp"]),
            _peer_max_abs_error("DCEmodels", method, "fp") + 1e-6,
            label,
            "fp",
        )
        _assert_close(
            _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, fp_per_sec),
            float(row["ps"]),
            _peer_max_abs_error("DCEmodels", method, "ps") + 1e-6,
            label,
            "ps",
        )
        return

    raise KeyError(f"Unsupported fast backend model '{model_name}'.")
