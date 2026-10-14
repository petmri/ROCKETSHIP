"""Helpers for OSIPI fast backend pass/fail tests (pycpufit/pygpufit)."""

from __future__ import annotations

import csv
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

from osipi_official_tolerances import official_abs_tol  # noqa: E402


OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
DCE_DATA_DIR = OSIPI_ROOT / "dce_models"
REFERENCE_DIR = OSIPI_ROOT / "reference"

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


def _model_param_checks(model_name: str, fit: np.ndarray, row: dict[str, str]) -> list[tuple[str, float, float]]:
    """Return [(param, actual, expected)] in OSIPI comparison units for one fit vector."""
    if model_name == "tofts":
        return [("Ktrans", float(fit[0]) * 60.0, float(row["Ktrans"])),
                ("ve", float(fit[1]), float(row["ve"]))]
    if model_name == "ex_tofts":
        return [("Ktrans", float(fit[0]) * 60.0, float(row["Ktrans"])),
                ("ve", float(fit[1]), float(row["ve"])),
                ("vp", float(fit[2]), float(row["vp"]))]
    if model_name == "patlak":
        return [("ps", float(fit[0]) * 60.0, float(row["ps"])),
                ("vp", float(fit[1]), float(row["vp"]))]
    if model_name == "2cxm":
        kt, fp = float(fit[0]), float(fit[3])
        return [("ve", float(fit[1]), float(row["ve"])),
                ("vp", float(fit[2]), float(row["vp"])),
                ("fp", fp * 60.0 * 100.0, float(row["fp"])),
                ("ps", _ps_per_min_from_ktrans_fp_per_sec(kt, fp), float(row["ps"]))]
    if model_name == "tissue_uptake":
        kt, fp = float(fit[0]), float(fit[1])
        return [("vp", float(fit[2]), float(row["vp"])),
                ("fp", fp * 60.0 * 100.0, float(row["fp"])),
                ("ps", _ps_per_min_from_ktrans_fp_per_sec(kt, fp), float(row["ps"]))]
    raise KeyError(f"Unsupported fast backend model '{model_name}'.")


def assert_backend_model_sweep(model_name: str, acceleration_backend: str) -> None:
    """Assert an accelerated backend fits the FULL OSIPI DRO sweep within OSIPI tolerances.

    Every case of the model's DRO dataset is fit and each parameter checked against the
    OSIPI official acceptance tolerance (``a_tol + r_tol*|ref|``). Fails with a per-case
    breakdown of every out-of-tolerance parameter.
    """
    if model_name not in FAST_BACKEND_CASES:
        raise KeyError(f"Unsupported fast backend model '{model_name}'.")
    case = FAST_BACKEND_CASES[model_name]
    method = case["peer_method"]
    rows_ = _rows(DCE_DATA_DIR / case["dataset"])

    failures: list[str] = []
    n_checks = 0
    for row in rows_:
        try:
            fit = _accelerated_fit_row(
                model_name=model_name, row=row, signal_col=case["signal_col"],
                aif_col=case["aif_col"], time_col=case["time_col"], acceleration_backend=acceleration_backend,
            )
        except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
            failures.append(f"{row['label']}: fit raised {exc!r}")
            continue
        for param, actual, expected in _model_param_checks(model_name, fit, row):
            n_checks += 1
            tol = official_abs_tol(method, param, expected)
            if not (math.isfinite(actual) and abs(actual - expected) <= tol):
                failures.append(
                    f"{row['label']} {param}: |{actual:.6g}-{expected:.6g}|={abs(actual - expected):.6g} > tol {tol:.6g}"
                )

    if failures:
        shown = "\n  ".join(failures[:12])
        more = f"\n  ... and {len(failures) - 12} more" if len(failures) > 12 else ""
        raise AssertionError(
            f"{model_name} ({acceleration_backend}) OSIPI full sweep: {len(failures)} of {n_checks} "
            f"parameter checks outside OSIPI tolerance across {len(rows_)} cases:\n  {shown}{more}"
        )
