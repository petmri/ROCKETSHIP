"""Shared OSIPI primary DCE reliability summary helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import model_extended_tofts_fit, model_patlak_fit, model_tofts_fit  # noqa: E402


OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
DCE_DATA_DIR = OSIPI_ROOT / "dce_models"
REFERENCE_DIR = OSIPI_ROOT / "reference"
PEER_ERROR_SUMMARY = json.loads((REFERENCE_DIR / "osipi_peer_error_summary.json").read_text())


def _rows(csv_file: Path) -> list[dict[str, str]]:
    with csv_file.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split()]


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"n": 0.0, "mae": math.nan, "p95_abs_error": math.nan, "max_abs_error": math.nan}
    n = float(len(ordered))
    mae = float(sum(ordered) / len(ordered))
    p95 = float(ordered[int(round(0.95 * (len(ordered) - 1)))])
    max_err = float(ordered[-1])
    return {"n": n, "mae": mae, "p95_abs_error": p95, "max_abs_error": max_err}


def peer_dce_primary_metrics() -> dict[str, dict[str, dict[str, float]]]:
    dce_peer = PEER_ERROR_SUMMARY["metrics"]["DCEmodels"]
    return {
        "tofts": {
            "Ktrans": {
                "n": float(dce_peer["tofts"]["Ktrans"]["n"]),
                "mae": float(dce_peer["tofts"]["Ktrans"]["mae"]),
                "p95_abs_error": float(dce_peer["tofts"]["Ktrans"]["p95_abs_error"]),
                "max_abs_error": float(dce_peer["tofts"]["Ktrans"]["max_abs_error"]),
            },
            "ve": {
                "n": float(dce_peer["tofts"]["ve"]["n"]),
                "mae": float(dce_peer["tofts"]["ve"]["mae"]),
                "p95_abs_error": float(dce_peer["tofts"]["ve"]["p95_abs_error"]),
                "max_abs_error": float(dce_peer["tofts"]["ve"]["max_abs_error"]),
            },
        },
        "etofts": {
            "Ktrans": {
                "n": float(dce_peer["etofts"]["Ktrans"]["n"]),
                "mae": float(dce_peer["etofts"]["Ktrans"]["mae"]),
                "p95_abs_error": float(dce_peer["etofts"]["Ktrans"]["p95_abs_error"]),
                "max_abs_error": float(dce_peer["etofts"]["Ktrans"]["max_abs_error"]),
            },
            "ve": {
                "n": float(dce_peer["etofts"]["ve"]["n"]),
                "mae": float(dce_peer["etofts"]["ve"]["mae"]),
                "p95_abs_error": float(dce_peer["etofts"]["ve"]["p95_abs_error"]),
                "max_abs_error": float(dce_peer["etofts"]["ve"]["max_abs_error"]),
            },
            "vp": {
                "n": float(dce_peer["etofts"]["vp"]["n"]),
                "mae": float(dce_peer["etofts"]["vp"]["mae"]),
                "p95_abs_error": float(dce_peer["etofts"]["vp"]["p95_abs_error"]),
                "max_abs_error": float(dce_peer["etofts"]["vp"]["max_abs_error"]),
            },
        },
        "patlak": {
            "ps": {
                "n": float(dce_peer["patlak"]["ps"]["n"]),
                "mae": float(dce_peer["patlak"]["ps"]["mae"]),
                "p95_abs_error": float(dce_peer["patlak"]["ps"]["p95_abs_error"]),
                "max_abs_error": float(dce_peer["patlak"]["ps"]["max_abs_error"]),
            },
            "vp": {
                "n": float(dce_peer["patlak"]["vp"]["n"]),
                "mae": float(dce_peer["patlak"]["vp"]["mae"]),
                "p95_abs_error": float(dce_peer["patlak"]["vp"]["p95_abs_error"]),
                "max_abs_error": float(dce_peer["patlak"]["vp"]["max_abs_error"]),
            },
        },
    }


def compute_dce_primary_metrics() -> dict[str, dict[str, dict[str, float]]]:
    tofts_errs = {"Ktrans": [], "ve": []}
    for row in _rows(DCE_DATA_DIR / "dce_DRO_data_tofts.csv"):
        fit = model_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))
        tofts_errs["Ktrans"].append(abs((float(fit[0]) * 60.0) - float(row["Ktrans"])))
        tofts_errs["ve"].append(abs(float(fit[1]) - float(row["ve"])))

    ex_errs = {"Ktrans": [], "ve": [], "vp": []}
    for row in _rows(DCE_DATA_DIR / "dce_DRO_data_extended_tofts.csv"):
        fit = model_extended_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))
        ex_errs["Ktrans"].append(abs((float(fit[0]) * 60.0) - float(row["Ktrans"])))
        ex_errs["ve"].append(abs(float(fit[1]) - float(row["ve"])))
        ex_errs["vp"].append(abs(float(fit[2]) - float(row["vp"])))

    patlak_errs = {"ps": [], "vp": []}
    for row in _rows(DCE_DATA_DIR / "patlak_sd_0.02_delay_0.csv"):
        fit = model_patlak_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))
        patlak_errs["ps"].append(abs((float(fit[0]) * 60.0) - float(row["ps"])))
        patlak_errs["vp"].append(abs(float(fit[1]) - float(row["vp"])))

    return {
        "tofts": {param: _summary(values) for param, values in tofts_errs.items()},
        "etofts": {param: _summary(values) for param, values in ex_errs.items()},
        "patlak": {param: _summary(values) for param, values in patlak_errs.items()},
    }


def strict_peer_max_limit(peer_max_abs_error: float) -> float:
    peer_max = float(peer_max_abs_error)
    epsilon = max(1e-12, abs(peer_max) * 1e-6)
    return peer_max + epsilon


def evaluate_dce_primary_gate(
    ours: dict[str, dict[str, dict[str, float]]], peer: dict[str, dict[str, dict[str, float]]]
) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    for method, method_metrics in ours.items():
        peer_method = peer[method]
        for param, ours_metrics in method_metrics.items():
            peer_metrics = peer_method[param]
            limit = strict_peer_max_limit(float(peer_metrics["max_abs_error"]))
            ours_max = float(ours_metrics["max_abs_error"])
            checks.append(
                {
                    "method": method,
                    "param": param,
                    "ours_max_abs_error": ours_max,
                    "peer_max_abs_error": float(peer_metrics["max_abs_error"]),
                    "limit_max_abs_error": limit,
                    "pass": ours_max <= limit,
                }
            )
    return bool(all(bool(c["pass"]) for c in checks)), checks


def as_summary_payload(
    ours: dict[str, dict[str, dict[str, float]]], peer: dict[str, dict[str, dict[str, float]]]
) -> dict[str, Any]:
    passed, checks = evaluate_dce_primary_gate(ours, peer)
    return {
        "suite": "osipi_dce_primary",
        "passed": passed,
        "metrics": {"ours": ours, "peer": peer},
        "checks": checks,
    }
