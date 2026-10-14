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

from osipi_official_tolerances import official_abs_tol


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


def _param_metrics(method: str, param: str, cases: list[tuple[float, float]]) -> dict[str, float]:
    """Aggregate stats + OSIPI official-tolerance gate for one model/param.

    ``cases`` is a list of (reference, fitted) pairs. The official gate mirrors OSIPI's
    per-case ``assert_allclose(atol=a_tol, rtol=r_tol)``.
    """
    errs = [abs(fit - ref) for ref, fit in cases]
    worst_frac = 0.0
    passed = True
    for ref, fit in cases:
        tol = official_abs_tol(method, param, ref)
        ratio = abs(fit - ref) / tol if tol > 0 else math.inf
        worst_frac = max(worst_frac, ratio)
        if not (math.isfinite(fit) and abs(fit - ref) <= tol):
            passed = False
    out = _summary(errs)
    out["official_worst_frac"] = float(worst_frac)
    out["official_pass"] = float(1.0 if passed else 0.0)
    return out


def compute_dce_primary_metrics() -> dict[str, dict[str, dict[str, float]]]:
    tofts: dict[str, list[tuple[float, float]]] = {"Ktrans": [], "ve": []}
    for row in _rows(DCE_DATA_DIR / "dce_DRO_data_tofts.csv"):
        fit = model_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))
        tofts["Ktrans"].append((float(row["Ktrans"]), float(fit[0]) * 60.0))
        tofts["ve"].append((float(row["ve"]), float(fit[1])))

    ex: dict[str, list[tuple[float, float]]] = {"Ktrans": [], "ve": [], "vp": []}
    for row in _rows(DCE_DATA_DIR / "dce_DRO_data_extended_tofts.csv"):
        fit = model_extended_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))
        ex["Ktrans"].append((float(row["Ktrans"]), float(fit[0]) * 60.0))
        ex["ve"].append((float(row["ve"]), float(fit[1])))
        ex["vp"].append((float(row["vp"]), float(fit[2])))

    patlak: dict[str, list[tuple[float, float]]] = {"ps": [], "vp": []}
    for row in _rows(DCE_DATA_DIR / "patlak_sd_0.02_delay_0.csv"):
        fit = model_patlak_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))
        patlak["ps"].append((float(row["ps"]), float(fit[0]) * 60.0))
        patlak["vp"].append((float(row["vp"]), float(fit[1])))

    return {
        "tofts": {p: _param_metrics("tofts", p, cases) for p, cases in tofts.items()},
        "etofts": {p: _param_metrics("etofts", p, cases) for p, cases in ex.items()},
        "patlak": {p: _param_metrics("patlak", p, cases) for p, cases in patlak.items()},
    }


def evaluate_dce_primary_gate(
    ours: dict[str, dict[str, dict[str, float]]], peer: dict[str, dict[str, dict[str, float]]]
) -> tuple[bool, list[dict[str, Any]]]:
    """Hard gate on OSIPI official acceptance tolerances; peer max reported for context."""
    checks: list[dict[str, Any]] = []
    for method, method_metrics in ours.items():
        peer_method = peer[method]
        for param, ours_metrics in method_metrics.items():
            passed = bool(ours_metrics.get("official_pass", 0.0))
            checks.append(
                {
                    "method": method,
                    "param": param,
                    "ours_max_abs_error": float(ours_metrics["max_abs_error"]),
                    "official_worst_frac": float(ours_metrics.get("official_worst_frac", math.nan)),
                    "peer_max_abs_error": float(peer_method[param]["max_abs_error"]),
                    "pass": passed,
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
