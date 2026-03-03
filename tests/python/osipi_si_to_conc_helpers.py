"""Shared SI-to-concentration OSIPI reliability helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_signal import signal_to_concentration_spgr  # noqa: E402


OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
SI_DATA_CSV = OSIPI_ROOT / "si_to_conc" / "SI2Conc_data.csv"
REFERENCE_DIR = OSIPI_ROOT / "reference"
PEER_ERROR_SUMMARY = json.loads((REFERENCE_DIR / "osipi_peer_error_summary.json").read_text())


def _rows() -> list[dict[str, str]]:
    with SI_DATA_CSV.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _series(raw: str) -> np.ndarray:
    return np.asarray([float(x) for x in str(raw).split()], dtype=np.float64)


def peer_si_to_conc_metrics() -> dict[str, float]:
    peer = PEER_ERROR_SUMMARY["metrics"]["SI_to_Conc"][""]["conc"]
    return {
        "n": float(peer["n"]),
        "mae": float(peer["mae"]),
        "p95_abs_error": float(peer["p95_abs_error"]),
        "max_abs_error": float(peer["max_abs_error"]),
    }


def compute_si_to_conc_metrics() -> dict[str, float]:
    errors: list[float] = []
    for row in _rows():
        signal = _series(row["s"])
        conc_ref = _series(row["conc"])

        fa_deg = float(row["FA"])
        tr_sec = float(row["TR"])
        t10_sec = float(row["T1base"])
        relaxivity = float(row["r1"])
        n_baseline = int(float(row["numbaselinepts"]))

        conc = signal_to_concentration_spgr(
            signal=signal,
            baseline_indices=np.arange(1, n_baseline, dtype=np.int64),
            tr_sec=tr_sec,
            fa_deg=fa_deg,
            t10_sec=t10_sec,
            relaxivity_r1=relaxivity,
        )

        abs_err = np.abs(conc[1:] - conc_ref[1:])
        errors.extend(float(v) for v in abs_err)

    errors_arr = np.asarray(errors, dtype=np.float64)
    return {
        "n": float(errors_arr.size),
        "mae": float(np.mean(errors_arr)),
        "p95_abs_error": float(np.percentile(errors_arr, 95)),
        "max_abs_error": float(np.max(errors_arr)),
    }


def evaluate_si_to_conc_gate(
    ours: dict[str, float], peer: dict[str, float], *, epsilon: float = 1e-12
) -> tuple[bool, dict[str, bool], dict[str, float]]:
    limits = {
        "mae": float(peer["mae"]) + float(epsilon),
        "p95_abs_error": float(peer["p95_abs_error"]) + float(epsilon),
        "max_abs_error": float(peer["max_abs_error"]) + float(epsilon),
    }
    checks = {
        "mae": float(ours["mae"]) <= limits["mae"],
        "p95_abs_error": float(ours["p95_abs_error"]) <= limits["p95_abs_error"],
        "max_abs_error": float(ours["max_abs_error"]) <= limits["max_abs_error"],
    }
    return bool(all(checks.values())), checks, limits


def as_summary_payload(ours: dict[str, float], peer: dict[str, float], epsilon: float = 1e-12) -> dict[str, Any]:
    passed, checks, limits = evaluate_si_to_conc_gate(ours, peer, epsilon=epsilon)
    return {
        "suite": "osipi_si_to_conc",
        "passed": passed,
        "epsilon": float(epsilon),
        "metrics": {"ours": ours, "peer": peer, "limits": limits},
        "checks": checks,
    }
