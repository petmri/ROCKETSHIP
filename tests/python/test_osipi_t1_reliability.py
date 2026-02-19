"""OSIPI-labeled T1 reliability tests using imported OSIPI reference datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import t1_fa_linear_fit  # noqa: E402


OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
T1_DATA_DIR = OSIPI_ROOT / "t1_mapping"
REFERENCE_DIR = OSIPI_ROOT / "reference"

PEER_ERROR_SUMMARY = json.loads((REFERENCE_DIR / "osipi_peer_error_summary.json").read_text())


def _rows(csv_file: Path) -> list[dict[str, str]]:
    with csv_file.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split()]


def _iter_t1_cases() -> list[tuple[str, dict[str, str]]]:
    rows: list[tuple[str, dict[str, str]]] = []
    rows.extend(("brain", row) for row in _rows(T1_DATA_DIR / "t1_brain_data.csv"))
    rows.extend(("quiba", row) for row in _rows(T1_DATA_DIR / "t1_quiba_data.csv"))
    rows.extend(("prostate", row) for row in _rows(T1_DATA_DIR / "t1_prostate_data.csv"))
    return rows


def _reference_r1(case_type: str, row: dict[str, str]) -> tuple[float, float]:
    # Model API returns T1 in ms. Return (TR_ms, R1_ref_per_s).
    if case_type == "prostate":
        tr_ms = float(str(row["TR"]).split()[0])
        r1_ref = 1000.0 / float(row[" T1 nonlinear"])
        return tr_ms, r1_ref

    if case_type == "quiba":
        tr_ms = float(str(row["TR"]).split()[0]) * 1000.0
        r1_ref = float(row["R1"]) * 1000.0
        return tr_ms, r1_ref

    tr_ms = float(str(row["TR"]).split()[0]) * 1000.0
    r1_ref = float(row["R1"])
    return tr_ms, r1_ref


@pytest.mark.osipi
def test_osipi_t1_linear_error_distribution_matches_peer_results() -> None:
    cases = _iter_t1_cases()

    peer_linear = PEER_ERROR_SUMMARY["metrics"]["T1mapping"]["linear"]["r1"]
    peer_p95 = float(peer_linear["p95_abs_error"])
    peer_max = float(peer_linear["max_abs_error"])

    near_peer_band = peer_p95 * 1.2

    errors: list[float] = []
    for case_type, row in cases:
        tr_ms, r1_ref = _reference_r1(case_type, row)
        t1_ms = float(t1_fa_linear_fit(_series(row["FA"]), _series(row["s"]), tr_ms)[0])
        r1_measured = 1000.0 / t1_ms
        errors.append(abs(r1_measured - r1_ref))

    within_band = sum(1 for err in errors if err <= near_peer_band)
    within_fraction = within_band / len(errors)

    assert max(errors) <= (peer_max + 1e-6), (
        f"Max OSIPI T1 linear error {max(errors):.8g} exceeded peer max {peer_max:.8g}"
    )
    assert within_fraction >= 0.95, (
        "OSIPI T1 linear error distribution fell below peer-aligned threshold: "
        f"{within_fraction:.3f} within {near_peer_band:.8g}"
    )
