"""Unit tests for DCE signal/concentration conversion helpers."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_signal import enhancement_to_concentration_spgr, signal_to_concentration_spgr, signal_to_enhancement  # noqa: E402


@pytest.mark.unit
def test_signal_to_enhancement_uses_baseline_mean() -> None:
    signal = np.array([10.0, 10.0, 12.0, 8.0], dtype=np.float64)
    enh = signal_to_enhancement(signal, baseline_indices=[0, 1])
    np.testing.assert_allclose(enh, np.array([0.0, 0.0, 20.0, -20.0], dtype=np.float64), rtol=0.0, atol=1e-12)


@pytest.mark.unit
def test_enhancement_to_concentration_spgr_zero_enhancement_is_zero_concentration() -> None:
    enhancement = np.zeros(5, dtype=np.float64)
    conc = enhancement_to_concentration_spgr(
        enhancement_percent=enhancement,
        tr_sec=0.003,
        fa_deg=10.0,
        t10_sec=1.2,
        relaxivity_r1=4.5,
    )
    np.testing.assert_allclose(conc, np.zeros_like(conc), rtol=0.0, atol=1e-12)


@pytest.mark.unit
def test_signal_to_concentration_spgr_matches_osipi_reference_curve() -> None:
    data_csv = REPO_ROOT / "tests" / "data" / "osipi" / "si_to_conc" / "SI2Conc_data.csv"
    with data_csv.open(newline="", encoding="utf-8-sig") as handle:
        first_row = next(csv.DictReader(handle))

    signal = np.array([float(x) for x in str(first_row["s"]).split()], dtype=np.float64)
    conc_ref = np.array([float(x) for x in str(first_row["conc"]).split()], dtype=np.float64)

    fa_deg = float(first_row["FA"])
    tr_sec = float(first_row["TR"])
    t10_sec = float(first_row["T1base"])
    relaxivity = float(first_row["r1"])
    n_baseline = int(float(first_row["numbaselinepts"]))

    conc = signal_to_concentration_spgr(
        signal=signal,
        baseline_indices=np.arange(1, n_baseline, dtype=np.int64),
        tr_sec=tr_sec,
        fa_deg=fa_deg,
        t10_sec=t10_sec,
        relaxivity_r1=relaxivity,
    )

    np.testing.assert_allclose(conc[1:], conc_ref[1:], rtol=0.0, atol=2e-11)
