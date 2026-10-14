"""OSIPI DCE reliability tests — python backend (``model_*_fit``).

Full sweep of every OSIPI DRO case, gated on OSIPI's official acceptance tolerances.
The cpufit/gpufit backends are covered by test_osipi_pycpufit.py / test_osipi_pygpufit.py.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import (  # noqa: E402
    model_2cxm_fit,
    model_extended_tofts_fit,
    model_patlak_fit,
    model_tissue_uptake_fit,
    model_tofts_fit,
)
from osipi_official_tolerances import official_abs_tol


OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
DCE_DATA_DIR = OSIPI_ROOT / "dce_models"
REFERENCE_DIR = OSIPI_ROOT / "reference"


def _rows(csv_file: Path) -> list[dict[str, str]]:
    with csv_file.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split()]


def _assert_within_official(actual: float, expected: float, method: str, param: str, label: str) -> None:
    """Hard gate: OSIPI official acceptance tolerance (abs(a-e) <= a_tol + r_tol*|e|)."""
    tol = official_abs_tol(method, param, expected)
    if not math.isfinite(actual):
        pytest.fail(f"OSIPI {label} {method} {param} produced non-finite value: {actual!r}")
    err = abs(actual - expected)
    assert err <= tol, (
        f"OSIPI {label} {method} {param} abs error {err:.8g} exceeded OSIPI official tolerance "
        f"{tol:.8g}. actual={actual:.8g}, expected={expected:.8g}"
    )
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


@pytest.mark.osipi
def test_osipi_tofts_reliability_against_reference_values() -> None:
    rows = _rows(DCE_DATA_DIR / "dce_DRO_data_tofts.csv")

    for row in rows:
        fit = model_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))

        # OSIPI DCE datasets are in seconds and /min; model_tofts_fit returns /s for this input.
        ktrans_per_min = float(fit[0]) * 60.0
        ve = float(fit[1])

        _assert_within_official(ktrans_per_min, float(row["Ktrans"]), "tofts", "Ktrans", row["label"])
        _assert_within_official(ve, float(row["ve"]), "tofts", "ve", row["label"])


@pytest.mark.osipi
def test_osipi_extended_tofts_reliability_against_reference_values() -> None:
    rows = _rows(DCE_DATA_DIR / "dce_DRO_data_extended_tofts.csv")

    for row in rows:
        fit = model_extended_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))

        # OSIPI DCE datasets are in seconds and /min; model_extended_tofts_fit returns /s for this input.
        ktrans_per_min = float(fit[0]) * 60.0
        ve = float(fit[1])
        vp = float(fit[2])

        _assert_within_official(ktrans_per_min, float(row["Ktrans"]), "etofts", "Ktrans", row["label"])
        _assert_within_official(ve, float(row["ve"]), "etofts", "ve", row["label"])
        _assert_within_official(vp, float(row["vp"]), "etofts", "vp", row["label"])


@pytest.mark.osipi
def test_osipi_patlak_delay_reference_values_are_imported() -> None:
    delay_0_rows = _rows(DCE_DATA_DIR / "patlak_sd_0.02_delay_0.csv")
    delay_5_rows = _rows(DCE_DATA_DIR / "patlak_sd_0.02_delay_5.csv")

    delay_0_lookup = {row["label"]: float(row["arterial_delay"]) for row in delay_0_rows}
    delay_5_lookup = {row["label"].replace("_delayed", ""): float(row["arterial_delay"]) for row in delay_5_rows}

    manifest = json.loads((REFERENCE_DIR / "patlak_delay_reference_values.json").read_text())

    for label, delay_0_value in delay_0_lookup.items():
        assert label in manifest["cases"], f"Missing {label} from Patlak delay manifest"
        assert label in delay_5_lookup, f"Missing delayed counterpart for {label}"

        manifest_delay_0 = float(
            manifest["cases"][label]["patlak_sd_0.02_delay_0.csv"]["arterial_delay_s"]
        )
        manifest_delay_5 = float(
            manifest["cases"][label]["patlak_sd_0.02_delay_5.csv"]["arterial_delay_s"]
        )

        assert manifest_delay_0 == delay_0_value
        assert manifest_delay_5 == delay_5_lookup[label]


@pytest.mark.osipi
def test_osipi_patlak_reliability_delay0_against_reference_values() -> None:
    rows = _rows(DCE_DATA_DIR / "patlak_sd_0.02_delay_0.csv")

    for row in rows:
        fit = model_patlak_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))

        # OSIPI DCE datasets are in seconds and /min; model_patlak_fit returns /s for this input.
        ps_per_min = float(fit[0]) * 60.0
        vp = float(fit[1])

        _assert_within_official(ps_per_min, float(row["ps"]), "patlak", "ps", row["label"])
        _assert_within_official(vp, float(row["vp"]), "patlak", "vp", row["label"])


@pytest.mark.osipi
@pytest.mark.slow
def test_osipi_2cxm_reliability_delay0_against_reference_values() -> None:
    rows = _rows(DCE_DATA_DIR / "2cxm_sd_0.001_delay_0.csv")

    for row in rows:
        fit = model_2cxm_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))

        ktrans_per_sec = float(fit[0])
        ve = float(fit[1])
        vp = float(fit[2])
        fp_per_100ml_per_min = float(fit[3]) * 60.0 * 100.0
        ps_per_min = _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, float(fit[3]))

        _assert_within_official(ve, float(row["ve"]), "2CXM", "ve", row["label"])
        _assert_within_official(vp, float(row["vp"]), "2CXM", "vp", row["label"])
        _assert_within_official(fp_per_100ml_per_min, float(row["fp"]), "2CXM", "fp", row["label"])
        _assert_within_official(ps_per_min, float(row["ps"]), "2CXM", "ps", row["label"])


@pytest.mark.osipi
@pytest.mark.slow
def test_osipi_2cum_reliability_delay0_against_reference_values() -> None:
    rows = _rows(DCE_DATA_DIR / "2cum_sd_0.0025_delay_0.csv")

    for row in rows:
        fit = model_tissue_uptake_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))

        ktrans_per_sec = float(fit[0])
        fp_per_sec = float(fit[1])
        vp = float(fit[2])
        fp_per_100ml_per_min = fp_per_sec * 60.0 * 100.0
        ps_per_min = _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, fp_per_sec)

        _assert_within_official(vp, float(row["vp"]), "2CUM", "vp", row["label"])
        _assert_within_official(fp_per_100ml_per_min, float(row["fp"]), "2CUM", "fp", row["label"])
        _assert_within_official(ps_per_min, float(row["ps"]), "2CUM", "ps", row["label"])
