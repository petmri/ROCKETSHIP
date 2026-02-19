"""OSIPI-labeled DCE reliability tests using imported OSIPI reference datasets."""

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


OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
DCE_DATA_DIR = OSIPI_ROOT / "dce_models"
REFERENCE_DIR = OSIPI_ROOT / "reference"

PEER_ERROR_SUMMARY = json.loads((REFERENCE_DIR / "osipi_peer_error_summary.json").read_text())
SLOW_SKIP_MSG = "Use --osipi-slow to run long OSIPI reliability fits."


def _rows(csv_file: Path) -> list[dict[str, str]]:
    with csv_file.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split()]


def _peer_max_abs_error(category: str, method: str, param: str) -> float:
    return float(PEER_ERROR_SUMMARY["metrics"][category][method][param]["max_abs_error"])


def _assert_close(actual: float, expected: float, tol: float, label: str, param: str) -> None:
    if not math.isfinite(actual):
        pytest.fail(f"OSIPI {label} {param} produced non-finite value: {actual!r}")
    err = abs(actual - expected)
    assert err <= tol, (
        f"OSIPI {label} {param} abs error {err:.8g} exceeded tolerance {tol:.8g}. "
        f"actual={actual:.8g}, expected={expected:.8g}"
    )


def _require_osipi_slow(run_osipi_slow: bool) -> None:
    if not run_osipi_slow:
        pytest.skip(SLOW_SKIP_MSG)


def _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec: float, fp_per_sec: float) -> float:
    if abs(fp_per_sec - ktrans_per_sec) < 1e-12:
        return float("inf")
    return (ktrans_per_sec * fp_per_sec / (fp_per_sec - ktrans_per_sec)) * 60.0


@pytest.mark.osipi
def test_osipi_tofts_reliability_against_reference_values() -> None:
    rows = _rows(DCE_DATA_DIR / "dce_DRO_data_tofts.csv")

    ktrans_tol = _peer_max_abs_error("DCEmodels", "tofts", "Ktrans") + 1e-6
    ve_tol = _peer_max_abs_error("DCEmodels", "tofts", "ve") + 1e-6

    for row in rows:
        fit = model_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))

        # OSIPI DCE datasets are in seconds and /min; model_tofts_fit returns /s for this input.
        ktrans_per_min = float(fit[0]) * 60.0
        ve = float(fit[1])

        _assert_close(ktrans_per_min, float(row["Ktrans"]), ktrans_tol, row["label"], "Ktrans")
        _assert_close(ve, float(row["ve"]), ve_tol, row["label"], "ve")


@pytest.mark.osipi
def test_osipi_extended_tofts_reliability_against_reference_values() -> None:
    rows = _rows(DCE_DATA_DIR / "dce_DRO_data_extended_tofts.csv")

    ktrans_tol = _peer_max_abs_error("DCEmodels", "etofts", "Ktrans") + 1e-6
    ve_tol = _peer_max_abs_error("DCEmodels", "etofts", "ve") + 1e-6
    vp_tol = _peer_max_abs_error("DCEmodels", "etofts", "vp") + 1e-6

    for row in rows:
        fit = model_extended_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))

        # OSIPI DCE datasets are in seconds and /min; model_extended_tofts_fit returns /s for this input.
        ktrans_per_min = float(fit[0]) * 60.0
        ve = float(fit[1])
        vp = float(fit[2])

        _assert_close(ktrans_per_min, float(row["Ktrans"]), ktrans_tol, row["label"], "Ktrans")
        _assert_close(ve, float(row["ve"]), ve_tol, row["label"], "ve")
        _assert_close(vp, float(row["vp"]), vp_tol, row["label"], "vp")


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

    ps_tol = _peer_max_abs_error("DCEmodels", "patlak", "ps") + 1e-6
    vp_tol = _peer_max_abs_error("DCEmodels", "patlak", "vp") + 1e-6

    for row in rows:
        fit = model_patlak_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))

        # OSIPI DCE datasets are in seconds and /min; model_patlak_fit returns /s for this input.
        ps_per_min = float(fit[0]) * 60.0
        vp = float(fit[1])

        _assert_close(ps_per_min, float(row["ps"]), ps_tol, row["label"], "ps")
        _assert_close(vp, float(row["vp"]), vp_tol, row["label"], "vp")


@pytest.mark.osipi
@pytest.mark.osipi_slow
@pytest.mark.slow
def test_osipi_2cxm_reliability_delay0_against_reference_values(run_osipi_slow: bool) -> None:
    _require_osipi_slow(run_osipi_slow)
    rows = _rows(DCE_DATA_DIR / "2cxm_sd_0.001_delay_0.csv")

    ve_tol = _peer_max_abs_error("DCEmodels", "2CXM", "ve") + 1e-6
    vp_tol = _peer_max_abs_error("DCEmodels", "2CXM", "vp") + 1e-6
    fp_tol = _peer_max_abs_error("DCEmodels", "2CXM", "fp") + 1e-6
    ps_tol = _peer_max_abs_error("DCEmodels", "2CXM", "ps") + 1e-6

    for row in rows:
        fit = model_2cxm_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))

        ktrans_per_sec = float(fit[0])
        ve = float(fit[1])
        vp = float(fit[2])
        fp_per_100ml_per_min = float(fit[3]) * 60.0 * 100.0
        ps_per_min = _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, float(fit[3]))

        _assert_close(ve, float(row["ve"]), ve_tol, row["label"], "ve")
        _assert_close(vp, float(row["vp"]), vp_tol, row["label"], "vp")
        _assert_close(fp_per_100ml_per_min, float(row["fp"]), fp_tol, row["label"], "fp")
        _assert_close(ps_per_min, float(row["ps"]), ps_tol, row["label"], "ps")


@pytest.mark.osipi
@pytest.mark.osipi_slow
@pytest.mark.slow
def test_osipi_2cum_reliability_delay0_against_reference_values(run_osipi_slow: bool) -> None:
    _require_osipi_slow(run_osipi_slow)
    rows = _rows(DCE_DATA_DIR / "2cum_sd_0.0025_delay_0.csv")

    vp_tol = _peer_max_abs_error("DCEmodels", "2CUM", "vp") + 1e-6
    fp_tol = _peer_max_abs_error("DCEmodels", "2CUM", "fp") + 1e-6
    ps_tol = _peer_max_abs_error("DCEmodels", "2CUM", "ps") + 1e-6

    for row in rows:
        fit = model_tissue_uptake_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))

        ktrans_per_sec = float(fit[0])
        fp_per_sec = float(fit[1])
        vp = float(fit[2])
        fp_per_100ml_per_min = fp_per_sec * 60.0 * 100.0
        ps_per_min = _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, fp_per_sec)

        _assert_close(vp, float(row["vp"]), vp_tol, row["label"], "vp")
        _assert_close(fp_per_100ml_per_min, float(row["fp"]), fp_tol, row["label"], "fp")
        _assert_close(ps_per_min, float(row["ps"]), ps_tol, row["label"], "ps")
