"""Generate OSIPI accuracy summary markdown and comparison figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _apply_model_specific_prefs,
    _fit_stage_d_model_accelerated,
    _stage_d_fit_prefs,
    probe_acceleration_backend,
)
from rocketship import (  # noqa: E402
    model_2cxm_fit,
    model_extended_tofts_fit,
    model_patlak_fit,
    model_tissue_uptake_fit,
    model_tofts_fit,
    t1_fa_linear_fit,
)


OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
DCE_DATA_DIR = OSIPI_ROOT / "dce_models"
T1_DATA_DIR = OSIPI_ROOT / "t1_mapping"
REFERENCE_DIR = OSIPI_ROOT / "reference"
FIG_DIR = REFERENCE_DIR / "figures"
SUMMARY_MD = REPO_ROOT / "osipi_summary.md"
PEER_SUMMARY_JSON = REFERENCE_DIR / "osipi_peer_error_summary.json"
SUMMARY_CONFIG = DcePipelineConfig(
    subject_source_path=REPO_ROOT,
    subject_tp_path=REPO_ROOT,
    output_dir=REPO_ROOT,
    backend="cpu",
)
BASE_STAGE_D_PREFS = _stage_d_fit_prefs(SUMMARY_CONFIG)

PREFS_2CXM = {
    "lower_limit_ktrans": 1e-7,
    "upper_limit_ktrans": 2.0,
    "initial_value_ktrans": 2e-4,
    "lower_limit_ve": 0.05,
    "upper_limit_ve": 1.0,
    "initial_value_ve": 0.15,
    "lower_limit_vp": 1e-3,
    "upper_limit_vp": 1.0,
    "initial_value_vp": 0.02,
    "lower_limit_fp": 1e-3,
    "upper_limit_fp": 20.0,
    "initial_value_fp": 0.35,
    "max_nfev": 140,
    "tol_fun": 1e-12,
    "tol_x": 1e-6,
    "robust": "off",
}

PREFS_2CUM = {
    "lower_limit_ktrans": 1e-7,
    "upper_limit_ktrans": 2.0,
    "initial_value_ktrans": 2e-4,
    "lower_limit_fp": 1e-3,
    "upper_limit_fp": 20.0,
    "initial_value_fp": 0.35,
    "lower_limit_tp": 0.0,
    "upper_limit_tp": 1.5,
    "initial_value_tp": 0.12,
    "max_nfev": 120,
    "tol_fun": 1e-12,
    "tol_x": 1e-6,
    "robust": "off",
}


def _cpufit_backend_available() -> str | None:
    probe_acceleration_backend.cache_clear()
    probe = probe_acceleration_backend()
    if bool(probe.get("pycpufit_imported", False)):
        return "cpufit_cpu"
    return None


def _stage_d_prefs_for_model(model_name: str) -> dict[str, Any]:
    prefs = dict(BASE_STAGE_D_PREFS)
    if model_name in {"2cxm", "tissue_uptake"}:
        return _apply_model_specific_prefs(prefs, model_name)
    return prefs


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
    if out is None or out.shape[0] == 0:
        raise RuntimeError(f"Accelerated fit failed for model={model_name}")
    return np.asarray(out[0], dtype=np.float64)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split()]


def _pct(values: list[float], p: float) -> float:
    vals = sorted(values)
    idx = (len(vals) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(vals[lo])
    frac = idx - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _summary(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    n_fail = len(values) - len(finite)
    cleaned = finite if finite else [1.0e9]
    return {
        "n": float(len(values)),
        "n_fail": float(n_fail),
        "mae": float(statistics.mean(cleaned)),
        "p95": float(_pct(cleaned, 0.95)),
        "max": float(max(cleaned)),
    }


def _ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec: float, fp_per_sec: float) -> float:
    if abs(fp_per_sec - ktrans_per_sec) < 1e-12:
        return 6.0e9
    return (ktrans_per_sec * fp_per_sec / (fp_per_sec - ktrans_per_sec)) * 60.0


def _add_row(
    table_rows: list[dict[str, Any]],
    *,
    model: str,
    dataset_slice: str,
    param: str,
    errors: list[float],
    peer_summary: dict[str, Any],
    peer_category: str,
    peer_method: str,
    peer_param: str,
    notes: str = "",
) -> None:
    ours = _summary(errors)
    peer = peer_summary["metrics"][peer_category][peer_method][peer_param]

    peer_mae = float(peer["mae"])
    peer_p95 = float(peer["p95_abs_error"])
    peer_max = float(peer["max_abs_error"])

    mae_ratio = ours["mae"] / peer_mae if peer_mae > 0 else float("inf")
    p95_ratio = ours["p95"] / peer_p95 if peer_p95 > 0 else float("inf")
    max_ratio = ours["max"] / peer_max if peer_max > 0 else float("inf")
    n_fail = int(ours["n_fail"])

    notes_out = notes
    if n_fail > 0:
        fail_note = f"nonfinite fit failures={n_fail}"
        notes_out = f"{notes}; {fail_note}" if notes else fail_note

    table_rows.append(
        {
            "model": model,
            "slice": dataset_slice,
            "param": param,
            "n": int(ours["n"]),
            "n_fail": n_fail,
            "our_mae": ours["mae"],
            "our_p95": ours["p95"],
            "our_max": ours["max"],
            "peer_mae": peer_mae,
            "peer_p95": peer_p95,
            "peer_max": peer_max,
            "mae_ratio": mae_ratio,
            "p95_ratio": p95_ratio,
            "max_ratio": max_ratio,
            "within_peer_max": n_fail == 0 and ours["max"] <= (peer_max + 1e-12),
            "notes": notes_out,
        }
    )


def _compute_table_rows() -> list[dict[str, Any]]:
    peer_summary = json.loads(PEER_SUMMARY_JSON.read_text())
    table_rows: list[dict[str, Any]] = []
    cpufit_backend = _cpufit_backend_available()

    # Tofts
    k_errors: list[float] = []
    ve_errors: list[float] = []
    for row in _rows(DCE_DATA_DIR / "dce_DRO_data_tofts.csv"):
        fit = model_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))
        k_errors.append(abs(float(fit[0]) * 60.0 - float(row["Ktrans"])))
        ve_errors.append(abs(float(fit[1]) - float(row["ve"])))

    _add_row(
        table_rows,
        model="tofts",
        dataset_slice="OSIPI Tofts DRO",
        param="Ktrans",
        errors=k_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="tofts",
        peer_param="Ktrans",
    )
    _add_row(
        table_rows,
        model="tofts",
        dataset_slice="OSIPI Tofts DRO",
        param="ve",
        errors=ve_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="tofts",
        peer_param="ve",
    )

    # Extended Tofts
    k_errors = []
    ve_errors = []
    vp_errors: list[float] = []
    for row in _rows(DCE_DATA_DIR / "dce_DRO_data_extended_tofts.csv"):
        fit = model_extended_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))
        k_errors.append(abs(float(fit[0]) * 60.0 - float(row["Ktrans"])))
        ve_errors.append(abs(float(fit[1]) - float(row["ve"])))
        vp_errors.append(abs(float(fit[2]) - float(row["vp"])))

    _add_row(
        table_rows,
        model="etofts",
        dataset_slice="OSIPI Extended Tofts DRO",
        param="Ktrans",
        errors=k_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="etofts",
        peer_param="Ktrans",
    )
    _add_row(
        table_rows,
        model="etofts",
        dataset_slice="OSIPI Extended Tofts DRO",
        param="ve",
        errors=ve_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="etofts",
        peer_param="ve",
    )
    _add_row(
        table_rows,
        model="etofts",
        dataset_slice="OSIPI Extended Tofts DRO",
        param="vp",
        errors=vp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="etofts",
        peer_param="vp",
    )

    # Patlak delay=0
    ps_errors: list[float] = []
    vp_errors = []
    for row in _rows(DCE_DATA_DIR / "patlak_sd_0.02_delay_0.csv"):
        fit = model_patlak_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))
        ps_errors.append(abs(float(fit[0]) * 60.0 - float(row["ps"])))
        vp_errors.append(abs(float(fit[1]) - float(row["vp"])))

    _add_row(
        table_rows,
        model="patlak",
        dataset_slice="OSIPI Patlak delay=0",
        param="ps",
        errors=ps_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="patlak",
        peer_param="ps",
    )
    _add_row(
        table_rows,
        model="patlak",
        dataset_slice="OSIPI Patlak delay=0",
        param="vp",
        errors=vp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="patlak",
        peer_param="vp",
    )

    # Patlak delay=5 (without delay fitting enabled yet)
    ps_errors = []
    vp_errors = []
    for row in _rows(DCE_DATA_DIR / "patlak_sd_0.02_delay_5.csv"):
        fit = model_patlak_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))
        ps_errors.append(abs(float(fit[0]) * 60.0 - float(row["ps"])))
        vp_errors.append(abs(float(fit[1]) - float(row["vp"])))

    _add_row(
        table_rows,
        model="patlak",
        dataset_slice="OSIPI Patlak delay=5",
        param="ps",
        errors=ps_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="patlak",
        peer_param="ps",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )
    _add_row(
        table_rows,
        model="patlak",
        dataset_slice="OSIPI Patlak delay=5",
        param="vp",
        errors=vp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="patlak",
        peer_param="vp",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )

    # 2CXM delay=0
    ve_errors: list[float] = []
    vp_errors = []
    fp_errors: list[float] = []
    ps_errors = []
    for row in _rows(DCE_DATA_DIR / "2cxm_sd_0.001_delay_0.csv"):
        if cpufit_backend is not None:
            fit = _accelerated_fit_row(
                model_name="2cxm",
                row=row,
                signal_col="C_t",
                aif_col="cp_aif",
                time_col="t",
                acceleration_backend=cpufit_backend,
            )
        else:
            fit = np.asarray(
                model_2cxm_fit(
                    _series(row["C_t"]),
                    _series(row["cp_aif"]),
                    _series(row["t"]),
                    dict(PREFS_2CXM),
                ),
                dtype=np.float64,
            )

        ktrans_per_sec = float(fit[0])
        ve_errors.append(abs(float(fit[1]) - float(row["ve"])))
        vp_errors.append(abs(float(fit[2]) - float(row["vp"])))
        fp_errors.append(abs(float(fit[3]) * 60.0 * 100.0 - float(row["fp"])))
        ps_errors.append(abs(_ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, float(fit[3])) - float(row["ps"])))

    _add_row(
        table_rows,
        model="2cxm",
        dataset_slice="OSIPI 2CXM delay=0",
        param="ve",
        errors=ve_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CXM",
        peer_param="ve",
    )
    _add_row(
        table_rows,
        model="2cxm",
        dataset_slice="OSIPI 2CXM delay=0",
        param="vp",
        errors=vp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CXM",
        peer_param="vp",
    )
    _add_row(
        table_rows,
        model="2cxm",
        dataset_slice="OSIPI 2CXM delay=0",
        param="fp",
        errors=fp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CXM",
        peer_param="fp",
    )
    _add_row(
        table_rows,
        model="2cxm",
        dataset_slice="OSIPI 2CXM delay=0",
        param="ps",
        errors=ps_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CXM",
        peer_param="ps",
    )

    # 2CXM delay=5 (without delay fitting enabled yet)
    ve_errors = []
    vp_errors = []
    fp_errors = []
    ps_errors = []
    for row in _rows(DCE_DATA_DIR / "2cxm_sd_0.001_delay_5.csv"):
        if cpufit_backend is not None:
            fit = _accelerated_fit_row(
                model_name="2cxm",
                row=row,
                signal_col="C_t",
                aif_col="cp_aif",
                time_col="t",
                acceleration_backend=cpufit_backend,
            )
        else:
            fit = np.asarray(
                model_2cxm_fit(
                    _series(row["C_t"]),
                    _series(row["cp_aif"]),
                    _series(row["t"]),
                    dict(PREFS_2CXM),
                ),
                dtype=np.float64,
            )

        ktrans_per_sec = float(fit[0])
        ve_errors.append(abs(float(fit[1]) - float(row["ve"])))
        vp_errors.append(abs(float(fit[2]) - float(row["vp"])))
        fp_errors.append(abs(float(fit[3]) * 60.0 * 100.0 - float(row["fp"])))
        ps_errors.append(abs(_ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, float(fit[3])) - float(row["ps"])))

    _add_row(
        table_rows,
        model="2cxm",
        dataset_slice="OSIPI 2CXM delay=5",
        param="ve",
        errors=ve_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CXM",
        peer_param="ve",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )
    _add_row(
        table_rows,
        model="2cxm",
        dataset_slice="OSIPI 2CXM delay=5",
        param="vp",
        errors=vp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CXM",
        peer_param="vp",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )
    _add_row(
        table_rows,
        model="2cxm",
        dataset_slice="OSIPI 2CXM delay=5",
        param="fp",
        errors=fp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CXM",
        peer_param="fp",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )
    _add_row(
        table_rows,
        model="2cxm",
        dataset_slice="OSIPI 2CXM delay=5",
        param="ps",
        errors=ps_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CXM",
        peer_param="ps",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )

    # 2CUM delay=0
    vp_errors = []
    fp_errors = []
    ps_errors = []
    for row in _rows(DCE_DATA_DIR / "2cum_sd_0.0025_delay_0.csv"):
        if cpufit_backend is not None:
            fit = _accelerated_fit_row(
                model_name="tissue_uptake",
                row=row,
                signal_col="C_t",
                aif_col="cp_aif",
                time_col="t",
                acceleration_backend=cpufit_backend,
            )
        else:
            fit = np.asarray(
                model_tissue_uptake_fit(
                    _series(row["C_t"]),
                    _series(row["cp_aif"]),
                    _series(row["t"]),
                    dict(PREFS_2CUM),
                ),
                dtype=np.float64,
            )

        ktrans_per_sec = float(fit[0])
        fp_per_sec = float(fit[1])
        vp_errors.append(abs(float(fit[2]) - float(row["vp"])))
        fp_errors.append(abs(fp_per_sec * 60.0 * 100.0 - float(row["fp"])))
        ps_errors.append(abs(_ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, fp_per_sec) - float(row["ps"])))

    _add_row(
        table_rows,
        model="2cum",
        dataset_slice="OSIPI 2CUM delay=0",
        param="vp",
        errors=vp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CUM",
        peer_param="vp",
    )
    _add_row(
        table_rows,
        model="2cum",
        dataset_slice="OSIPI 2CUM delay=0",
        param="fp",
        errors=fp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CUM",
        peer_param="fp",
    )
    _add_row(
        table_rows,
        model="2cum",
        dataset_slice="OSIPI 2CUM delay=0",
        param="ps",
        errors=ps_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CUM",
        peer_param="ps",
    )

    # 2CUM delay=5 (without delay fitting enabled yet)
    vp_errors = []
    fp_errors = []
    ps_errors = []
    for row in _rows(DCE_DATA_DIR / "2cum_sd_0.0025_delay_5.csv"):
        if cpufit_backend is not None:
            fit = _accelerated_fit_row(
                model_name="tissue_uptake",
                row=row,
                signal_col="C_t",
                aif_col="cp_aif",
                time_col="t",
                acceleration_backend=cpufit_backend,
            )
        else:
            fit = np.asarray(
                model_tissue_uptake_fit(
                    _series(row["C_t"]),
                    _series(row["cp_aif"]),
                    _series(row["t"]),
                    dict(PREFS_2CUM),
                ),
                dtype=np.float64,
            )

        ktrans_per_sec = float(fit[0])
        fp_per_sec = float(fit[1])
        vp_errors.append(abs(float(fit[2]) - float(row["vp"])))
        fp_errors.append(abs(fp_per_sec * 60.0 * 100.0 - float(row["fp"])))
        ps_errors.append(abs(_ps_per_min_from_ktrans_fp_per_sec(ktrans_per_sec, fp_per_sec) - float(row["ps"])))

    _add_row(
        table_rows,
        model="2cum",
        dataset_slice="OSIPI 2CUM delay=5",
        param="vp",
        errors=vp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CUM",
        peer_param="vp",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )
    _add_row(
        table_rows,
        model="2cum",
        dataset_slice="OSIPI 2CUM delay=5",
        param="fp",
        errors=fp_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CUM",
        peer_param="fp",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )
    _add_row(
        table_rows,
        model="2cum",
        dataset_slice="OSIPI 2CUM delay=5",
        param="ps",
        errors=ps_errors,
        peer_summary=peer_summary,
        peer_category="DCEmodels",
        peer_method="2CUM",
        peer_param="ps",
        notes="delay fitting not implemented yet; run shown for gap visibility",
    )

    # T1 linear across imported datasets
    r1_errors: list[float] = []
    for dataset_name, csv_name in [
        ("brain", "t1_brain_data.csv"),
        ("quiba", "t1_quiba_data.csv"),
        ("prostate", "t1_prostate_data.csv"),
    ]:
        for row in _rows(T1_DATA_DIR / csv_name):
            fa = _series(row["FA"])
            signal = _series(row["s"])

            if dataset_name == "prostate":
                tr_ms = float(str(row["TR"]).split()[0])
                r1_ref = 1000.0 / float(row[" T1 nonlinear"])
            elif dataset_name == "quiba":
                tr_ms = float(str(row["TR"]).split()[0]) * 1000.0
                r1_ref = float(row["R1"]) * 1000.0
            else:
                tr_ms = float(str(row["TR"]).split()[0]) * 1000.0
                r1_ref = float(row["R1"])

            t1_ms = float(t1_fa_linear_fit(fa, signal, tr_ms)[0])
            r1_measured = 1000.0 / t1_ms
            r1_errors.append(abs(r1_measured - r1_ref))

    _add_row(
        table_rows,
        model="t1_linear",
        dataset_slice="OSIPI T1 (brain+quiba+prostate)",
        param="r1",
        errors=r1_errors,
        peer_summary=peer_summary,
        peer_category="T1mapping",
        peer_method="linear",
        peer_param="r1",
    )

    return table_rows


def _row_label(row: dict[str, Any], *, compact: bool = False) -> str:
    if row["model"] == "patlak":
        delay = "d0" if "delay=0" in str(row["slice"]) else "d5"
        return f"{delay} {row['param']}" if compact else f"{delay}\n{row['param']}"
    if row["model"] == "t1_linear":
        return "t1 r1" if compact else "t1\nr1"
    return f"{row['model']} {row['param']}" if compact else f"{row['model']}\n{row['param']}"


def _plot_group(rows: list[dict[str, Any]], *, title: str, outfile: Path) -> None:
    labels = [_row_label(row) for row in rows]

    ours_mae = np.array([float(row["our_mae"]) for row in rows], dtype=float)
    ours_p95 = np.array([float(row["our_p95"]) for row in rows], dtype=float)
    ours_max = np.array([float(row["our_max"]) for row in rows], dtype=float)

    peer_mae = np.array([float(row["peer_mae"]) for row in rows], dtype=float)
    peer_p95 = np.array([float(row["peer_p95"]) for row in rows], dtype=float)
    peer_max = np.array([float(row["peer_max"]) for row in rows], dtype=float)

    ours_upper = np.maximum(0.0, ours_p95 - ours_mae)
    peer_upper = np.maximum(0.0, peer_p95 - peer_mae)

    # Show error bar as [0, p95] around the MAE bar using asymmetric yerr.
    ours_yerr = np.vstack([ours_mae, ours_upper])
    peer_yerr = np.vstack([peer_mae, peer_upper])

    x = np.arange(len(rows), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(rows)), 5.0))

    ax.bar(
        x - width / 2.0,
        ours_mae,
        width,
        yerr=ours_yerr,
        capsize=4,
        label="Ours MAE (error bar to P95)",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2.0,
        peer_mae,
        width,
        yerr=peer_yerr,
        capsize=4,
        label="Peer MAE (error bar to P95)",
        color="#ff7f0e",
        alpha=0.85,
    )

    ax.scatter(x - width / 2.0, ours_max, marker="x", color="#0b3a62", s=60, label="Ours Max")
    ax.scatter(x + width / 2.0, peer_max, marker="x", color="#8a3f00", s=60, label="Peer Max")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute error")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def _write_markdown(table_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# OSIPI Accuracy Summary")
    lines.append("")
    lines.append("Computed from ROCKETSHIP fits against imported OSIPI datasets and compared to OSIPI posted peer-result aggregates.")
    lines.append("")
    lines.append("- ROCKETSHIP datasets: `tests/data/osipi/...`")
    lines.append("- Peer reference summary: `tests/data/osipi/reference/osipi_peer_error_summary.json`")
    lines.append("- Peer source: https://github.com/OSIPI/DCE-DSC-MRI_TestResults (commit `23d3714797045d8103d5b5fa4f4c016840094dc0`)")
    lines.append("- Figures:")
    lines.append("  - `tests/data/osipi/reference/figures/osipi_accuracy_dros.png`")
    lines.append("  - `tests/data/osipi/reference/figures/osipi_accuracy_patlak_delay.png`")
    lines.append("  - `tests/data/osipi/reference/figures/osipi_accuracy_t1.png`")
    lines.append("")
    lines.append(
        "| Model | Dataset slice | Param | N | Our MAE | Our P95 | Our Max | Peer MAE | Peer P95 | Peer Max | MAE Ratio (Our/Peer) | Max Ratio (Our/Peer) | Within Peer Max | Notes |"
    )
    lines.append(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- |"
    )

    for row in table_rows:
        lines.append(
            "| {model} | {slice} | {param} | {n} | {our_mae:.6g} | {our_p95:.6g} | {our_max:.6g} | "
            "{peer_mae:.6g} | {peer_p95:.6g} | {peer_max:.6g} | {mae_ratio:.3f} | {max_ratio:.3f} | {within} | {notes} |".format(
                model=row["model"],
                slice=row["slice"],
                param=row["param"],
                n=row["n"],
                our_mae=row["our_mae"],
                our_p95=row["our_p95"],
                our_max=row["our_max"],
                peer_mae=row["peer_mae"],
                peer_p95=row["peer_p95"],
                peer_max=row["peer_max"],
                mae_ratio=row["mae_ratio"],
                max_ratio=row["max_ratio"],
                within="yes" if row["within_peer_max"] else "no",
                notes=row["notes"],
            )
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    table_rows = _compute_table_rows()

    dro_rows = [row for row in table_rows if row["model"] not in {"patlak", "t1_linear"}]
    patlak_rows = [row for row in table_rows if row["model"] == "patlak"]
    t1_rows = [row for row in table_rows if row["model"] == "t1_linear"]

    _plot_group(
        dro_rows,
        title="OSIPI DROs: ROCKETSHIP vs Peer MAE (P95 bars, Max as X)",
        outfile=FIG_DIR / "osipi_accuracy_dros.png",
    )
    _plot_group(
        patlak_rows,
        title="OSIPI Patlak Delay Cases: ROCKETSHIP vs Peer MAE (P95 bars, Max as X)",
        outfile=FIG_DIR / "osipi_accuracy_patlak_delay.png",
    )
    _plot_group(
        t1_rows,
        title="OSIPI T1 Linear: ROCKETSHIP vs Peer MAE (P95 bars, Max as X)",
        outfile=FIG_DIR / "osipi_accuracy_t1.png",
    )

    _write_markdown(table_rows)

    print(f"wrote {SUMMARY_MD}")
    print(f"wrote {FIG_DIR / 'osipi_accuracy_dros.png'}")
    print(f"wrote {FIG_DIR / 'osipi_accuracy_patlak_delay.png'}")
    print(f"wrote {FIG_DIR / 'osipi_accuracy_t1.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
