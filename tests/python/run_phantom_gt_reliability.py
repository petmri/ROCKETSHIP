#!/usr/bin/env python3
"""Run synthetic phantom GT reliability summaries (exploratory; tolerances are provisional until recalibrated)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "python"))

from phantom_gt_helpers import (  # noqa: E402
    PHANTOM_LABELS,
    PHANTOM_GT_TOLERANCES_PATH,
    _load_array,
    build_phantom_gt_tolerance_profile,
    run_phantom_gt_bids_summary,
)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "BIDS_test",
        help="BIDS root containing phantom sessions and gt folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "out" / "phantom_gt_reliability",
        help="Directory for per-session temp outputs and summary artifacts.",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "auto"],
        action="append",
        dest="backends",
        help="Backend(s) to run. Repeatable. Defaults to cpu+auto.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        default=None,
        help="Limit to one or more phantom subject IDs (repeatable), e.g. --subject sub-08phantom.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full summary JSON for each backend.",
    )
    parser.add_argument(
        "--write-summary-json",
        type=Path,
        default=None,
        help="Write combined summary JSON to this path.",
    )
    parser.add_argument(
        "--write-tolerance-suggestions",
        type=Path,
        nargs="?",
        const=PHANTOM_GT_TOLERANCES_PATH,
        default=None,
        help="Write a tolerance profile derived from observed max MAE values (default path: tests/data/BIDS_test/phantom_gt_mae_tolerances.json).",
    )
    parser.add_argument(
        "--write-scatter-plots",
        action="store_true",
        help="Write per-model/parameter GT-vs-fit scatter plots (PNG) with region subplots and linear regression annotations.",
    )
    return parser.parse_args(argv)


_MODEL_ORDER = {"t1": 0, "tofts": 1, "ex_tofts": 2, "patlak": 3, "2cxm": 4, "tissue_uptake": 5}
_PARAM_ORDER = {"T1": 0, "Ktrans": 1, "ve": 2, "vp": 3, "fp": 4}
_REGION_ORDER = {"muscle_fat": 0, "brain": 1, "vessel": 2}
_PARAM_UNITS = {"T1": "ms", "Ktrans": "/min", "fp": "/min", "ve": "frac", "vp": "frac"}
_REGION_NAMES = [v for _, v in sorted(PHANTOM_LABELS.items())]


def _fmt_float(value: Any, *, digits: int = 4) -> str:
    try:
        f = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    af = abs(f)
    if af != 0.0 and (af < 1e-3 or af >= 1e4):
        return f"{f:.3e}"
    return f"{f:.{digits}f}"


def _fmt_ratio(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.3f}"


def _fmt_percent(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.1f}"


def _session_rows(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for session in summary.get("sessions", []):
        sid = str(session.get("session_id", ""))
        backend_used = str(session.get("backend_used", ""))

        for region_name, stats in dict(session.get("t1", {})).items():
            rows.append(
                {
                    "session": sid,
                    "used": backend_used,
                    "model": "t1",
                    "param": "T1",
                    "region": str(region_name),
                    "unit": _PARAM_UNITS.get("T1", ""),
                    "n": str(int(stats.get("n", 0))),
                    "mae": _fmt_float(stats.get("mae"), digits=2),
                    "mae_pct": _fmt_percent(stats.get("mae_pct_gt_median_abs")),
                    "bias": _fmt_float(stats.get("bias"), digits=2),
                    "bias_med_pct": _fmt_percent(stats.get("median_bias_pct_gt_median")),
                    "fit_median": _fmt_float(stats.get("pred_median"), digits=2),
                    "gt_abs_median": _fmt_float(stats.get("gt_median_abs"), digits=2),
                }
            )

        for model_name, model_metrics in dict(session.get("dce", {})).items():
            for param_name, region_metrics in dict(model_metrics).items():
                for region_name, stats in dict(region_metrics).items():
                    rows.append(
                        {
                            "session": sid,
                            "used": backend_used,
                            "model": str(model_name),
                            "param": str(param_name),
                            "region": str(region_name),
                            "unit": _PARAM_UNITS.get(str(param_name), ""),
                            "n": str(int(stats.get("n", 0))),
                            "mae": _fmt_float(stats.get("mae"), digits=4),
                            "mae_pct": _fmt_percent(stats.get("mae_pct_gt_median_abs")),
                            "bias": _fmt_float(stats.get("bias"), digits=4),
                            "bias_med_pct": _fmt_percent(stats.get("median_bias_pct_gt_median")),
                            "fit_median": _fmt_float(stats.get("pred_median"), digits=4),
                            "gt_abs_median": _fmt_float(stats.get("gt_median_abs"), digits=4),
                        }
                    )

    rows.sort(
        key=lambda r: (
            r["session"],
            _MODEL_ORDER.get(r["model"], 99),
            r["model"],
            _PARAM_ORDER.get(r["param"], 99),
            r["param"],
            _REGION_ORDER.get(r["region"], 99),
            r["region"],
        )
    )
    return rows


def _print_table(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("(no rows)")
        return
    columns = [
        ("session", "session"),
        ("used", "used"),
        ("model", "model"),
        ("param", "param"),
        ("region", "region"),
        ("unit", "unit"),
        ("n", "n"),
        ("mae", "MAE"),
        ("mae_pct", "%GT"),
        ("bias", "bias"),
        ("bias_med_pct", "bias%(med)"),
        ("fit_median", "median(fit)"),
        ("gt_abs_median", "median|GT|"),
    ]
    widths: Dict[str, int] = {}
    for key, header in columns:
        widths[key] = len(header)
    for row in rows:
        for key, _ in columns:
            widths[key] = max(widths[key], len(str(row.get(key, ""))))

    header = "  ".join(f"{hdr:<{widths[key]}}" for key, hdr in columns)
    sep = "  ".join("-" * widths[key] for key, _ in columns)
    print(header)
    print(sep)
    numeric_cols = {"n", "mae", "mae_pct", "bias", "bias_med_pct", "fit_median", "gt_abs_median"}
    last_session = None
    for row in rows:
        session = row.get("session")
        if last_session is not None and session != last_session:
            print()
        rendered: List[str] = []
        for key, _ in columns:
            cell = str(row.get(key, ""))
            if key in numeric_cols:
                rendered.append(f"{cell:>{widths[key]}}")
            else:
                rendered.append(f"{cell:<{widths[key]}}")
        print("  ".join(rendered))
        last_session = session


def _print_condensed(summary_by_backend: Dict[str, Dict[str, Any]]) -> None:
    for backend_req, summary in summary_by_backend.items():
        aif_rows = _aif_rows(summary)
        rows = _session_rows(summary)
        print(f"\n== backend_requested={backend_req} ==")
        if aif_rows:
            print("\nAIF comparison (compact; GT vs ROCKETSHIP Stage-A/B curves, GT AIF is `aif_gd_mM`):")
            _print_aif_table(aif_rows)
            print()
        _print_table(rows)
        print("\nNotes: `MAE`, `bias` are voxelwise over the region (compute per-voxel error, then average).")
        print("Notes: `%GT` = 100 * MAE / median(|GT|) within the tissue region.")
        print("Notes: `bias%(med)` = 100 * (median(fit) - median(GT)) / median(GT).")
        print("Notes: Phantom runs currently align Stage-A baseline to GT `baseline_images` for diagnostics only.")


def _aif_rows(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for session in summary.get("sessions", []):
        sid = str(session.get("session_id", ""))
        backend_used = str(session.get("backend_used", ""))
        aif = dict(session.get("aif", {}))
        if not aif:
            continue
        gt = dict(aif.get("gt", {}))
        fit = dict(aif.get("fit", {}))
        cmp_ = dict(aif.get("compare", {}))
        cp_use = dict(cmp_.get("cp_use", cmp_.get("cp", {})))
        cb_stage_a = dict(cmp_.get("cb_stage_a", {}))
        rows.append(
            {
                "session": sid,
                "used": backend_used,
                "n": str(int(cmp_.get("n_compare", 0))),
                "relax": _fmt_float(fit.get("relaxivity"), digits=3),
                "hct": _fmt_float(fit.get("hematocrit"), digits=3),
                "dt_gt_s": _fmt_float(cmp_.get("timer_dt_gt_sec"), digits=3),
                "dt_fit_s": _fmt_float(cmp_.get("timer_dt_fit_sec"), digits=3),
                "dt_ratio": _fmt_ratio(cmp_.get("timer_dt_ratio_fit_over_gt")),
                "time_off_s": _fmt_float(cmp_.get("timer_offset_sec"), digits=2),
                "time_mae_s": _fmt_float(cmp_.get("timer_mae_after_offset_sec"), digits=4),
                "base_gt": str(gt.get("baseline_images", "n/a")),
                "base_fit": (
                    str(int(fit.get("steady_state_n"))) if fit.get("steady_state_n") is not None else "n/a"
                ),
                "cp_mae": _fmt_float(cp_use.get("mae"), digits=4),
                "cp_pct": _fmt_percent(cp_use.get("mae_pct_gt_median_abs")),
                "cp_bias": _fmt_float(cp_use.get("bias"), digits=4),
                "cp_bias_med_pct": _fmt_percent(cp_use.get("median_bias_pct_gt_median")),
                "cp_corr": _fmt_ratio(cp_use.get("correlation", cmp_.get("cp_correlation"))),
                "cp_peak_gt": _fmt_float(cp_use.get("peak_gt_mM", gt.get("cp_peak_mM")), digits=4),
                "cp_peak_fit": _fmt_float(cp_use.get("peak_fit_mM", fit.get("cp_peak_mM")), digits=4),
                "cp_peak_pct": _fmt_percent(
                    None
                    if cp_use.get("peak_ratio_fit_over_gt") is None
                    else (float(cp_use.get("peak_ratio_fit_over_gt")) - 1.0) * 100.0
                ),
                "cb_mae": _fmt_float(cb_stage_a.get("mae"), digits=4),
                "cb_pct": _fmt_percent(cb_stage_a.get("mae_pct_gt_median_abs")),
                "cb_bias": _fmt_float(cb_stage_a.get("bias"), digits=4),
                "cb_bias_med_pct": _fmt_percent(cb_stage_a.get("median_bias_pct_gt_median")),
                "cb_corr": _fmt_ratio(cb_stage_a.get("correlation")),
                "cb_peak_fit": _fmt_float(cb_stage_a.get("peak_fit_mM"), digits=4),
                "cb_peak_pct": _fmt_percent(
                    None
                    if cb_stage_a.get("peak_ratio_fit_over_gt") is None
                    else (float(cb_stage_a.get("peak_ratio_fit_over_gt")) - 1.0) * 100.0
                ),
                "cp_base_gt": _fmt_float(gt.get("cp_baseline_mean_mM"), digits=5),
                "cp_base_fit_gtn": _fmt_float(fit.get("cp_baseline_mean_mM_at_gt_baseline_n"), digits=5),
            }
        )
    rows.sort(key=lambda r: r["session"])
    return rows


def _print_aif_table(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("(no AIF rows)")
        return
    columns = [
        ("session", "session"),
        ("used", "used"),
        ("n", "n"),
        ("relax", "r1"),
        ("hct", "hct"),
        ("dt_ratio", "dt_ratio"),
        ("time_mae_s", "t_mae(s)"),
        ("base_gt", "baseGT"),
        ("base_fit", "baseFit"),
        ("cp_mae", "Cp_MAE"),
        ("cp_bias", "Cp_bias"),
        ("cp_corr", "Cp_r"),
        ("cp_peak_pct", "Cp_peak%"),
        ("cp_base_gt", "Cp_base_gt"),
        ("cp_base_fit_gtn", "Cp_base_fit@GTn"),
    ]
    widths: Dict[str, int] = {k: len(h) for k, h in columns}
    for row in rows:
        for key, _ in columns:
            widths[key] = max(widths[key], len(str(row.get(key, ""))))
    print("  ".join(f"{hdr:<{widths[k]}}" for k, hdr in columns))
    print("  ".join("-" * widths[k] for k, _ in columns))
    numeric = {
        "n",
        "dt_ratio",
        "time_mae_s",
        "relax",
        "hct",
        "cp_mae",
        "cp_bias",
        "cp_corr",
        "cp_peak_pct",
        "cp_base_gt",
        "cp_base_fit_gtn",
    }
    for row in rows:
        cells = []
        for key, _ in columns:
            cell = str(row.get(key, ""))
            cells.append(f"{cell:>{widths[key]}}" if key in numeric else f"{cell:<{widths[key]}}")
        print("  ".join(cells))


def _safe_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(text))


def _format_coef(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.3g}"


def _linear_regression(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if x.size < 2 or y.size < 2:
        return None
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or y.size < 2:
        return None
    if float(np.nanmax(x) - np.nanmin(x)) <= 0.0:
        return None
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return None
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0.0 and math.isfinite(ss_tot) else float("nan")
    return float(slope), float(intercept), r2


def _linear_axis_limits(vals: np.ndarray, *, zero_floor: bool = False) -> Optional[Tuple[float, float]]:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return None
    if vmax <= vmin:
        pad = max(abs(vmax), 1.0) * 0.05
    else:
        pad = (vmax - vmin) * 0.05
    if not math.isfinite(pad) or pad <= 0.0:
        pad = 1e-6
    lo = vmin - pad
    hi = vmax + pad
    if zero_floor:
        lo = max(0.0, lo)
        # Fraction parameters should stay in a readable range; keep a little headroom.
        if vmax <= 1.05:
            hi = min(1.05, max(hi, vmax * 1.02 + 1e-6))
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0) * 0.05
    return float(lo), float(hi)


def _log_axis_limits(vals: np.ndarray) -> Optional[Tuple[float, float]]:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return None
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmin <= 0.0 or vmax <= 0.0:
        return None
    lo = vmin / 1.25
    hi = vmax * 1.25
    if hi <= lo:
        hi = lo * 10.0
    return float(lo), float(hi)


def _collect_scatter_points(summary: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Return nested dict: model -> param -> region -> {"gt": ndarray, "fit": ndarray}.
    Arrays are concatenated across sessions.
    """
    gt_key_by_param = {"T1": "t1", "Ktrans": "ktrans", "ve": "ve", "vp": "vp", "fp": "fp"}
    label_by_region = {label_name: int(label_val) for label_val, label_name in PHANTOM_LABELS.items()}
    cache: Dict[str, np.ndarray] = {}
    bucket_lists: Dict[str, Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]] = {}

    def _arr(path_str: str) -> np.ndarray:
        key = str(path_str)
        if key not in cache:
            cache[key] = np.asarray(_load_array(Path(key)), dtype=np.float64)
        return cache[key]

    def _append(model_name: str, param_name: str, region_name: str, gt_vals: np.ndarray, fit_vals: np.ndarray) -> None:
        dst = (
            bucket_lists.setdefault(model_name, {})
            .setdefault(param_name, {})
            .setdefault(region_name, {"gt": [], "fit": []})
        )
        if gt_vals.size > 0:
            dst["gt"].append(np.asarray(gt_vals, dtype=np.float64).reshape(-1))
            dst["fit"].append(np.asarray(fit_vals, dtype=np.float64).reshape(-1))

    for session in summary.get("sessions", []):
        artifacts = dict(session.get("artifacts", {}))
        gt_paths = dict(artifacts.get("gt_paths", {}))
        if not gt_paths:
            continue
        seg_path = gt_paths.get("seg")
        if not seg_path:
            continue
        seg = _arr(str(seg_path))

        pred_t1_path = artifacts.get("pred_t1_map_path")
        gt_t1_path = gt_paths.get("t1")
        if pred_t1_path and gt_t1_path:
            pred_t1 = _arr(str(pred_t1_path))
            gt_t1 = _arr(str(gt_t1_path))
            for region_name in _REGION_NAMES:
                label_val = label_by_region[region_name]
                mask = (seg == float(label_val)) & np.isfinite(pred_t1) & np.isfinite(gt_t1)
                if np.any(mask):
                    _append("t1", "T1", region_name, gt_t1[mask], pred_t1[mask])

        dce_map_paths = dict(artifacts.get("dce_map_paths", {}))
        for model_name, model_paths_any in dce_map_paths.items():
            model_paths = dict(model_paths_any)
            for param_name, pred_map_path in model_paths.items():
                gt_key = gt_key_by_param.get(str(param_name))
                if not gt_key or gt_key not in gt_paths:
                    continue
                pred_map = _arr(str(pred_map_path))
                gt_map = _arr(str(gt_paths[gt_key]))
                for region_name in _REGION_NAMES:
                    label_val = label_by_region[region_name]
                    # Match DCE summary masking: exclude zeroed background/non-fit voxels.
                    mask = (
                        (seg == float(label_val))
                        & (pred_map != 0.0)
                        & np.isfinite(pred_map)
                        & np.isfinite(gt_map)
                    )
                    if np.any(mask):
                        _append(str(model_name), str(param_name), region_name, gt_map[mask], pred_map[mask])

    out: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}
    for model_name, model_dict in bucket_lists.items():
        out[model_name] = {}
        for param_name, param_dict in model_dict.items():
            out[model_name][param_name] = {}
            for region_name, pair_lists in param_dict.items():
                gt_cat = np.concatenate(pair_lists["gt"]) if pair_lists["gt"] else np.zeros((0,), dtype=np.float64)
                fit_cat = (
                    np.concatenate(pair_lists["fit"]) if pair_lists["fit"] else np.zeros((0,), dtype=np.float64)
                )
                out[model_name][param_name][region_name] = {"gt": gt_cat, "fit": fit_cat}
    return out


def _write_scatter_plots(summary: Dict[str, Any], *, output_dir: Path, backend_label: str) -> List[Path]:
    points = _collect_scatter_points(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for model_name in sorted(points.keys(), key=lambda m: (_MODEL_ORDER.get(m, 99), m)):
        for param_name in sorted(points[model_name].keys(), key=lambda p: (_PARAM_ORDER.get(p, 99), p)):
            region_data = points[model_name][param_name]
            fig, axes = plt.subplots(1, len(_REGION_NAMES), figsize=(15.5, 4.8), constrained_layout=True)
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])
            unit = _PARAM_UNITS.get(param_name, "")
            for ax, region_name in zip(axes, _REGION_NAMES):
                pair = region_data.get(region_name, {})
                x = np.asarray(pair.get("gt", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
                y = np.asarray(pair.get("fit", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
                finite = np.isfinite(x) & np.isfinite(y)
                x = x[finite]
                y = y[finite]
                use_log = param_name == "Ktrans" and region_name in {"brain", "vessel"}
                if use_log:
                    pos = (x > 0.0) & (y > 0.0)
                    x = x[pos]
                    y = y[pos]
                    if x.size >= 2:
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                if x.size == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(region_name)
                    ax.grid(True, alpha=0.25)
                    continue

                ax.scatter(x, y, s=6, alpha=0.25, edgecolors="none", color="#1f77b4")

                x_min = float(np.min(x))
                x_max = float(np.max(x))
                y_min = float(np.min(y))
                y_max = float(np.max(y))
                x_lim: Optional[Tuple[float, float]]
                if use_log:
                    x_lim = _log_axis_limits(x)
                else:
                    x_lim = _linear_axis_limits(x, zero_floor=(param_name in {"ve", "vp"}))
                if x_lim is not None:
                    ax.set_xlim(*x_lim)

                # Keep y autoscaling driven by fit values, but avoid identity-line range from
                # expanding x-axis (which made vp and vessel-ve plots hard to read).
                if x_lim is not None:
                    lo_x, hi_x = x_lim
                    if use_log and lo_x > 0.0 and hi_x > lo_x:
                        xx = np.geomspace(lo_x, hi_x, 200)
                    else:
                        xx = np.linspace(lo_x, hi_x, 200)
                    ax.plot(xx, xx, linestyle="--", linewidth=1.0, color="0.35", alpha=0.8, label="y=x")

                reg = _linear_regression(x, y)
                if reg is not None:
                    slope, intercept, r2 = reg
                    if math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min:
                        if use_log and x_min > 0.0:
                            x_line = np.geomspace(x_min, x_max, 200)
                        else:
                            x_line = np.linspace(x_min, x_max, 200)
                        y_line = slope * x_line + intercept
                        finite_line = np.isfinite(y_line)
                        if use_log:
                            finite_line = finite_line & (y_line > 0.0)
                        if np.any(finite_line):
                            ax.plot(
                                x_line[finite_line],
                                y_line[finite_line],
                                linewidth=1.5,
                                color="#d62728",
                                alpha=0.9,
                                label="linear fit",
                            )
                    eq_text = f"y = {_format_coef(slope)}x + {_format_coef(intercept)}\n$R^2$ = {_format_coef(r2)}\nn = {x.size}"
                else:
                    eq_text = f"regression unavailable\nn = {x.size}"

                ax.text(
                    0.03,
                    0.97,
                    eq_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
                )
                ax.set_title(f"{region_name}{' (log)' if use_log else ''}")
                xlabel = f"GT {param_name}" + (f" ({unit})" if unit else "")
                ylabel = f"Fit {param_name}" + (f" ({unit})" if unit else "")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", alpha=0.25)

            subject_ids = [str(s.get("subject", "")) for s in summary.get("sessions", [])]
            subject_ids = [s for s in subject_ids if s]
            subj_text = ", ".join(subject_ids)
            fig.suptitle(f"Phantom GT Scatter: {model_name} / {param_name}  (backend={backend_label}; {subj_text})", fontsize=12)
            out_path = output_dir / f"{_safe_name(model_name)}__{_safe_name(param_name)}_scatter.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved.append(out_path)

    return saved


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    backends = args.backends or ["cpu", "auto"]

    combined: Dict[str, Dict[str, Any]] = {}
    for backend in backends:
        out_root = args.output_root / str(backend)
        summary = run_phantom_gt_bids_summary(
            bids_root=args.bids_root,
            output_root=out_root,
            backend=str(backend),
            subjects=args.subjects,
        )
        combined[str(backend)] = summary
        if args.print_json:
            print(json.dumps(summary, indent=2))
        if args.write_scatter_plots:
            plot_dir = out_root / "scatter_plots"
            plot_paths = _write_scatter_plots(summary, output_dir=plot_dir, backend_label=str(backend))
            print(f"Wrote {len(plot_paths)} scatter plots: {plot_dir}")

    _print_condensed(combined)

    if args.write_summary_json is not None:
        target = Path(args.write_summary_json).expanduser()
        if not target.is_absolute():
            target = (REPO_ROOT / target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote summary JSON: {target}")

    if args.write_tolerance_suggestions is not None:
        target = Path(args.write_tolerance_suggestions).expanduser()
        if not target.is_absolute():
            target = (REPO_ROOT / target).resolve()
        profile = build_phantom_gt_tolerance_profile(combined)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote tolerance suggestions: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
