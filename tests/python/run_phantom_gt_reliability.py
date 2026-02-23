#!/usr/bin/env python3
"""Run synthetic phantom GT reliability summaries (exploratory; tolerances are provisional until recalibrated)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "python"))

from phantom_gt_helpers import (  # noqa: E402
    PHANTOM_GT_TOLERANCES_PATH,
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
    return parser.parse_args(argv)


_MODEL_ORDER = {"t1": 0, "tofts": 1, "ex_tofts": 2, "patlak": 3, "2cxm": 4, "tissue_uptake": 5}
_PARAM_ORDER = {"T1": 0, "Ktrans": 1, "ve": 2, "vp": 3, "fp": 4}
_REGION_ORDER = {"muscle_fat": 0, "brain": 1, "vessel": 2}
_PARAM_UNITS = {"T1": "ms", "Ktrans": "/min", "fp": "/min", "ve": "frac", "vp": "frac"}


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
