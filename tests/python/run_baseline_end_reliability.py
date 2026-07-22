#!/usr/bin/env python3
"""Evaluate automatic end-baseline (steady-state end) detectors against human-rated
ground truth (`SteadyStateEndTimeIndex` in AIF-mask JSON sidecars, e.g. from AIFArtist).

On-demand diagnostic tool, not part of the automated test suite: walks a BIDS derivatives
tree for AIF-mask sidecars with ground truth, finds the matching raw dynamic DCE series in
a separate BIDS raw tree, runs all 4 auto-detectors (piecewise_constant, legacy_sobel, glr,
tv) on each session's AIF-mask curve, and writes a per-algorithm accuracy/MSE summary plus
one figure per session."""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "python"))

from baseline_end_reliability_helpers import (  # noqa: E402
    DETECTORS,
    AlgorithmStats,
    SessionResult,
    compute_algorithm_stats,
    discover_aif_sidecars,
    is_ground_truth_valid,
    process_session,
)


_ALGO_COLORS = {
    "piecewise_constant": "#d62728",
    "legacy_sobel": "#ff7f0e",
    "glr": "#2ca02c",
    "tv": "#9467bd",
}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        required=True,
        help="BIDS derivatives root, searched recursively for AIF-mask JSON sidecars with a "
        "SteadyStateEndTimeIndex field. Point this at a single pipeline folder (e.g. "
        "derivatives/AIFArtist) to avoid double-counting sibling pipelines rating the same session.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="BIDS raw root containing rawdata-style sub-*/ses-*/dce/*.nii(.gz) dynamic series.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the summary text file, per-session CSV, and per-session PNG figures.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        default=None,
        help="Limit to one or more subject IDs (repeatable), e.g. --subject sub-1101608.",
    )
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=None,
        help="Optional: also report accuracy within +/- N frames of ground truth, alongside exact-match accuracy.",
    )
    parser.add_argument(
        "--no-per-session-csv",
        action="store_true",
        help="Skip writing per_session_details.csv.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing per-session PNG figures.",
    )
    parser.add_argument(
        "--all",
        dest="use_all_voxels",
        action="store_true",
        help="Run detectors on the mean of ALL voxels in the dynamic image instead of the AIF "
        "mask (either/or, not both). Output filenames get an '_allvoxels' suffix so both modes "
        "can be written into the same --output-dir without overwriting each other.",
    )
    return parser.parse_args(argv)


def _plot_session(result: SessionResult, output_dir: Path, *, mode_suffix: str, signal_label: str) -> Optional[Path]:
    if result.status != "ok" or result.mean_curve is None:
        return None

    curve = result.mean_curve
    n = int(curve.size)
    frames = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frames, curve, color="0.2", linewidth=1.5, label=f"mean {signal_label} signal")

    gt_valid = is_ground_truth_valid(result)
    if gt_valid:
        ax.axvline(
            result.ground_truth_1b, color="black", linestyle="-", linewidth=2.0, label=f"GT: {result.ground_truth_1b}"
        )

    for name in DETECTORS:
        value = result.predictions.get(name)
        if value is None:
            continue
        ax.axvline(
            value, color=_ALGO_COLORS.get(name, "gray"), linestyle="--", linewidth=1.5, label=f"{name}: {value}"
        )

    title = result.id if gt_valid else f"{result.id}  (GT out of range)"
    ax.set_title(f"{title}  [{signal_label}]")
    ax.set_xlabel("Frame index (1-based)")
    ax.set_ylabel(f"Mean {signal_label} signal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.text(0.01, 0.01, f"n_timepoints={n}   dynamic={result.dynamic_path}", fontsize=6, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_path = output_dir / f"{result.id}_end_ss{mode_suffix}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _write_summary_txt(
    results: List[SessionResult],
    stats: Dict[str, AlgorithmStats],
    args: argparse.Namespace,
    output_dir: Path,
    *,
    mode_suffix: str,
    signal_label: str,
) -> Path:
    n_total = len(results)
    n_ok = sum(1 for r in results if r.status == "ok")
    skipped = [r for r in results if r.status == "skipped"]
    reason_counts = collections.Counter(r.reason for r in skipped)

    lines: List[str] = []
    lines.append("End-baseline detector reliability summary")
    lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Signal source: {signal_label}")
    lines.append(f"Derivatives root: {Path(args.derivatives_root).expanduser().resolve()}")
    lines.append(f"Raw root: {Path(args.raw_root).expanduser().resolve()}")
    lines.append(f"Subject filter: {', '.join(args.subjects) if args.subjects else '(none)'}")
    lines.append("")
    lines.append(f"Discovered sidecars (with SteadyStateEndTimeIndex): {n_total}")
    lines.append(f"  ok: {n_ok}")
    lines.append(f"  skipped: {len(skipped)}")
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"    - {count}x: {reason}")
    lines.append("")

    header = f"{'Algorithm':<20}{'N_valid':>10}{'Accuracy%':>12}{'MSE(frames^2)':>16}"
    if args.tolerance_frames is not None:
        header += f"{'Within+/-' + str(args.tolerance_frames) + '%':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in DETECTORS:
        s = stats[name]
        row = f"{name:<20}{s.n_valid:>10}{s.accuracy_pct:>12.1f}{s.mse:>16.3f}"
        if args.tolerance_frames is not None:
            row += f"{(s.tolerance_pct if s.tolerance_pct is not None else float('nan')):>14.1f}"
        lines.append(row)

    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- Accuracy% = % of valid sessions (ground truth index within [1, n_timepoints]) where the "
        "detector's predicted end_ss_1b exactly matches SteadyStateEndTimeIndex."
    )
    lines.append(
        "- MSE is in squared-frame-index units; it is not time-normalized across sessions with "
        "different temporal resolution."
    )
    lines.append("- Sessions with no usable dynamic file (see skip breakdown above) are excluded entirely.")

    summary_path = output_dir / f"baseline_end_summary{mode_suffix}.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _write_per_session_csv(results: List[SessionResult], output_dir: Path, *, mode_suffix: str) -> Path:
    csv_path = output_dir / f"per_session_details{mode_suffix}.csv"
    fieldnames = [
        "subject",
        "session",
        "sidecar_path",
        "dynamic_path",
        "dynamic_source",
        "status",
        "reason",
        "ground_truth_1b",
        "n_timepoints",
        "gt_valid",
        *DETECTORS.keys(),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: Dict[str, Any] = {
                "subject": r.subject,
                "session": r.session or "",
                "sidecar_path": str(r.sidecar_path),
                "dynamic_path": str(r.dynamic_path) if r.dynamic_path else "",
                "dynamic_source": r.dynamic_source,
                "status": r.status,
                "reason": r.reason or "",
                "ground_truth_1b": r.ground_truth_1b if r.ground_truth_1b is not None else "",
                "n_timepoints": r.n_timepoints if r.n_timepoints is not None else "",
                "gt_valid": is_ground_truth_valid(r),
            }
            for name in DETECTORS:
                row[name] = r.predictions.get(name, "")
            writer.writerow(row)
    return csv_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for AIF-mask sidecars with SteadyStateEndTimeIndex under {args.derivatives_root}...")
    records = discover_aif_sidecars(Path(args.derivatives_root), subjects=args.subjects)
    if not records:
        print(
            f"No AIF-mask sidecars with SteadyStateEndTimeIndex found under {args.derivatives_root}",
            file=sys.stderr,
        )
        return 1

    print(f"Discovered {len(records)} AIF-mask sidecars with SteadyStateEndTimeIndex under {args.derivatives_root}")

    mode_suffix = "_allvoxels" if args.use_all_voxels else ""
    signal_label = "all-voxel (whole image)" if args.use_all_voxels else "AIF-mask"

    results: List[SessionResult] = []
    plotted = 0
    n_records = len(records)
    for i, record in enumerate(records, start=1):
        result = process_session(record, Path(args.raw_root), use_all_voxels=args.use_all_voxels)
        results.append(result)
        if not args.no_plots and _plot_session(result, output_dir, mode_suffix=mode_suffix, signal_label=signal_label) is not None:
            plotted += 1
        if i % 10 == 0 or i == n_records:
            print(f"Processed {i}/{n_records} sessions...")

    if not args.no_plots:
        print(f"Wrote {plotted} per-session figures to {output_dir}")

    stats = compute_algorithm_stats(results, tolerance_frames=args.tolerance_frames)
    summary_path = _write_summary_txt(results, stats, args, output_dir, mode_suffix=mode_suffix, signal_label=signal_label)
    print(f"Wrote summary: {summary_path}")

    if not args.no_per_session_csv:
        csv_path = _write_per_session_csv(results, output_dir, mode_suffix=mode_suffix)
        print(f"Wrote per-session details: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
