#!/usr/bin/env python3
"""Evaluate automatic end-baseline (steady-state end) detectors against human-rated
ground truth (`SteadyStateEndTimeIndex` in AIF-mask JSON sidecars, e.g. from AIFArtist).

On-demand diagnostic tool, not part of the automated test suite: walks a BIDS derivatives
tree for AIF-mask sidecars with ground truth, finds the matching raw dynamic DCE series in
a separate BIDS raw tree, runs all 5 auto-detectors (piecewise_constant, legacy_sobel, glr,
tv, biexp_fit) on each session's AIF-mask curve, and writes a per-algorithm accuracy/MSE
summary plus one figure per session.

`biexp_fit` is the production default. Unlike the other four it is a model fit rather than a
signal-shape heuristic, so it also yields a fractional injection end (`t0_exp`) and a fitted
curve; both are overlaid on the per-session figures and its extra diagnostics land in the
per-session CSV."""

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
    DEFAULT_AIF_MASK_PATTERN,
    DEFAULT_CONFIG_TEMPLATE,
    DETECTOR_NAMES,
    PIPELINE_DYNAMIC_PATTERN,
    AgreementStats,
    AlgorithmStats,
    SessionResult,
    biexp_fitted_curve,
    build_detectors,
    compute_algorithm_stats,
    compute_detector_agreement,
    discover_aif_masks,
    load_biexp_config,
    discover_aif_sidecars,
    is_ground_truth_valid,
    process_session,
)

AGREEMENT_REFERENCE = "biexp_fit"


_ALGO_COLORS = {
    "piecewise_constant": "#d62728",
    "legacy_sobel": "#ff7f0e",
    "glr": "#2ca02c",
    "tv": "#9467bd",
    "biexp_fit": "#1f77b4",
}

# Extra per-session columns the `biexp_fit` detector produces that the others have no analogue
# for: (CSV column, key in the detector's details dict).
_BIEXP_DETAIL_COLUMNS = (
    ("biexp_mode", "mode"),
    ("biexp_end_injection_1b", "end_injection_1b"),
    ("biexp_t_base_end_frames", "fit_t_base_end_frames"),
    ("biexp_t0_exp_frames", "fit_t0_exp_frames"),
    ("biexp_delta_frames", "fit_delta_frames"),
    ("biexp_rsquare_adj", "fit_rsquare_adj"),
    ("biexp_seed_end_ss_1b", "seed_end_ss_1b"),
)


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
        "--dynamic-pattern",
        default=None,
        help="Glob pinning the dynamic series within each session's dce/ folder, e.g. "
        f"'{PIPELINE_DYNAMIC_PATTERN}'. Required when --raw-root points at a derivatives tree, "
        "where the default one-dynamic-per-session heuristic has many candidates and picks "
        "alphabetically. Use the pipeline's own pattern so the detectors see the curve "
        "production feeds them.",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Discover AIF masks by filename instead of requiring a SteadyStateEndTimeIndex "
        "sidecar. Use on datasets that were never rated in AIFArtist: all detectors still run "
        "and all figures are still written, but accuracy/MSE are replaced by a cross-detector "
        f"agreement table against '{AGREEMENT_REFERENCE}'.",
    )
    parser.add_argument(
        "--aif-mask-pattern",
        default=DEFAULT_AIF_MASK_PATTERN,
        help=f"Glob for --no-ground-truth mask discovery (default: {DEFAULT_AIF_MASK_PATTERN}, the "
        "same pattern the pipeline resolves aif_files from).",
    )
    parser.add_argument(
        "--config-template",
        type=Path,
        default=DEFAULT_CONFIG_TEMPLATE,
        help="JSON config whose stage_overrides supply the aif_* settings for the biexp_fit "
        f"detector (default: {DEFAULT_CONFIG_TEMPLATE}). The other detectors ignore it.",
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

    # The biexponential detector is the only one that produces a fitted curve, so overlay it:
    # a detector line landing on the right frame for the wrong reason is visible here and
    # nowhere else. Drawn before the vertical lines so it sits underneath them.
    biexp_details = result.detector_details.get("biexp_fit", {})
    fitted = biexp_fitted_curve(biexp_details, n)
    if fitted is not None:
        rsq = biexp_details.get("fit_rsquare_adj")
        rsq_text = f"  (adj R²={float(rsq):.3f})" if isinstance(rsq, (int, float)) else ""
        ax.plot(
            frames,
            fitted,
            color=_ALGO_COLORS["biexp_fit"],
            linewidth=1.2,
            alpha=0.85,
            label=f"biexp_fit curve{rsq_text}",
        )

    gt_valid = is_ground_truth_valid(result)
    if gt_valid:
        ax.axvline(
            result.ground_truth_1b, color="black", linestyle="-", linewidth=2.0, label=f"GT: {result.ground_truth_1b}"
        )

    for name in DETECTOR_NAMES:
        value = result.predictions.get(name)
        if value is None:
            continue
        ax.axvline(
            value, color=_ALGO_COLORS.get(name, "gray"), linestyle="--", linewidth=1.5, label=f"{name}: {value}"
        )

    # t0_exp is the end of the linear upslope, not a baseline end, so it is not a prediction of
    # the ground truth and is drawn in a different style to keep that clear.
    t0_exp_1b = biexp_details.get("end_injection_1b")
    if isinstance(t0_exp_1b, (int, float)) and np.isfinite(t0_exp_1b):
        ax.axvline(
            float(t0_exp_1b),
            color=_ALGO_COLORS["biexp_fit"],
            linestyle=":",
            linewidth=1.5,
            label=f"biexp_fit t0_exp: {float(t0_exp_1b):.2f}",
        )

    if gt_valid:
        title = result.id
    elif result.ground_truth_1b is None:
        title = f"{result.id}  (no GT)"
    else:
        title = f"{result.id}  (GT out of range)"
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
    stats: Optional[Dict[str, AlgorithmStats]],
    args: argparse.Namespace,
    output_dir: Path,
    *,
    mode_suffix: str,
    signal_label: str,
    agreement: Optional[Dict[str, AgreementStats]] = None,
) -> Path:
    """Write the summary. Exactly one of `stats` / `agreement` is populated."""
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
    lines.append(f"biexp_fit config template: {Path(args.config_template).expanduser().resolve()}")
    lines.append("")
    if args.no_ground_truth:
        lines.append(f"Discovered AIF masks ({args.aif_mask_pattern}, unrated): {n_total}")
    else:
        lines.append(f"Discovered sidecars (with SteadyStateEndTimeIndex): {n_total}")
    lines.append(f"  ok: {n_ok}")
    lines.append(f"  skipped: {len(skipped)}")
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"    - {count}x: {reason}")
    lines.append("")

    if agreement is not None:
        # No ratings exist, so accuracy is undefined. Report how the detectors relate to each
        # other instead -- the only comparison the data actually supports.
        lines.append(f"No ground truth available; comparing detectors against '{AGREEMENT_REFERENCE}'.")
        lines.append("")
        header = f"{'Algorithm':<20}{'N_compared':>12}{'Identical%':>12}{'MeanOffset':>12}{'MaxAbsOffset':>14}"
        lines.append(header)
        lines.append("-" * len(header))
        for name in DETECTOR_NAMES:
            a = agreement[name]
            lines.append(
                f"{name:<20}{a.n_compared:>12}{a.identical_pct:>12.1f}"
                f"{a.mean_signed_offset:>12.2f}{a.max_abs_offset:>14.1f}"
            )
    else:
        header = f"{'Algorithm':<20}{'N_valid':>10}{'Accuracy%':>12}{'MSE(frames^2)':>16}"
        if args.tolerance_frames is not None:
            header += f"{'Within+/-' + str(args.tolerance_frames) + '%':>14}"
        lines.append(header)
        lines.append("-" * len(header))
        for name in DETECTOR_NAMES:
            s = stats[name]
            row = f"{name:<20}{s.n_valid:>10}{s.accuracy_pct:>12.1f}{s.mse:>16.3f}"
            if args.tolerance_frames is not None:
                row += f"{(s.tolerance_pct if s.tolerance_pct is not None else float('nan')):>14.1f}"
            lines.append(row)

    # With so few sessions an aggregate hides more than it shows, so list the raw calls too.
    if n_ok > 0 and n_ok <= 20:
        lines.append("")
        lines.append("Per-session predictions (end_ss, 1-based):")
        gt_col = "" if args.no_ground_truth else f"{'GT':>6}"
        lines.append(f"{'Session':<28}{gt_col}" + "".join(f"{name:>20}" for name in DETECTOR_NAMES))
        for r in results:
            if r.status != "ok":
                continue
            gt_cell = "" if args.no_ground_truth else f"{r.ground_truth_1b if r.ground_truth_1b is not None else '-':>6}"
            cells = "".join(f"{str(r.predictions.get(name, '-')):>20}" for name in DETECTOR_NAMES)
            lines.append(f"{r.id:<28}{gt_cell}{cells}")

    # Whether biexp_fit actually fitted matters for reading its row above: every fallback is a
    # silent vote for `tv`, so a high fallback count means the two rows are not independent.
    biexp_modes = collections.Counter(
        str(r.detector_details.get("biexp_fit", {}).get("mode", "(not run)")) for r in results if r.status == "ok"
    )
    if biexp_modes:
        lines.append("")
        lines.append("biexp_fit outcome breakdown (ok sessions):")
        for mode, count in sorted(biexp_modes.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {count}x: {mode}")

    lines.append("")
    lines.append("Notes:")
    if agreement is not None:
        lines.append(
            f"- Identical% / offsets are measured against '{AGREEMENT_REFERENCE}', not against truth. "
            "They say how far the detectors are from each other, and say nothing about which is right."
        )
        lines.append(
            "- Offsets are signed (detector - reference) in frames, so a consistent bias is "
            "distinguishable from scatter."
        )
    else:
        lines.append(
            "- Accuracy% = % of valid sessions (ground truth index within [1, n_timepoints]) where the "
            "detector's predicted end_ss_1b exactly matches SteadyStateEndTimeIndex."
        )
        lines.append(
            "- MSE is in squared-frame-index units; it is not time-normalized across sessions with "
            "different temporal resolution."
        )
    lines.append("- Sessions with no usable dynamic file (see skip breakdown above) are excluded entirely.")
    lines.append(
        "- biexp_fit is seeded from tv and falls back to it when the fit cannot run or does not "
        "converge, so any mode other than 'fit' above means that session's biexp_fit prediction "
        "is just tv's."
    )

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
        *DETECTOR_NAMES,
        *(column for column, _ in _BIEXP_DETAIL_COLUMNS),
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
            for name in DETECTOR_NAMES:
                row[name] = r.predictions.get(name, "")
            biexp_details = r.detector_details.get("biexp_fit", {})
            for column, detail_key in _BIEXP_DETAIL_COLUMNS:
                row[column] = biexp_details.get(detail_key, "")
            writer.writerow(row)
    return csv_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    biexp_config = load_biexp_config(args.config_template)
    detectors = build_detectors(biexp_config)
    print(
        f"biexp_fit settings from {Path(args.config_template).expanduser().resolve()}: "
        f"aif_Robust={biexp_config.stage_overrides.get('aif_Robust', '(default)')}, "
        f"aif_peak_weight_exponent={biexp_config.stage_overrides.get('aif_peak_weight_exponent', '(default)')}"
    )

    if args.no_ground_truth:
        what = f"AIF masks matching {args.aif_mask_pattern}"
        print(f"Searching for {what} under {args.derivatives_root}...")
        records = discover_aif_masks(
            Path(args.derivatives_root), subjects=args.subjects, pattern=args.aif_mask_pattern
        )
    else:
        what = "AIF-mask sidecars with SteadyStateEndTimeIndex"
        print(f"Searching for {what} under {args.derivatives_root}...")
        records = discover_aif_sidecars(Path(args.derivatives_root), subjects=args.subjects)

    if not records:
        print(f"No {what} found under {args.derivatives_root}", file=sys.stderr)
        if not args.no_ground_truth:
            print("Hint: pass --no-ground-truth to run the detectors on an unrated dataset.", file=sys.stderr)
        return 1

    print(f"Discovered {len(records)} {what} under {args.derivatives_root}")

    mode_suffix = "_allvoxels" if args.use_all_voxels else ""
    signal_label = "all-voxel (whole image)" if args.use_all_voxels else "AIF-mask"

    results: List[SessionResult] = []
    plotted = 0
    n_records = len(records)
    for i, record in enumerate(records, start=1):
        result = process_session(
            record,
            Path(args.raw_root),
            detectors,
            use_all_voxels=args.use_all_voxels,
            dynamic_pattern=args.dynamic_pattern,
        )
        results.append(result)
        if not args.no_plots and _plot_session(result, output_dir, mode_suffix=mode_suffix, signal_label=signal_label) is not None:
            plotted += 1
        if i % 10 == 0 or i == n_records:
            print(f"Processed {i}/{n_records} sessions...")

    if not args.no_plots:
        print(f"Wrote {plotted} per-session figures to {output_dir}")

    # Exactly one of the two tables applies: without ratings there is no accuracy to compute.
    if args.no_ground_truth:
        stats, agreement = None, compute_detector_agreement(results, reference=AGREEMENT_REFERENCE)
    else:
        stats, agreement = compute_algorithm_stats(results, tolerance_frames=args.tolerance_frames), None
    summary_path = _write_summary_txt(
        results,
        stats,
        args,
        output_dir,
        mode_suffix=mode_suffix,
        signal_label=signal_label,
        agreement=agreement,
    )
    print(f"Wrote summary: {summary_path}")

    if not args.no_per_session_csv:
        csv_path = _write_per_session_csv(results, output_dir, mode_suffix=mode_suffix)
        print(f"Wrote per-session details: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
