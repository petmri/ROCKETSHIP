#!/usr/bin/env python3
"""Run DCE Part E post-fit comparisons from Stage-D `*_postfit_arrays.npz` files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_postfit_analysis import (  # noqa: E402
    DceFitStats,
    load_dce_fit_stats_from_npz,
    run_aic_analysis,
    run_ftest_analysis,
)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=["ftest", "aic"],
        required=True,
        help="Comparison to run.",
    )
    parser.add_argument(
        "--region",
        choices=["voxel", "roi"],
        required=True,
        help="Operate on voxel-level or ROI-level results.",
    )
    parser.add_argument(
        "--result",
        dest="results",
        action="append",
        required=True,
        help="Path to a Stage-D postfit array file (`*_postfit_arrays.npz`) (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where analysis artifacts are written.",
    )
    parser.add_argument(
        "--print-summary-json",
        action="store_true",
        help="Print the output summary JSON payload to stdout.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional plot PNG generation.",
    )
    return parser.parse_args(argv)


def _load_models(result_paths: List[str]) -> List[DceFitStats]:
    return [load_dce_fit_stats_from_npz(Path(raw)) for raw in result_paths]


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    models = _load_models(args.results)

    if args.analysis == "ftest":
        if len(models) != 2:
            raise ValueError("ftest requires exactly two --result files (lower model then higher model)")
        summary = run_ftest_analysis(
            models[0],
            models[1],
            region=args.region,
            output_dir=args.output_dir,
            write_plots=(not args.no_plots),
        )
    else:
        if len(models) < 2:
            raise ValueError("aic requires at least two --result files")
        summary = run_aic_analysis(
            models,
            region=args.region,
            output_dir=args.output_dir,
            write_plots=(not args.no_plots),
        )

    summary_json = json.dumps(summary, indent=2)
    if args.print_summary_json:
        print(summary_json)
    else:
        print(f"Wrote summary: {summary['summary_json_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
