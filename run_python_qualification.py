#!/usr/bin/env python3
"""Run Python T1 + DCE qualification across all datasets in a BIDS root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from qualification import QualificationRunConfig, run_bids_qualification  # noqa: E402


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "BIDS_test",
        help="BIDS root with rawdata/ + derivatives/ (default: tests/data/BIDS_test).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "out" / "python_qualification",
        help="Directory for qualification artifacts.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "gpufit"],
        default="auto",
        help="DCE backend to request.",
    )
    parser.add_argument(
        "--dce-models",
        default="tofts,ex_tofts,patlak",
        help="Comma-separated DCE models to run.",
    )
    parser.add_argument("--skip-t1", action="store_true", help="Skip parametric T1 qualification.")
    parser.add_argument("--skip-dce", action="store_true", help="Skip DCE qualification.")
    parser.add_argument(
        "--no-postfit-arrays",
        action="store_true",
        help="Disable Stage-D postfit NPZ export during DCE qualification.",
    )
    parser.add_argument(
        "--print-summary-json",
        action="store_true",
        help="Print full summary JSON to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    models = [token.strip().lower() for token in str(args.dce_models).split(",") if token.strip()]
    cfg = QualificationRunConfig(
        bids_root=args.bids_root,
        output_root=args.output_root,
        backend=args.backend,
        dce_models=models,
        run_t1=not bool(args.skip_t1),
        run_dce=not bool(args.skip_dce),
        write_postfit_arrays=not bool(args.no_postfit_arrays),
    )
    summary = run_bids_qualification(cfg)

    if args.print_summary_json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary["meta"], indent=2))

    return 0 if str(summary["meta"]["status"]) == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

