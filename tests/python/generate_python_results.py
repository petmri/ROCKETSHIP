#!/usr/bin/env python3
"""Generate Python model outputs in parity-runner JSON format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("tests/contracts/baselines/matlab_reference_v1.json"),
        help="MATLAB baseline JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/python/python_results.json"),
        help="Output JSON path for Python results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))

    from rocketship import model_patlak_cfit, model_tofts_cfit  # pylint: disable=import-outside-toplevel

    baseline = json.loads(args.baseline.read_text())

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]

    # Use MATLAB-fit values from baseline to avoid hard-coding fixture internals.
    tofts_fit = baseline["dce"]["inverse"]["tofts_fit"]
    ktrans = float(tofts_fit[0])
    ve = float(tofts_fit[1])
    patlak_fit = baseline["dce"]["inverse"]["patlak_linear"]
    patlak_ktrans = float(patlak_fit[0])
    patlak_vp = float(patlak_fit[1])

    results = {
        "meta": {
            "source": "generate_python_results.py",
            "baseline": str(args.baseline),
            "models": ["model_tofts_cfit", "model_patlak_cfit"],
        },
        "results": {
            "tofts_forward": model_tofts_cfit(ktrans, ve, cp, timer),
            "patlak_forward": model_patlak_cfit(patlak_ktrans, patlak_vp, cp, timer),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote Python results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
