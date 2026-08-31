#!/usr/bin/env python3
"""Fail if a freshly regenerated MATLAB baseline drifts from the committed one.

The committed ``matlab_reference_v1.json`` is the source of truth for every
Python-vs-MATLAB parity check. If MATLAB algorithm code changes but the baseline
is not regenerated, parity is silently validated against a stale snapshot. This
guard closes that loop: CI regenerates the baseline with the *current* MATLAB
(``export_parity_baseline``) into a temp file, then runs this script to compare
that candidate against the committed baseline.

Tolerances are deliberately loose enough to absorb nonlinear-optimizer
nondeterminism across MATLAB releases (observed at ~1e-8 on confidence-interval
columns) while still catching any real algorithm change, which moves values by
orders of magnitude more than that.

Usage:
    python tests/contracts/check_baseline_drift.py --candidate /tmp/fresh.json
    python tests/contracts/check_baseline_drift.py --candidate a.json --reference b.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = REPO_ROOT / "tests" / "contracts" / "baselines" / "matlab_reference_v1.json"

# Absorbs cross-release optimizer noise; a genuine algorithm change is far larger.
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4


def _load(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline JSON: {path}")
    return json.loads(path.read_text())


def _drifts(reference: Any, candidate: Any, atol: float, rtol: float, path: str = "") -> List[Tuple[str, float, float, float, float]]:
    """Return [(path, ref, cand, abs_err, rel_err)] for every element out of tolerance."""
    out: List[Tuple[str, float, float, float, float]] = []

    if isinstance(reference, dict) and isinstance(candidate, dict):
        for key in reference:
            # meta: generated_utc always differs. noisy: input curves come from
            # MATLAB randn, whose sequence we do not want to couple to this guard;
            # the same fitters are covered by the noise-free inverse contracts.
            if key in ("meta", "noisy"):
                continue
            child = f"{path}.{key}" if path else key
            if key not in candidate:
                out.append((child, math.nan, math.nan, math.inf, math.inf))
            else:
                out += _drifts(reference[key], candidate[key], atol, rtol, child)
        return out

    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            out.append((f"{path} (len {len(reference)}->{len(candidate)})", math.nan, math.nan, math.inf, math.inf))
            return out
        for i, (r, c) in enumerate(zip(reference, candidate)):
            out += _drifts(r, c, atol, rtol, f"{path}[{i}]")
        return out

    # Scalar comparison (numbers only; strings/bools/null are ignored as non-numeric).
    try:
        rv, cv = float(reference), float(candidate)
    except (TypeError, ValueError):
        return out
    abs_err = abs(rv - cv)
    scale = abs(rv)
    rel_err = abs_err / scale if scale > 0 else (0.0 if abs_err == 0 else math.inf)
    if abs_err > (atol + rtol * scale):
        out.append((path or "<root>", rv, cv, abs_err, rel_err))
    return out


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True, help="Freshly regenerated baseline JSON to check.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Committed baseline JSON (source of truth).")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args(argv)

    reference = _load(args.reference)
    candidate = _load(args.candidate)

    drifts = _drifts(reference, candidate, args.atol, args.rtol)

    print(f"baseline drift check: reference={args.reference.name} candidate={args.candidate.name} "
          f"atol={args.atol:g} rtol={args.rtol:g}")

    if not drifts:
        print("OK: candidate matches committed baseline within tolerance.")
        return 0

    drifts.sort(key=lambda d: -(d[3] if math.isfinite(d[3]) else float("inf")))
    print(f"DRIFT DETECTED: {len(drifts)} element(s) exceed tolerance (worst first):")
    for p, rv, cv, abs_err, rel_err in drifts[:30]:
        print(f"  {p}: committed={rv:.8g} candidate={cv:.8g} abs={abs_err:.3e} rel={rel_err:.3e}")
    print(
        "\nThe committed MATLAB baseline no longer matches current MATLAB output.\n"
        "Regenerate and commit it:\n"
        "  matlab -batch \"addpath('tests/matlab'); addpath('tests/matlab/helpers'); "
        "export_parity_baseline();\"\n"
        "Then review the diff and commit tests/contracts/baselines/matlab_reference_v1.{json,mat}."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
