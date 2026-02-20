"""Run OSIPI reliability summary checks with explicit threshold reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from osipi_si_to_conc_helpers import as_summary_payload, compute_si_to_conc_metrics, peer_si_to_conc_metrics


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["si-to-conc"],
        default="si-to-conc",
        help="OSIPI reliability suite to run (default: si-to-conc).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Numerical slack added to peer thresholds (default: 1e-12).",
    )
    parser.add_argument(
        "--summary-json",
        default=".pytest_cache/osipi_summaries/osipi_si_to_conc_summary.json",
        help="Write summary JSON to this path.",
    )
    return parser.parse_args()


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _render_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    widths = {h: len(h) for h in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def _line(values: dict[str, str]) -> str:
        return "  ".join(str(values.get(h, "")).ljust(widths[h]) for h in headers)

    lines = [_line({h: h for h in headers}), _line({h: "-" * widths[h] for h in headers})]
    lines.extend(_line(row) for row in rows)
    return "\n".join(lines)


def _print_si_to_conc_summary(payload: dict[str, Any]) -> None:
    ours = payload["metrics"]["ours"]
    peer = payload["metrics"]["peer"]
    limits = payload["metrics"]["limits"]
    checks = payload["checks"]

    rows = [
        {
            "metric": "MAE",
            "ours": _format_float(ours["mae"]),
            "peer": _format_float(peer["mae"]),
            "limit": _format_float(limits["mae"]),
            "pass": str(bool(checks["mae"])).lower(),
        },
        {
            "metric": "P95 abs err",
            "ours": _format_float(ours["p95_abs_error"]),
            "peer": _format_float(peer["p95_abs_error"]),
            "limit": _format_float(limits["p95_abs_error"]),
            "pass": str(bool(checks["p95_abs_error"])).lower(),
        },
        {
            "metric": "Max abs err",
            "ours": _format_float(ours["max_abs_error"]),
            "peer": _format_float(peer["max_abs_error"]),
            "limit": _format_float(limits["max_abs_error"]),
            "pass": str(bool(checks["max_abs_error"])).lower(),
        },
    ]
    print("[OSIPI-RELIABILITY] suite=si-to-conc")
    print(f"[OSIPI-RELIABILITY] n={int(float(ours['n']))} epsilon={_format_float(payload['epsilon'])}")
    print(_render_table(rows, ["metric", "ours", "peer", "limit", "pass"]))


def main() -> int:
    root = _repo_root()
    args = _parse_args()

    if args.suite != "si-to-conc":
        print(f"Unsupported suite: {args.suite}", file=sys.stderr)
        return 2

    ours = compute_si_to_conc_metrics()
    peer = peer_si_to_conc_metrics()
    payload = as_summary_payload(ours, peer, epsilon=float(args.epsilon))

    summary_path = Path(args.summary_json).expanduser()
    if not summary_path.is_absolute():
        summary_path = (root / summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_si_to_conc_summary(payload)
    print(f"[OSIPI-RELIABILITY] summary_json={summary_path}")

    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
