#!/usr/bin/env python3
"""Discover BIDS sessions and emit a reusable manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import BidsSession, discover_bids_sessions  # noqa: E402


def _session_to_dict(session: BidsSession) -> Dict[str, Any]:
    return {
        "id": session.id,
        "subject": session.subject,
        "session": session.session,
        "rawdata_path": str(session.rawdata_path),
        "derivatives_path": str(session.derivatives_path),
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "BIDS_test",
        help="BIDS root containing rawdata/ and derivatives/ directories.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path to write the discovery manifest as JSON.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full manifest JSON instead of a concise text summary.",
    )
    parser.add_argument(
        "--require-sessions",
        action="store_true",
        help="Return non-zero when no sessions are discovered.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    bids_root = Path(args.bids_root).expanduser().resolve()

    try:
        sessions = discover_bids_sessions(bids_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    manifest = {
        "bids_root": str(bids_root),
        "session_count": len(sessions),
        "sessions": [_session_to_dict(session) for session in sessions],
    }

    if args.output_json is not None:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.print_json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"discovered {manifest['session_count']} sessions under {manifest['bids_root']}")
        for entry in manifest["sessions"]:
            print(
                f"- {entry['id']}: raw={entry['rawdata_path']} "
                f"derivatives={entry['derivatives_path']}"
            )

    if args.require_sessions and not sessions:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
