#!/usr/bin/env python3
"""Run the Python DCE in-memory CLI pipeline scaffold."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "python"))

    from rocketship.dce_cli import main as cli_main  # pylint: disable=import-outside-toplevel

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
