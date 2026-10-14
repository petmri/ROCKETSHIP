#!/usr/bin/env python3
"""Run the PySide6 GUI for the Python DCE pipeline."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "python"))

    from dce_gui import main as gui_main  # pylint: disable=import-outside-toplevel

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())

