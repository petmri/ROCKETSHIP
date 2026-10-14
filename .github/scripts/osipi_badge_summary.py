#!/usr/bin/env python3
"""Summarize a pytest JUnit XML run of the OSIPI suite into shields.io badge fields.

Writes `message`, `color` and `failed` as GitHub Actions step outputs (stdout, meant to
be redirected into `$GITHUB_OUTPUT`).

Skipped tests are excluded from the count rather than counted as passes. Six of the
OSIPI tests skip on a GitHub-hosted runner even with the acceleration wheels installed,
because they need real CUDA hardware; counting those as passes would advertise a
conformance level that was never measured.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict


def _totals(results_path: Path) -> Dict[str, int]:
    root = ET.parse(results_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.get(key) or 0)
    return totals


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <junit-xml>", file=sys.stderr)
        return 2

    totals = _totals(Path(argv[1]))
    failed = totals["failures"] + totals["errors"]
    ran = totals["tests"] - totals["skipped"]
    passed = ran - failed

    if ran <= 0:
        message = "no tests run"
        color = "lightgrey"
    else:
        message = f"{passed}/{ran} passing"
        if totals["skipped"]:
            message += f" ({totals['skipped']} skipped)"
        color = "brightgreen" if failed == 0 else "red"

    print(f"message={message}")
    print(f"color={color}")
    print(f"failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
