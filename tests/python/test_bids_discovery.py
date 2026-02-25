"""Tests for reusable BIDS discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import discover_bids_sessions  # noqa: E402


@pytest.mark.unit
def test_discover_bids_sessions_on_bids_example() -> None:
    bids_root = REPO_ROOT / "tests" / "data" / "BIDS_example"
    sessions = discover_bids_sessions(bids_root)
    assert len(sessions) == 1
    assert sessions[0].subject == "sub-01"
    assert sessions[0].session == "ses-01"
    assert sessions[0].rawdata_path.exists()
    assert sessions[0].derivatives_path.exists()


@pytest.mark.unit
def test_discover_bids_sessions_on_bids_test_multi_dataset() -> None:
    bids_root = REPO_ROOT / "tests" / "data" / "BIDS_test"
    sessions = discover_bids_sessions(bids_root)
    assert len(sessions) >= 3
    for entry in sessions:
        assert entry.subject.startswith("sub-")
        assert entry.rawdata_path.exists()
        assert entry.derivatives_path.exists()


@pytest.mark.unit
def test_run_bids_discovery_script_writes_manifest(tmp_path: Path) -> None:
    script_path = REPO_ROOT / "run_bids_discovery.py"
    output_path = tmp_path / "bids_manifest.json"
    bids_root = REPO_ROOT / "tests" / "data" / "BIDS_example"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--bids-root",
            str(bids_root),
            "--output-json",
            str(output_path),
            "--require-sessions",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["session_count"] == 1
    assert payload["sessions"][0]["id"] == "sub-01_ses-01"
