"""Integration checks for MATLAB baseline contract parity runner scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "tests" / "contracts"
BASELINE_JSON = CONTRACTS_DIR / "baselines" / "matlab_reference_v1.json"


@pytest.mark.integration
@pytest.mark.parity
def test_contract_runner_includes_parametric_t1_fa_fit(tmp_path: Path) -> None:
    results_path = tmp_path / "contract_runner_results.json"

    generate_cmd = [
        sys.executable,
        str(CONTRACTS_DIR / "generate_python_results.py"),
        "--baseline",
        str(BASELINE_JSON),
        "--output",
        str(results_path),
    ]
    generate = subprocess.run(generate_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert generate.returncode == 0, generate.stdout + "\n" + generate.stderr

    compare_cmd = [
        sys.executable,
        str(CONTRACTS_DIR / "compare_with_matlab_baseline.py"),
        "--baseline",
        str(BASELINE_JSON),
        "--contracts-dir",
        str(CONTRACTS_DIR),
        "--python-results",
        str(results_path),
        "--require-all",
    ]
    compare = subprocess.run(compare_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert compare.returncode == 0, compare.stdout + "\n" + compare.stderr
    assert "t1_fa_fit | PASS" in compare.stdout
