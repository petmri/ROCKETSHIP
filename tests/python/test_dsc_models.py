"""Unit tests for Python DSC model ports."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import dsc_convolution_ssvd  # noqa: E402


def _within_tol(actual: float, expected: float, atol: float, rtol: float) -> bool:
    return abs(actual - expected) <= (atol + rtol * abs(expected))


def _compare_nested(test_case: unittest.TestCase, actual, expected, atol: float, rtol: float) -> None:
    if isinstance(expected, list):
        test_case.assertIsInstance(actual, list)
        test_case.assertEqual(len(actual), len(expected))
        for a_item, e_item in zip(actual, expected):
            _compare_nested(test_case, a_item, e_item, atol, rtol)
        return

    test_case.assertTrue(
        _within_tol(float(actual), float(expected), atol, rtol),
        msg=f"Mismatch: actual={actual} expected={expected}",
    )


class TestDscModels(unittest.TestCase):
    def test_ssvd_matches_matlab_baseline_fixture(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["map_regression"]
        atol = float(tol["atol"])
        rtol = float(tol["rtol"])

        time_index = list(range(10))
        concentration = []
        for ix in range(2):
            row = []
            for iy in range(2):
                trace = [math.exp(-(((t - (2 + (ix + 1) + (iy + 1) / 2.0)) ** 2) / 6.0)) for t in time_index]
                row.append(trace)
            concentration.append(row)
        aif = [math.exp(-(((t - 2) ** 2) / 4.0)) for t in time_index]

        cbf, cbv, mtt = dsc_convolution_ssvd(concentration, aif, 0.1, 0.73, 1.04, 20, 1)

        expected = baseline["dsc"]["ssvd_deconvolution"]
        _compare_nested(self, cbf, expected["CBF"], atol, rtol)
        _compare_nested(self, cbv, expected["CBV"], atol, rtol)
        _compare_nested(self, mtt, expected["MTT"], atol, rtol)


if __name__ == "__main__":
    unittest.main()
