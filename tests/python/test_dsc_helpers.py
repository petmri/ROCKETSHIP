"""Unit tests for Python DSC helper ports."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import import_aif, previous_aif  # noqa: E402
from dsc_helpers import last_dim_len, matlab_reshape_linspace  # noqa: E402


def _within_tol(actual: float, expected: float, atol: float, rtol: float) -> bool:
    return abs(actual - expected) <= (atol + rtol * abs(expected))


def _compare_nested_lists(actual, expected, atol: float, rtol: float) -> None:
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _compare_nested_lists(a, e, atol, rtol)
        return

    assert _within_tol(float(actual), float(expected), atol, rtol), (
        f"Mismatch: actual={actual} expected={expected}"
    )


@pytest.mark.unit
def test_import_and_previous_match_matlab_baseline() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["forward_exact"]
    atol = float(tol["atol"])
    rtol = float(tol["rtol"])

    mean_aif = [0.0 + (1.1 / 13.0) * i for i in range(14)]
    bolus_time = 3
    time_vect = [0.0 + 0.1 * i for i in range(19)]
    concentration = matlab_reshape_linspace(0.05, 0.6, 2 * 2 * len(time_vect), (2, 2, len(time_vect)))

    import_out = import_aif(mean_aif, bolus_time, time_vect, concentration, 3.4, 0.03)
    prev_out = previous_aif(import_out[0], import_out[3], bolus_time, import_out[1], import_out[2])

    expected_import = baseline["dsc"]["import_aif"]
    expected_prev = baseline["dsc"]["previous_aif"]

    _compare_nested_lists(import_out[0], expected_import["meanAIF_adjusted"], atol, rtol)
    _compare_nested_lists(import_out[1], expected_import["time_vect"], atol, rtol)
    _compare_nested_lists(import_out[2], expected_import["concentration_array"], atol, rtol)
    _compare_nested_lists(import_out[3], expected_import["meanSignal"], atol, rtol)

    _compare_nested_lists(prev_out[0], expected_prev["meanAIF_adjusted"], atol, rtol)
    _compare_nested_lists(prev_out[1], expected_prev["time_vect"], atol, rtol)
    _compare_nested_lists(prev_out[2], expected_prev["concentration_array"], atol, rtol)


@pytest.mark.unit
def test_import_truncates_last_dimension_when_time_longer_than_aif() -> None:
    mean_aif = [0.2, 0.3, 0.4, 0.5]
    bolus_time = 2
    time_vect = [0.0, 1.0, 2.0, 3.0, 4.0]
    concentration = matlab_reshape_linspace(1.0, 20.0, 2 * 2 * len(time_vect), (2, 2, len(time_vect)))

    adjusted, out_time, out_conc, _ = import_aif(mean_aif, bolus_time, time_vect, concentration, 3.4, 0.03)
    assert len(adjusted) == len(mean_aif) - (bolus_time - 1)
    assert len(out_time) == len(adjusted)
    assert last_dim_len(out_conc) == len(adjusted)
