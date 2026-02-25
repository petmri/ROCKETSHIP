"""Unit tests for Python parametric model ports."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import t1_fa_linear_fit, t1_fa_nonlinear_fit, t1_fa_two_point_fit, t2_linear_fast  # noqa: E402


def _within_tol(actual: float, expected: float, atol: float, rtol: float) -> bool:
    return abs(actual - expected) <= (atol + rtol * abs(expected))


@pytest.mark.unit
@pytest.mark.parity
def test_t2_linear_fast_matches_matlab_baseline() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    te = [10.0, 20.0, 40.0, 60.0]
    true_t2 = 85.0
    rho = 900.0
    si_t2 = [rho * math.exp(-t / true_t2) for t in te]

    expected = baseline["parametric"]["t2_linear_fast"]
    actual = t2_linear_fast(te, si_t2)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
@pytest.mark.parity
def test_t1_fa_linear_fit_matches_matlab_baseline() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    fa = [2.0, 5.0, 10.0, 15.0]
    tr = 8.0
    true_t1 = 1300.0
    m0 = 1100.0
    theta = [f * (math.pi / 180.0) for f in fa]
    si_t1 = [
        m0
        * ((1.0 - math.exp(-tr / true_t1)) * math.sin(th))
        / (1.0 - math.exp(-tr / true_t1) * math.cos(th))
        for th in theta
    ]

    expected = baseline["parametric"]["t1_fa_linear_fit"]
    actual = t1_fa_linear_fit(fa, si_t1, tr)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_t1_fa_nonlinear_fit_recovers_synthetic_t1() -> None:
    fa = [2.0, 5.0, 10.0, 15.0]
    tr = 8.0
    true_t1 = 1300.0
    m0 = 1100.0
    theta = [f * (math.pi / 180.0) for f in fa]
    si_t1 = [
        m0
        * ((1.0 - math.exp(-tr / true_t1)) * math.sin(th))
        / (1.0 - math.exp(-tr / true_t1) * math.cos(th))
        for th in theta
    ]

    actual = t1_fa_nonlinear_fit(fa, si_t1, tr)
    assert _within_tol(float(actual[0]), true_t1, atol=1e-1, rtol=1e-3)


@pytest.mark.unit
@pytest.mark.parity
def test_t1_fa_nonlinear_fit_matches_matlab_reference() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    fa = [2.0, 5.0, 10.0, 15.0]
    tr = 8.0
    true_t1 = 1300.0
    m0 = 1100.0
    theta = [f * (math.pi / 180.0) for f in fa]
    si_t1 = [
        m0
        * ((1.0 - math.exp(-tr / true_t1)) * math.sin(th))
        / (1.0 - math.exp(-tr / true_t1) * math.cos(th))
        for th in theta
    ]

    actual = t1_fa_nonlinear_fit(fa, si_t1, tr)
    expected = baseline["parametric"]["t1_fa_fit"]
    # Compare [T1, M0, r_squared, CI_low, CI_high, sse]. CI from SciPy and MATLAB
    # can differ in implementation details, so parity check currently gates
    # T1/M0/r_squared/sse and leaves CI for future tightening.
    for idx in (0, 1, 2, 5):
        a = actual[idx]
        e = expected[idx]
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_t1_fa_two_point_fit_recovers_synthetic_t1() -> None:
    fa = [2.0, 5.0, 10.0, 15.0]
    tr = 8.0
    true_t1 = 1300.0
    m0 = 1100.0
    theta = [f * (math.pi / 180.0) for f in fa]
    si_t1 = [
        m0
        * ((1.0 - math.exp(-tr / true_t1)) * math.sin(th))
        / (1.0 - math.exp(-tr / true_t1) * math.cos(th))
        for th in theta
    ]

    actual = t1_fa_two_point_fit(fa, si_t1, tr)
    assert _within_tol(float(actual[0]), true_t1, atol=1e-1, rtol=1e-3)
