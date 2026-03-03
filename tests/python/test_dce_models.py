"""Unit tests for Python DCE model ports."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import (  # noqa: E402
    model_2cxm_cfit,
    model_2cxm_fit,
    model_extended_tofts_cfit,
    model_extended_tofts_fit,
    model_fxr_cfit,
    model_fxr_fit,
    model_patlak_cfit,
    model_patlak_fit,
    model_patlak_linear,
    model_tissue_uptake_cfit,
    model_tissue_uptake_fit,
    model_tofts_cfit,
    model_tofts_fit,
    model_vp_cfit,
    model_vp_fit,
)


def _within_tol(actual: float, expected: float, atol: float, rtol: float) -> bool:
    return abs(actual - expected) <= (atol + rtol * abs(expected))


def _mse(a: list[float], b: list[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / len(a)


@pytest.mark.unit
def test_tofts_zero_cp_returns_zero() -> None:
    timer = [0.0, 0.1, 0.2, 0.3]
    cp = [0.0, 0.0, 0.0, 0.0]
    out = model_tofts_cfit(0.03, 0.25, cp, timer)
    assert len(out) == len(timer)
    assert all(v == 0.0 for v in out)


@pytest.mark.unit
def test_tofts_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["forward_exact"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    expected = baseline["dce"]["forward"]["tofts"]

    ktrans = float(baseline["dce"]["inverse"]["tofts_fit"][0])
    ve = float(baseline["dce"]["inverse"]["tofts_fit"][1])

    actual = model_tofts_cfit(ktrans, ve, cp, timer)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_patlak_zero_cp_returns_zero() -> None:
    timer = [0.0, 0.1, 0.2, 0.3]
    cp = [0.0, 0.0, 0.0, 0.0]
    out = model_patlak_cfit(0.03, 0.05, cp, timer)
    assert len(out) == len(timer)
    assert all(v == 0.0 for v in out)


@pytest.mark.unit
def test_patlak_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["forward_exact"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    expected = baseline["dce"]["forward"]["patlak"]

    ktrans = float(baseline["dce"]["inverse"]["patlak_linear"][0])
    vp = float(baseline["dce"]["inverse"]["patlak_linear"][1])

    actual = model_patlak_cfit(ktrans, vp, cp, timer)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_extended_tofts_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["forward_exact"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    expected = baseline["dce"]["forward"]["extended_tofts"]

    fit_vals = baseline["dce"]["inverse"]["extended_tofts_fit"]
    ktrans = float(fit_vals[0])
    ve = float(fit_vals[1])
    vp = float(fit_vals[2])

    actual = model_extended_tofts_cfit(ktrans, ve, vp, cp, timer)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_vp_forward_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["forward_exact"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    vp = float(baseline["dce"]["params"]["vp"])
    expected = baseline["dce"]["forward"]["vp"]

    actual = model_vp_cfit(vp, cp, timer)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_tissue_uptake_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["forward_exact"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    params = baseline["dce"]["params"]
    ktrans = float(params["ktrans"])
    fp = float(params["fp"])
    tp = float(params["tp"])
    expected = baseline["dce"]["forward"]["tissue_uptake"]

    actual = model_tissue_uptake_cfit(ktrans, fp, tp, cp, timer)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_twocxm_forward_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["forward_exact"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    params = baseline["dce"]["params"]
    ktrans = float(params["ktrans"])
    ve = float(params["ve"])
    vp = float(params["vp"])
    fp = float(params["fp"])
    expected = baseline["dce"]["forward"]["twocxm"]

    actual = model_2cxm_cfit(ktrans, ve, vp, fp, cp, timer)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_fxr_forward_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["forward_exact"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    params = baseline["dce"]["params"]
    expected = baseline["dce"]["forward"]["fxr"]

    actual = model_fxr_cfit(
        float(params["ktrans"]),
        float(params["ve"]),
        float(params["tau"]),
        cp,
        timer,
        float(params["R1o"]),
        float(params["R1i"]),
        float(params["r1"]),
        float(params["fw"]),
    )
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_patlak_linear_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    ct = baseline["dce"]["forward"]["patlak"]
    expected = baseline["dce"]["inverse"]["patlak_linear"]

    actual = model_patlak_linear(ct, cp, timer)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_patlak_fit_recovers_forward_params_and_improves_sse() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    ct = baseline["dce"]["forward"]["patlak"]
    params = baseline["dce"]["params"]
    ktrans_true = float(params["ktrans"])
    vp_true = float(params["vp"])

    fit_nonlinear = model_patlak_fit(ct, cp, timer)
    fit_linear = model_patlak_linear(ct, cp, timer)

    assert _within_tol(float(fit_nonlinear[0]), ktrans_true, atol=5e-4, rtol=0.05)
    assert _within_tol(float(fit_nonlinear[1]), vp_true, atol=5e-3, rtol=0.1)
    assert float(fit_nonlinear[2]) <= float(fit_linear[2]) + 1e-12


@pytest.mark.unit
def test_tofts_fit_inverse_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    ct = baseline["dce"]["forward"]["tofts"]
    expected = baseline["dce"]["inverse"]["tofts_fit"]

    actual = model_tofts_fit(ct, cp, timer)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_vp_fit_inverse_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    ct = baseline["dce"]["forward"]["vp"]
    expected = baseline["dce"]["inverse"]["vp_fit"]

    actual = model_vp_fit(ct, cp, timer)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_tissue_uptake_fit_inverse_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    ct = baseline["dce"]["forward"]["tissue_uptake"]
    expected = baseline["dce"]["inverse"]["tissue_uptake_fit"]

    actual = model_tissue_uptake_fit(ct, cp, timer)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_twocxm_fit_inverse_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    ct = baseline["dce"]["forward"]["twocxm"]
    expected = baseline["dce"]["inverse"]["twocxm_fit"]

    actual = model_2cxm_fit(ct, cp, timer)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_fxr_fit_inverse_matches_matlab_baseline_profile() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
    tol = tolerances["fit_recovery"]

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    ct = baseline["dce"]["forward"]["fxr"]
    params = baseline["dce"]["params"]
    expected = baseline["dce"]["inverse"]["fxr_fit"]

    actual = model_fxr_fit(
        ct,
        cp,
        timer,
        float(params["R1o"]),
        float(params["R1i"]),
        float(params["r1"]),
        float(params["fw"]),
    )

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])), (
            f"Mismatch: actual={a} expected={e}"
        )


@pytest.mark.unit
def test_tissue_uptake_fit_recovers_synthetic_no_and_low_noise() -> None:
    timer = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.2]
    cp = [0.0, 0.4, 0.9, 1.2, 1.0, 0.82, 0.63, 0.49, 0.35, 0.25, 0.18]
    ktrans_true = 0.045
    fp_true = 0.36
    tp_true = 0.18

    clean = model_tissue_uptake_cfit(ktrans_true, fp_true, tp_true, cp, timer)
    vp_true = (fp_true + (ktrans_true * fp_true / (fp_true - ktrans_true))) * tp_true

    rng = random.Random(7)
    for noise_std in (0.0, 2.5e-4):
        noisy = [float(v + rng.gauss(0.0, noise_std)) for v in clean]
        fit = model_tissue_uptake_fit(noisy, cp, timer)

        assert _within_tol(float(fit[0]), ktrans_true, atol=2e-3, rtol=0.1)
        assert _within_tol(float(fit[1]), fp_true, atol=3e-2, rtol=0.15)
        assert _within_tol(float(fit[2]), vp_true, atol=2e-2, rtol=0.2)


@pytest.mark.unit
def test_tissue_uptake_fit_is_unit_consistent_for_timer_and_prefs() -> None:
    timer_min = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.2]
    timer_sec = [t * 60.0 for t in timer_min]
    cp = [0.0, 0.4, 0.9, 1.2, 1.0, 0.82, 0.63, 0.49, 0.35, 0.25, 0.18]
    ct = model_tissue_uptake_cfit(0.045, 0.36, 0.18, cp, timer_min)

    prefs_min = {
        "time_unit": "minutes",
        "initial_value_ktrans": 0.04,
        "initial_value_fp": 0.34,
        "initial_value_tp": 0.16,
        "lower_limit_ktrans": 1e-7,
        "upper_limit_ktrans": 2.0,
        "lower_limit_fp": 1e-3,
        "upper_limit_fp": 100.0,
        "lower_limit_tp": 0.0,
        "upper_limit_tp": 1e6,
    }
    prefs_sec = {
        "time_unit": "seconds",
        "initial_value_ktrans": prefs_min["initial_value_ktrans"] / 60.0,
        "initial_value_fp": prefs_min["initial_value_fp"] / 60.0,
        "initial_value_tp": prefs_min["initial_value_tp"] * 60.0,
        "lower_limit_ktrans": prefs_min["lower_limit_ktrans"] / 60.0,
        "upper_limit_ktrans": prefs_min["upper_limit_ktrans"] / 60.0,
        "lower_limit_fp": prefs_min["lower_limit_fp"] / 60.0,
        "upper_limit_fp": prefs_min["upper_limit_fp"] / 60.0,
        "lower_limit_tp": prefs_min["lower_limit_tp"] * 60.0,
        "upper_limit_tp": prefs_min["upper_limit_tp"] * 60.0,
    }

    fit_min = model_tissue_uptake_fit(ct, cp, timer_min, prefs_min)
    fit_sec = model_tissue_uptake_fit(ct, cp, timer_sec, prefs_sec)

    assert _within_tol(float(fit_sec[0]) * 60.0, float(fit_min[0]), atol=5e-4, rtol=0.05)
    assert _within_tol(float(fit_sec[1]) * 60.0, float(fit_min[1]), atol=5e-3, rtol=0.05)
    assert _within_tol(float(fit_sec[2]), float(fit_min[2]), atol=5e-3, rtol=0.05)


@pytest.mark.unit
def test_twocxm_fit_is_unit_consistent_for_timer_and_prefs() -> None:
    timer_min = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.2]
    timer_sec = [t * 60.0 for t in timer_min]
    cp = [0.0, 0.4, 0.9, 1.2, 1.0, 0.82, 0.63, 0.49, 0.35, 0.25, 0.18]
    ct = model_2cxm_cfit(0.03, 0.24, 0.045, 0.31, cp, timer_min)

    prefs_min = {
        "time_unit": "minutes",
        "initial_value_ktrans": 0.03,
        "initial_value_ve": 0.2,
        "initial_value_vp": 0.04,
        "initial_value_fp": 0.3,
        "lower_limit_ktrans": 1e-7,
        "upper_limit_ktrans": 2.0,
        "lower_limit_ve": 0.02,
        "upper_limit_ve": 1.0,
        "lower_limit_vp": 1e-3,
        "upper_limit_vp": 1.0,
        "lower_limit_fp": 1e-3,
        "upper_limit_fp": 100.0,
    }
    prefs_sec = {
        "time_unit": "seconds",
        "initial_value_ktrans": prefs_min["initial_value_ktrans"] / 60.0,
        "initial_value_ve": prefs_min["initial_value_ve"],
        "initial_value_vp": prefs_min["initial_value_vp"],
        "initial_value_fp": prefs_min["initial_value_fp"] / 60.0,
        "lower_limit_ktrans": prefs_min["lower_limit_ktrans"] / 60.0,
        "upper_limit_ktrans": prefs_min["upper_limit_ktrans"] / 60.0,
        "lower_limit_ve": prefs_min["lower_limit_ve"],
        "upper_limit_ve": prefs_min["upper_limit_ve"],
        "lower_limit_vp": prefs_min["lower_limit_vp"],
        "upper_limit_vp": prefs_min["upper_limit_vp"],
        "lower_limit_fp": prefs_min["lower_limit_fp"] / 60.0,
        "upper_limit_fp": prefs_min["upper_limit_fp"] / 60.0,
    }

    fit_min = model_2cxm_fit(ct, cp, timer_min, prefs_min)
    fit_sec = model_2cxm_fit(ct, cp, timer_sec, prefs_sec)

    assert _within_tol(float(fit_sec[0]) * 60.0, float(fit_min[0]), atol=5e-4, rtol=0.05)
    assert _within_tol(float(fit_sec[1]), float(fit_min[1]), atol=5e-3, rtol=0.05)
    assert _within_tol(float(fit_sec[2]), float(fit_min[2]), atol=5e-3, rtol=0.05)
    assert _within_tol(float(fit_sec[3]) * 60.0, float(fit_min[3]), atol=5e-3, rtol=0.05)


@pytest.mark.unit
def test_model_fit_rejects_algorithm_override_prefs() -> None:
    timer = [0.0, 0.1, 0.2, 0.3]
    cp = [0.0, 0.4, 0.2, 0.1]
    ct = [0.0, 0.01, 0.015, 0.017]

    with pytest.raises(ValueError, match="does not support 'fit_algorithm'"):
        model_2cxm_fit(ct, cp, timer, {"fit_algorithm": "legacy"})

    with pytest.raises(ValueError, match="does not support 'fit_algorithm'"):
        model_tissue_uptake_fit(ct, cp, timer, {"fit_algorithm": "osipi_quick"})


@pytest.mark.unit
def test_primary_dce_fits_handle_nonuniform_timer_and_low_snr() -> None:
    timer = [0.0, 0.07, 0.11, 0.19, 0.31, 0.47, 0.66, 0.88, 1.17, 1.53, 1.96]
    cp = [0.0, 0.25, 0.65, 1.05, 1.2, 1.08, 0.9, 0.72, 0.53, 0.38, 0.27]
    rng = random.Random(31)

    tofts_clean = model_tofts_cfit(0.045, 0.32, cp, timer)
    ex_clean = model_extended_tofts_cfit(0.04, 0.28, 0.06, cp, timer)
    patlak_clean = model_patlak_cfit(0.018, 0.04, cp, timer)

    def _noisy(signal: list[float], frac: float = 0.012) -> list[float]:
        sigma = max(signal) * frac
        return [float(v + rng.gauss(0.0, sigma)) for v in signal]

    tofts_noisy = _noisy(tofts_clean)
    tofts_fit = model_tofts_fit(tofts_noisy, cp, timer)
    tofts_pred = model_tofts_cfit(float(tofts_fit[0]), float(tofts_fit[1]), cp, timer)
    tofts_baseline = model_tofts_cfit(2e-4, 0.2, cp, timer)

    ex_noisy = _noisy(ex_clean)
    ex_fit = model_extended_tofts_fit(ex_noisy, cp, timer)
    ex_pred = model_extended_tofts_cfit(float(ex_fit[0]), float(ex_fit[1]), float(ex_fit[2]), cp, timer)
    ex_baseline = model_extended_tofts_cfit(2e-4, 0.2, 0.02, cp, timer)

    patlak_noisy = _noisy(patlak_clean)
    patlak_fit = model_patlak_fit(patlak_noisy, cp, timer)
    patlak_pred = model_patlak_cfit(float(patlak_fit[0]), float(patlak_fit[1]), cp, timer)
    patlak_baseline = model_patlak_cfit(2e-4, 0.02, cp, timer)

    assert all(math.isfinite(float(v)) for v in tofts_fit[:2])
    assert all(math.isfinite(float(v)) for v in ex_fit[:3])
    assert all(math.isfinite(float(v)) for v in patlak_fit[:2])

    # Regression guard: even on low-SNR inputs with irregular timing, optimized fits
    # should materially improve reconstruction versus default seed parameters.
    assert _mse(tofts_pred, tofts_noisy) <= (_mse(tofts_baseline, tofts_noisy) * 0.05)
    assert _mse(ex_pred, ex_noisy) <= (_mse(ex_baseline, ex_noisy) * 0.05)
    assert _mse(patlak_pred, patlak_noisy) <= (_mse(patlak_baseline, patlak_noisy) * 0.05)


@pytest.mark.unit
def test_primary_dce_fits_honor_custom_bounds() -> None:
    timer = [0.0, 0.07, 0.11, 0.19, 0.31, 0.47, 0.66, 0.88, 1.17, 1.53, 1.96]
    cp = [0.0, 0.25, 0.65, 1.05, 1.2, 1.08, 0.9, 0.72, 0.53, 0.38, 0.27]
    rng = random.Random(23)

    def _noisy(signal: list[float], frac: float = 0.012) -> list[float]:
        sigma = max(signal) * frac
        return [float(v + rng.gauss(0.0, sigma)) for v in signal]

    tofts_ct = _noisy(model_tofts_cfit(0.045, 0.32, cp, timer))
    tofts_prefs = {
        "lower_limit_ktrans": 0.03,
        "upper_limit_ktrans": 0.06,
        "initial_value_ktrans": 0.04,
        "lower_limit_ve": 0.22,
        "upper_limit_ve": 0.45,
        "initial_value_ve": 0.3,
    }
    tofts_fit = model_tofts_fit(tofts_ct, cp, timer, tofts_prefs)
    assert tofts_prefs["lower_limit_ktrans"] <= float(tofts_fit[0]) <= tofts_prefs["upper_limit_ktrans"]
    assert tofts_prefs["lower_limit_ve"] <= float(tofts_fit[1]) <= tofts_prefs["upper_limit_ve"]

    ex_ct = _noisy(model_extended_tofts_cfit(0.04, 0.28, 0.06, cp, timer))
    ex_prefs = {
        "lower_limit_ktrans": 0.025,
        "upper_limit_ktrans": 0.06,
        "initial_value_ktrans": 0.038,
        "lower_limit_ve": 0.12,
        "upper_limit_ve": 0.8,
        "initial_value_ve": 0.25,
        "lower_limit_vp": 0.03,
        "upper_limit_vp": 0.12,
        "initial_value_vp": 0.05,
    }
    ex_fit = model_extended_tofts_fit(ex_ct, cp, timer, ex_prefs)
    assert ex_prefs["lower_limit_ktrans"] <= float(ex_fit[0]) <= ex_prefs["upper_limit_ktrans"]
    assert ex_prefs["lower_limit_ve"] <= float(ex_fit[1]) <= ex_prefs["upper_limit_ve"]
    assert ex_prefs["lower_limit_vp"] <= float(ex_fit[2]) <= ex_prefs["upper_limit_vp"]

    patlak_ct = _noisy(model_patlak_cfit(0.018, 0.04, cp, timer))
    patlak_prefs = {
        "lower_limit_ktrans": 0.01,
        "upper_limit_ktrans": 0.03,
        "initial_value_ktrans": 0.015,
        "lower_limit_vp": 0.02,
        "upper_limit_vp": 0.08,
        "initial_value_vp": 0.03,
    }
    patlak_fit = model_patlak_fit(patlak_ct, cp, timer, patlak_prefs)
    assert patlak_prefs["lower_limit_ktrans"] <= float(patlak_fit[0]) <= patlak_prefs["upper_limit_ktrans"]
    assert patlak_prefs["lower_limit_vp"] <= float(patlak_fit[1]) <= patlak_prefs["upper_limit_vp"]
