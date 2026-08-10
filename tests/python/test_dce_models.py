"""Unit tests for Python DCE model ports."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import (  # noqa: E402
    cxm2_curve,
    cxm2_curve_batch,
    model_2cxm_fit,
    model_extended_tofts_cfit,
    model_extended_tofts_cfit_batch,
    model_extended_tofts_fit,
    model_fxr_cfit,
    model_fxr_cfit_batch,
    model_fxr_fit,
    model_patlak_cfit,
    model_patlak_cfit_batch,
    model_patlak_fit,
    model_patlak_linear,
    model_patlak_linear_batch,
    model_tissue_uptake_cfit,
    model_tissue_uptake_cfit_batch,
    model_tissue_uptake_fit,
    model_tofts_cfit,
    model_tofts_cfit_batch,
    model_tofts_fit,
    model_vp_cfit,
    model_vp_fit,
)


def _within_tol(actual: float, expected: float, atol: float, rtol: float) -> bool:
    return abs(actual - expected) <= (atol + rtol * abs(expected))


def _mse(a: list[float], b: list[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / len(a)


def _twocxm_samples(ktrans, ve, vp, fp, cp, timer) -> list[float]:
    """One 2CXM curve at the acquired times, via the single forward model.

    2CXM is evaluated on a dense 0.1 s grid rather than the acquired timebase
    (see `dce_models.DenseResampleGrid`), so producing acquired-time samples
    means building the grid and interpolating back. `cxm2_curve` is
    parametrized on E = Ktrans / Fp.
    """
    from dce_models import build_dense_resample_grid

    grid = build_dense_resample_grid(cp, list(timer))
    assert grid is not None, "test timer cannot support a dense grid"
    dense = cxm2_curve(vp, ve, fp, float(ktrans) / float(fp), grid)
    return [float(v) for v in grid.to_samples(dense)]


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

    # Forward parity uses the known fixture parameters (what MATLAB synthesized the
    # baseline from), not MATLAB's recovered fit, so this isolates forward-model
    # parity from fit-recovery behavior.
    ktrans = float(baseline["dce"]["params"]["ktrans"])
    ve = float(baseline["dce"]["params"]["ve"])

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

    # Forward parity uses the known fixture parameters, not MATLAB's recovered fit.
    ktrans = float(baseline["dce"]["params"]["ktrans"])
    vp = float(baseline["dce"]["params"]["vp"])

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

    # Forward parity uses the known fixture parameters, not MATLAB's recovered fit.
    ktrans = float(baseline["dce"]["params"]["ktrans"])
    ve = float(baseline["dce"]["params"]["ve"])
    vp = float(baseline["dce"]["params"]["vp"])

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
    tol = tolerances["fit_recovery_strict"]

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
def test_patlak_linear_batch_is_bit_identical_to_scalar() -> None:
    """The batch seeder replaced a per-voxel scalar loop; drift here shifts fit seeds.

    Summation order is the fragile part -- see `_sequential_sum_over_time` in
    `python/dce_models.py`. Exact equality is the assertion on purpose.
    """
    rng = np.random.default_rng(1234)
    for _ in range(25):
        n_time = int(rng.integers(2, 48))
        n_voxels = int(rng.integers(1, 24))
        timer = np.sort(rng.random(n_time)) * 12.0
        cp = rng.random(n_time) * 3.0
        cp[rng.random(n_time) < 0.2] = 0.0  # forces the |cp| <= tiny branch
        ct = rng.normal(0.0, 1.0, (n_time, n_voxels))
        ct[rng.random((n_time, n_voxels)) < 0.05] = np.nan

        with np.errstate(all="ignore"):
            batch = model_patlak_linear_batch(ct, cp, timer)
        for i in range(n_voxels):
            scalar = np.asarray(
                model_patlak_linear(list(ct[:, i]), list(cp), list(timer)), dtype=np.float64
            )
            assert np.array_equal(scalar, batch[i], equal_nan=True), (
                f"voxel {i}: scalar={scalar} batch={batch[i]}"
            )


@pytest.mark.unit
def test_patlak_linear_batch_accepts_single_voxel_vector() -> None:
    baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]
    ct = baseline["dce"]["forward"]["patlak"]

    batch = model_patlak_linear_batch(np.asarray(ct, dtype=np.float64), cp, timer)
    assert batch.shape == (1, 7)
    scalar = np.asarray(model_patlak_linear(ct, cp, timer), dtype=np.float64)
    assert np.array_equal(scalar, batch[0], equal_nan=True)


def _degenerate_params(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    """Random parameters salted with the values that trip the scalar guards."""
    v = rng.uniform(lo, hi, n)
    hit = rng.random(n) < 0.35
    pick = rng.integers(0, 4, n)
    v = np.where(hit & (pick == 0), 0.0, v)
    v = np.where(hit & (pick == 1), -v, v)
    v = np.where(hit & (pick == 2), 1e-18, v)
    v = np.where(hit & (pick == 3), 1e12, v)
    return v


def _scalar_column(fn, n_time: int, *args) -> np.ndarray:
    """Evaluate a scalar cfit the way `_predict_curves_batch`'s callers do.

    The float() casts matter: numpy scalars return inf on divide-by-zero where
    Python floats raise, and the batch functions reproduce the Python-float
    behaviour because that is what the pipeline actually passes.
    """
    cast = [float(a) if isinstance(a, (int, float, np.floating)) else a for a in args]
    try:
        out = np.asarray(fn(*cast), dtype=np.float64)
    except Exception:
        return np.full(n_time, np.nan)
    return out if out.shape == (n_time,) else np.full(n_time, np.nan)


# Values chosen to trip every guard in the batched curve evaluators: exact zeros,
# negatives (negative lambda -> exp overflow), denormal and huge magnitudes
# (float ** 2 overflow, zero denominators), plus two ordinary values.
_PATHOLOGICAL = (0.0, -0.5, 1e-18, 1e-200, 0.3, 1.0, 1e12, 1e200)


def _assert_batch_matches_scalar(name, batch, scalar_fn, combos, tail, n_time) -> None:
    assert batch.shape == (n_time, len(combos)), name
    for i, params in enumerate(combos):
        expected = _scalar_column(scalar_fn, n_time, *params, *tail)
        actual = batch[:, i]
        assert np.array_equal(np.isnan(expected), np.isnan(actual)), (
            f"{name} params={params}: validity differs\n"
            f"  scalar={expected}\n  batch={actual}"
        )
        good = ~np.isnan(expected)
        # Values are bit-identical in practice; the tolerance exists only so a
        # last-bit difference in a libm routine cannot fail this on some other
        # platform. It is orders of magnitude tighter than any real regression.
        np.testing.assert_allclose(
            actual[good], expected[good], rtol=1e-9, atol=0.0,
            err_msg=f"{name} params={params}",
        )


@pytest.mark.unit
def test_cfit_batch_matches_scalar_on_pathological_grid() -> None:
    """The batched curve evaluators replaced a per-voxel loop in the post-fit path.

    The fragile part is agreeing on *which* voxels are unusable. The scalar path
    raises where numpy stays silent -- ZeroDivisionError on float division by zero,
    OverflowError from `math.exp` and from squaring a finite float past DBL_MAX,
    ValueError from a negative `math.sqrt` -- and each of those is a separate mask
    in the batch versions. Random parameters reach some of those branches only
    rarely, so this walks an exhaustive grid of pathological values instead.

    The validity mask is what is asserted strictly; values carry a loose tolerance
    because a last-bit disagreement between libm and numpy is not a regression.
    """
    timer = np.array([0.0, 0.5, 1.1, 1.9, 2.4, 3.3], dtype=np.float64)
    cp = np.array([0.0, 2.1, 3.4, 2.2, 1.5, 0.9], dtype=np.float64)
    cp_l, t_l = list(cp), list(timer)
    n_time = timer.size
    relaxivity, fw = 4.5, 0.8

    def grid(n: int) -> tuple[list[tuple[float, ...]], list[np.ndarray]]:
        combos = list(itertools.product(_PATHOLOGICAL, repeat=n))
        cols = [np.array([c[j] for c in combos], dtype=np.float64) for j in range(n)]
        return combos, cols

    two, (k2, x2) = grid(2)
    three, (k3, x3, y3) = grid(3)
    four, (k4, x4, y4, z4) = grid(4)

    _assert_batch_matches_scalar(
        "tofts", model_tofts_cfit_batch(k2, x2, cp, timer),
        model_tofts_cfit, two, (cp_l, t_l), n_time,
    )
    _assert_batch_matches_scalar(
        "patlak", model_patlak_cfit_batch(k2, x2, cp, timer),
        model_patlak_cfit, two, (cp_l, t_l), n_time,
    )
    _assert_batch_matches_scalar(
        "ex_tofts", model_extended_tofts_cfit_batch(k3, x3, y3, cp, timer),
        model_extended_tofts_cfit, three, (cp_l, t_l), n_time,
    )
    _assert_batch_matches_scalar(
        "tissue_uptake", model_tissue_uptake_cfit_batch(k3, x3, y3, cp, timer),
        model_tissue_uptake_cfit, three, (cp_l, t_l), n_time,
    )
    # fxr's fourth grid axis is r1o, which it also uses as r1i.
    _assert_batch_matches_scalar(
        "fxr",
        model_fxr_cfit_batch(k4, x4, y4, cp, timer, z4, z4, relaxivity, fw),
        lambda kk, vv, tt, rr, c, t: model_fxr_cfit(
            kk, vv, tt, c, t, rr, rr, relaxivity, fw
        ),
        four, (cp_l, t_l), n_time,
    )


@pytest.mark.unit
def test_cxm2_batch_matches_single_voxel() -> None:
    """`cxm2_curve_batch` must agree column-for-column with the single-voxel model.

    2CXM sits outside `_assert_batch_matches_scalar` because it has no
    acquired-grid scalar counterpart -- the batch evaluates on the dense grid
    and interpolates back, exactly as `_twocxm_samples` does.
    """
    rng = np.random.default_rng(20260802)
    timer = np.linspace(0.0, 5.0, 40)
    cp = 6.0 * np.exp(-1.6 * timer) + 1.4 * np.exp(-0.13 * timer)
    cp[0] = 0.0

    n_vox = 12
    fp = rng.uniform(0.3, 1.6, n_vox)
    ktrans = fp * rng.uniform(0.02, 0.9, n_vox)  # keeps E in range
    ve = rng.uniform(0.05, 0.6, n_vox)
    vp = rng.uniform(0.005, 0.1, n_vox)
    # Degenerate voxels must come back NaN rather than as numbers.
    ktrans[0], fp[1], ve[2], vp[3] = np.nan, 0.0, 0.0, 0.0

    batch = cxm2_curve_batch(ktrans, ve, vp, fp, cp, timer)
    assert batch.shape == (timer.size, n_vox)
    assert np.all(np.isnan(batch[:, :4])), "degenerate parameters must yield NaN columns"

    for i in range(4, n_vox):
        expected = _twocxm_samples(ktrans[i], ve[i], vp[i], fp[i], cp, timer)
        np.testing.assert_allclose(batch[:, i], expected, rtol=1e-12, atol=0.0)


@pytest.mark.unit
def test_cfit_batch_matches_scalar_on_random_params() -> None:
    """Complements the pathological grid with ordinary values at many time lengths."""
    rng = np.random.default_rng(20260730)
    for _ in range(20):
        n_time = int(rng.integers(2, 48))
        n_vox = int(rng.integers(1, 20))
        timer = np.sort(rng.random(n_time)) * 10.0
        cp = rng.random(n_time) * 4.0
        cp_l, t_l = list(cp), list(timer)

        k = _degenerate_params(rng, n_vox, 0.01, 1.0)
        ve = _degenerate_params(rng, n_vox, 0.01, 0.9)
        vp = _degenerate_params(rng, n_vox, 0.001, 0.3)
        fp = _degenerate_params(rng, n_vox, 0.05, 2.0)
        tp = _degenerate_params(rng, n_vox, 0.01, 5.0)
        tau = _degenerate_params(rng, n_vox, 0.01, 3.0)
        r1o = _degenerate_params(rng, n_vox, 0.1, 2.0)
        relaxivity, fw = 4.5, 0.8

        combos2 = list(zip(k, vp))
        _assert_batch_matches_scalar(
            "tofts", model_tofts_cfit_batch(k, ve, cp, timer), model_tofts_cfit,
            list(zip(k, ve)), (cp_l, t_l), n_time,
        )
        _assert_batch_matches_scalar(
            "patlak", model_patlak_cfit_batch(k, vp, cp, timer), model_patlak_cfit,
            combos2, (cp_l, t_l), n_time,
        )
        _assert_batch_matches_scalar(
            "ex_tofts", model_extended_tofts_cfit_batch(k, ve, vp, cp, timer),
            model_extended_tofts_cfit, list(zip(k, ve, vp)), (cp_l, t_l), n_time,
        )
        _assert_batch_matches_scalar(
            "tissue_uptake", model_tissue_uptake_cfit_batch(k, fp, tp, cp, timer),
            model_tissue_uptake_cfit, list(zip(k, fp, tp)), (cp_l, t_l), n_time,
        )
        _assert_batch_matches_scalar(
            "fxr",
            model_fxr_cfit_batch(k, ve, tau, cp, timer, r1o, r1o, relaxivity, fw),
            lambda kk, vv, tt, rr, c, t: model_fxr_cfit(
                kk, vv, tt, c, t, rr, rr, relaxivity, fw
            ),
            list(zip(k, ve, tau, r1o)), (cp_l, t_l), n_time,
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
    tol = tolerances["fit_recovery_strict"]

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
    tol = tolerances["fit_recovery_strict"]

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
    tol = tolerances["fit_recovery_strict"]

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
    # 2CXM parameters (ve/vp/Fp) trade off even on noise-free data, so its inverse
    # is intrinsically ill-conditioned; keep the loose profile here (see
    # tolerance_profiles.json "fit_recovery").
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
    tol = tolerances["fit_recovery_strict"]

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
    ct = _twocxm_samples(0.03, 0.24, 0.045, 0.31, cp, timer_min)

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
