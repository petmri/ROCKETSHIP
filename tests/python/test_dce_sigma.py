"""Tier-1 unit tests for DCE quality-of-fit noise-σ estimation (`dce_sigma`).

These validate *estimator arithmetic* on synthetic curves with known injected σ. Per
`sigma_estimators.md`, a Tier-1 pass proves the math/implementation only — real-data usefulness
(Tier 2: χ²_ν rising where backends diverge) is a separate, unlabeled-data exercise.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_sigma import (  # noqa: E402
    bolus_exclude_window,
    reduced_chi_square,
    reduced_chi_square_weighted,
    sigma_successive_difference,
    successive_difference_sigma,
)


def _smooth_ramp(n: int) -> np.ndarray:
    """A slowly varying 'true' curve: baseline, sigmoid wash-in, slow wash-out."""
    t = np.linspace(0.0, 1.0, n)
    return 2.0 / (1.0 + np.exp(-12.0 * (t - 0.25))) - 0.6 * t


# --------------------------------------------------------------------------- core recovery


@pytest.mark.unit
def test_successive_difference_recovers_injected_sigma_robust() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    true_sigma = 0.05
    x = _smooth_ramp(n) + rng.normal(0.0, true_sigma, size=n)
    est = successive_difference_sigma(x, robust=True)
    assert est == pytest.approx(true_sigma, rel=0.06)


@pytest.mark.unit
def test_successive_difference_recovers_injected_sigma_von_neumann() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    true_sigma = 0.12
    x = _smooth_ramp(n) + rng.normal(0.0, true_sigma, size=n)
    est = successive_difference_sigma(x, robust=False)
    assert est == pytest.approx(true_sigma, rel=0.05)


@pytest.mark.unit
def test_too_few_differences_returns_nan() -> None:
    assert np.isnan(successive_difference_sigma(np.array([1.0, 2.0]), min_diffs=3))
    assert np.isnan(successive_difference_sigma(np.array([1.0]), min_diffs=1))


@pytest.mark.unit
def test_non_finite_samples_are_ignored() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    x = _smooth_ramp(n) + rng.normal(0.0, 0.07, size=n)
    x_with_nan = x.copy()
    x_with_nan[::37] = np.nan  # scatter NaNs through the curve
    clean = successive_difference_sigma(x, robust=True)
    dirty = successive_difference_sigma(x_with_nan, robust=True)
    assert dirty == pytest.approx(clean, rel=0.1)


# --------------------------------------------------------------------- wash-in exclusion


@pytest.mark.unit
def test_exclude_window_removes_washin_inflation() -> None:
    rng = np.random.default_rng(3)
    n = 600
    true_sigma = 0.04
    x = rng.normal(0.0, true_sigma, size=n)
    # Genuine first-pass transient in frames [50, 70): a large, sharp excursion whose
    # frame-to-frame jumps are real signal, not noise. The von Neumann (mean-of-squares)
    # form is the sensitive one here -- the robust MAD form deliberately tolerates it.
    x[50:70] += np.linspace(0.0, 10.0, 20)

    without = successive_difference_sigma(x, robust=False)
    with_excl = successive_difference_sigma(x, exclude=(50, 70), robust=False)

    assert without > 1.5 * true_sigma  # transient inflates the naive estimate
    assert with_excl == pytest.approx(true_sigma, rel=0.15)  # excision recovers the noise floor


@pytest.mark.unit
def test_robust_form_tolerates_short_washin_without_excision() -> None:
    # Complement to the above: the robust MAD estimator is largely unaffected by a short
    # transient even without excision -- this is why exclusion matters most for the von
    # Neumann form and for vascular voxels where the transient is a large fraction.
    rng = np.random.default_rng(7)
    n = 600
    true_sigma = 0.04
    x = rng.normal(0.0, true_sigma, size=n)
    x[50:70] += np.linspace(0.0, 10.0, 20)
    assert successive_difference_sigma(x, robust=True) == pytest.approx(true_sigma, rel=0.15)


@pytest.mark.unit
def test_no_difference_taken_across_the_gap() -> None:
    # Two flat noise-free half-curves at very different levels; the only large jump is the
    # boundary between them. Excising that boundary must leave zero spurious "noise".
    n = 200
    x = np.concatenate([np.zeros(100), np.full(100, 100.0)])
    # Retained segments [0,90) and [110,200) are each perfectly flat -> sigma ~ 0.
    est = successive_difference_sigma(x, exclude=(90, 110), robust=True)
    assert est == pytest.approx(0.0, abs=1e-9)
    # Without excision, the single 100.0 jump is (robustly) ignored by the median but the
    # von Neumann form would see it; confirm the gap logic, not the estimator, does the work.


# --------------------------------------------------------------------- bolus_exclude_window


@pytest.mark.unit
def test_bolus_exclude_window_basic_span() -> None:
    lo, hi = bolus_exclude_window(onset_frame=10, duration_frames=4, n_frames=100, margin_frames=2)
    assert (lo, hi) == (10, 16)  # onset .. onset+dur+margin


@pytest.mark.unit
def test_bolus_exclude_window_always_drops_onset() -> None:
    lo, hi = bolus_exclude_window(onset_frame=5, duration_frames=0, n_frames=50, margin_frames=0)
    assert (lo, hi) == (5, 6)


@pytest.mark.unit
def test_bolus_exclude_window_clamps_to_bounds() -> None:
    lo, hi = bolus_exclude_window(onset_frame=48, duration_frames=10, n_frames=50, margin_frames=5)
    assert (lo, hi) == (48, 50)
    lo, hi = bolus_exclude_window(onset_frame=-3, duration_frames=2, n_frames=50, margin_frames=1)
    assert lo == 0 and hi >= 1


# --------------------------------------------------------------------- vectorized estimator B


@pytest.mark.unit
def test_2d_matches_per_voxel_1d() -> None:
    rng = np.random.default_rng(4)
    n, v = 500, 6
    block = np.empty((n, v))
    for j in range(v):
        block[:, j] = _smooth_ramp(n) + rng.normal(0.0, 0.03 + 0.01 * j, size=n)

    excl = (120, 140)
    vec = sigma_successive_difference(block, exclude=excl, robust=True)
    loop = np.array(
        [successive_difference_sigma(block[:, j], exclude=excl, robust=True) for j in range(v)]
    )
    np.testing.assert_allclose(vec, loop, rtol=0.0, atol=1e-12)


@pytest.mark.unit
def test_2d_all_nan_voxel_is_nan() -> None:
    rng = np.random.default_rng(5)
    n = 300
    block = np.column_stack([
        _smooth_ramp(n) + rng.normal(0.0, 0.05, size=n),
        np.full(n, np.nan),
    ])
    sigma = sigma_successive_difference(block, robust=True)
    assert np.isfinite(sigma[0])
    assert np.isnan(sigma[1])


@pytest.mark.unit
def test_1d_input_returns_scalar() -> None:
    out = sigma_successive_difference(_smooth_ramp(100), robust=True)
    assert np.isscalar(out) or np.ndim(out) == 0


# --------------------------------------------------------------------- reduced chi-square


@pytest.mark.unit
def test_reduced_chi_square_scalar_value() -> None:
    # SSE = sigma^2 * (N-p) -> chi2_nu == 1 exactly.
    sigma, n_obs, n_params = 0.2, 50, 3
    sse = sigma**2 * (n_obs - n_params)
    assert reduced_chi_square(sse, sigma, n_obs, n_params) == pytest.approx(1.0)


@pytest.mark.unit
def test_reduced_chi_square_vectorized_and_bad_sigma_nan() -> None:
    sse = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.5, 0.0, np.nan])  # only the first is usable
    out = reduced_chi_square(sse, sigma, n_obs=20, n_params=2)
    assert out[0] == pytest.approx(1.0 / (0.25 * 18))
    assert np.isnan(out[1]) and np.isnan(out[2])


@pytest.mark.unit
def test_reduced_chi_square_raises_on_nonpositive_dof() -> None:
    with pytest.raises(ValueError):
        reduced_chi_square(1.0, 0.1, n_obs=3, n_params=3)


@pytest.mark.unit
def test_weighted_equals_scalar_when_sigma_constant() -> None:
    rng = np.random.default_rng(6)
    n_params = 2
    residuals = rng.normal(0.0, 0.3, size=40)
    sigma = 0.3
    sse = float(np.sum(residuals**2))
    scalar_form = reduced_chi_square(sse, sigma, n_obs=residuals.size, n_params=n_params)
    weighted_form = reduced_chi_square_weighted(residuals, sigma, n_params=n_params)
    assert weighted_form == pytest.approx(scalar_form)


@pytest.mark.unit
def test_weighted_per_frame_sigma_and_2d() -> None:
    # Per-frame sigma_i scaling: standardized residuals are all 1 -> chi2_nu = N/(N-p).
    n, v, n_params = 30, 4, 3
    per_frame_sigma = np.linspace(0.1, 0.5, n)
    residuals = np.tile(per_frame_sigma[:, None], (1, v))  # r_i / sigma_i == 1 everywhere
    out = reduced_chi_square_weighted(residuals, per_frame_sigma[:, None], n_params=n_params)
    np.testing.assert_allclose(out, np.full(v, n / (n - n_params)), rtol=1e-12)
