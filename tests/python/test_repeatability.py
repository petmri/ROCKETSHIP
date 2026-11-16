"""Repeatability statistics on synthetic pairs with known answers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import repeatability as rp  # noqa: E402

pytestmark = pytest.mark.unit


def simulate_pairs(n, between_sd, within_sd, seed, mean=10.0, session_bias=0.0):
    rng = np.random.default_rng(seed)
    truth = mean + between_sd * rng.standard_normal(n)
    a = truth + within_sd * rng.standard_normal(n)
    b = truth + within_sd * rng.standard_normal(n) + session_bias
    return a, b


def test_wsd_by_hand():
    # d = [2, -2, 0, 4] -> mean(d^2) = 6 -> wSD = sqrt(3)
    a = [1.0, 5.0, 3.0, 2.0]
    b = [3.0, 3.0, 3.0, 6.0]
    assert rp.within_subject_sd(a, b) == pytest.approx(math.sqrt(3.0))
    assert rp.repeatability_coefficient(a, b) == pytest.approx(1.96 * math.sqrt(2.0) * math.sqrt(3.0))
    assert rp.REPEATABILITY_COEFFICIENT_FACTOR == pytest.approx(2.77, abs=0.005)


def test_wsd_recovers_the_generating_value():
    a, b = simulate_pairs(20000, between_sd=3.0, within_sd=0.7, seed=1)
    assert rp.within_subject_sd(a, b) == pytest.approx(0.7, rel=0.02)


def test_wsd_counts_a_systematic_session_shift():
    a, b = simulate_pairs(20000, between_sd=3.0, within_sd=0.7, seed=2, session_bias=1.0)
    # mean(d^2)/2 = within^2 + bias^2/2
    assert rp.within_subject_sd(a, b) == pytest.approx(math.sqrt(0.7**2 + 0.5), rel=0.02)


def test_wcv_log_recovers_the_generating_value():
    rng = np.random.default_rng(3)
    truth = np.exp(rng.normal(0.0, 0.5, 20000))
    cv = 0.15
    sigma = math.sqrt(math.log(1.0 + cv**2))
    a = truth * np.exp(sigma * rng.standard_normal(truth.size))
    b = truth * np.exp(sigma * rng.standard_normal(truth.size))
    assert rp.within_subject_cv_log(a, b) == pytest.approx(cv, rel=0.03)


def test_wcv_log_is_nan_when_a_value_is_not_positive():
    assert math.isnan(rp.within_subject_cv_log([1.0, 0.0, 2.0], [1.0, 1.0, 2.0]))
    assert math.isnan(rp.within_subject_cv_log([1.0, -0.1, 2.0], [1.0, 1.0, 2.0]))


def test_icc_matches_the_variance_components():
    a, b = simulate_pairs(40000, between_sd=2.0, within_sd=1.0, seed=4)
    assert rp.icc_2_1(a, b) == pytest.approx(4.0 / 5.0, abs=0.01)


def test_icc_absolute_agreement_penalises_a_session_shift():
    a, b = simulate_pairs(40000, between_sd=2.0, within_sd=1.0, seed=5, session_bias=2.0)
    # the shift adds bias^2/2 = 2 to the disagreement variance
    assert rp.icc_2_1(a, b) == pytest.approx(4.0 / (4.0 + 1.0 + 2.0), abs=0.01)


def test_icc_shrout_fleiss_first_two_judges():
    # Shrout & Fleiss (1979) Table 2, judges 1 and 2. Mean squares worked by hand:
    # SSR = 23.4167 (5 df), SSC = 80.0833 (1 df), SSE = 106.9167 - SSR - SSC = 3.4167 (5 df)
    a = [9.0, 6.0, 8.0, 7.0, 10.0, 6.0]
    b = [2.0, 1.0, 4.0, 1.0, 5.0, 2.0]
    ms_rows, ms_cols, ms_error, n, k = 4.6833, 80.0833, 0.6833, 6, 2
    expected = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    assert expected == pytest.approx(0.1257, abs=1e-4)
    assert rp.icc_2_1(a, b) == pytest.approx(expected, abs=1e-4)


def test_bland_altman_bias_and_limits():
    a, b = simulate_pairs(40000, between_sd=2.0, within_sd=1.0, seed=6, session_bias=0.5)
    ba = rp.bland_altman(a, b)
    assert ba.bias == pytest.approx(0.5, abs=0.03)
    assert ba.sd_diff == pytest.approx(math.sqrt(2.0), rel=0.02)
    assert ba.loa_high - ba.loa_low == pytest.approx(2 * 1.96 * math.sqrt(2.0), rel=0.02)
    assert ba.spread_vs_mean_p > 0.001


def test_bland_altman_flags_spread_that_grows_with_the_mean():
    rng = np.random.default_rng(7)
    truth = rng.uniform(1.0, 10.0, 400)
    a = truth * (1.0 + 0.1 * rng.standard_normal(400))
    b = truth * (1.0 + 0.1 * rng.standard_normal(400))
    ba = rp.bland_altman(a, b)
    assert ba.spread_vs_mean_rho > 0.3
    assert ba.spread_vs_mean_p < 1e-6


def test_paired_metrics_rejects_incomplete_pairs():
    with pytest.raises(ValueError, match="finite"):
        rp.paired_metrics([1.0, float("nan"), 2.0], [1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="two subjects"):
        rp.paired_metrics([1.0], [1.0])


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    a, b = simulate_pairs(40, between_sd=2.0, within_sd=1.0, seed=8)
    first = rp.bootstrap_ci(rp.within_subject_sd, a, b, n_boot=1000, seed=11)
    again = rp.bootstrap_ci(rp.within_subject_sd, a, b, n_boot=1000, seed=11)
    assert first == again
    assert first[0] < rp.within_subject_sd(a, b) < first[1]


def two_pipelines(n, true_ratio, shared, seed):
    """Test-retest differences of two pipelines on the same subjects.

    ``shared`` is the fraction of the reference's rescan variance that both pipelines see
    (biology and acquisition); the rest is each pipeline's own.
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n, 2))
    own_ref = rng.standard_normal((n, 2))
    own_new = rng.standard_normal((n, 2))
    truth = 5.0 + rng.standard_normal(n)[:, None]
    ref = truth + math.sqrt(shared) * common + math.sqrt(1.0 - shared) * own_ref
    new_own_variance = true_ratio**2 - shared
    new = truth + math.sqrt(shared) * common + math.sqrt(new_own_variance) * own_new
    return new[:, 0], new[:, 1], ref[:, 0], ref[:, 1]


def test_noninferiority_decisions_at_large_n():
    equal = rp.wsd_ratio_noninferiority(
        *two_pipelines(3000, 1.0, 0.8, seed=9), margin=1.1, confidence=0.95, n_boot=600
    )
    assert equal.ratio == pytest.approx(1.0, abs=0.03)
    assert equal.non_inferior
    assert equal.ci_low < 1.0 < equal.ci_high

    worse = rp.wsd_ratio_noninferiority(
        *two_pipelines(3000, 1.3, 0.8, seed=10), margin=1.1, confidence=0.95, n_boot=600
    )
    assert worse.ratio == pytest.approx(1.3, abs=0.04)
    assert not worse.non_inferior
    # the analytic interval agrees with the bootstrap when there is no session shift
    assert worse.pitman_ratio == pytest.approx(worse.ratio, rel=0.01)
    assert worse.pitman_ci_high == pytest.approx(worse.ci_high, rel=0.03)


def test_noninferiority_requires_an_explicit_margin_and_confidence():
    args = two_pipelines(30, 1.0, 0.8, seed=12)
    with pytest.raises(TypeError):
        rp.wsd_ratio_noninferiority(*args)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="exceed 1"):
        rp.wsd_ratio_noninferiority(*args, margin=0.9, confidence=0.95)


@pytest.mark.slow
@pytest.mark.parametrize("n", [19, 24])
def test_interval_coverage_at_cohort_size(n):
    """At the cohort sizes the 95% intervals should cover the true ratio about 95% of the time."""
    true_ratio, reps = 1.0, 400
    pitman_hits = bootstrap_hits = 0
    for rep in range(reps):
        new_a, new_b, ref_a, ref_b = two_pipelines(n, true_ratio, 0.8, seed=1000 + rep)
        _, low, high = rp.pitman_sd_ratio_ci(new_b - new_a, ref_b - ref_a)
        pitman_hits += low <= true_ratio <= high
        low, high = rp.bootstrap_ci(rp.wsd_ratio, new_a, new_b, ref_a, ref_b, n_boot=400, seed=rep)
        bootstrap_hits += low <= true_ratio <= high
    assert 0.92 <= pitman_hits / reps <= 0.98
    # the percentile bootstrap is known to run a little narrow at this n
    assert 0.86 <= bootstrap_hits / reps <= 0.98


def test_processing_sd_equals_wsd_for_two_levels():
    a, b = simulate_pairs(50, between_sd=2.0, within_sd=1.0, seed=13)
    assert rp.processing_sd(list(zip(a, b))) == pytest.approx(rp.within_subject_sd(a, b))


def test_processing_sd_recovers_the_generating_value_with_three_levels():
    rng = np.random.default_rng(14)
    session_value = rng.normal(5.0, 2.0, 5000)[:, None]
    levels = session_value + 0.3 * rng.standard_normal((5000, 3))
    assert rp.processing_sd(levels) == pytest.approx(0.3, rel=0.03)


def test_variance_budget_orders_factors_and_reports_ratios():
    budget = rp.variance_budget({"backend": 0.01, "aif_rater": 0.5, "start_t": 0.1}, rescan_wsd=1.0)
    assert [row["factor"] for row in budget] == ["aif_rater", "start_t", "backend"]
    assert budget[0]["sd_ratio"] == pytest.approx(0.5)
    assert budget[0]["variance_ratio"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        rp.variance_budget({"backend": 0.01}, rescan_wsd=0.0)


def test_stratified_split_is_reproducible_and_balanced():
    strata = {"RIDER": [f"r{i:02d}" for i in range(19)], "QIN-GBM": [f"q{i:02d}" for i in range(24)]}
    split = rp.stratified_split(strata, development_fraction=1 / 3, seed=5)
    assert split == rp.stratified_split(strata, development_fraction=1 / 3, seed=5)
    assert sum(split[s] == "development" for s in strata["RIDER"]) == 6
    assert sum(split[s] == "development" for s in strata["QIN-GBM"]) == 8
    assert split != rp.stratified_split(strata, development_fraction=1 / 3, seed=6)


def test_pair_sessions_pairs_in_session_order_and_logs_drops():
    rows = [
        {"subject": "s1", "session": "ses-02", "pipeline": "python", "value": 2.0},
        {"subject": "s1", "session": "ses-01", "pipeline": "python", "value": 1.0},
        {"subject": "s2", "session": "ses-01", "pipeline": "python", "value": 3.0},
        {"subject": "s3", "session": "ses-01", "pipeline": "python", "value": float("nan")},
        {"subject": "s3", "session": "ses-02", "pipeline": "python", "value": 3.0},
        {"subject": "s1", "session": "ses-01", "pipeline": "matlab", "value": 1.5},
        {"subject": "s1", "session": "ses-02", "pipeline": "matlab", "value": 2.5},
    ]
    pairs, dropped = rp.pair_sessions(rows, group_keys=("pipeline",))
    assert pairs[("python",)] == {"s1": (1.0, 2.0)}
    assert pairs[("matlab",)] == {"s1": (1.5, 2.5)}
    assert {(d["subject"], d["reason"]) for d in dropped} == {
        ("s2", "1 session(s), need 2"),
        ("s3", "non-finite value"),
    }
    with pytest.raises(ValueError, match="duplicate"):
        rp.pair_sessions(rows + [rows[0]], group_keys=("pipeline",))
