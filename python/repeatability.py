"""Scan-rescan repeatability statistics (QIBA conventions).

Every function takes paired values: ``a[i]`` and ``b[i]`` are the two sessions of subject
``i``. The subject is the unit of analysis throughout, so bootstrap intervals resample
subjects, never voxels or ROIs.

Plan and definitions: project-management/projects/test-retest-validation/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# 1.96 * sqrt(2): the smallest difference between two scans that exceeds measurement noise
# with 95% confidence
REPEATABILITY_COEFFICIENT_FACTOR = 1.96 * math.sqrt(2.0)


def _as_pairs(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"a and b must be 1-D and the same length, got {a.shape} and {b.shape}")
    if a.size < 2:
        raise ValueError("repeatability needs at least two subjects")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("a and b must be finite; drop incomplete pairs before calling")
    return a, b


def within_subject_sd(a: Sequence[float], b: Sequence[float]) -> float:
    """wSD = sqrt(mean(d^2) / 2) for two measurements per subject, d = b - a."""
    a, b = _as_pairs(a, b)
    return float(np.sqrt(np.mean((b - a) ** 2) / 2.0))


def repeatability_coefficient(a: Sequence[float], b: Sequence[float]) -> float:
    """RC = 2.77 x wSD."""
    return REPEATABILITY_COEFFICIENT_FACTOR * within_subject_sd(a, b)


def within_subject_cv_log(a: Sequence[float], b: Sequence[float]) -> float:
    """wCV from log-transformed values: sqrt(exp(s^2) - 1), s = wSD of ln(value).

    NaN unless every value is strictly positive. Near zero the result is unstable even when
    it is defined, so whether to report it for a parameter is the caller's decision.
    """
    a, b = _as_pairs(a, b)
    if (a <= 0).any() or (b <= 0).any():
        return float("nan")
    log_wsd = within_subject_sd(np.log(a), np.log(b))
    return float(np.sqrt(np.exp(log_wsd**2) - 1.0))


def icc_2_1(a: Sequence[float], b: Sequence[float]) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    a, b = _as_pairs(a, b)
    data = np.column_stack([a, b])
    n, k = data.shape
    grand = data.mean()
    ss_rows = k * np.sum((data.mean(axis=1) - grand) ** 2)
    ss_cols = n * np.sum((data.mean(axis=0) - grand) ** 2)
    ss_error = np.sum((data - grand) ** 2) - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator == 0:
        return float("nan")
    return float((ms_rows - ms_error) / denominator)


@dataclass(frozen=True)
class BlandAltman:
    bias: float
    sd_diff: float
    loa_low: float
    loa_high: float
    # Do the differences grow with the mean? Spearman correlation of |d| with the pair mean.
    spread_vs_mean_rho: float
    spread_vs_mean_p: float
    # Does the bias drift with the mean? Slope of d on the pair mean.
    trend_slope: float
    trend_p: float


def bland_altman(a: Sequence[float], b: Sequence[float]) -> BlandAltman:
    """Bias, 95% limits of agreement, and two checks against the pair mean (d = b - a)."""
    a, b = _as_pairs(a, b)
    diff = b - a
    mean = (a + b) / 2.0
    bias = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    if a.size >= 3 and np.ptp(mean) > 0:
        rho, rho_p = stats.spearmanr(np.abs(diff), mean)
        trend = stats.linregress(mean, diff)
        slope, slope_p = float(trend.slope), float(trend.pvalue)
    else:
        rho = rho_p = slope = slope_p = float("nan")
    return BlandAltman(
        bias=bias,
        sd_diff=sd_diff,
        loa_low=bias - 1.96 * sd_diff,
        loa_high=bias + 1.96 * sd_diff,
        spread_vs_mean_rho=float(rho),
        spread_vs_mean_p=float(rho_p),
        trend_slope=slope,
        trend_p=slope_p,
    )


def paired_metrics(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Every point estimate for one set of pairs, as a flat dict."""
    a, b = _as_pairs(a, b)
    ba = bland_altman(a, b)
    wsd = within_subject_sd(a, b)
    return {
        "n": int(a.size),
        "mean": float(np.mean(np.concatenate([a, b]))),
        "wSD": wsd,
        "RC": REPEATABILITY_COEFFICIENT_FACTOR * wsd,
        "wCV_log": within_subject_cv_log(a, b),
        "ICC_2_1": icc_2_1(a, b),
        "BA_bias": ba.bias,
        "BA_loa_low": ba.loa_low,
        "BA_loa_high": ba.loa_high,
        "BA_spread_vs_mean_rho": ba.spread_vs_mean_rho,
        "BA_spread_vs_mean_p": ba.spread_vs_mean_p,
        "BA_trend_slope": ba.trend_slope,
        "BA_trend_p": ba.trend_p,
    }


def bootstrap_ci(
    statistic: Callable[..., float],
    *paired_arrays: Sequence[float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    strata: Optional[Sequence[object]] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap interval for ``statistic(*paired_arrays)``, resampling subjects.

    Every array is indexed by subject and resampled with the same indices, so pairing --
    between sessions and between pipelines -- is kept inside each replicate. With
    ``strata`` (one label per subject, e.g. the collection) subjects are resampled within
    their stratum, so every replicate keeps the cohort sizes.
    """
    arrays = [np.asarray(x, dtype=float) for x in paired_arrays]
    n = arrays[0].size
    if any(x.size != n for x in arrays):
        raise ValueError("all arrays must have one entry per subject")
    if strata is None:
        groups = [np.arange(n)]
    else:
        labels = np.asarray([str(label) for label in strata])
        if labels.size != n:
            raise ValueError("strata must have one label per subject")
        groups = [np.flatnonzero(labels == label) for label in sorted(set(labels.tolist()))]
    rng = np.random.default_rng(seed)
    replicates = np.full(n_boot, np.nan)
    for i in range(n_boot):
        index = np.concatenate([rng.choice(group, group.size) for group in groups])
        try:
            replicates[i] = statistic(*(x[index] for x in arrays))
        except (ValueError, ZeroDivisionError):
            continue  # a degenerate resample, e.g. one subject drawn n times
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size < 0.9 * n_boot:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(replicates, [tail, 1.0 - tail])
    return float(low), float(high)


def wsd_ratio(
    new_a: Sequence[float], new_b: Sequence[float], ref_a: Sequence[float], ref_b: Sequence[float]
) -> float:
    """wSD(new) / wSD(reference) on the same subjects."""
    reference = within_subject_sd(ref_a, ref_b)
    if reference == 0:
        raise ZeroDivisionError("reference wSD is zero")
    return within_subject_sd(new_a, new_b) / reference


def pitman_sd_ratio_ci(
    new_diff: Sequence[float], ref_diff: Sequence[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Ratio of the SDs of two correlated samples, with Pitman's exact-normal interval.

    Used on the per-subject test-retest differences of two pipelines run on the same scans,
    as an analytic cross-check of the bootstrap. It centres each sample, so it ignores a
    systematic session effect that wSD includes.

    Returns (ratio, low, high).
    """
    x = np.asarray(new_diff, dtype=float)
    y = np.asarray(ref_diff, dtype=float)
    n = x.size
    if n != y.size or n < 4:
        raise ValueError("need the same subjects in both samples, at least four")
    variance_ratio = x.var(ddof=1) / y.var(ddof=1)
    r = np.corrcoef(x, y)[0, 1]
    t = stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, n - 2)
    k = 1.0 + 2.0 * (1.0 - r**2) * t**2 / (n - 2)
    half = math.sqrt(k**2 - 1.0)
    return (
        float(math.sqrt(variance_ratio)),
        float(math.sqrt(variance_ratio * (k - half))),
        float(math.sqrt(variance_ratio * (k + half))),
    )


@dataclass(frozen=True)
class NonInferiorityResult:
    ratio: float
    ci_low: float
    ci_high: float
    confidence: float
    margin: float
    non_inferior: bool
    pitman_ratio: float
    pitman_ci_low: float
    pitman_ci_high: float
    n: int


def wsd_ratio_noninferiority(
    new_a: Sequence[float],
    new_b: Sequence[float],
    ref_a: Sequence[float],
    ref_b: Sequence[float],
    *,
    margin: float,
    confidence: float,
    n_boot: int = 5000,
    seed: int = 0,
    strata: Optional[Sequence[object]] = None,
) -> NonInferiorityResult:
    """Is the new pipeline's wSD no worse than the reference's by more than ``margin``?

    Non-inferior when the upper limit of the two-sided ``confidence`` interval of
    wSD(new)/wSD(ref) is below ``margin``. The four arrays are indexed by the same subjects.
    ``margin`` and ``confidence`` have no defaults: both are fixed in the pre-registration
    file before unblinding, not chosen here.
    """
    new_a, new_b = _as_pairs(new_a, new_b)
    ref_a, ref_b = _as_pairs(ref_a, ref_b)
    if new_a.size != ref_a.size:
        raise ValueError("both pipelines must cover the same subjects")
    if margin <= 1.0:
        raise ValueError("a non-inferiority margin on the wSD ratio must exceed 1")
    ratio = wsd_ratio(new_a, new_b, ref_a, ref_b)
    low, high = bootstrap_ci(
        wsd_ratio, new_a, new_b, ref_a, ref_b,
        n_boot=n_boot, confidence=confidence, seed=seed, strata=strata,
    )
    pitman = pitman_sd_ratio_ci(new_b - new_a, ref_b - ref_a, confidence=confidence)
    return NonInferiorityResult(
        ratio=ratio,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        margin=margin,
        non_inferior=bool(np.isfinite(high) and high < margin),
        pitman_ratio=pitman[0],
        pitman_ci_low=pitman[1],
        pitman_ci_high=pitman[2],
        n=int(new_a.size),
    )


def processing_sd(values_by_session: Sequence[Sequence[float]]) -> float:
    """Spread one processing factor adds: the pooled within-session SD across its levels.

    ``values_by_session[i]`` holds one session's value under each level of the factor
    (two or more). For two levels this equals the wSD of the pair, so it is directly
    comparable with the scan-rescan wSD.
    """
    variances: List[float] = []
    for values in values_by_session:
        values = np.asarray(values, dtype=float)
        if values.size < 2 or not np.isfinite(values).all():
            raise ValueError("each session needs a finite value at two or more levels")
        variances.append(float(values.var(ddof=1)))
    if not variances:
        raise ValueError("no sessions")
    return float(np.sqrt(np.mean(variances)))


def variance_budget(
    factor_sd: Mapping[str, float], rescan_wsd: float
) -> List[Dict[str, float | str]]:
    """Each processing factor's spread relative to the scan-rescan wSD, largest first.

    ``variance_ratio`` is the squared ratio. The factors are varied one at a time, so the
    ratios are not shares of one total and need not sum to anything.
    """
    if not rescan_wsd > 0:
        raise ValueError("scan-rescan wSD must be positive")
    rows = [
        {
            "factor": name,
            "processing_sd": float(sd),
            "sd_ratio": float(sd) / rescan_wsd,
            "variance_ratio": (float(sd) / rescan_wsd) ** 2,
        }
        for name, sd in factor_sd.items()
    ]
    return sorted(rows, key=lambda row: -float(row["sd_ratio"]))


def stratified_split(
    subjects_by_stratum: Mapping[str, Iterable[str]], development_fraction: float, seed: int
) -> Dict[str, str]:
    """Assign each subject to 'development' or 'held_out', the same fraction per stratum."""
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    assignment: Dict[str, str] = {}
    for stratum in sorted(subjects_by_stratum):
        subjects = sorted(set(subjects_by_stratum[stratum]))
        n_development = int(round(development_fraction * len(subjects)))
        chosen = set(rng.permutation(len(subjects))[:n_development].tolist())
        for i, subject in enumerate(subjects):
            assignment[subject] = "development" if i in chosen else "held_out"
    return assignment


def pair_sessions(
    rows: Iterable[Mapping[str, object]],
    group_keys: Sequence[str],
    value_key: str = "value",
) -> Tuple[Dict[Tuple[object, ...], Dict[str, Tuple[float, float]]], List[Dict[str, object]]]:
    """Pair each subject's two sessions within every group of a tidy table.

    Returns ``(pairs, dropped)``. ``pairs[group][subject] = (first, second)`` with sessions
    in sorted order. A subject without exactly two finite values in a group is left out of
    that group and listed in ``dropped`` with the reason.
    """
    by_group: Dict[Tuple[object, ...], Dict[str, Dict[str, float]]] = {}
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        sessions = by_group.setdefault(group, {}).setdefault(str(row["subject"]), {})
        session = str(row["session"])
        if session in sessions:
            raise ValueError(f"duplicate row for {row['subject']} {session} in group {group}")
        sessions[session] = float(row[value_key])  # type: ignore[arg-type]

    pairs: Dict[Tuple[object, ...], Dict[str, Tuple[float, float]]] = {}
    dropped: List[Dict[str, object]] = []
    for group, subjects in by_group.items():
        for subject, sessions in sorted(subjects.items()):
            values = [sessions[name] for name in sorted(sessions)]
            if len(values) != 2:
                reason: Optional[str] = f"{len(values)} session(s), need 2"
            elif not all(math.isfinite(v) for v in values):
                reason = "non-finite value"
            else:
                reason = None
            if reason:
                dropped.append({**dict(zip(group_keys, group)), "subject": subject, "reason": reason})
                continue
            pairs.setdefault(group, {})[subject] = (values[0], values[1])
    return pairs, dropped
