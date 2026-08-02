"""DCE model ports from MATLAB."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, least_squares
from scipy.stats import t as _student_t


def _safe_float_setting(settings: Dict[str, object], key: str, default: float) -> float:
    raw = settings.get(key, default)
    try:
        out = float(raw)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _timer_looks_like_seconds(t_vec: List[float]) -> bool:
    """Heuristic to detect second-based timer vectors (vs minute-based)."""
    finite = [float(v) for v in t_vec if math.isfinite(float(v))]
    if len(finite) < 2:
        return False

    span = max(finite) - min(finite)
    if span > 100.0:
        return True

    diffs = [abs(finite[i] - finite[i - 1]) for i in range(1, len(finite))]
    positive = sorted(d for d in diffs if d > 0.0)
    if not positive:
        return False
    median_step = positive[len(positive) // 2]
    return median_step > 0.25


def _resolve_timer_unit(t_vec: List[float], explicit: Optional[object] = None) -> str:
    """Return canonical timer unit label: 'seconds' or 'minutes'."""
    if explicit is not None:
        unit = str(explicit).strip().lower()
        if unit in {"s", "sec", "second", "seconds"}:
            return "seconds"
        if unit in {"m", "min", "minute", "minutes"}:
            return "minutes"
    return "seconds" if _timer_looks_like_seconds(t_vec) else "minutes"


def _canonical_time_context(
    t_vec: List[float],
    prefs: Optional[Dict[str, float]] = None,
) -> tuple[List[float], str, float, float]:
    """Map timer/prefs units to canonical fit units.

    Canonical internal units:
      - time: minutes
      - rate constants (Ktrans, Fp, PS): per-minute

    Returns:
      timer_min, input_timer_unit, rate_in_to_min, rate_min_to_output
    """
    explicit_unit = None
    if prefs:
        explicit_unit = prefs.get("time_unit", prefs.get("timer_unit"))

    unit = _resolve_timer_unit(t_vec, explicit=explicit_unit)
    if unit == "seconds":
        timer_min = [float(v) / 60.0 for v in t_vec]
        rate_in_to_min = 60.0
        rate_min_to_output = 1.0 / 60.0
    else:
        timer_min = [float(v) for v in t_vec]
        rate_in_to_min = 1.0
        rate_min_to_output = 1.0
    return timer_min, unit, rate_in_to_min, rate_min_to_output


def _merge_prefs_in_canonical_units(
    defaults: Dict[str, float],
    prefs: Optional[Dict[str, float]],
    *,
    rate_keys: List[str],
    time_constant_keys: Optional[List[str]] = None,
    rate_in_to_min: float = 1.0,
) -> Dict[str, float]:
    """Merge caller prefs into canonical-unit defaults.

    Input preference units are assumed to follow the input timer unit.
    """
    settings = dict(defaults)
    if not prefs:
        return settings

    ignored = {"time_unit", "timer_unit"}
    time_scale = 1.0 / rate_in_to_min if rate_in_to_min != 0.0 else 1.0
    for key, raw in prefs.items():
        if key in ignored:
            continue
        if key in rate_keys:
            settings[key] = float(raw) * rate_in_to_min
        elif key in (time_constant_keys or []):
            settings[key] = float(raw) * time_scale
        else:
            settings[key] = raw
    return settings


def _reject_algorithm_override(prefs: Optional[Dict[str, float]], model_name: str) -> None:
    """Disallow algorithm switching to keep one code path per model."""
    if not prefs:
        return
    for key in ("fit_algorithm", "algorithm"):
        if key in prefs:
            raise ValueError(
                f"{model_name} does not support '{key}' overrides; "
                "algorithm selection is fixed."
            )


def _loss_from_robust(value: object) -> str:
    mode = str(value).strip().lower()
    if mode in {"", "off", "none", "linear"}:
        return "linear"
    if mode == "lar":
        return "soft_l1"
    if mode == "bisquare":
        return "cauchy"
    return "linear"


def _least_squares_kwargs(settings: Dict[str, object], default_max_nfev: int) -> Dict[str, object]:
    max_nfev = int(_safe_float_setting(settings, "max_nfev", float(default_max_nfev)))
    tol_floor = 1e-15
    ftol = max(_safe_float_setting(settings, "tol_fun", 1e-12), tol_floor)
    xtol = max(_safe_float_setting(settings, "tol_x", 1e-6), tol_floor)
    kwargs: Dict[str, object] = {
        "method": "trf",
        "max_nfev": max_nfev,
        "ftol": ftol,
        "xtol": xtol,
    }
    loss = _loss_from_robust(settings.get("robust", "off"))
    if loss != "linear":
        kwargs["loss"] = loss
    return kwargs


@lru_cache(maxsize=64)
def _student_t_quantile(level: float, dof: int) -> float:
    """Two-sided t quantile, memoized on (level, dof).

    `dof = n_obs - n_params` is constant for a whole run, but CIs are built per
    voxel per candidate start, so this was the single most-repeated scipy call
    in a fit. Memoizing a deterministic scalar is exact.
    """
    return float(_student_t.ppf(1.0 - (1.0 - level) / 2.0, dof))


def _ci_stderrs_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Standard errors from a parameter covariance matrix (NaN where undefined)."""
    var = np.asarray(np.diag(cov), dtype=float)
    return np.sqrt(np.where(np.isfinite(var) & (var >= 0.0), var, np.nan))


def _ci_from_stderrs(
    estimates: np.ndarray,
    stderrs: np.ndarray,
    dof: int,
    *,
    level: float = 0.95,
) -> tuple[List[float], List[float]]:
    """Two-sided CI bounds: estimate +/- t(1-alpha/2, dof) * stderr.

    Returns (lows, highs) in the estimates' own units. NaN entries where the
    standard error is undefined, mirroring MATLAB confint on a degenerate fit.
    """
    est = np.asarray(estimates, dtype=float)
    se = np.asarray(stderrs, dtype=float)
    if dof <= 0:
        nan = [float("nan")] * est.size
        return nan, nan
    tval = _student_t_quantile(level, int(dof))
    lows: List[float] = []
    highs: List[float] = []
    for i in range(est.size):
        s = float(se[i])
        if not math.isfinite(s):
            lows.append(float("nan"))
            highs.append(float("nan"))
        else:
            lows.append(float(est[i] - tval * s))
            highs.append(float(est[i] + tval * s))
    return lows, highs


def _ci_bounds_from_fit(fit, *, level: float = 0.95) -> tuple[List[float], List[float]]:
    """95% CIs for a scipy ``least_squares`` result, matching MATLAB confint/nlparci.

    Covariance = MSE * inv(J^T J) with MSE = SSE / (n_obs - n_params); the CI half
    width is t(1-alpha/2, dof) * sqrt(diag(cov)). Returns (lows, highs) in the fit's
    own parameter units, or all-NaN where the covariance is undefined (dof <= 0,
    singular normal matrix, or non-finite variance).
    """
    x = np.asarray(fit.x, dtype=float)
    p = int(x.size)
    res = np.asarray(fit.fun, dtype=float).reshape(-1)
    n_obs = int(res.size)
    nan = [float("nan")] * p
    dof = n_obs - p
    if dof <= 0:
        return nan, nan
    jac = np.asarray(fit.jac, dtype=float)
    if jac.ndim != 2 or jac.shape[0] != n_obs or jac.shape[1] != p:
        return nan, nan
    mse = float(np.dot(res, res)) / dof
    try:
        cov = mse * np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        return nan, nan
    return _ci_from_stderrs(x, _ci_stderrs_from_covariance(cov), dof, level=level)


def _cumulative_trapz_values(y: List[float], t: List[float]) -> List[float]:
    """Return cumulative trapezoid integral with output aligned to sample indices.

    Callers pass already-coerced float sequences, so the per-sample `float()`
    calls this used to make were pure overhead on the hottest path in the
    codebase. Operation order is unchanged and the results are bit-identical.
    """
    n = len(t)
    out = [0.0] * n
    prev = 0.0
    for i in range(1, n):
        prev = prev + (0.5 * (t[i] - t[i - 1]) * (y[i - 1] + y[i]))
        out[i] = prev
    return out


def _exp_weighted_cumulative_trapz_values(y: List[float], t: List[float], lam: float) -> List[float]:
    """Compute trapz(t, y * exp(-lam * (t_k - t))) for each endpoint k in O(n).

    As in `_cumulative_trapz_values`, the per-sample `float()` coercions are
    dropped and the running total is carried in a local instead of re-read from
    the output list. Same operations in the same order, bit-identical results.
    """
    n = len(t)
    out = [0.0] * n
    prev = 0.0
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        decay = math.exp(-lam * dt)
        # Trapezoid on the new interval, weighted at t_k.
        interval = 0.5 * dt * ((y[k - 1] * decay) + y[k])
        prev = (decay * prev) + interval
        out[k] = prev
    return out


def _exp_weighted_cumulative_trapz_batch(
    y: np.ndarray, t: np.ndarray, lam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batched `_exp_weighted_cumulative_trapz_values`: shared y/t, one lam per voxel.

    The recurrence is sequential in time and cannot vectorize over it, but `lam` is
    the only per-voxel quantity, so this runs n_time vector steps instead of
    n_voxels * n_time scalar ones. Every operation is in the same order and
    association as the scalar function, so the result is bit-identical.

    Returns `(out, overflowed)`, both (n_time, n_voxels) and (n_voxels,). A large
    negative `lam` makes the scalar `math.exp` raise OverflowError where `np.exp`
    quietly returns inf, so callers need the mask to reject exactly those voxels.
    """
    n_time = int(t.size)
    out = np.zeros((n_time, int(lam.size)), dtype=np.float64)
    overflowed = np.zeros(int(lam.size), dtype=bool)
    for k in range(1, n_time):
        dt = float(t[k] - t[k - 1])
        decay = np.exp(-lam * dt)
        overflowed |= np.isinf(decay)
        interval = 0.5 * dt * ((float(y[k - 1]) * decay) + float(y[k]))
        out[k] = (decay * out[k - 1]) + interval
    return out, overflowed


class OsipiResampleGrid:
    """The dense resample grid OSIPI LEK reference implementations fit 2CXM on.

    Those implementations quadratically resample to 0.1 s before fitting, so a
    five-minute study becomes a ~3000-point curve. Everything about that grid
    except the tissue curve -- the sample times, the resampled AIF, and the
    spline solve behind it -- depends only on cp and the timer, which are the
    same for every voxel and every multistart candidate. Building it once and
    reusing it takes the AIF spline out of the per-voxel path entirely; only
    `resample_ct` still runs per voxel, and its output is a single ~3000-point
    vector rather than a whole (n_interp, n_voxels) matrix, which for a
    full-ROI Stage-D batch would be hundreds of megabytes.

    Holds only the arrays it needs, never the caller's cp/ct/settings scope.
    """

    __slots__ = ("t_sample", "t_interp", "cp_interp", "n_samples")

    def __init__(
        self,
        t_sample: np.ndarray,
        t_interp: np.ndarray,
        cp_interp: np.ndarray,
        n_samples: int,
    ) -> None:
        self.t_sample = t_sample
        self.t_interp = t_interp
        self.cp_interp = cp_interp
        # Count of *acquired* samples, kept for the CI degrees-of-freedom
        # rescale in `_fit_2cxm_osipi_canonical` -- see the note there.
        self.n_samples = n_samples

    def resample_ct(self, ct_vec: Iterable[float]) -> np.ndarray:
        """Resample one voxel's tissue curve onto the grid."""
        ct_arr = np.asarray(ct_vec, dtype=np.float64)
        ct_interp_fn = interp1d(
            self.t_sample,
            ct_arr,
            kind="quadratic",
            bounds_error=False,
            fill_value=(0.0, float(ct_arr[-1])),
        )
        return np.asarray(ct_interp_fn(self.t_interp), dtype=np.float64)


def build_osipi_resample_grid(
    cp_vec: Iterable[float], t_min_vec: List[float]
) -> Optional[OsipiResampleGrid]:
    """Build the OSIPI resample grid, or None if the timer can't support one.

    Rejection depends only on the timer (fewer than three samples, non-
    increasing, or shorter than one interpolation step), so a caller that gets
    None here knows no voxel in the batch is fittable.

    Args:
      cp_vec: Plasma concentration samples.
      t_min_vec: Time samples in minutes.
    """
    if len(t_min_vec) < 3:
        return None

    t_arr = np.asarray(t_min_vec, dtype=np.float64)
    cp_arr = np.asarray(cp_vec, dtype=np.float64)

    t_shift = t_arr - float(t_arr[0])
    dt = np.diff(t_shift)
    if np.any(dt <= 0.0):
        return None

    # OSIPI LEK implementations resample to 0.1 s; canonical timer here is minutes.
    dt_interp = 0.1 / 60.0
    t_end = float(t_shift[-1])
    if t_end <= dt_interp:
        return None

    t_interp = np.arange(0.0, t_end + 0.5 * dt_interp, dt_interp, dtype=np.float64)
    cp_interp_fn = interp1d(
        t_shift,
        cp_arr,
        kind="quadratic",
        bounds_error=False,
        fill_value=(0.0, float(cp_arr[-1])),
    )
    cp_interp = np.asarray(cp_interp_fn(t_interp), dtype=np.float64)
    return OsipiResampleGrid(t_shift, t_interp, cp_interp, len(t_min_vec))


def _two_cxm_curve_osipi(
    vp: float,
    ve: float,
    fp: float,
    e: float,
    cp: np.ndarray,
    t_min: np.ndarray,
) -> np.ndarray:
    eps = 1e-12
    fp_use = max(float(fp), eps)
    e_use = min(max(float(e), eps), 1.0 - eps)

    tp = (float(vp) / fp_use) * (1.0 - e_use)
    te = float(ve) * (1.0 - e_use) / (e_use * fp_use)
    tb = float(vp) / fp_use

    if tp <= eps or te <= eps or tb <= eps:
        return np.zeros_like(cp, dtype=np.float64)

    root_arg = ((1.0 / tp) + (1.0 / te)) ** 2 - (4.0 / (te * tb))
    if root_arg < 0.0:
        root_arg = 0.0
    root = math.sqrt(root_arg)

    k_plus = 0.5 * ((1.0 / tp) + (1.0 / te) + root)
    k_minus = 0.5 * ((1.0 / tp) + (1.0 / te) - root)
    denom = k_plus - k_minus
    if abs(denom) < eps:
        denom = eps if denom >= 0.0 else -eps
    a_coeff = (k_plus - (1.0 / tb)) / denom

    exp_plus = np.exp(-t_min * k_plus)
    exp_minus = np.exp(-t_min * k_minus)
    imp = exp_plus + a_coeff * (exp_minus - exp_plus)

    dt = float(t_min[1] - t_min[0]) if t_min.size > 1 else 0.0
    conv = np.convolve(cp, imp, mode="full")[: t_min.size] * dt
    out = fp_use * conv
    out[0] = 0.0
    return out


def _e_space_bounds(ktrans_lo: float, ktrans_hi: float, fp_lo: float, fp_hi: float) -> tuple[float, float]:
    """Map (Ktrans, Fp) bounds to extraction-fraction bounds E=Ktrans/Fp in (0, 1).

    Lives here, next to the 2CXM E-parametrization it belongs to, and is used by
    both this module's canonical 2CXM fit and the accelerated tissue_uptake/2cxm
    runners in `dce_fit_backends`. Kept separate from the per-candidate E
    initial-value clip, which each caller applies per voxel/candidate.
    """
    e_lo = min(max(ktrans_lo / max(fp_hi, 1e-12), 0.0), 1.0 - 1e-10)
    e_hi = min(max(ktrans_hi / max(fp_lo, 1e-12), e_lo + 1e-10), 1.0 - 1e-8)
    return float(e_lo), float(e_hi)


def _fit_2cxm_osipi_canonical(
    ct_interp: np.ndarray,
    grid: OsipiResampleGrid,
    settings: Optional[Dict[str, float]] = None,
) -> Optional[List[float]]:
    """Return OSIPI LEK 2CXM fit in canonical units.

    Takes the tissue curve already resampled onto `grid` (see
    `OsipiResampleGrid.resample_ct`) rather than raw samples, so a caller
    fitting many candidates for one voxel resamples once, and a caller
    fitting many voxels reuses one grid.

    Canonical units:
      - timer: minutes
      - ktrans/fp: per-minute
    """
    t_interp = grid.t_interp
    cp_interp = grid.cp_interp

    settings = settings or {}
    vp0 = float(settings.get("initial_value_vp", 0.01))
    ve0 = float(settings.get("initial_value_ve", 0.2))
    fp0 = float(settings.get("initial_value_fp", 20.0 / 100.0))
    fp0 = max(fp0, 1e-12)
    ktrans0 = float(settings.get("initial_value_ktrans", 0.03))

    fp_lo = max(float(settings.get("lower_limit_fp", 0.0)), 0.0)
    fp_hi = max(float(settings.get("upper_limit_fp", 200.0 / 100.0)), fp_lo + 1e-12)
    vp_lo = max(float(settings.get("lower_limit_vp", 0.0)), 0.0)
    vp_hi = max(float(settings.get("upper_limit_vp", 1.0)), vp_lo + 1e-12)
    ve_lo = max(float(settings.get("lower_limit_ve", 0.0)), 0.0)
    ve_hi = max(float(settings.get("upper_limit_ve", 1.0)), ve_lo + 1e-12)
    ktrans_lo = max(float(settings.get("lower_limit_ktrans", 0.0)), 0.0)
    ktrans_hi = max(float(settings.get("upper_limit_ktrans", 2.0)), ktrans_lo + 1e-12)

    e0 = ktrans0 / max(fp0, 1e-12)
    e0 = min(max(e0, 1e-8), 1.0 - 1e-8)
    e_lo, e_hi = _e_space_bounds(ktrans_lo, ktrans_hi, fp_lo, fp_hi)
    e0 = min(max(e0, e_lo + 1e-10), e_hi - 1e-10)
    maxfev = int(_safe_float_setting(settings, "max_nfev", 4000.0))

    try:
        fit, pcov = curve_fit(
            lambda _t, vp, ve, fp, e: _two_cxm_curve_osipi(vp, ve, fp, e, cp_interp, t_interp),
            t_interp,
            ct_interp,
            p0=(vp0, ve0, fp0, e0),
            bounds=((vp_lo, ve_lo, fp_lo, e_lo), (vp_hi, ve_hi, fp_hi, e_hi)),
            maxfev=maxfev,
        )
    except Exception:
        return None

    vp = float(fit[0])
    ve = float(fit[1])
    fp_per_min_ml_per_ml = float(fit[2])
    e = min(max(float(fit[3]), 1e-12), 1.0 - 1e-12)
    ktrans_per_min = e * fp_per_min_ml_per_ml

    pred = _two_cxm_curve_osipi(vp, ve, fp_per_min_ml_per_ml, e, cp_interp, t_interp)
    sse = float(np.sum((pred - ct_interp) ** 2))

    # CIs from curve_fit's covariance (fit coefficients are [vp, ve, fp, e]).
    # Ktrans = E * Fp is derived, so its variance comes from the delta method.
    #
    # curve_fit (absolute_sigma=False) internally scales pcov by
    # sse / (len(ct_interp) - n_params) -- i.e. by the *interpolated* grid's
    # point count, not the number of actually acquired, independent DCE time
    # points. ct_interp is a dense quadratic-resampled curve (OSIPI LEK
    # convention, ~0.1s steps) that carries no more real information than the
    # original samples it was interpolated from, so using its length as dof
    # systematically understates the variance (and hence CI width). Rescale
    # pcov to the real dof: cov_real = pcov * dof_interp / dof_real.
    cov = np.asarray(pcov, dtype=float)
    dof_interp = int(len(ct_interp)) - 4
    dof = int(grid.n_samples) - 4
    if dof > 0 and dof_interp > 0:
        cov = cov * (float(dof_interp) / float(dof))
    else:
        cov = np.full_like(cov, np.nan)

    def _safe_se(value: float) -> float:
        return math.sqrt(value) if (math.isfinite(value) and value >= 0.0) else float("nan")

    var_ktrans = (
        (e * e) * float(cov[2, 2])
        + (fp_per_min_ml_per_ml ** 2) * float(cov[3, 3])
        + 2.0 * e * fp_per_min_ml_per_ml * float(cov[2, 3])
    )
    estimates = np.array([ktrans_per_min, ve, vp, fp_per_min_ml_per_ml], dtype=float)
    stderrs = np.array(
        [
            _safe_se(var_ktrans),
            _safe_se(float(cov[1, 1])),
            _safe_se(float(cov[0, 0])),
            _safe_se(float(cov[2, 2])),
        ],
        dtype=float,
    )
    ci_lo, ci_hi = _ci_from_stderrs(estimates, stderrs, dof)

    return [
        ktrans_per_min,
        ve,
        vp,
        fp_per_min_ml_per_ml,
        sse,
        ci_lo[0],
        ci_hi[0],
        ci_lo[1],
        ci_hi[1],
        ci_lo[2],
        ci_hi[2],
        ci_lo[3],
        ci_hi[3],
    ]


class ForwardInputs:
    """Pre-coerced cp/timer plus the integrals that don't vary with fit parameters.

    `cp` and the timer are fixed for an entire fit, so coercing them to float
    lists -- and integrating cp over the timer -- is loop-invariant across every
    residual evaluation, every candidate start, and every voxel sharing an AIF.
    Runners build one of these per voxel set and pass it to the `_*_curve`
    forward models; the public `model_*_cfit` functions build a throwaway one so
    their standalone signatures and validation are unchanged.

    Holds only the two vectors it needs rather than closing over the caller's
    scope, so it never pins a voxel-sized array alive.

    cp is kept in both forms on purpose: the O(n) recurrences below index it as
    a list (faster than element access on an ndarray), while the forward models
    combine their results with it as an array.

    `cp` and `t` must already be lists of Python floats -- callers coerce.
    """

    __slots__ = ("cp", "t", "n", "cp_arr", "_cum")

    def __init__(self, cp: List[float], t: List[float]) -> None:
        self.cp = cp
        self.t = t
        self.n = len(t)
        self.cp_arr = np.asarray(cp, dtype=np.float64)
        self._cum: Optional[np.ndarray] = None

    @property
    def cum(self) -> np.ndarray:
        """Cumulative trapezoid of cp over t, computed at most once per instance."""
        if self._cum is None:
            self._cum = np.asarray(_cumulative_trapz_values(self.cp, self.t), dtype=np.float64)
        return self._cum


def _coerce_forward_inputs(cp: Iterable[float], t: Iterable[float], t_name: str) -> ForwardInputs:
    """Build a `ForwardInputs` from arbitrary iterables, with the public length check."""
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in t]
    if len(cp_vec) != len(t_vec):
        raise ValueError(f"cp and {t_name} lengths differ: {len(cp_vec)} vs {len(t_vec)}")
    return ForwardInputs(cp_vec, t_vec)


def _empty_curve() -> np.ndarray:
    """Zero-length curve, returned fresh so callers may write into the result."""
    return np.zeros(0, dtype=np.float64)


# The `_*_curve` forward models below return float64 ndarrays, not lists: their
# only hot caller is a `least_squares`/`curve_fit` residual, which needs an
# array and would otherwise re-convert element by element on every evaluation.
# The sequential O(n) recurrences they build on stay in pure Python -- numpy
# would reassociate them -- so only the final elementwise combination is
# vectorized, which is bit-for-bit the same arithmetic in the same order. The
# public `model_*_cfit` wrappers hand back lists, as the MATLAB ports they are
# documented against do.


def _tofts_curve(ktrans: float, ve: float, fi: ForwardInputs) -> np.ndarray:
    if fi.n == 0:
        return _empty_curve()
    if ve == 0:
        raise ValueError("ve must be non-zero")

    kep = float(ktrans) / float(ve)
    conv = _exp_weighted_cumulative_trapz_values(fi.cp, fi.t, kep)
    return float(ktrans) * np.asarray(conv, dtype=np.float64)


def model_tofts_cfit(ktrans: float, ve: float, cp: Iterable[float], t1: Iterable[float]) -> List[float]:
    """Port of `dce/model_tofts_cfit.m`.

    MATLAB reference:
      Ct(k) = Ktrans * trapz(T, CP .* exp((-Ktrans/ve) * (T(end)-T)))
      with Ct(1) = 0

    Args:
      ktrans: Ktrans parameter.
      ve: ve parameter.
      cp: Plasma concentration time series.
      t1: Time vector.

    Returns:
      Ct vector as a Python list (same length as inputs).
    """
    return _tofts_curve(ktrans, ve, _coerce_forward_inputs(cp, t1, "t1")).tolist()


def _patlak_curve(ktrans: float, vp: float, fi: ForwardInputs) -> np.ndarray:
    if fi.n == 0:
        return _empty_curve()

    return (float(ktrans) * fi.cum) + (float(vp) * fi.cp_arr)


def model_patlak_cfit(ktrans: float, vp: float, cp: Iterable[float], time: Iterable[float]) -> List[float]:
    """Port of `dce/model_patlak_cfit.m`.

    MATLAB reference:
      Ct(k) = Ktrans * trapz(T, CP) + vp * Cp(k)
      with Ct(1) = vp * Cp(1) since trapz term is zero at first sample.

    Args:
      ktrans: Ktrans parameter.
      vp: vp parameter.
      cp: Plasma concentration time series.
      time: Time vector.

    Returns:
      Ct vector as a Python list (same length as inputs).
    """
    return _patlak_curve(ktrans, vp, _coerce_forward_inputs(cp, time, "time")).tolist()


def _vp_curve(vp: float, fi: ForwardInputs) -> np.ndarray:
    return vp * fi.cp_arr


def model_vp_cfit(vp: float, cp: Iterable[float], time: Iterable[float]) -> List[float]:
    """Port of `dce/model_vp_cfit.m`."""
    return _vp_curve(vp, _coerce_forward_inputs(cp, time, "time")).tolist()


def _tissue_uptake_curve(ktrans: float, fp: float, tp: float, fi: ForwardInputs) -> np.ndarray:
    if fi.n == 0:
        return _empty_curve()
    if tp == 0:
        raise ValueError("tp must be non-zero")

    weighted = _exp_weighted_cumulative_trapz_values(fi.cp, fi.t, 1.0 / float(tp))
    scale_exp = float(fp) - float(ktrans)
    return (float(ktrans) * fi.cum) + (scale_exp * np.asarray(weighted, dtype=np.float64))


def model_tissue_uptake_cfit(
    ktrans: float,
    fp: float,
    tp: float,
    cp: Iterable[float],
    t: Iterable[float],
) -> List[float]:
    """Port of `dce/model_tissue_uptake_cfit.m`."""
    return _tissue_uptake_curve(ktrans, fp, tp, _coerce_forward_inputs(cp, t, "t")).tolist()


def _2cxm_curve(ktrans: float, ve: float, vp: float, fp: float, fi: ForwardInputs) -> np.ndarray:
    cp_vec = fi.cp
    t_vec = fi.t

    if fi.n == 0:
        return _empty_curve()
    if vp + ve == 0:
        raise ValueError("vp + ve must be non-zero")

    if ktrans >= fp:
        ps = 1e8
    else:
        if fp - ktrans == 0:
            raise ValueError("fp - ktrans must be non-zero")
        ps = ktrans * fp / (fp - ktrans)

    e_big = ps / (ps + fp)
    e_small = ve / (vp + ve)

    root_num = 4.0 * e_big * e_small * (1.0 - e_big) * (1.0 - e_small)
    root_den = (e_big - e_big * e_small + e_small) ** 2
    root_arg = 1.0 - (root_num / root_den)
    if root_arg < 0.0 and abs(root_arg) < 1e-15:
        root_arg = 0.0
    sqrt_term = math.sqrt(root_arg)

    tau_scale = (e_big - e_big * e_small + e_small) / (2.0 * e_big)
    tau_plus = tau_scale * (1.0 + sqrt_term)
    tau_minus = tau_scale * (1.0 - sqrt_term)

    k_plus = fp / ((vp + ve) * tau_minus)
    k_minus = fp / ((vp + ve) * tau_plus)

    tau_diff = tau_plus - tau_minus
    if tau_diff == 0:
        raise ValueError("tau_plus and tau_minus collapse to same value")

    f_plus = fp * (tau_plus - 1.0) / tau_diff
    f_minus = -fp * (tau_minus - 1.0) / tau_diff

    conv_plus = _exp_weighted_cumulative_trapz_values(cp_vec, t_vec, float(k_plus))
    conv_minus = _exp_weighted_cumulative_trapz_values(cp_vec, t_vec, float(k_minus))

    ct = (float(f_plus) * np.asarray(conv_plus, dtype=np.float64)) + (
        float(f_minus) * np.asarray(conv_minus, dtype=np.float64)
    )
    # Overflow in the exponentials can leave inf/nan; those samples read as zero.
    ct[~np.isfinite(ct)] = 0.0
    return ct


def model_2cxm_cfit(
    ktrans: float,
    ve: float,
    vp: float,
    fp: float,
    cp: Iterable[float],
    t1: Iterable[float],
) -> List[float]:
    """Port of `dce/model_2cxm_cfit.m`."""
    return _2cxm_curve(ktrans, ve, vp, fp, _coerce_forward_inputs(cp, t1, "t1")).tolist()


def _fxr_curve(
    ktrans: float,
    ve: float,
    tau: float,
    fi: ForwardInputs,
    r1o: float,
    r1i: float,
    r1: float,
    fw: float,
) -> np.ndarray:
    cp_vec = fi.cp
    t_vec = fi.t

    if fi.n == 0:
        return _empty_curve()
    if ve == 0:
        raise ValueError("ve must be non-zero")
    if fw == 0:
        raise ValueError("fw must be non-zero")
    if tau <= 0:
        raise ValueError("tau must be positive")

    po = ve / fw
    if po == 0:
        raise ValueError("ve/fw must be non-zero")

    conv = _exp_weighted_cumulative_trapz_values(cp_vec, t_vec, float(ktrans) / float(ve))
    ct = float(ktrans) * np.asarray(conv, dtype=np.float64)

    xx = (r1o - r1i + (1.0 / tau)) / po
    with np.errstate(over="ignore", invalid="ignore"):
        inner = (2.0 / tau) - (r1 * ct) - xx
        yy = (inner * inner) + (4.0 * (1.0 - po) / (tau * tau * po))
    # `x ** 2` on a Python float raises OverflowError when the square is not
    # representable, and `model_fxr_fit`'s residual leans on that to bail out of
    # a runaway trial step. numpy quietly yields inf instead, so squaring a
    # finite value into infinity is re-raised here to keep that control flow.
    if np.any(np.isinf(yy) & np.isfinite(inner)):
        raise OverflowError("fxr: squared term is too large")
    np.maximum(yy, 0.0, out=yy)
    return 0.5 * ((2.0 * r1i) + (r1 * ct) + xx - np.sqrt(yy))


def model_fxr_cfit(
    ktrans: float,
    ve: float,
    tau: float,
    cp: Iterable[float],
    t1: Iterable[float],
    r1o: float,
    r1i: float,
    r1: float,
    fw: float,
) -> List[float]:
    """Port of `dce/model_fxr_cfit.m`."""
    return _fxr_curve(
        ktrans, ve, tau, _coerce_forward_inputs(cp, t1, "t1"), r1o, r1i, r1, fw
    ).tolist()


def _extended_tofts_curve(ktrans: float, ve: float, vp: float, fi: ForwardInputs) -> np.ndarray:
    if fi.n == 0:
        return _empty_curve()
    if ve == 0:
        raise ValueError("ve must be non-zero")

    conv = _exp_weighted_cumulative_trapz_values(fi.cp, fi.t, float(ktrans) / float(ve))
    return (float(ktrans) * np.asarray(conv, dtype=np.float64)) + (float(vp) * fi.cp_arr)


def model_extended_tofts_cfit(
    ktrans: float,
    ve: float,
    vp: float,
    cp: Iterable[float],
    t1: Iterable[float],
) -> List[float]:
    """Port of `dce/model_extended_tofts_cfit.m`.

    MATLAB reference:
      Ct(k) = Ktrans * trapz(T, CP .* exp((-Ktrans/ve) * (T(end)-T))) + vp * Cp(k)
      with the trapz term set to zero at first sample.
    """
    return _extended_tofts_curve(ktrans, ve, vp, _coerce_forward_inputs(cp, t1, "t1")).tolist()


# --- Batched curve evaluation -------------------------------------------------
#
# The `*_cfit_batch` functions below evaluate one forward curve per voxel for many
# voxels sharing a single cp/timer. They exist for the post-fit residual path,
# which re-evaluates every fitted voxel in Python regardless of which backend did
# the fitting.
#
# Two conventions differ from the scalar functions, both deliberate:
#   - Parameters that would make the scalar function raise (a zero denominator, a
#     negative sqrt) yield a NaN column instead, so one bad voxel cannot abort a
#     batch of 160k.
#   - Python raises ZeroDivisionError on float division by exact zero where numpy
#     returns inf, so each such divide is masked rather than left to propagate.
#     That is what keeps a degenerate voxel NaN here and skipped there.


def _cfit_batch_inputs(
    cp: Iterable[float], t: Iterable[float], *params: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Coerce the shared cp/timer and the per-voxel parameter vectors."""
    cp_arr = np.asarray(cp, dtype=np.float64).reshape(-1)
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    if cp_arr.size != t_arr.size:
        raise ValueError(f"cp and t lengths differ: {cp_arr.size} vs {t_arr.size}")
    rows = [np.asarray(p, dtype=np.float64).reshape(-1) for p in params]
    if any(r.size != rows[0].size for r in rows):
        raise ValueError("parameter vectors have differing lengths")
    return cp_arr, t_arr, rows


def _safe_denominator(den: np.ndarray) -> np.ndarray:
    """Replace exact zeros with 1.0 so a masked-off voxel cannot divide by zero."""
    return np.where(den != 0.0, den, 1.0)


def model_tofts_cfit_batch(
    ktrans: Iterable[float], ve: Iterable[float], cp: Iterable[float], t1: Iterable[float]
) -> np.ndarray:
    """Batched `model_tofts_cfit`. Returns (n_time, n_voxels)."""
    cp_arr, t_arr, (k, v) = _cfit_batch_inputs(cp, t1, ktrans, ve)
    ok = v != 0.0
    with np.errstate(all="ignore"):
        conv, overflowed = _exp_weighted_cumulative_trapz_batch(
            cp_arr, t_arr, k / _safe_denominator(v)
        )
        out = k[None, :] * conv
    out[:, ~ok | overflowed] = np.nan
    return out


def model_extended_tofts_cfit_batch(
    ktrans: Iterable[float],
    ve: Iterable[float],
    vp: Iterable[float],
    cp: Iterable[float],
    t1: Iterable[float],
) -> np.ndarray:
    """Batched `model_extended_tofts_cfit`. Returns (n_time, n_voxels)."""
    cp_arr, t_arr, (k, v, p) = _cfit_batch_inputs(cp, t1, ktrans, ve, vp)
    ok = v != 0.0
    with np.errstate(all="ignore"):
        conv, overflowed = _exp_weighted_cumulative_trapz_batch(
            cp_arr, t_arr, k / _safe_denominator(v)
        )
        out = (k[None, :] * conv) + (p[None, :] * cp_arr[:, None])
    out[:, ~ok | overflowed] = np.nan
    return out


def model_patlak_cfit_batch(
    ktrans: Iterable[float], vp: Iterable[float], cp: Iterable[float], time: Iterable[float]
) -> np.ndarray:
    """Batched `model_patlak_cfit`. Returns (n_time, n_voxels).

    No exponential weighting, so the integral is voxel-independent and the whole
    model collapses to two outer products.
    """
    cp_arr, t_arr, (k, p) = _cfit_batch_inputs(cp, time, ktrans, vp)
    cum = np.asarray(_cumulative_trapz_values(cp_arr, t_arr), dtype=np.float64)
    with np.errstate(all="ignore"):
        return (k[None, :] * cum[:, None]) + (p[None, :] * cp_arr[:, None])


def model_tissue_uptake_cfit_batch(
    ktrans: Iterable[float],
    fp: Iterable[float],
    tp: Iterable[float],
    cp: Iterable[float],
    t: Iterable[float],
) -> np.ndarray:
    """Batched `model_tissue_uptake_cfit`. Returns (n_time, n_voxels)."""
    cp_arr, t_arr, (k, f, tp_arr) = _cfit_batch_inputs(cp, t, ktrans, fp, tp)
    ok = tp_arr != 0.0
    cum = np.asarray(_cumulative_trapz_values(cp_arr, t_arr), dtype=np.float64)
    with np.errstate(all="ignore"):
        weighted, overflowed = _exp_weighted_cumulative_trapz_batch(
            cp_arr, t_arr, 1.0 / _safe_denominator(tp_arr)
        )
        out = (k[None, :] * cum[:, None]) + ((f - k)[None, :] * weighted)
    out[:, ~ok | overflowed] = np.nan
    return out


def model_2cxm_cfit_batch(
    ktrans: Iterable[float],
    ve: Iterable[float],
    vp: Iterable[float],
    fp: Iterable[float],
    cp: Iterable[float],
    t1: Iterable[float],
) -> np.ndarray:
    """Batched `model_2cxm_cfit`. Returns (n_time, n_voxels).

    Each guard the scalar function raises on becomes a mask; `ok` accumulates them
    and the masked-off columns are set to NaN at the end. Non-finite curve values
    are clamped to 0.0, matching the scalar function rather than the masks.
    """
    cp_arr, t_arr, (k, v, p, f) = _cfit_batch_inputs(cp, t1, ktrans, ve, vp, fp)
    v_sum = p + v
    ok = v_sum != 0.0

    with np.errstate(all="ignore"):
        # ktrans >= fp saturates PS rather than dividing by a non-positive gap.
        saturated = k >= f
        ps = np.where(saturated, 1e8, (k * f) / _safe_denominator(np.where(saturated, 1.0, f - k)))

        den_e = ps + f
        ok &= den_e != 0.0
        e_big = ps / _safe_denominator(den_e)
        e_small = v / _safe_denominator(v_sum)

        scale_num = (e_big - (e_big * e_small)) + e_small
        root_num = (((4.0 * e_big) * e_small) * (1.0 - e_big)) * (1.0 - e_small)
        # np.power(x, 2.0) routes to libm pow, matching the scalar `** 2`; `x ** 2`
        # on an array becomes x*x and differs in the last ULP.
        root_den = np.power(scale_num, 2.0)
        # Python's `**` raises OverflowError when a finite base squares past
        # DBL_MAX; np.power returns inf. (`inf ** 2` returns inf in both.)
        ok &= ~(np.isfinite(scale_num) & np.isinf(root_den))
        ok &= root_den != 0.0
        root_arg = 1.0 - (root_num / _safe_denominator(root_den))
        root_arg = np.where((root_arg < 0.0) & (np.abs(root_arg) < 1e-15), 0.0, root_arg)
        # math.sqrt raises on a negative; numpy would return NaN.
        ok &= ~(root_arg < 0.0)
        sqrt_term = np.sqrt(np.where(root_arg < 0.0, 0.0, root_arg))

        den_tau = 2.0 * e_big
        ok &= den_tau != 0.0
        tau_scale = scale_num / _safe_denominator(den_tau)
        tau_plus = tau_scale * (1.0 + sqrt_term)
        tau_minus = tau_scale * (1.0 - sqrt_term)

        den_plus = v_sum * tau_minus
        den_minus = v_sum * tau_plus
        ok &= (den_plus != 0.0) & (den_minus != 0.0)
        k_plus = f / _safe_denominator(den_plus)
        k_minus = f / _safe_denominator(den_minus)

        tau_diff = tau_plus - tau_minus
        ok &= tau_diff != 0.0
        safe_diff = _safe_denominator(tau_diff)
        f_plus = (f * (tau_plus - 1.0)) / safe_diff
        f_minus = ((-f) * (tau_minus - 1.0)) / safe_diff

        conv_plus, over_plus = _exp_weighted_cumulative_trapz_batch(cp_arr, t_arr, k_plus)
        conv_minus, over_minus = _exp_weighted_cumulative_trapz_batch(cp_arr, t_arr, k_minus)
        ok &= ~(over_plus | over_minus)
        ct = (f_plus[None, :] * conv_plus) + (f_minus[None, :] * conv_minus)
        ct = np.where(np.isfinite(ct), ct, 0.0)

    ct[:, ~ok] = np.nan
    return ct


def model_fxr_cfit_batch(
    ktrans: Iterable[float],
    ve: Iterable[float],
    tau: Iterable[float],
    cp: Iterable[float],
    t1: Iterable[float],
    r1o: Iterable[float],
    r1i: Iterable[float],
    r1: float,
    fw: float,
) -> np.ndarray:
    """Batched `model_fxr_cfit`. Returns (n_time, n_voxels).

    `r1o`/`r1i` are per-voxel; `r1` (relaxivity) and `fw` are shared scalars.
    """
    cp_arr, t_arr, (k, v, tau_arr, r1o_arr, r1i_arr) = _cfit_batch_inputs(
        cp, t1, ktrans, ve, tau, r1o, r1i
    )
    if float(fw) == 0.0:
        return np.full((t_arr.size, k.size), np.nan, dtype=np.float64)

    # po == 0 exactly when ve == 0, so it needs no separate mask.
    ok = (v != 0.0) & (tau_arr > 0.0)
    safe_tau = _safe_denominator(np.where(ok, tau_arr, 1.0))
    with np.errstate(all="ignore"):
        po = v / float(fw)
        safe_po = _safe_denominator(po)
        conv, overflowed = _exp_weighted_cumulative_trapz_batch(
            cp_arr, t_arr, k / _safe_denominator(v)
        )
        ok &= ~overflowed
        ct = k[None, :] * conv

        xx = ((r1o_arr - r1i_arr) + (1.0 / safe_tau)) / safe_po
        r1_ct = float(r1) * ct
        term = ((2.0 / safe_tau)[None, :] - r1_ct) - xx[None, :]
        term_sq = np.power(term, 2.0)
        # See the same guard in model_2cxm_cfit_batch: finite base, overflowing square.
        ok &= ~np.any(np.isfinite(term) & np.isinf(term_sq), axis=0)
        yy = term_sq + ((4.0 * (1.0 - safe_po)) / ((safe_tau * safe_tau) * safe_po))[None, :]
        yy = np.where(yy < 0.0, 0.0, yy)
        out = 0.5 * ((((2.0 * r1i_arr)[None, :] + r1_ct) + xx[None, :]) - np.sqrt(yy))

    out[:, ~ok] = np.nan
    return out


def model_patlak_linear(ct: Iterable[float], cp: Iterable[float], timer: Iterable[float]) -> List[float]:
    """Port of `dce/model_patlak_linear.m`.

    Returns MATLAB-style 7-value result vector:
      [ktrans, vp, sse, -1, -1, -1, -1]
    """
    ct_vec = [float(v) for v in ct]
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in timer]

    if not (len(ct_vec) == len(cp_vec) == len(t_vec)):
        raise ValueError(
            f"ct/cp/timer lengths differ: {len(ct_vec)} / {len(cp_vec)} / {len(t_vec)}"
        )
    if len(t_vec) < 2:
        return [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]

    tiny = 1e-12
    y_value = [float("nan")] * len(t_vec)
    x_value = [float("nan")] * len(t_vec)
    cum = _cumulative_trapz_values(cp_vec, t_vec)

    for t in range(len(t_vec)):
        denom = cp_vec[t]
        if abs(denom) > tiny:
            y_value[t] = ct_vec[t] / denom
            x_value[t] = cum[t] / denom

    # Trim to after injection (MATLAB drops first element)
    x_trim = x_value[1:]
    y_trim = y_value[1:]
    valid = [math.isfinite(x_trim[i]) and math.isfinite(y_trim[i]) for i in range(len(x_trim))]
    x_trim = [x_trim[i] for i in range(len(x_trim)) if valid[i]]
    y_trim = [y_trim[i] for i in range(len(y_trim)) if valid[i]]
    if len(x_trim) < 2:
        return [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]

    x_bar = sum(x_trim) / len(x_trim)
    y_bar = sum(y_trim) / len(y_trim)

    x_centered = [x - x_bar for x in x_trim]
    y_centered = [y - y_bar for y in y_trim]

    # Each reduction is bound once; `model_patlak_linear_batch` already does the
    # same, and the two must stay structurally aligned to keep their last-ULP
    # agreement.
    sum_x2 = sum(x * x for x in x_centered)
    sum_y2 = sum(y * y for y in y_centered)
    sum_xy = sum(x * y for x, y in zip(x_centered, y_centered))

    slope = 0.0 if sum_x2 == 0.0 else sum_xy / sum_x2
    intercept = y_bar - (slope * x_bar)

    denom = math.sqrt(sum_x2 * sum_y2)
    r_squared = 0.0 if denom == 0.0 else (sum_xy / denom) ** 2

    sum_squared_error = (1.0 - r_squared) * sum_y2

    return [slope, intercept, sum_squared_error, -1.0, -1.0, -1.0, -1.0]


def _sequential_sum_over_time(values: np.ndarray) -> np.ndarray:
    """Left-to-right sum along axis 0, one row at a time.

    Do not replace with `.sum(axis=0)`: numpy blocks axis-0 reductions once an
    array is wide enough, which reorders the summation and breaks the last-ULP
    agreement with `model_patlak_linear`. Row-by-row costs the same.
    """
    total = np.zeros(values.shape[1], dtype=np.float64)
    for row in values:
        total += row
    return total


def model_patlak_linear_batch(
    ct: np.ndarray, cp: Iterable[float], timer: Iterable[float]
) -> np.ndarray:
    """Vectorized `model_patlak_linear` over many voxels sharing one cp/timer.

    `ct` is (n_time,) or (n_time, n_voxels); returns (n_voxels, 7) rows in the
    same layout the scalar function returns, bit-identical to calling the scalar
    function per column.
    """
    cp_arr = np.asarray(cp, dtype=np.float64).reshape(-1)
    t_arr = np.asarray(timer, dtype=np.float64).reshape(-1)
    ct_arr = np.asarray(ct, dtype=np.float64)
    if ct_arr.ndim == 1:
        ct_arr = ct_arr[:, None]

    n_time = t_arr.size
    if not (ct_arr.shape[0] == cp_arr.size == n_time):
        raise ValueError(
            f"ct/cp/timer lengths differ: {ct_arr.shape[0]} / {cp_arr.size} / {n_time}"
        )

    out = np.zeros((ct_arr.shape[1], 7), dtype=np.float64)
    out[:, 3:] = -1.0
    if n_time < 2:
        return out

    # Sequential accumulation, matching _cumulative_trapz_values exactly.
    cum = np.zeros(n_time, dtype=np.float64)
    np.cumsum(0.5 * np.diff(t_arr) * (cp_arr[:-1] + cp_arr[1:]), out=cum[1:])

    usable = np.abs(cp_arr) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        x_row = np.where(usable, cum / cp_arr, np.nan)
        y_all = np.where(usable[:, None], ct_arr / cp_arr[:, None], np.nan)

    # MATLAB model_patlak_linear.m drops the first frame.
    x_row = x_row[1:]
    y_all = y_all[1:]
    x_all = np.broadcast_to(x_row[:, None], y_all.shape)
    valid = np.isfinite(x_row)[:, None] & np.isfinite(y_all)
    counts = valid.sum(axis=0)

    # Invalid slots are zeroed rather than compacted; adding 0.0 is exact, so the
    # sums match the scalar function's sums over its filtered lists.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x_bar = _sequential_sum_over_time(np.where(valid, x_all, 0.0)) / counts
        y_bar = _sequential_sum_over_time(np.where(valid, y_all, 0.0)) / counts
        x_centered = np.where(valid, x_all - x_bar, 0.0)
        y_centered = np.where(valid, y_all - y_bar, 0.0)
        sum_x2 = _sequential_sum_over_time(x_centered * x_centered)
        sum_y2 = _sequential_sum_over_time(y_centered * y_centered)
        sum_xy = _sequential_sum_over_time(x_centered * y_centered)

        slope = np.where(sum_x2 == 0.0, 0.0, sum_xy / sum_x2)
        intercept = y_bar - (slope * x_bar)
        denom = np.sqrt(sum_x2 * sum_y2)
        # np.power(x, 2.0) routes to libm pow, matching the scalar function's `** 2`;
        # `x ** 2` on an array would become x*x, which differs in the last ULP.
        r_squared = np.where(denom == 0.0, 0.0, np.power(sum_xy / denom, 2.0))
        sum_squared_error = (1.0 - r_squared) * sum_y2

    fitted = counts >= 2
    out[fitted, 0] = slope[fitted]
    out[fitted, 1] = intercept[fitted]
    out[fitted, 2] = sum_squared_error[fitted]
    return out


def model_patlak_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_patlak.m`.

    Thin single-voxel wrapper over the shared Stage-D fit machinery in
    `dce_fit_backends` (the same candidate-assembly/multi-start code path
    used by the accelerated cpufit/gpufit backends for this model).
    """
    from dce_fit_backends import fit_patlak_stage_d  # local: avoids an import cycle

    row = fit_patlak_stage_d(
        np.asarray([float(v) for v in ct], dtype=np.float64),
        np.asarray([float(v) for v in cp], dtype=np.float64),
        np.asarray([float(v) for v in timer], dtype=np.float64),
        prefs,
        backend="python",
    )
    return [float(v) for v in row]


def model_tofts_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_tofts.m`.

    Thin single-voxel wrapper over the shared Stage-D fit machinery in
    `dce_fit_backends` (the same candidate-assembly/multi-start code path
    used by the accelerated cpufit/gpufit backends for this model).

    Returns MATLAB-style 7-value output:
      [Ktrans, ve, sse, ktrans_ci_low, ktrans_ci_high, ve_ci_low, ve_ci_high]
    """
    from dce_fit_backends import fit_tofts_stage_d  # local: avoids an import cycle

    row = fit_tofts_stage_d(
        np.asarray([float(v) for v in ct], dtype=np.float64),
        np.asarray([float(v) for v in cp], dtype=np.float64),
        np.asarray([float(v) for v in timer], dtype=np.float64),
        prefs,
        backend="python",
    )
    return [float(v) for v in row]


def model_extended_tofts_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_extended_tofts.m`.

    Thin single-voxel wrapper over the shared Stage-D fit machinery in
    `dce_fit_backends` (the same candidate-assembly/multi-start code path
    used by the accelerated cpufit/gpufit backends for this model).
    """
    from dce_fit_backends import fit_ex_tofts_stage_d  # local: avoids an import cycle

    row = fit_ex_tofts_stage_d(
        np.asarray([float(v) for v in ct], dtype=np.float64),
        np.asarray([float(v) for v in cp], dtype=np.float64),
        np.asarray([float(v) for v in timer], dtype=np.float64),
        prefs,
        backend="python",
    )
    return [float(v) for v in row]


def _clip_start_to_bounds(start: List[float], lb: List[float], ub: List[float]) -> List[float]:
    out = []
    for i, val in enumerate(start):
        lo = float(lb[i])
        hi = float(ub[i])
        out.append(min(max(float(val), lo), hi))
    return out


def model_vp_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_vp.m`."""
    ct_vec = [float(v) for v in ct]
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in timer]

    if not (len(ct_vec) == len(cp_vec) == len(t_vec)):
        raise ValueError(
            f"ct/cp/timer lengths differ: {len(ct_vec)} / {len(cp_vec)} / {len(t_vec)}"
        )
    if len(t_vec) == 0:
        raise ValueError("ct/cp/timer must be non-empty")

    settings = {
        "lower_limit_vp": 1e-3,
        "upper_limit_vp": 1.0,
        "initial_value_vp": 0.02,
        "max_nfev": 2000,
        "tol_fun": 1e-12,
        "tol_x": 1e-6,
        "robust": "off",
    }
    if prefs:
        settings.update(prefs)

    fi = ForwardInputs(cp_vec, t_vec)
    ct_arr = np.asarray(ct_vec, dtype=np.float64)

    def residual(params: List[float]) -> np.ndarray:
        return _vp_curve(params[0], fi) - ct_arr

    lb = [float(settings["lower_limit_vp"])]
    ub = [float(settings["upper_limit_vp"])]

    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)
    fit = least_squares(
        residual,
        x0=_clip_start_to_bounds([float(settings["initial_value_vp"])], lb, ub),
        bounds=(lb, ub),
        **lsq_kwargs,
    )
    sse = float(sum(v * v for v in fit.fun.tolist()))
    vp = float(fit.x[0])

    ci_lo, ci_hi = _ci_bounds_from_fit(fit)
    return [vp, sse, ci_lo[0], ci_hi[0]]


def model_tissue_uptake_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_tissue_uptake.m`.

    Thin single-voxel wrapper over the shared Stage-D fit machinery in
    `dce_fit_backends` (the same candidate-assembly/multi-start code path
    used by the accelerated cpufit/gpufit backends for this model).
    """
    from dce_fit_backends import fit_tissue_uptake_stage_d  # local: avoids an import cycle

    row = fit_tissue_uptake_stage_d(
        np.asarray([float(v) for v in ct], dtype=np.float64),
        np.asarray([float(v) for v in cp], dtype=np.float64),
        np.asarray([float(v) for v in timer], dtype=np.float64),
        prefs,
        backend="python",
    )
    return [float(v) for v in row]


def model_2cxm_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_2cxm.m`.

    Thin single-voxel wrapper over the shared Stage-D fit machinery in
    `dce_fit_backends` (the same candidate-assembly/multi-start code path
    used by the accelerated cpufit/gpufit backends for this model).
    """
    from dce_fit_backends import fit_2cxm_stage_d  # local: avoids an import cycle

    row = fit_2cxm_stage_d(
        np.asarray([float(v) for v in ct], dtype=np.float64),
        np.asarray([float(v) for v in cp], dtype=np.float64),
        np.asarray([float(v) for v in timer], dtype=np.float64),
        prefs,
        backend="python",
    )
    if not math.isfinite(float(row[0])):
        raise ValueError("2cxm fit failed (requires >=3 strictly increasing time points).")
    return [float(v) for v in row]


def model_fxr_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    r1o: float,
    r1i: float,
    r1: float,
    fw: float,
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_fxr.m`."""
    ct_vec = [float(v) for v in ct]
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in timer]

    if not (len(ct_vec) == len(cp_vec) == len(t_vec)):
        raise ValueError(
            f"ct/cp/timer lengths differ: {len(ct_vec)} / {len(cp_vec)} / {len(t_vec)}"
        )
    if len(t_vec) == 0:
        raise ValueError("ct/cp/timer must be non-empty")

    settings = {
        "lower_limit_ktrans": 1e-7,
        "upper_limit_ktrans": 2.0,
        "initial_value_ktrans": 2e-4,
        "lower_limit_ve": 0.02,
        "upper_limit_ve": 1.0,
        "initial_value_ve": 0.2,
        "lower_limit_tau": 0.0,
        "upper_limit_tau": 100.0,
        "initial_value_tau": 0.01,
        "max_nfev": 2000,
        "tol_fun": 1e-12,
        "tol_x": 1e-6,
        "robust": "off",
    }
    if prefs:
        settings.update(prefs)

    fi = ForwardInputs(cp_vec, t_vec)
    ct_arr = np.asarray(ct_vec, dtype=np.float64)

    def residual(params: List[float]) -> np.ndarray:
        ktrans, ve, tau = params
        try:
            return _fxr_curve(ktrans, ve, tau, fi, r1o, r1i, r1, fw) - ct_arr
        except (ValueError, OverflowError):
            return np.full(ct_arr.size, 1e12, dtype=np.float64)

    lb = [
        float(settings["lower_limit_ktrans"]),
        float(settings["lower_limit_ve"]),
        float(settings["lower_limit_tau"]),
    ]
    ub = [
        float(settings["upper_limit_ktrans"]),
        float(settings["upper_limit_ve"]),
        float(settings["upper_limit_tau"]),
    ]
    x0 = _clip_start_to_bounds(
        [
            float(settings["initial_value_ktrans"]),
            float(settings["initial_value_ve"]),
            float(settings["initial_value_tau"]),
        ],
        lb,
        ub,
    )

    fit = least_squares(
        residual,
        x0=x0,
        bounds=(lb, ub),
        **_least_squares_kwargs(settings, default_max_nfev=2000),
    )

    ktrans = float(fit.x[0])
    ve = float(fit.x[1])
    tau = float(fit.x[2])
    sse = float(sum(v * v for v in fit.fun.tolist()))

    ci_lo, ci_hi = _ci_bounds_from_fit(fit)
    return [
        ktrans,
        ve,
        tau,
        sse,
        ci_lo[0],
        ci_hi[0],
        ci_lo[1],
        ci_hi[1],
        ci_lo[2],
        ci_hi[2],
    ]
