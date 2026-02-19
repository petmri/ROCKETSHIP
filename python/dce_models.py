"""DCE model ports from MATLAB."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, least_squares


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


def _cumulative_trapz_values(y: List[float], t: List[float]) -> List[float]:
    """Return cumulative trapezoid integral with output aligned to sample indices."""
    out = [0.0] * len(t)
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        out[i] = out[i - 1] + (0.5 * dt * (float(y[i - 1]) + float(y[i])))
    return out


def _exp_weighted_cumulative_trapz_values(y: List[float], t: List[float], lam: float) -> List[float]:
    """Compute trapz(t, y * exp(-lam * (t_k - t))) for each endpoint k in O(n)."""
    out = [0.0] * len(t)
    for k in range(1, len(t)):
        dt = float(t[k] - t[k - 1])
        decay = math.exp(-lam * dt)
        # Trapezoid on the new interval, weighted at t_k.
        interval = 0.5 * dt * ((float(y[k - 1]) * decay) + float(y[k]))
        out[k] = (decay * out[k - 1]) + interval
    return out


def _resample_for_osipi_style_fit(
    ct_vec: List[float],
    cp_vec: List[float],
    t_min_vec: List[float],
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Quadratic resampling used by OSIPI LEK reference implementations.

    Args:
      ct_vec: Tissue concentration samples.
      cp_vec: Plasma concentration samples.
      t_min_vec: Time samples in minutes.
    """
    if len(t_min_vec) < 3:
        return None

    t_arr = np.asarray(t_min_vec, dtype=np.float64)
    cp_arr = np.asarray(cp_vec, dtype=np.float64)
    ct_arr = np.asarray(ct_vec, dtype=np.float64)

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
    ct_interp_fn = interp1d(
        t_shift,
        ct_arr,
        kind="quadratic",
        bounds_error=False,
        fill_value=(0.0, float(ct_arr[-1])),
    )
    return t_interp, np.asarray(cp_interp_fn(t_interp), dtype=np.float64), np.asarray(
        ct_interp_fn(t_interp), dtype=np.float64
    )


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


def _fit_2cxm_osipi_canonical(
    ct_vec: List[float],
    cp_vec: List[float],
    t_min_vec: List[float],
    settings: Optional[Dict[str, float]] = None,
) -> Optional[List[float]]:
    """Return OSIPI LEK 2CXM fit in canonical units.

    Canonical units:
      - timer: minutes
      - ktrans/fp: per-minute
    """
    sample = _resample_for_osipi_style_fit(ct_vec, cp_vec, t_min_vec)
    if sample is None:
        return None
    t_interp, cp_interp, ct_interp = sample

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
    e_lo = min(max(ktrans_lo / max(fp_hi, 1e-12), 0.0), 1.0 - 1e-10)
    e_hi = min(max(ktrans_hi / max(fp_lo, 1e-12), e_lo + 1e-10), 1.0 - 1e-8)
    e0 = min(max(e0, e_lo + 1e-10), e_hi - 1e-10)
    maxfev = int(_safe_float_setting(settings, "max_nfev", 4000.0))

    try:
        fit, _ = curve_fit(
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

    return [
        ktrans_per_min,
        ve,
        vp,
        fp_per_min_ml_per_ml,
        sse,
        ktrans_per_min,
        ktrans_per_min,
        ve,
        ve,
        vp,
        vp,
        fp_per_min_ml_per_ml,
        fp_per_min_ml_per_ml,
    ]


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
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in t1]

    if len(cp_vec) != len(t_vec):
        raise ValueError(f"cp and t1 lengths differ: {len(cp_vec)} vs {len(t_vec)}")
    if len(t_vec) == 0:
        return []
    if ve == 0:
        raise ValueError("ve must be non-zero")

    kep = float(ktrans) / float(ve)
    conv = _exp_weighted_cumulative_trapz_values(cp_vec, t_vec, kep)
    return [float(ktrans) * conv[i] for i in range(len(conv))]


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
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in time]

    if len(cp_vec) != len(t_vec):
        raise ValueError(f"cp and time lengths differ: {len(cp_vec)} vs {len(t_vec)}")
    if len(t_vec) == 0:
        return []

    cum = _cumulative_trapz_values(cp_vec, t_vec)
    return [(float(ktrans) * cum[i]) + (float(vp) * cp_vec[i]) for i in range(len(t_vec))]


def model_vp_cfit(vp: float, cp: Iterable[float], time: Iterable[float]) -> List[float]:
    """Port of `dce/model_vp_cfit.m`."""
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in time]

    if len(cp_vec) != len(t_vec):
        raise ValueError(f"cp and time lengths differ: {len(cp_vec)} vs {len(t_vec)}")

    return [vp * cp_vec[i] for i in range(len(t_vec))]


def model_tissue_uptake_cfit(
    ktrans: float,
    fp: float,
    tp: float,
    cp: Iterable[float],
    t: Iterable[float],
) -> List[float]:
    """Port of `dce/model_tissue_uptake_cfit.m`."""
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in t]

    if len(cp_vec) != len(t_vec):
        raise ValueError(f"cp and t lengths differ: {len(cp_vec)} vs {len(t_vec)}")
    if len(t_vec) == 0:
        return []
    if tp == 0:
        raise ValueError("tp must be non-zero")

    cum = _cumulative_trapz_values(cp_vec, t_vec)
    weighted = _exp_weighted_cumulative_trapz_values(cp_vec, t_vec, 1.0 / float(tp))
    scale_exp = float(fp) - float(ktrans)
    return [float(ktrans) * cum[i] + (scale_exp * weighted[i]) for i in range(len(t_vec))]


def model_2cxm_cfit(
    ktrans: float,
    ve: float,
    vp: float,
    fp: float,
    cp: Iterable[float],
    t1: Iterable[float],
) -> List[float]:
    """Port of `dce/model_2cxm_cfit.m`."""
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in t1]

    if len(cp_vec) != len(t_vec):
        raise ValueError(f"cp and t1 lengths differ: {len(cp_vec)} vs {len(t_vec)}")
    if len(t_vec) == 0:
        return []
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

    ct = [0.0] * len(t_vec)
    for i in range(len(t_vec)):
        cti = (float(f_plus) * conv_plus[i]) + (float(f_minus) * conv_minus[i])
        if not math.isfinite(cti):
            cti = 0.0
        ct[i] = cti
    return ct


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
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in t1]

    if len(cp_vec) != len(t_vec):
        raise ValueError(f"cp and t1 lengths differ: {len(cp_vec)} vs {len(t_vec)}")
    if len(t_vec) == 0:
        return []
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
    ct = [float(ktrans) * conv[i] for i in range(len(t_vec))]

    xx = (r1o - r1i + (1.0 / tau)) / po
    r1t = [0.0] * len(ct)
    for i, ct_i in enumerate(ct):
        yy = ((2.0 / tau - r1 * ct_i - xx) ** 2) + (4.0 * (1.0 - po) / (tau * tau * po))
        if yy < 0:
            yy = 0.0
        r1t[i] = 0.5 * (2.0 * r1i + r1 * ct_i + xx - math.sqrt(yy))

    return r1t


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
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in t1]

    if len(cp_vec) != len(t_vec):
        raise ValueError(f"cp and t1 lengths differ: {len(cp_vec)} vs {len(t_vec)}")
    if len(t_vec) == 0:
        return []
    if ve == 0:
        raise ValueError("ve must be non-zero")

    conv = _exp_weighted_cumulative_trapz_values(cp_vec, t_vec, float(ktrans) / float(ve))
    return [(float(ktrans) * conv[i]) + (float(vp) * cp_vec[i]) for i in range(len(t_vec))]


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
        if abs(denom) > tiny:
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

    sum_x2 = sum(x * x for x in x_centered)
    if sum_x2 == 0.0:
        slope = 0.0
    else:
        slope = sum(x * y for x, y in zip(x_centered, y_centered)) / sum_x2
    intercept = y_bar - (slope * x_bar)

    numer = sum(x * y for x, y in zip(x_centered, y_centered))
    denom = math.sqrt(sum(x * x for x in x_centered) * sum(y * y for y in y_centered))
    if denom == 0.0:
        r_squared = 0.0
    else:
        r_squared = (numer / denom) ** 2

    sum_squared_error = (1.0 - r_squared) * sum(y * y for y in y_centered)

    if not math.isfinite(r_squared):
        r_squared = 0.0

    return [slope, intercept, sum_squared_error, -1.0, -1.0, -1.0, -1.0]


def model_patlak_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_patlak.m`.

    Mirrors MATLAB CPU behavior: linear Patlak estimate for start-point,
    then nonlinear least-squares on the forward Patlak model.
    """
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

    # Match MATLAB path: use linear Patlak estimate as nonlinear start point.
    try:
        estimate = model_patlak_linear(ct_vec, cp_vec, t_vec)
        ktrans_start = float(estimate[0])
        vp_start = float(estimate[1])
        if math.isfinite(ktrans_start):
            settings["initial_value_ktrans"] = ktrans_start
        if math.isfinite(vp_start):
            settings["initial_value_vp"] = vp_start
    except Exception:
        pass

    def residual(params: List[float]) -> List[float]:
        ktrans, vp = params
        pred = model_patlak_cfit(ktrans, vp, cp_vec, t_vec)
        return [pred[i] - ct_vec[i] for i in range(len(ct_vec))]

    lb = [
        float(settings["lower_limit_ktrans"]),
        float(settings["lower_limit_vp"]),
    ]
    ub = [
        float(settings["upper_limit_ktrans"]),
        float(settings["upper_limit_vp"]),
    ]

    starts = [
        [
            float(settings["initial_value_ktrans"]),
            float(settings["initial_value_vp"]),
        ],
        [
            float(settings["initial_value_ktrans"]) * 10.0,
            float(settings["initial_value_vp"]),
        ],
        [
            float(settings["initial_value_ktrans"]) * 100.0,
            float(settings["initial_value_vp"]),
        ],
    ]

    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)
    fit, sse = _best_fit_over_starts(residual, starts, lb, ub, lsq_kwargs)
    ktrans = float(fit.x[0])
    vp = float(fit.x[1])

    return [ktrans, vp, sse, ktrans, ktrans, vp, vp]


def model_tofts_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_tofts.m`.

    Returns MATLAB-style 7-value output:
      [Ktrans, ve, sse, ktrans_ci_low, ktrans_ci_high, ve_ci_low, ve_ci_high]

    Confidence interval values are approximated as the fit estimates for now.
    This preserves output shape and keeps parity checks stable for synthetic data.
    """
    ct_vec = [float(v) for v in ct]
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in timer]

    if not (len(ct_vec) == len(cp_vec) == len(t_vec)):
        raise ValueError(
            f"ct/cp/timer lengths differ: {len(ct_vec)} / {len(cp_vec)} / {len(t_vec)}"
        )
    if len(t_vec) == 0:
        raise ValueError("ct/cp/timer must be non-empty")

    # Defaults aligned with tests/matlab/helpers/default_dce_fit_prefs.m
    settings = {
        "lower_limit_ktrans": 1e-7,
        "upper_limit_ktrans": 2.0,
        "initial_value_ktrans": 2e-4,
        "lower_limit_ve": 0.02,
        "upper_limit_ve": 1.0,
        "initial_value_ve": 0.2,
        "max_nfev": 2000,
        "tol_fun": 1e-12,
        "tol_x": 1e-6,
        "robust": "off",
    }
    if prefs:
        settings.update(prefs)

    def residual(params: List[float]) -> List[float]:
        ktrans, ve = params
        pred = model_tofts_cfit(ktrans, ve, cp_vec, t_vec)
        return [pred[i] - ct_vec[i] for i in range(len(ct_vec))]

    x0 = [float(settings["initial_value_ktrans"]), float(settings["initial_value_ve"])]
    lb = [float(settings["lower_limit_ktrans"]), float(settings["lower_limit_ve"])]
    ub = [float(settings["upper_limit_ktrans"]), float(settings["upper_limit_ve"])]
    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)

    fit = least_squares(
        residual,
        x0=x0,
        bounds=(lb, ub),
        **lsq_kwargs,
    )

    ktrans = float(fit.x[0])
    ve = float(fit.x[1])
    sse = float(sum(v * v for v in fit.fun))

    # Placeholder CI values: match output shape expected by parity contracts.
    return [ktrans, ve, sse, ktrans, ktrans, ve, ve]


def model_extended_tofts_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_extended_tofts.m`."""
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

    def residual(params: List[float]) -> List[float]:
        ktrans, ve, vp = params
        pred = model_extended_tofts_cfit(ktrans, ve, vp, cp_vec, t_vec)
        return [pred[i] - ct_vec[i] for i in range(len(ct_vec))]

    lb = [
        float(settings["lower_limit_ktrans"]),
        float(settings["lower_limit_ve"]),
        float(settings["lower_limit_vp"]),
    ]
    ub = [
        float(settings["upper_limit_ktrans"]),
        float(settings["upper_limit_ve"]),
        float(settings["upper_limit_vp"]),
    ]
    starts = [
        [
            float(settings["initial_value_ktrans"]),
            float(settings["initial_value_ve"]),
            float(settings["initial_value_vp"]),
        ],
        [
            float(settings["initial_value_ktrans"]) * 10.0,
            float(settings["initial_value_ve"]),
            float(settings["initial_value_vp"]),
        ],
        [
            float(settings["initial_value_ktrans"]) * 100.0,
            float(settings["initial_value_ve"]),
            float(settings["initial_value_vp"]),
        ],
    ]

    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)
    fit, sse = _best_fit_over_starts(residual, starts, lb, ub, lsq_kwargs)
    ktrans = float(fit.x[0])
    ve = float(fit.x[1])
    vp = float(fit.x[2])

    # Placeholder CI values: match MATLAB output shape.
    return [ktrans, ve, vp, sse, ktrans, ktrans, ve, ve, vp, vp]


def _clip_start_to_bounds(start: List[float], lb: List[float], ub: List[float]) -> List[float]:
    out = []
    for i, val in enumerate(start):
        lo = float(lb[i])
        hi = float(ub[i])
        out.append(min(max(float(val), lo), hi))
    return out


def _best_fit_over_starts(
    residual_fn,
    starts: List[List[float]],
    lb: List[float],
    ub: List[float],
    lsq_kwargs: Dict[str, object],
):
    best_fit = None
    best_sse = math.inf
    for start in starts:
        x0 = _clip_start_to_bounds(start, lb, ub)
        fit = least_squares(
            residual_fn,
            x0=x0,
            bounds=(lb, ub),
            **lsq_kwargs,
        )
        sse = float(sum(v * v for v in fit.fun))
        if sse < best_sse:
            best_fit = fit
            best_sse = sse
    return best_fit, best_sse


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

    def residual(params: List[float]) -> List[float]:
        vp = params[0]
        pred = model_vp_cfit(vp, cp_vec, t_vec)
        return [pred[i] - ct_vec[i] for i in range(len(ct_vec))]

    lb = [float(settings["lower_limit_vp"])]
    ub = [float(settings["upper_limit_vp"])]
    starts = [[float(settings["initial_value_vp"])]]

    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)
    fit, sse = _best_fit_over_starts(residual, starts, lb, ub, lsq_kwargs)
    vp = float(fit.x[0])

    # Placeholder CI values: match MATLAB output shape.
    return [vp, sse, vp, vp]


def model_tissue_uptake_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_tissue_uptake.m`.

    Internal canonical units:
      - time in minutes
      - ktrans/fp in per-minute

    Returned ktrans/fp values are converted back to match input timer units.
    """
    ct_vec = [float(v) for v in ct]
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in timer]

    if not (len(ct_vec) == len(cp_vec) == len(t_vec)):
        raise ValueError(
            f"ct/cp/timer lengths differ: {len(ct_vec)} / {len(cp_vec)} / {len(t_vec)}"
        )
    if len(t_vec) == 0:
        raise ValueError("ct/cp/timer must be non-empty")

    _reject_algorithm_override(prefs, "tissue_uptake")
    timer_min, _, rate_in_to_min, rate_min_to_output = _canonical_time_context(
        t_vec,
        prefs,
    )

    defaults = {
        "lower_limit_ktrans": 1e-7,
        "upper_limit_ktrans": 2.0,
        "initial_value_ktrans": 2e-4,
        "lower_limit_fp": 1e-3,
        "upper_limit_fp": 100.0,
        "initial_value_fp": 0.2,
        "lower_limit_tp": 0.0,
        "upper_limit_tp": 1e6,
        "initial_value_tp": 0.05,
        "max_nfev": 2000,
        "tol_fun": 1e-12,
        "tol_x": 1e-6,
        "robust": "off",
    }
    settings = _merge_prefs_in_canonical_units(
        defaults,
        prefs,
        rate_keys=[
            "lower_limit_ktrans",
            "upper_limit_ktrans",
            "initial_value_ktrans",
            "lower_limit_fp",
            "upper_limit_fp",
            "initial_value_fp",
        ],
        time_constant_keys=[
            "lower_limit_tp",
            "upper_limit_tp",
            "initial_value_tp",
        ],
        rate_in_to_min=rate_in_to_min,
    )

    def residual(params: List[float]) -> List[float]:
        ktrans, fp, tp = params
        pred = model_tissue_uptake_cfit(ktrans, fp, tp, cp_vec, timer_min)
        return [pred[i] - ct_vec[i] for i in range(len(ct_vec))]

    lb = [
        float(settings["lower_limit_ktrans"]),
        float(settings["lower_limit_fp"]),
        float(settings["lower_limit_tp"]),
    ]
    ub = [
        float(settings["upper_limit_ktrans"]),
        float(settings["upper_limit_fp"]),
        float(settings["upper_limit_tp"]),
    ]
    k_seed = float(settings["initial_value_ktrans"])
    fp_seed = max(float(settings["initial_value_fp"]), k_seed * 1.25)
    tp_seed = float(settings["initial_value_tp"])

    starts = [
        [
            k_seed,
            fp_seed,
            tp_seed,
        ],
        [
            k_seed * 2.0,
            max(fp_seed * 1.5, k_seed * 1.6),
            tp_seed,
        ],
        [
            max(k_seed * 0.5, float(settings["lower_limit_ktrans"])),
            max(fp_seed * 0.8, k_seed * 1.25),
            min(tp_seed * 2.0, float(settings["upper_limit_tp"])),
        ],
        [
            min(k_seed * 4.0, float(settings["upper_limit_ktrans"])),
            max(fp_seed * 2.0, k_seed * 2.5),
            tp_seed,
        ],
    ]

    try:
        patlak = model_patlak_linear(ct_vec, cp_vec, timer_min)
        patlak_k = float(patlak[0])
        if math.isfinite(patlak_k):
            patlak_k = min(max(patlak_k, lb[0]), ub[0])
            starts.append(
                [
                    patlak_k,
                    min(max(max(fp_seed, patlak_k * 1.25), lb[1]), ub[1]),
                    min(max(tp_seed, lb[2]), ub[2]),
                ]
            )
    except Exception:
        pass

    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)
    fit, sse = _best_fit_over_starts(residual, starts, lb, ub, lsq_kwargs)
    ktrans = float(fit.x[0])
    fp = float(fit.x[1])
    tp = float(fit.x[2])

    if abs(fp - ktrans) < 1e-12:
        ps = 1e8
    else:
        ps = ktrans * fp / (fp - ktrans)
    vp = (fp + ps) * tp
    ktrans_out = ktrans * rate_min_to_output
    fp_out = fp * rate_min_to_output

    # Placeholder CI values: match MATLAB output shape.
    return [
        ktrans_out,
        fp_out,
        vp,
        sse,
        ktrans_out,
        ktrans_out,
        fp_out,
        fp_out,
        vp,
        vp,
    ]


def model_2cxm_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_2cxm.m`.

    Uses a single OSIPI LEK-style fit path with canonical internal units:
      - time in minutes
      - ktrans/fp in per-minute

    Returned ktrans/fp values are converted back to match input timer units.
    """
    ct_vec = [float(v) for v in ct]
    cp_vec = [float(v) for v in cp]
    t_vec = [float(v) for v in timer]

    if not (len(ct_vec) == len(cp_vec) == len(t_vec)):
        raise ValueError(
            f"ct/cp/timer lengths differ: {len(ct_vec)} / {len(cp_vec)} / {len(t_vec)}"
        )
    if len(t_vec) == 0:
        raise ValueError("ct/cp/timer must be non-empty")

    _reject_algorithm_override(prefs, "2cxm")
    timer_min, _, rate_in_to_min, rate_min_to_output = _canonical_time_context(
        t_vec,
        prefs,
    )

    defaults = {
        "lower_limit_ktrans": 1e-7,
        "upper_limit_ktrans": 2.0,
        "initial_value_ktrans": 2e-4,
        "lower_limit_ve": 0.02,
        "upper_limit_ve": 1.0,
        "initial_value_ve": 0.2,
        "lower_limit_vp": 1e-3,
        "upper_limit_vp": 1.0,
        "initial_value_vp": 0.02,
        "lower_limit_fp": 1e-3,
        "upper_limit_fp": 2.0,
        "initial_value_fp": 20.0 / 100.0,
        "max_nfev": 4000,
    }
    settings = _merge_prefs_in_canonical_units(
        defaults,
        prefs,
        rate_keys=[
            "lower_limit_ktrans",
            "upper_limit_ktrans",
            "initial_value_ktrans",
            "lower_limit_fp",
            "upper_limit_fp",
            "initial_value_fp",
        ],
        rate_in_to_min=rate_in_to_min,
    )

    quick = _fit_2cxm_osipi_canonical(ct_vec, cp_vec, timer_min, settings=settings)
    if quick is None:
        raise ValueError("2cxm fit failed (requires >=3 strictly increasing time points).")
    for idx in (0, 3, 5, 6, 11, 12):
        quick[idx] = float(quick[idx]) * rate_min_to_output
    return quick


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

    def residual(params: List[float]) -> List[float]:
        ktrans, ve, tau = params
        try:
            pred = model_fxr_cfit(ktrans, ve, tau, cp_vec, t_vec, r1o, r1i, r1, fw)
            return [pred[i] - ct_vec[i] for i in range(len(ct_vec))]
        except (ValueError, OverflowError):
            return [1e12] * len(ct_vec)

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
    sse = float(sum(v * v for v in fit.fun))

    # Placeholder CI values: match output shape expected by parity contracts.
    return [ktrans, ve, tau, sse, ktrans, ktrans, ve, ve, tau, tau]
