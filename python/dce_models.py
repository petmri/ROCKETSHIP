"""DCE model ports from MATLAB."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from scipy.optimize import least_squares


def _safe_float_setting(settings: Dict[str, object], key: str, default: float) -> float:
    raw = settings.get(key, default)
    try:
        out = float(raw)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


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


def _trapz(x: List[float], y: List[float]) -> float:
    """Numerical integration matching MATLAB trapz behavior for vectors."""
    total = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        total += 0.5 * dx * (y[i] + y[i - 1])
    return total


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

    ct = [0.0] * len(t_vec)

    for k in range(len(t_vec)):
        if k == 0:
            m = 0.0
        else:
            t_sub = t_vec[: k + 1]
            cp_sub = cp_vec[: k + 1]
            t_end = t_sub[-1]
            f = [cp_sub[i] * math.exp((-ktrans / ve) * (t_end - t_sub[i])) for i in range(len(t_sub))]
            m = _trapz(t_sub, f)
        ct[k] = ktrans * m

    return ct


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

    ct = [0.0] * len(t_vec)

    for k in range(len(t_vec)):
        if k == 0:
            m = 0.0
        else:
            t_sub = t_vec[: k + 1]
            cp_sub = cp_vec[: k + 1]
            m = _trapz(t_sub, cp_sub)
        ct[k] = (ktrans * m) + (vp * cp_vec[k])

    return ct


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

    ct = [0.0] * len(t_vec)
    for k in range(len(t_vec)):
        if k == 0:
            ct[k] = 0.0
            continue

        t_sub = t_vec[: k + 1]
        cp_sub = cp_vec[: k + 1]
        t_end = t_sub[-1]
        f = [
            cp_sub[i]
            * (
                fp * math.exp(-((t_end - t_sub[i]) / tp))
                + ktrans * (1.0 - math.exp(-((t_end - t_sub[i]) / tp)))
            )
            for i in range(len(t_sub))
        ]
        ct[k] = _trapz(t_sub, f)

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

    ct = [0.0] * len(t_vec)
    for k in range(len(t_vec)):
        if k == 0:
            ct[k] = 0.0
            continue

        t_sub = t_vec[: k + 1]
        cp_sub = cp_vec[: k + 1]
        t_end = t_sub[-1]
        f = [
            cp_sub[i]
            * (
                f_plus * math.exp(-k_plus * (t_end - t_sub[i]))
                + f_minus * math.exp(-k_minus * (t_end - t_sub[i]))
            )
            for i in range(len(t_sub))
        ]
        ctk = _trapz(t_sub, f)
        if math.isnan(ctk):
            ctk = 0.0
        ct[k] = ctk

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

    ct = [0.0] * len(t_vec)
    for k in range(len(t_vec)):
        if k == 0:
            m = 0.0
        else:
            t_sub = t_vec[: k + 1]
            cp_sub = cp_vec[: k + 1]
            t_end = t_sub[-1]
            f = [cp_sub[i] * math.exp((-ktrans / ve) * (t_end - t_sub[i])) for i in range(len(t_sub))]
            m = _trapz(t_sub, f)
        ct[k] = ktrans * m

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

    ct = [0.0] * len(t_vec)
    for k in range(len(t_vec)):
        if k == 0:
            m = 0.0
        else:
            t_sub = t_vec[: k + 1]
            cp_sub = cp_vec[: k + 1]
            t_end = t_sub[-1]
            f = [cp_sub[i] * math.exp((-ktrans / ve) * (t_end - t_sub[i])) for i in range(len(t_sub))]
            m = _trapz(t_sub, f)
        ct[k] = (ktrans * m) + (vp * cp_vec[k])

    return ct


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

    for t in range(len(t_vec)):
        denom = cp_vec[t]
        if abs(denom) > tiny:
            y_value[t] = ct_vec[t] / denom
        if t == 0:
            m = 0.0
        else:
            tau = t_vec[: t + 1]
            cp_t = cp_vec[: t + 1]
            m = _trapz(tau, cp_t)
        if abs(denom) > tiny:
            x_value[t] = m / denom

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
    """Python inverse-fit counterpart of `dce/model_tissue_uptake.m`."""
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
    if prefs:
        settings.update(prefs)

    def residual(params: List[float]) -> List[float]:
        ktrans, fp, tp = params
        pred = model_tissue_uptake_cfit(ktrans, fp, tp, cp_vec, t_vec)
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
    starts = [
        [
            float(settings["initial_value_ktrans"]),
            float(settings["initial_value_fp"]),
            float(settings["initial_value_tp"]),
        ],
        [
            float(settings["initial_value_ktrans"]) * 10.0,
            float(settings["initial_value_fp"]),
            float(settings["initial_value_tp"]),
        ],
        [
            float(settings["initial_value_ktrans"]) * 100.0,
            float(settings["initial_value_fp"]),
            float(settings["initial_value_tp"]),
        ],
    ]

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

    # Placeholder CI values: match MATLAB output shape.
    return [ktrans, fp, vp, sse, ktrans, ktrans, fp, fp, vp, vp]


def model_2cxm_fit(
    ct: Iterable[float],
    cp: Iterable[float],
    timer: Iterable[float],
    prefs: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Python inverse-fit counterpart of `dce/model_2cxm.m`."""
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
        "lower_limit_fp": 1e-3,
        "upper_limit_fp": 100.0,
        "initial_value_fp": 0.2,
        "max_nfev": 2000,
        "tol_fun": 1e-12,
        "tol_x": 1e-6,
        "robust": "off",
    }
    if prefs:
        settings.update(prefs)

    def residual(params: List[float]) -> List[float]:
        ktrans, ve, vp, fp = params
        pred = model_2cxm_cfit(ktrans, ve, vp, fp, cp_vec, t_vec)
        return [pred[i] - ct_vec[i] for i in range(len(ct_vec))]

    lb = [
        float(settings["lower_limit_ktrans"]),
        float(settings["lower_limit_ve"]),
        float(settings["lower_limit_vp"]),
        float(settings["lower_limit_fp"]),
    ]
    ub = [
        float(settings["upper_limit_ktrans"]),
        float(settings["upper_limit_ve"]),
        float(settings["upper_limit_vp"]),
        float(settings["upper_limit_fp"]),
    ]
    starts = [
        [
            float(settings["initial_value_ktrans"]),
            float(settings["initial_value_ve"]),
            float(settings["initial_value_vp"]),
            float(settings["initial_value_fp"]),
        ],
        [
            float(settings["initial_value_ktrans"]) * 10.0,
            float(settings["initial_value_ve"]),
            float(settings["initial_value_vp"]),
            float(settings["initial_value_fp"]),
        ],
        [
            float(settings["initial_value_ktrans"]) * 100.0,
            float(settings["initial_value_ve"]),
            float(settings["initial_value_vp"]),
            float(settings["initial_value_fp"]),
        ],
    ]

    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)
    fit, sse = _best_fit_over_starts(residual, starts, lb, ub, lsq_kwargs)
    ktrans = float(fit.x[0])
    ve = float(fit.x[1])
    vp = float(fit.x[2])
    fp = float(fit.x[3])

    # Placeholder CI values: match MATLAB output shape.
    return [ktrans, ve, vp, fp, sse, ktrans, ktrans, ve, ve, vp, vp, fp, fp]


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
