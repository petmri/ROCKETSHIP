"""Parametric model ports from MATLAB fitParameter.m."""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
from scipy.optimize import least_squares


def _linear_fit_stats(x_values: List[float], y_values: List[float]) -> tuple[float, float, float, float]:
    """Return slope, intercept, r_squared, sse using MATLAB-equivalent formulas."""
    x_bar = sum(x_values) / len(x_values)
    y_bar = sum(y_values) / len(y_values)

    x = [v - x_bar for v in x_values]
    y = [v - y_bar for v in y_values]

    sum_x2 = sum(v * v for v in x)
    if sum_x2 == 0.0:
        slope = 0.0
    else:
        slope = sum(a * b for a, b in zip(x, y)) / sum_x2

    intercept = y_bar - (slope * x_bar)

    numer = sum(a * b for a, b in zip(x, y))
    denom = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    if denom == 0.0:
        r_squared = 0.0
    else:
        r_squared = (numer / denom) ** 2

    sse = (1.0 - r_squared) * sum(v * v for v in y)

    if not math.isfinite(r_squared):
        r_squared = 0.0

    return slope, intercept, r_squared, sse


def t2_linear_fast(parameter: Iterable[float], si: Iterable[float]) -> List[float]:
    """Port of `fitParameter(..., 't2_linear_fast', ...)`.

    Returns MATLAB-style fit_output:
      [exponential_fit, rho_fit, r_squared, -1, -1, sse]
    """
    te = [float(v) for v in parameter]
    signal = [float(v) for v in si]

    if len(te) != len(signal):
        raise ValueError(f"parameter and si lengths differ: {len(te)} vs {len(signal)}")
    if len(te) == 0:
        raise ValueError("parameter/si must be non-empty")

    ln_si = [math.log(v) for v in signal]

    slope, intercept, r_squared, sse = _linear_fit_stats(te, ln_si)

    exponential_fit = -1.0 / slope if slope != 0.0 else math.inf
    rho_fit = intercept

    return [exponential_fit, rho_fit, r_squared, -1.0, -1.0, sse]


def t1_fa_linear_fit(parameter: Iterable[float], si: Iterable[float], tr: float) -> List[float]:
    """Port of `fitParameter(..., 't1_fa_linear_fit', ...)`.

    Returns MATLAB-style fit_output:
      [exponential_fit, rho_fit, r_squared, -1, -1, sse]
    """
    fa_deg = [float(v) for v in parameter]
    signal = [float(v) for v in si]
    tr = float(tr)

    if len(fa_deg) != len(signal):
        raise ValueError(f"parameter and si lengths differ: {len(fa_deg)} vs {len(signal)}")
    if len(fa_deg) == 0:
        raise ValueError("parameter/si must be non-empty")

    y_lin = [signal[i] / math.sin(math.pi / 180.0 * fa_deg[i]) for i in range(len(fa_deg))]
    x_lin = [signal[i] / math.tan(math.pi / 180.0 * fa_deg[i]) for i in range(len(fa_deg))]

    slope, intercept, r_squared, sse = _linear_fit_stats(x_lin, y_lin)

    rho_fit = intercept
    if slope < 0.0:
        exponential_fit = -0.4
    else:
        if slope == 1.0:
            exponential_fit = math.inf
        else:
            exponential_fit = -tr / math.log(slope)

    if exponential_fit < 0.0:
        exponential_fit = -0.5

    return [exponential_fit, rho_fit, r_squared, -1.0, -1.0, sse]


def _spgr_signal_ms(s0: float, t1_ms: float, tr_ms: float, fa_deg: np.ndarray) -> np.ndarray:
    """SPGR signal model with T1/TR in milliseconds and flip angle in degrees."""
    fa_rad = np.deg2rad(fa_deg)
    e1 = np.exp(-tr_ms / t1_ms)
    numer = (1.0 - e1) * np.sin(fa_rad)
    denom = 1.0 - e1 * np.cos(fa_rad)
    return np.abs(s0 * (numer / denom))


def _fit_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return (r_squared, sse)."""
    residual = y_true - y_pred
    sse = float(np.sum(residual * residual))
    centered = y_true - float(np.mean(y_true))
    sst = float(np.sum(centered * centered))
    if sst <= 0.0:
        return 0.0, sse
    r_squared = 1.0 - (sse / sst)
    if not math.isfinite(r_squared):
        r_squared = 0.0
    return float(r_squared), sse


def t1_fa_nonlinear_fit(parameter: Iterable[float], si: Iterable[float], tr: float) -> List[float]:
    """Port of `fitParameter(..., 't1_fa_fit', ...)` with ms-based inputs.

    Args:
        parameter: Flip angles in degrees.
        si: Signal intensities (arbitrary units).
        tr: Repetition time in milliseconds.

    Returns MATLAB-style fit_output:
      [exponential_fit_ms, rho_fit, r_squared, -1, -1, sse]
    """
    fa_deg = np.asarray([float(v) for v in parameter], dtype=np.float64)
    signal = np.asarray([float(v) for v in si], dtype=np.float64)
    tr_ms = float(tr)

    if fa_deg.shape[0] != signal.shape[0]:
        raise ValueError(f"parameter and si lengths differ: {fa_deg.shape[0]} vs {signal.shape[0]}")
    if fa_deg.size == 0:
        raise ValueError("parameter/si must be non-empty")
    if tr_ms <= 0.0:
        raise ValueError("tr must be positive")

    scale_max = float(np.max(signal))
    if scale_max > 0.0:
        signal_scaled = signal / scale_max
    else:
        signal_scaled = signal.copy()

    x0 = np.asarray([float(np.max(signal_scaled)) * 10.0, 500.0], dtype=np.float64)
    x0 = np.maximum(x0, 1e-8)

    def _residual(x: np.ndarray) -> np.ndarray:
        a, t1_ms = float(x[0]), float(x[1])
        return _spgr_signal_ms(a, t1_ms, tr_ms, fa_deg) - signal_scaled

    result = least_squares(
        _residual,
        x0,
        bounds=([0.0, 0.0], [math.inf, 10000.0]),
        method="trf",
        x_scale=x0,
    )
    if not result.success:
        raise ArithmeticError(f"Unable to fit VFA non-linear model: {result.message}")

    rho_fit_scaled = float(result.x[0])
    exponential_fit = float(result.x[1])
    rho_fit = rho_fit_scaled * scale_max

    pred = _spgr_signal_ms(rho_fit_scaled, exponential_fit, tr_ms, fa_deg)
    r_squared, sse = _fit_r_squared(signal_scaled, pred)

    if exponential_fit < 0.0:
        exponential_fit = -0.5

    return [exponential_fit, rho_fit, r_squared, -1.0, -1.0, sse]


def t1_fa_two_point_fit(parameter: Iterable[float], si: Iterable[float], tr: float) -> List[float]:
    """Two-point VFA T1 estimate using first/last flip-angle samples.

    Inputs/outputs follow the same units as other T1 fitters in this module:
    - flip angle: degrees
    - TR: milliseconds
    - returned T1: milliseconds
    """
    fa_deg = np.asarray([float(v) for v in parameter], dtype=np.float64)
    signal = np.asarray([float(v) for v in si], dtype=np.float64)
    tr_ms = float(tr)

    if fa_deg.shape[0] != signal.shape[0]:
        raise ValueError(f"parameter and si lengths differ: {fa_deg.shape[0]} vs {signal.shape[0]}")
    if fa_deg.size < 2:
        raise ValueError("two-point fit requires at least two flip-angle samples")
    if tr_ms <= 0.0:
        raise ValueError("tr must be positive")

    # Match OSIPI two-FA convention: use the first and last flip-angle points.
    fa1 = float(np.deg2rad(fa_deg[0]))
    fa2 = float(np.deg2rad(fa_deg[-1]))
    s1 = float(signal[0])
    s2 = float(signal[-1])
    if s2 == 0.0:
        raise ArithmeticError("Unable to estimate two-point VFA T1: second signal sample is zero")

    sr = s1 / s2
    numer = sr * math.sin(fa2) * math.cos(fa1) - math.sin(fa1) * math.cos(fa2)
    denom = sr * math.sin(fa2) - math.sin(fa1)
    ratio = numer / denom if denom != 0.0 else math.nan
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ArithmeticError("Unable to estimate two-point VFA T1: invalid log ratio")

    exponential_fit = tr_ms / math.log(ratio)
    if not math.isfinite(exponential_fit) or exponential_fit <= 0.0:
        raise ArithmeticError("Unable to estimate two-point VFA T1: non-positive T1")

    e1 = math.exp(-tr_ms / exponential_fit)
    denom_s0 = (1.0 - e1) * math.sin(fa1)
    if denom_s0 == 0.0:
        raise ArithmeticError("Unable to estimate two-point VFA T1: invalid S0 denominator")
    rho_fit = s1 * ((1.0 - e1 * math.cos(fa1)) / denom_s0)
    if not math.isfinite(rho_fit) or rho_fit <= 0.0:
        raise ArithmeticError("Unable to estimate two-point VFA T1: invalid S0")

    pred = _spgr_signal_ms(rho_fit, exponential_fit, tr_ms, fa_deg)
    r_squared, sse = _fit_r_squared(signal, pred)
    return [exponential_fit, rho_fit, r_squared, -1.0, -1.0, sse]
