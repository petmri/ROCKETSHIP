"""Parametric model ports from MATLAB fitParameter.m."""

from __future__ import annotations

import math
from typing import Iterable, List


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
