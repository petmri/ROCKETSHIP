"""DCE model ports from MATLAB."""

from __future__ import annotations

import math
from typing import Iterable, List


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
