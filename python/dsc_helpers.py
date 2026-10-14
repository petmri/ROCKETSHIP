"""DSC helper ports from MATLAB."""

from __future__ import annotations

import math
from typing import Any, List, Tuple


def _truncate_last_dim(data: Any, new_len: int) -> Any:
    """Truncate along last dimension for nested-list arrays."""
    if not isinstance(data, list):
        return data
    if not data:
        return data
    if not isinstance(data[0], list):
        return data[:new_len]
    return [_truncate_last_dim(item, new_len) for item in data]


def _last_dim_len(data: Any) -> int:
    cur = data
    while isinstance(cur, list) and cur and isinstance(cur[0], list):
        cur = cur[0]
    if isinstance(cur, list):
        return len(cur)
    raise ValueError("Expected nested list array")


def import_aif(
    mean_aif: List[float],
    bolus_time: int,
    time_vect: List[float],
    concentration_array: Any,
    r2_star: float,
    te: float,
) -> Tuple[List[float], List[float], Any, List[float]]:
    """Port of `dsc/import_AIF.m`.

    Args:
      mean_aif: Imported mean AIF signal.
      bolus_time: MATLAB-style 1-based index where AIF starts.
      time_vect: Time vector.
      concentration_array: Nested-list concentration array (3D/4D style).
      r2_star: Relaxivity constant.
      te: Echo time.

    Returns:
      (meanAIF_adjusted, time_vect, concentration_array, meanSignal)
    """
    if not mean_aif:
        return [], time_vect, concentration_array, []

    # MATLAB indexing semantics: meanAIF = meanAIF(bolus_time:end)
    bolus_idx = int(bolus_time)
    if bolus_idx < 1:
        bolus_idx = 1
    mean_aif_post = mean_aif[bolus_idx - 1 :]

    if len(time_vect) <= len(mean_aif_post):
        mean_aif_adjusted = [mean_aif_post[i] for i in range(len(time_vect))]
    else:
        mean_aif_adjusted = [mean_aif_post[i] for i in range(len(mean_aif_post))]
        time_vect = time_vect[: len(mean_aif_post)]
        concentration_array = _truncate_last_dim(concentration_array, len(mean_aif_post))

    # Keep MATLAB behavior exactly: meanSignal uses mean_aif_post (not mean_aif_adjusted).
    mean_signal = [math.exp(-r2_star * v * te) for v in mean_aif_post]

    return mean_aif_adjusted, time_vect, concentration_array, mean_signal


def previous_aif(
    mean_aif: List[float],
    mean_signal: List[float],
    bolus_time: int,
    time_vect: List[float],
    concentration_array: Any,
) -> Tuple[List[float], List[float], Any]:
    """Port of `dsc/previous_AIF.m`.

    Args:
      mean_aif: Previous mean AIF.
      mean_signal: Previous mean signal (unused in MATLAB implementation).
      bolus_time: Previous bolus time (unused in MATLAB implementation).
      time_vect: Time vector.
      concentration_array: Nested-list concentration array.

    Returns:
      (meanAIF_adjusted, time_vect, concentration_array)
    """
    # Keep the same unused args for API parity.
    _ = mean_signal
    _ = bolus_time

    if len(time_vect) <= len(mean_aif):
        mean_aif_adjusted = [mean_aif[i] for i in range(len(time_vect))]
    else:
        mean_aif_adjusted = [mean_aif[i] for i in range(len(mean_aif))]
        time_vect = time_vect[: len(mean_aif)]
        concentration_array = _truncate_last_dim(concentration_array, len(mean_aif))

    return mean_aif_adjusted, time_vect, concentration_array


def matlab_reshape_linspace(start: float, stop: float, count: int, dims: Tuple[int, ...]) -> Any:
    """Create MATLAB-compatible reshape(linspace(...), dims) as nested lists.

    MATLAB reshapes in column-major order; this helper reproduces that layout.
    """
    if count <= 1:
        values = [float(start)]
    else:
        step = (stop - start) / float(count - 1)
        values = [start + step * i for i in range(count)]

    if len(dims) != 3:
        raise ValueError("Only 3D dims are currently supported")

    d1, d2, d3 = dims
    if d1 * d2 * d3 != count:
        raise ValueError("dims product does not match count")

    out = [[[0.0 for _ in range(d3)] for _ in range(d2)] for _ in range(d1)]
    for n, val in enumerate(values):
        i = n % d1
        j = (n // d1) % d2
        k = n // (d1 * d2)
        out[i][j][k] = float(val)
    return out


def last_dim_len(data: Any) -> int:
    """Public helper for tests."""
    return _last_dim_len(data)
