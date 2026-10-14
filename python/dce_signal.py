"""Signal/intensity conversion helpers for DCE workflows."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def signal_to_enhancement(signal: Iterable[float], baseline_indices: Iterable[int]) -> np.ndarray:
    """Convert signal intensity to enhancement (%).

    Args:
        signal: Signal-intensity time series.
        baseline_indices: Integer indices used to compute baseline signal.

    Returns:
        Enhancement time series (%), same length as `signal`.
    """
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("signal contains non-finite values")

    base_idx = np.asarray(list(baseline_indices), dtype=np.int64)
    if base_idx.ndim != 1 or base_idx.size == 0:
        raise ValueError("baseline_indices must contain at least one index")
    if np.any(base_idx < 0) or np.any(base_idx >= s.size):
        raise ValueError("baseline_indices contains out-of-range entries")

    s_pre = float(np.mean(s[base_idx]))
    if s_pre <= 0.0:
        raise ArithmeticError("Baseline signal is zero or negative")

    return 100.0 * ((s - s_pre) / s_pre)


def enhancement_to_concentration_spgr(
    enhancement_percent: Iterable[float],
    tr_sec: float,
    fa_deg: float,
    t10_sec: float,
    relaxivity_r1: float,
    k_fa: float = 1.0,
) -> np.ndarray:
    """Convert enhancement (%) to concentration (mM) for SPGR.

    This follows the closed-form expression used by OSIPI's
    `EnhToConcSPGR` implementation.

    Args:
        enhancement_percent: Enhancement time series (%).
        tr_sec: Repetition time in seconds.
        fa_deg: Nominal flip angle in degrees.
        t10_sec: Baseline T1 in seconds.
        relaxivity_r1: R1 relaxivity in s^-1 mM^-1.
        k_fa: B1 correction factor (actual/nominal flip angle).

    Returns:
        Concentration time series (mM).
    """
    enh = np.asarray(enhancement_percent, dtype=np.float64)
    if enh.ndim != 1:
        raise ValueError(f"enhancement_percent must be 1D, got shape {enh.shape}")
    if not np.all(np.isfinite(enh)):
        raise ValueError("enhancement_percent contains non-finite values")

    tr_sec = float(tr_sec)
    fa_deg = float(fa_deg)
    t10_sec = float(t10_sec)
    relaxivity_r1 = float(relaxivity_r1)
    k_fa = float(k_fa)
    if tr_sec <= 0.0 or t10_sec <= 0.0 or relaxivity_r1 <= 0.0:
        raise ValueError("tr_sec, t10_sec, and relaxivity_r1 must be positive")

    fa_rad = fa_deg * np.pi / 180.0
    cos_fa_true = np.cos(k_fa * fa_rad)
    exp_r10_tr = np.exp(tr_sec / t10_sec)

    numerator = exp_r10_tr * (enh - 100.0 * cos_fa_true - enh * exp_r10_tr + 100.0)
    denominator = (
        100.0 * exp_r10_tr
        + enh * cos_fa_true
        - 100.0 * exp_r10_tr * cos_fa_true
        - enh * exp_r10_tr * cos_fa_true
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        concentration = -np.log(numerator / denominator) / (tr_sec * relaxivity_r1)

    if not np.all(np.isfinite(concentration)):
        raise ArithmeticError("Non-finite concentration values computed; check inputs")

    return concentration


def signal_to_concentration_spgr(
    signal: Iterable[float],
    baseline_indices: Iterable[int],
    tr_sec: float,
    fa_deg: float,
    t10_sec: float,
    relaxivity_r1: float,
    k_fa: float = 1.0,
) -> np.ndarray:
    """Convert signal intensity directly to concentration (mM) for SPGR."""
    enhancement = signal_to_enhancement(signal, baseline_indices)
    return enhancement_to_concentration_spgr(
        enhancement_percent=enhancement,
        tr_sec=tr_sec,
        fa_deg=fa_deg,
        t10_sec=t10_sec,
        relaxivity_r1=relaxivity_r1,
        k_fa=k_fa,
    )
