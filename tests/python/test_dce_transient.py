"""Unit tests for non-steady-state transient detection (`dce_transient`).

Synthetic series with a known number of contaminated leading frames, covering both
evidence channels (slice banding, mean elevation), the guard against a mislocated
arrival, and the `start_t` precedence the pipeline applies.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_transient import (  # noqa: E402
    detect_from_dynamic,
    detect_transient,
    oscillation_ratio,
    plateau_window,
    slice_curves,
)


N_SLICES = 14
ONSET = 5  # first post-arrival frame, 0-based


def _series(
    n_frames: int = 24,
    *,
    banded_frames: int = 0,
    band_amplitude: float = 6.0,
    elevated: tuple = (),
    noise: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """A `(X, Y, Z, T)` series: flat baseline, arrival at `ONSET`, optional transient."""
    rng = np.random.default_rng(seed)
    dyn = 100.0 + rng.normal(0.0, noise, (10, 10, N_SLICES, n_frames))
    dyn[..., ONSET:] += 40.0
    # Banding runs along the slice axis with a ~3.5-slice period, as observed on VIBE.
    band = np.cos(np.arange(N_SLICES) * 2.0 * np.pi / 3.5) * band_amplitude
    for f in range(banded_frames):
        dyn[:, :, :, f] += band[None, None, :]
    for frame, amount in elevated:
        dyn[..., frame] += amount
    return dyn


def _mask() -> np.ndarray:
    return np.ones((10, 10, N_SLICES), dtype=bool)


# --------------------------------------------------------------------------- channels


def test_clean_series_is_not_chopped() -> None:
    result = detect_from_dynamic(_series(), ONSET, mask=_mask())
    assert result.n_chop == 0
    assert result.trustworthy
    assert result.start_t_1b == 1


def test_banding_alone_triggers_a_chop() -> None:
    """Banding cancels in the volume mean, so this is the case the mean curve misses."""
    result = detect_from_dynamic(_series(banded_frames=1), ONSET, mask=_mask())
    assert result.n_chop == 1
    assert abs(float(result.deviations[0])) < 4.0  # mean channel would not have fired
    assert float(result.oscillation[0]) > 5.0


def test_mean_elevation_alone_triggers_a_chop() -> None:
    result = detect_from_dynamic(_series(elevated=((0, 12.0),)), ONSET, mask=_mask())
    assert result.n_chop == 1
    assert float(result.deviations[0]) > 4.0


def test_transient_extends_over_two_frames() -> None:
    """A decaying transient: banded frame 0, still-elevated frame 1."""
    result = detect_from_dynamic(
        _series(banded_frames=1, elevated=((0, 12.0), (1, 3.0))), ONSET, mask=_mask()
    )
    assert result.n_chop == 2


def test_chop_never_exceeds_max_chop() -> None:
    """Needs a long baseline: with arrival at frame 5 the reference already caps the
    search at two candidates, so the cap would not be what was under test."""
    rng = np.random.default_rng(1)
    dyn = 100.0 + rng.normal(0.0, 0.3, (10, 10, N_SLICES, 24))
    dyn[..., 8:] += 40.0
    for frame, amount in ((0, 12.0), (1, 6.0), (2, 4.0)):
        dyn[..., frame] += amount

    assert detect_from_dynamic(dyn, 8, mask=_mask()).n_chop == 3
    assert detect_from_dynamic(dyn, 8, mask=_mask(), max_chop=2).n_chop == 2


def test_isolated_later_outlier_is_not_a_start_transient() -> None:
    """A transient is contiguous from frame 0; a spike at frame 2 alone is not one."""
    result = detect_from_dynamic(_series(elevated=((2, 30.0),)), ONSET, mask=_mask())
    assert result.n_chop == 0


def test_low_first_frame_does_not_trigger_the_mean_channel() -> None:
    """The mean test is one-sided: pre-steady-state signal starts high, never low."""
    result = detect_from_dynamic(_series(elevated=((0, -12.0),)), ONSET, mask=_mask())
    assert float(result.deviations[0]) < -4.0
    assert result.n_chop == 0


# --------------------------------------------------------------------------- guards


def test_late_arrival_is_flagged_and_not_chopped() -> None:
    """A mislocated arrival puts post-contrast signal in the plateau, against which the
    real baseline reads as transient. Refusing beats chopping the baseline away."""
    result = detect_from_dynamic(_series(banded_frames=1), 12, mask=_mask())
    assert result.n_chop == 0
    assert not result.trustworthy
    assert "too late" in result.flag


def test_absent_arrival_is_flagged() -> None:
    dyn = _series()
    result = detect_from_dynamic(dyn, dyn.shape[3], mask=_mask())
    assert result.n_chop == 0
    assert "no contrast arrival" in result.flag


def test_too_few_pre_arrival_frames_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        detect_from_dynamic(_series(), 2, mask=_mask())


def test_few_slice_volume_falls_back_to_the_mean_channel() -> None:
    """Below ~7 slices a profile cannot show banding; the mean channel still works."""
    dyn = _series()[:, :, :3, :]
    dyn[..., 0] += 12.0
    result = detect_from_dynamic(dyn, ONSET, mask=np.ones(dyn.shape[:3], dtype=bool))
    assert result.oscillation is None
    assert result.n_chop == 1


def test_oscillation_is_all_nan_without_a_plateau() -> None:
    profiles = slice_curves(_series(), _mask())
    osc = oscillation_ratio(profiles, plateau_window(0, 24))
    assert np.all(np.isnan(osc))


def test_plateau_window_never_includes_frame_zero() -> None:
    """Frame 0 is the frame under test and must not normalise itself."""
    assert plateau_window(3, 24) == slice(1, 3)
    assert plateau_window(1, 24) == slice(1, 1)


def test_detect_transient_accepts_a_bare_curve() -> None:
    curve = np.full(24, 100.0)
    curve[ONSET:] = 140.0
    curve[0] = 118.0
    result = detect_transient(curve, ONSET)
    assert result.n_chop == 1
    assert result.oscillation is None
