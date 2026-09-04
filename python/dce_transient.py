"""Detect leading dynamic frames acquired before the MRI steady state.

The first frames of a dynamic series are often acquired before the magnetisation
reaches steady state. They carry transient high signal and slice-to-slice banding,
and they corrupt anything downstream that assumes a flat pre-contrast baseline.
This module decides how many leading frames to discard, as an alternative to
naming a fixed `start_t`.

Note "steady state" here means the *magnetisation* steady state, not the
pre-contrast baseline window that `steady_state_start`/`steady_state_end`
describe. The two are unrelated: this runs before contrast arrival, which is
what those keys bound.

Two evidence channels, because on 3D VIBE data the mean signal alone misses most
cases (evidence and thresholds:
`project-management/projects/transient-chop/transient_detection.md`):

- **Banding** decides *whether* frame 0 is transient. A frame acquired off
  steady state weights k-space partitions unevenly, so signal oscillates along
  the slice axis with a period of a few slices. That largely cancels in the
  volume mean, which is why the mean curve misses it.
- **Mean deviation** decides *how far* the transient extends, since a decaying
  transient leaves a second frame elevated but no longer banded.

Noise comes from `dce_sigma.successive_difference_sigma` -- the whole series has
far more samples than the three or four baseline frames, and lag-1 differencing
is blind to slow drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dce_sigma import bolus_exclude_window, successive_difference_sigma

# Consistency factor turning MAD into a Gaussian-σ estimate (1/Φ⁻¹(0.75)).
_MAD_TO_SIGMA = 1.4826
# How long the wash-in is assumed to last, for the noise exclusion window.
_ARRIVAL_FRAMES = 3
# Slices carrying less object than this are dropped from the banding profile.
_MIN_SLICE_VOXELS = 50
# Below this many usable slices a profile cannot show banding at all.
_MIN_SLICES = 7
# Moving-average width of the through-plane detrend.
_SMOOTH_SLICES = 5


@dataclass
class TransientResult:
    """What the detector concluded about one dynamic series."""

    n_chop: int                      # leading frames to discard
    onset_0b: int                    # first post-arrival frame, 0-based
    level: float                     # robust baseline level of the mean curve
    sigma: float                     # robust noise on that level
    deviations: np.ndarray           # (curve - level) / sigma, signed, per frame
    curve: np.ndarray                # mean signal per frame
    oscillation: Optional[np.ndarray]  # banding ratio per frame, None when unavailable
    flag: Optional[str] = None       # why the result is untrustworthy, if it is

    @property
    def start_t_1b(self) -> int:
        """The 1-based first frame to analyse -- what `start_t` would be set to."""
        return self.n_chop + 1

    @property
    def trustworthy(self) -> bool:
        return self.flag is None


def object_mask(dynamic: np.ndarray, thresh_frac: float = 0.25) -> np.ndarray:
    """Coarse foreground mask from the temporal mean, for when no ROI mask is given.

    Args:
        dynamic: `(X, Y, Z, T)` dynamic series.
        thresh_frac: keep voxels above this fraction of the 99th-percentile mean.

    Returns:
        `(X, Y, Z)` boolean mask.
    """
    mean_img = np.asarray(dynamic, dtype=np.float64).mean(axis=-1)
    hi = float(np.percentile(mean_img, 99))
    return mean_img > thresh_frac * hi


def signal_curve(dynamic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean signal per frame over `mask`.

    Args:
        dynamic: `(X, Y, Z, T)` dynamic series.
        mask: `(X, Y, Z)` boolean mask with at least one voxel.

    Returns:
        `(T,)` mean signal curve.

    Raises:
        ValueError: if the mask is empty.
    """
    if not np.any(mask):
        raise ValueError("transient detection needs a non-empty mask")
    return np.asarray(dynamic, dtype=np.float64)[mask].mean(axis=0)


def slice_curves(dynamic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean signal per (slice, frame), dropping slices with too little object.

    Args:
        dynamic: `(X, Y, Z, T)` dynamic series.
        mask: `(X, Y, Z)` boolean mask.

    Returns:
        `(S, T)` array over the retained slices; `S` may be 0.
    """
    arr = np.asarray(dynamic, dtype=np.float64)
    curves = [
        arr[:, :, s, :][mask[:, :, s]].mean(axis=0)
        for s in range(arr.shape[2])
        if int(mask[:, :, s].sum()) >= _MIN_SLICE_VOXELS
    ]
    if not curves:
        return np.empty((0, arr.shape[3]), dtype=np.float64)
    return np.asarray(curves, dtype=np.float64)


def plateau_window(onset_0b: int, n_frames: int, span: int = 3) -> slice:
    """Settled pre-arrival frames used to normalise the banding.

    Frame 0 is always excluded -- it is the frame under test and must not
    normalise itself.
    """
    onset = max(0, min(int(onset_0b), int(n_frames)))
    return slice(max(1, onset - int(span)), onset)


def oscillation_ratio(slice_arr: np.ndarray, plateau: slice) -> np.ndarray:
    """Per-frame slice-to-slice banding, as a multiple of the plateau frames' own level.

    Each slice is divided by its own plateau mean, which cancels anatomy exactly, then
    detrended along the slice axis so only banding survives. The detrend is a moving
    average rather than a second difference because the observed period is 3-4 slices,
    not every other slice, which a second difference would attenuate by about half.

    Args:
        slice_arr: `(S, T)` per-slice mean signal from :func:`slice_curves`.
        plateau: frames defining the settled reference, from :func:`plateau_window`.

    Returns:
        `(T,)` ratios; all-NaN when there are too few slices or no usable plateau,
        which callers must treat as no evidence rather than as zero banding.
    """
    arr = np.asarray(slice_arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < _MIN_SLICES or arr.shape[1] == 0:
        return np.full(arr.shape[1] if arr.ndim == 2 else 0, np.nan)

    reference = arr[:, plateau]
    if reference.shape[1] == 0:
        return np.full(arr.shape[1], np.nan)
    plateau_mean = reference.mean(axis=1, keepdims=True)
    if not np.all(np.isfinite(plateau_mean)) or not np.all(plateau_mean > 0):
        return np.full(arr.shape[1], np.nan)

    norm = arr / plateau_mean
    pad = _SMOOTH_SLICES // 2
    kernel = np.ones(_SMOOTH_SLICES) / _SMOOTH_SLICES
    smooth = np.apply_along_axis(
        lambda p: np.convolve(np.pad(p, pad, mode="edge"), kernel, mode="valid"), 0, norm
    )
    residual = norm - smooth
    osc = np.sqrt(np.mean(residual * residual, axis=0))

    base = float(np.mean(osc[plateau]))
    if not np.isfinite(base) or base <= 0:
        return np.full(arr.shape[1], np.nan)
    return osc / base


def _baseline_noise(reference: np.ndarray, curve: np.ndarray, onset_0b: int) -> float:
    """Robust noise on the baseline level.

    A MAD over the two or three pre-arrival plateau frames alone collapses to nearly
    zero whenever they happen to land close together, and ordinary scatter then reads
    as a transient. Take the larger of that and a successive-difference σ over the
    whole series, which has every frame to work with.
    """
    mad = _MAD_TO_SIGMA * float(np.median(np.abs(reference - np.median(reference))))
    window = bolus_exclude_window(onset_0b, _ARRIVAL_FRAMES, len(curve))
    sd = successive_difference_sigma(curve, exclude=window)
    if not np.isfinite(sd):
        sd = 0.0
    floor = 1e-6 * abs(float(np.median(reference)))
    return max(mad, sd, floor, float(np.finfo(float).tiny))


def detect_transient(
    curve: np.ndarray,
    onset_0b: int,
    *,
    oscillation: Optional[np.ndarray] = None,
    osc_z: float = 5.0,
    z: float = 4.0,
    z_ext: float = 2.0,
    max_chop: int = 3,
    max_baseline: int = 8,
    min_plateau: int = 3,
    onset_guard: int = 1,
) -> TransientResult:
    """Count the leading frames that have not settled to the baseline plateau.

    A frame is transient when its banding clears `osc_z`, or when its mean sits
    `z` above the plateau (`z_ext` once a transient is already established --
    a decaying transient makes a moderately high next frame far more likely
    residual than noise). The mean test is one-sided because pre-steady-state
    signal starts high and decays.

    Both channels measure a frame against a plateau taken from the rest of the
    pre-arrival segment, so both are only as good as `onset_0b`. An onset past
    `max_baseline` means arrival was mislocated or never found, and the
    "plateau" is then post-contrast signal against which the true baseline reads
    as transient -- which would chop the whole baseline away. Such a series comes
    back flagged with `n_chop = 0` instead.

    Args:
        curve: `(T,)` mean signal per frame.
        onset_0b: first post-arrival frame, 0-based (ROCKETSHIP's `end_ss_1b`).
        oscillation: `(T,)` banding ratios, or None when unavailable.
        osc_z: banding ratio above which a frame is transient.
        z: mean-curve σ threshold to trigger at frame 0.
        z_ext: mean-curve σ threshold to extend an established transient.
        max_chop: never chop more than this many frames.
        max_baseline: an onset beyond this frame is not believable as a baseline.
        min_plateau: preferred number of reference frames.
        onset_guard: frames dropped just before arrival, so a first hint of
            enhancement cannot contaminate the reference.

    Returns:
        A :class:`TransientResult`.

    Raises:
        ValueError: if there are fewer than 3 pre-arrival frames.
    """
    curve = np.asarray(curve, dtype=np.float64).reshape(-1)
    n_frames = len(curve)
    onset_0b = int(min(max(int(onset_0b), 0), n_frames))

    pre = curve[:onset_0b]
    if len(pre) < 3:
        raise ValueError(
            f"only {len(pre)} pre-arrival frame(s); need at least 3 to separate a "
            "transient from the baseline"
        )

    if onset_0b > int(max_baseline):
        reason = (
            "no contrast arrival found"
            if onset_0b >= n_frames
            else f"arrival at frame {onset_0b} is too late to be a baseline"
        )
        return TransientResult(
            n_chop=0,
            onset_0b=onset_0b,
            level=float(np.median(pre)),
            sigma=float("nan"),
            deviations=np.full(n_frames, np.nan),
            curve=curve,
            oscillation=oscillation,
            flag=f"{reason}; cannot locate a pre-contrast plateau",
        )

    # Drop the guard frame next to arrival only when enough frames remain.
    usable = pre[:-onset_guard] if len(pre) - onset_guard >= min_plateau + 1 else pre
    # Reserve the leading frames as chop candidates first, then let the reference
    # shrink -- never below two frames. With arrival at frame 3 or 4 there is not
    # room for both a 3-frame reference and any candidate, and a transient frame
    # inside its own reference can never be flagged.
    searchable = min(int(max_chop), max(0, len(usable) - 2))
    reference = usable[-min(int(min_plateau), len(usable) - searchable):]

    level = float(np.median(reference))
    sigma = _baseline_noise(reference, curve, onset_0b)
    deviations = (curve - level) / sigma

    # A settling transient is contiguous from frame 0, so stop at the first
    # settled frame; an isolated outlier deeper in cannot be a start-of-run
    # transient.
    n_chop = 0
    for i in range(searchable):
        banded = (
            oscillation is not None
            and i < len(oscillation)
            and np.isfinite(oscillation[i])
            and float(oscillation[i]) > float(osc_z)
        )
        if banded or deviations[i] > (z if i == 0 else z_ext):
            n_chop = i + 1
        else:
            break

    return TransientResult(
        n_chop=n_chop,
        onset_0b=onset_0b,
        level=level,
        sigma=sigma,
        deviations=deviations,
        curve=curve,
        oscillation=oscillation,
    )


def detect_from_dynamic(
    dynamic: np.ndarray,
    onset_0b: int,
    *,
    mask: Optional[np.ndarray] = None,
    **kwargs,
) -> TransientResult:
    """Run both channels on a 4-D dynamic series.

    Args:
        dynamic: `(X, Y, Z, T)` dynamic series, untrimmed.
        onset_0b: first post-arrival frame, 0-based.
        mask: `(X, Y, Z)` tissue mask; a coarse intensity mask is derived when None.
        **kwargs: passed to :func:`detect_transient`.

    Returns:
        A :class:`TransientResult`. The banding channel is skipped (and
        `oscillation` left None) when the volume has too few usable slices.
    """
    arr = np.asarray(dynamic)
    if arr.ndim != 4:
        raise ValueError(f"expected a 4D dynamic series, got shape {arr.shape}")
    if mask is None:
        mask = object_mask(arr)

    curve = signal_curve(arr, mask)
    profiles = slice_curves(arr, mask)
    osc: Optional[np.ndarray] = None
    if profiles.shape[0] >= _MIN_SLICES:
        window = plateau_window(onset_0b, len(curve))
        candidate = oscillation_ratio(profiles, window)
        if np.any(np.isfinite(candidate)):
            osc = candidate
    return detect_transient(curve, onset_0b, oscillation=osc, **kwargs)
