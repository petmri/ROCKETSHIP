"""Per-voxel noise-σ estimation and reduced χ² for DCE quality-of-fit (QoF).

Implements the day-1 pieces of the QoF reduced-χ² signal
(`docs/project-management/projects/batch-parity/sigma_estimators.md`):

- **Estimator B** — a robust successive-difference (von Neumann / lag-1) noise σ in
  concentration units, computed from the per-voxel `C(t)` the pipeline already writes
  (`ct_voxel_mM` in `*_postfit_arrays.npz`). No extra plumbing.
- The shared **successive-difference core** it is built on, which both B (on `C(t)`) and the
  future signal-domain estimator C (on `S(t)`) use.
- A **bolus wash-in exclusion window** so the successive-difference smoothness assumption is
  not violated by the genuine frame-to-frame jump of the first pass in vascular voxels.
- **Reduced χ²** helpers: a scalar-σ form `SSE/(σ²·(N−p))` (B's path) and a per-frame weighted
  form `Σ(r_i/σ_i)²/(N−p)` (for estimator C / heteroscedastic σ, using the precomputed
  `voxel_residuals`).
- **Empirical-Bayes variance moderation** (`eb_moderate_variance`): shrink per-unit σ² toward an
  inverse-gamma prior (limma/Smyth `squeezeVar`) and **clamp** σ² that is statistically
  incompatible with that fitted prior (the prior-predictive tail) — this both improves the σ
  estimate for the bulk and restores χ²_ν at contaminated (e.g. motion-artifact) voxels whose
  inflated σ would otherwise suppress χ²_ν. A general variance-stabilization tool (reusable beyond
  DCE — AIF voxels, bad time points).

Estimator C (signal-domain σ_S propagated through the SPGR Jacobian) is intentionally not here
yet — it needs raw signal / `R1t` + SPGR params that are not in the post-fit artifact (see the
plumbing probe in `sigma_estimators.md`).
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import f as f_dist


# Consistency factor turning MAD into a Gaussian-σ estimate (1/Φ⁻¹(0.75)).
_MAD_TO_SIGMA = 1.4826
# Lag-1 differencing doubles the variance (Var[xᵢ₊₁−xᵢ] = 2σ²); undo it.
_DIFF_VARIANCE_FACTOR = np.sqrt(2.0)


def _retained_diff_mask(n_frames: int, exclude: Optional[Tuple[int, int]]) -> np.ndarray:
    """Boolean mask over the `n_frames-1` lag-1 differences.

    A difference `d_i = x[i+1] - x[i]` is retained only when *both* endpoints are outside the
    excluded window, so no difference is ever taken *across* the excised gap (that jump is
    signal, not noise).
    """
    if n_frames < 2:
        return np.zeros(max(n_frames - 1, 0), dtype=bool)
    retained = np.ones(int(n_frames), dtype=bool)
    if exclude is not None:
        lo, hi = int(exclude[0]), int(exclude[1])
        lo = max(0, min(lo, n_frames))
        hi = max(0, min(hi, n_frames))
        if hi > lo:
            retained[lo:hi] = False
    return retained[:-1] & retained[1:]


def retained_diff_count(n_frames: int, exclude: Optional[Tuple[int, int]] = None) -> int:
    """Number of lag-1 differences retained after excising `exclude` — the raw sample count
    behind :func:`successive_difference_sigma`. Handy for deriving an effective σ dof."""
    return int(_retained_diff_mask(n_frames, exclude).sum())


def _robust_scale(diffs: np.ndarray) -> float:
    """Robust Gaussian-σ from lag-1 differences via MAD about the median."""
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med))
    return float(_MAD_TO_SIGMA * mad / _DIFF_VARIANCE_FACTOR)


def successive_difference_sigma(
    x: np.ndarray,
    *,
    exclude: Optional[Tuple[int, int]] = None,
    robust: bool = True,
    min_diffs: int = 3,
) -> float:
    """Estimate the noise σ of a 1-D time series from its lag-1 differences.

    Assumes the true signal is smooth frame-to-frame so that differences are noise-dominated;
    pass `exclude=(lo, hi)` (half-open, 0-based frames) to drop a fast transient such as the
    bolus wash-in (see :func:`bolus_exclude_window`). Non-finite samples are ignored.

    Args:
        x: 1-D time series (e.g. one voxel's `C(t)` or `S(t)`).
        exclude: optional half-open frame window whose samples are excluded from differencing.
        robust: `True` → MAD-based (default, tolerates residual outliers); `False` → von Neumann
            `sqrt(mean(d²)/2)`.
        min_diffs: minimum number of usable differences; below this, returns NaN.

    Returns:
        Estimated σ (same units as `x`), or NaN if too few usable differences.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    mask = _retained_diff_mask(arr.size, exclude)
    if not mask.any():
        return float("nan")
    diffs = np.diff(arr)[mask]
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < int(min_diffs):
        return float("nan")
    if robust:
        return _robust_scale(diffs)
    return float(np.sqrt(np.mean(diffs * diffs) / 2.0))


def sigma_successive_difference(
    ct: np.ndarray,
    *,
    exclude: Optional[Tuple[int, int]] = None,
    robust: bool = True,
    min_diffs: int = 3,
) -> Union[float, np.ndarray]:
    """Estimator **B**: successive-difference noise σ from concentration curves.

    Accepts a single voxel's curve `(T,)` → scalar σ, or a `(T, V)` frames-by-voxels block
    (the layout of `ct_voxel_mM`) → per-voxel σ array. The exclude window is shared across
    voxels (scan-level bolus timing), so the computation is vectorized.

    Args:
        ct: `(T,)` or `(T, V)` concentration time series (frames along axis 0).
        exclude: optional half-open `(lo, hi)` frame window (typically from
            :func:`bolus_exclude_window`).
        robust: MAD-based when `True` (default), else von Neumann.
        min_diffs: per-voxel minimum usable differences; below this the voxel's σ is NaN.

    Returns:
        Scalar σ for 1-D input, else a `(V,)` array of per-voxel σ (NaN where under-determined).
    """
    arr = np.asarray(ct, dtype=np.float64)
    if arr.ndim == 1:
        return successive_difference_sigma(arr, exclude=exclude, robust=robust, min_diffs=min_diffs)
    if arr.ndim != 2:
        raise ValueError(f"ct must be 1-D (T,) or 2-D (T, V); got shape {arr.shape}")

    n_frames = arr.shape[0]
    mask = _retained_diff_mask(n_frames, exclude)
    n_vox = arr.shape[1]
    if not mask.any():
        return np.full(n_vox, np.nan, dtype=np.float64)

    diffs = np.diff(arr, axis=0)[mask, :]  # (K, V)
    finite = np.isfinite(diffs)
    usable = finite.sum(axis=0)

    masked = np.where(finite, diffs, np.nan)
    with warnings.catch_warnings():
        # All-NaN voxels (no usable frames) are expected and handled below via `usable`.
        warnings.simplefilter("ignore", RuntimeWarning)
        if robust:
            med = np.nanmedian(masked, axis=0)
            mad = np.nanmedian(np.abs(masked - med[np.newaxis, :]), axis=0)
            sigma = _MAD_TO_SIGMA * mad / _DIFF_VARIANCE_FACTOR
        else:
            sigma = np.sqrt(np.nanmean(masked * masked, axis=0) / 2.0)

    sigma = np.asarray(sigma, dtype=np.float64)
    sigma[usable < int(min_diffs)] = np.nan
    return sigma


def bolus_exclude_window(
    onset_frame: float,
    duration_frames: float,
    n_frames: int,
    *,
    margin_frames: int = 2,
) -> Tuple[int, int]:
    """Half-open `(lo, hi)` 0-based frame window to excise around the bolus wash-in.

    Starts at `onset_frame` (the bolus-onset frame; the pipeline's `end_ss` /
    `start_injection`) and spans the injection duration plus a first-pass `margin_frames`.
    Always excludes at least the onset frame, and is clamped to `[0, n_frames]`.

    Callers derive the arguments from the Stage-B checkpoint (`b_out.json`):
    `onset_frame` from `start_time_index` (or `round(start_injection_min / time_resolution_min)`),
    `duration_frames` from `round((end_injection_min - start_injection_min) / time_resolution_min)`.

    Returns:
        `(lo, hi)` half-open frame range suitable for the `exclude` argument of the
        successive-difference estimators. `hi == lo` means an empty window.
    """
    n = int(n_frames)
    onset = max(0, min(int(round(float(onset_frame))), n))
    duration = max(0, int(round(float(duration_frames))))
    margin = max(0, int(round(float(margin_frames))))

    lo = onset
    hi = min(max(onset + duration + margin, onset + 1), n)  # always drop >= the onset frame
    if hi <= lo:
        return (lo, lo)
    return (lo, hi)


def reduced_chi_square(
    sse: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    n_obs: int,
    n_params: int,
) -> Union[float, np.ndarray]:
    """Reduced χ² from a scalar-per-voxel σ: `χ²_ν = SSE / (σ²·(N−p))`.

    This is estimator B's path — the fit already provides SSE, and B provides one σ per voxel.
    Voxels with non-positive or non-finite σ (or SSE) yield NaN.

    Args:
        sse: per-voxel sum of squared residuals (scalar or `(V,)`).
        sigma: per-voxel noise σ (scalar or `(V,)`), same shape as `sse` when array.
        n_obs: number of observations (time points) the fit used, `N`.
        n_params: number of free model parameters, `p`.

    Returns:
        Reduced χ² (scalar or `(V,)`), NaN where under-determined.

    Raises:
        ValueError: if `N − p <= 0`.
    """
    dof = int(n_obs) - int(n_params)
    if dof <= 0:
        raise ValueError(f"degrees of freedom N-p must be positive; got N={n_obs}, p={n_params}")

    sse_arr = np.asarray(sse, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    valid = (sigma_arr > 0) & np.isfinite(sigma_arr) & np.isfinite(sse_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2v = sse_arr / (sigma_arr * sigma_arr * dof)
    chi2v = np.where(valid, chi2v, np.nan)
    if chi2v.ndim == 0:
        return float(chi2v)
    return chi2v


def reduced_chi_square_weighted(
    residuals: np.ndarray,
    sigma: Union[float, np.ndarray],
    n_params: int,
    *,
    axis: int = 0,
) -> Union[float, np.ndarray]:
    """Weighted reduced χ² from per-frame residuals: `χ²_ν = Σ(r_i/σ_i)² / (N−p)`.

    Supports a per-frame σ (estimator C's heteroscedastic σ_i(t)) as well as a scalar/per-voxel
    σ (broadcast). Uses the precomputed `voxel_residuals`; `N` is counted per voxel from the
    finite standardized residuals, so NaN frames drop out naturally.

    Args:
        residuals: `(T,)` or `(T, V)` per-frame residuals `r_i = Ct − model` (frames on `axis`).
        sigma: scalar, `(T,)` per-frame, `(V,)` per-voxel, or `(T, V)` σ — broadcast against
            `residuals`.
        n_params: number of free model parameters, `p`.
        axis: time axis of `residuals` (default 0).

    Returns:
        Reduced χ² (scalar for 1-D residuals, else `(V,)`), NaN where `N − p <= 0`.
    """
    r = np.asarray(residuals, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = (r / s) ** 2
    valid = np.isfinite(z2) & (np.asarray(np.broadcast_to(s, z2.shape)) > 0)
    z2 = np.where(valid, z2, np.nan)

    sse_w = np.nansum(z2, axis=axis)
    n_obs = valid.sum(axis=axis)
    dof = n_obs - int(n_params)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2v = np.where(dof > 0, sse_w / dof, np.nan)
    if np.ndim(chi2v) == 0:
        return float(chi2v)
    return chi2v


# --------------------------------------------------------------------------- variance moderation


def _trigamma_inverse(x: np.ndarray) -> np.ndarray:
    """Solve `trigamma(y) = x` for `y` (Newton iteration; after limma's trigammaInverse)."""
    x = np.asarray(x, dtype=np.float64)
    y = 0.5 + 1.0 / x  # initial guess
    for _ in range(60):
        tri = polygamma(1, y)
        step = tri * (1.0 - tri / x) / polygamma(2, y)
        y = y + step
        if np.max(np.abs(step / y)) < 1e-10:
            break
    return y


def eb_fit_variance_prior(
    s2: np.ndarray,
    dof: float,
    *,
    robust: bool = True,
) -> dict:
    """Fit an inverse-gamma (scaled-inverse-χ²) prior to sample variances — limma/Smyth `fitFDist`.

    Models `s2_i | σ²_i ~ σ²_i · χ²_dof / dof` with `σ²_i` inverse-gamma. Returns the prior
    location `s0_2` and prior degrees of freedom `d0` (may be `inf` ⇒ all units share one
    variance). `robust=True` estimates the hyperparameters from the median/MAD of the log-variances
    so a heavy outlier tail (e.g. motion voxels) does not corrupt the prior.

    Args:
        s2: 1-D sample variances (> 0); non-finite / non-positive entries are ignored in the fit.
        dof: residual degrees of freedom of each `s2` estimate (scalar).
        robust: use median/MAD (default) vs mean/variance of the log-variances.

    Returns:
        ``{"d0": float, "s0_2": float, "dof": float}``.
    """
    arr = np.asarray(s2, dtype=np.float64).reshape(-1)
    good = np.isfinite(arr) & (arr > 0)
    if good.sum() < 2:
        raise ValueError("need at least 2 finite positive variances to fit the prior")
    df = float(dof)
    z = np.log(arr[good])
    e = z - digamma(df / 2.0) + np.log(df / 2.0)  # E[e] = log(s0^2)
    if robust:
        loc = float(np.median(e))
        evar = float((_MAD_TO_SIGMA * np.median(np.abs(e - loc))) ** 2)
    else:
        loc = float(np.mean(e))
        evar = float(np.var(e, ddof=1))
    evar_adj = evar - float(polygamma(1, df / 2.0))
    if evar_adj <= 0:
        # observed spread ≤ sampling spread ⇒ variances are effectively constant ⇒ full shrinkage.
        return {"d0": np.inf, "s0_2": float(np.exp(loc)), "dof": df}
    d0 = 2.0 * float(_trigamma_inverse(np.asarray(evar_adj)))
    s0_2 = float(np.exp(loc + digamma(d0 / 2.0) - np.log(d0 / 2.0)))
    return {"d0": d0, "s0_2": s0_2, "dof": df}


def eb_moderate_variance(
    s2: np.ndarray,
    dof: float,
    *,
    robust: bool = True,
    clamp_quantile: Optional[float] = 0.999,
) -> dict:
    """Empirical-Bayes moderated variances: shrink toward an inverse-gamma prior + clamp outliers.

    Two effects, both driven by the *same* fitted prior (no hand-tuned σ threshold):
    1. **Shrinkage** — every unit's `s2` is pulled toward the prior mean by
       `s2_post = (d0·s0² + dof·s2) / (d0 + dof)`, stabilizing noisy estimates.
    2. **Clamp** — units whose `s2` exceeds the prior-predictive upper quantile
       (`s0²·F_q(dof, d0)`) are statistically incompatible with the noise model (contamination —
       motion, etc.) and are set to the prior mean `s0²`. This restores a large χ²_ν at those
       units instead of letting an inflated σ suppress it. Pass `clamp_quantile=None` to skip.

    Non-finite / non-positive inputs pass through unchanged.

    Returns:
        ``{"s2": moderated array, "d0", "s0_2", "clamp_value", "clamp_mask", "dof"}``.
    """
    arr = np.asarray(s2, dtype=np.float64)
    flat = arr.reshape(-1)
    good = np.isfinite(flat) & (flat > 0)
    prior = eb_fit_variance_prior(flat, dof, robust=robust)
    d0, s0_2, df = prior["d0"], prior["s0_2"], prior["dof"]

    out = flat.copy()
    if np.isinf(d0):
        out[good] = s0_2
        clamp_value = s0_2
        clamp_mask_flat = good.copy()
    else:
        out[good] = (d0 * s0_2 + df * flat[good]) / (d0 + df)
        clamp_mask_flat = np.zeros_like(good)
        if clamp_quantile is not None:
            clamp_value = s0_2 * float(f_dist.ppf(clamp_quantile, df, d0))
            hits = good & (flat > clamp_value)
            out[hits] = s0_2  # contaminated: substitute the population noise level
            clamp_mask_flat = hits
        else:
            clamp_value = np.inf

    return {
        "s2": out.reshape(arr.shape),
        "d0": float(d0),
        "s0_2": float(s0_2),
        "clamp_value": float(clamp_value),
        "clamp_mask": clamp_mask_flat.reshape(arr.shape),
        "dof": float(df),
    }
