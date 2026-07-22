"""Shared Stage-D fitting machinery: one place to assemble fit inputs and run
multi-start optimization, shared by the CPU/python and accelerated
(cpufit/gpufit) backends.

Wired up for every accelerated-eligible model: patlak, tofts, ex_tofts,
tissue_uptake, 2cxm. Replaces dce_models._best_fit_over_starts and
dce_pipeline._accel_multistart_refine, both now dead and removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from dce_models import (
    _canonical_time_context,
    _ci_bounds_from_fit,
    _fit_2cxm_osipi_canonical,
    _least_squares_kwargs,
    _merge_prefs_in_canonical_units,
    _reject_algorithm_override,
    model_extended_tofts_cfit,
    model_patlak_cfit,
    model_patlak_linear,
    model_tissue_uptake_cfit,
    model_tofts_cfit,
)


@dataclass
class FitInputs:
    """Everything one Stage-D fit needs for N voxels of a single model."""

    ct: np.ndarray  # (n_time, n_voxels)
    cp: np.ndarray  # (n_time,)
    timer: np.ndarray  # (n_time,)
    bounds_row: np.ndarray  # flat [lo0, hi0, lo1, hi1, ...]
    prefs: Dict[str, Any]
    # Caller-supplied overrides only (pre-merge with hardcoded defaults), needed
    # by tissue_uptake/2cxm's python runners to convert bounds to canonical
    # per-minute units without re-scaling the (already-canonical) defaults
    # baked into `prefs`. See _run_tissue_uptake_python/_run_2cxm_python.
    raw_prefs: Optional[Dict[str, Any]] = None

    @property
    def n_voxels(self) -> int:
        return int(self.ct.shape[1])


def _patlak_settings(prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
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
    return settings


def _patlak_bounds_row(settings: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(settings["lower_limit_ktrans"]),
            float(settings["upper_limit_ktrans"]),
            float(settings["lower_limit_vp"]),
            float(settings["upper_limit_vp"]),
        ],
        dtype=np.float64,
    )


def assemble_patlak_candidates(inputs: FitInputs) -> np.ndarray:
    """Per-voxel linear-regression seed, then the prefs default, then default x100.

    Row 0 is the closed-form linear Patlak estimate per voxel (falling back to
    the prefs default when non-finite); rows 1-2 are the fixed prefs default
    ktrans, unscaled and x100, same vp default on both -- a fixed rescue
    candidate independent of the (occasionally bad) linear estimate. Returns
    shape (3, n_voxels, 2), columns [ktrans, vp].
    """
    settings = inputs.prefs
    default_k = float(settings["initial_value_ktrans"])
    default_vp = float(settings["initial_value_vp"])
    n_voxels = inputs.n_voxels
    base_k = np.full(n_voxels, default_k, dtype=np.float64)
    base_vp = np.full(n_voxels, default_vp, dtype=np.float64)

    cp_vec = [float(v) for v in inputs.cp]
    t_vec = [float(v) for v in inputs.timer]
    for i in range(n_voxels):
        try:
            estimate = model_patlak_linear([float(v) for v in inputs.ct[:, i]], cp_vec, t_vec)
            k0, vp0 = float(estimate[0]), float(estimate[1])
            if math.isfinite(k0):
                base_k[i] = k0
            if math.isfinite(vp0):
                base_vp[i] = vp0
        except Exception:
            continue

    default_k_row = np.full(n_voxels, default_k, dtype=np.float64)
    default_vp_row = np.full(n_voxels, default_vp, dtype=np.float64)

    return np.stack(
        [
            np.stack([base_k, base_vp], axis=-1),
            np.stack([default_k_row, default_vp_row], axis=-1),
            np.stack([default_k_row * 100, default_vp_row], axis=-1),
        ],
        axis=0,
    )


def _tofts_settings(prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
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
    return settings


def _tofts_bounds_row(settings: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(settings["lower_limit_ktrans"]),
            float(settings["upper_limit_ktrans"]),
            float(settings["lower_limit_ve"]),
            float(settings["upper_limit_ve"]),
        ],
        dtype=np.float64,
    )


def assemble_tofts_candidates(inputs: FitInputs) -> np.ndarray:
    """Single fixed candidate, broadcast to every voxel.

    Tofts has no per-voxel seeding or multi-start on either backend today --
    this is a mechanical move of the existing fixed-prefs start into the
    shared candidate-array shape, not a new strategy. Returns shape
    (1, n_voxels, 2), columns [ktrans, ve].
    """
    settings = inputs.prefs
    row = np.array([float(settings["initial_value_ktrans"]), float(settings["initial_value_ve"])], dtype=np.float64)
    return np.tile(row[None, None, :], (1, inputs.n_voxels, 1))


def _ex_tofts_settings(prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
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
    return settings


def _ex_tofts_bounds_row(settings: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(settings["lower_limit_ktrans"]),
            float(settings["upper_limit_ktrans"]),
            float(settings["lower_limit_ve"]),
            float(settings["upper_limit_ve"]),
            float(settings["lower_limit_vp"]),
            float(settings["upper_limit_vp"]),
        ],
        dtype=np.float64,
    )


def assemble_ex_tofts_candidates(inputs: FitInputs) -> np.ndarray:
    """Fixed x1/x10/x100 candidates on Ktrans, ve/vp held at prefs defaults.

    Same fixed-multiplier strategy `dce_models.model_extended_tofts_fit` has
    always used on the CPU path (no per-voxel seed, unlike patlak) -- this
    migration's only real behavior change is giving the accelerated backend
    this same multistart for the first time (it previously ran a single fixed
    start). Returns shape (3, n_voxels, 3), columns [ktrans, ve, vp].
    """
    settings = inputs.prefs
    k0 = float(settings["initial_value_ktrans"])
    ve0 = float(settings["initial_value_ve"])
    vp0 = float(settings["initial_value_vp"])
    n_voxels = inputs.n_voxels
    ve_row = np.full(n_voxels, ve0, dtype=np.float64)
    vp_row = np.full(n_voxels, vp0, dtype=np.float64)

    return np.stack(
        [
            np.stack([np.full(n_voxels, k0, dtype=np.float64), ve_row, vp_row], axis=-1),
            np.stack([np.full(n_voxels, k0 * 10.0, dtype=np.float64), ve_row, vp_row], axis=-1),
            np.stack([np.full(n_voxels, k0 * 100.0, dtype=np.float64), ve_row, vp_row], axis=-1),
        ],
        axis=0,
    )


def _log_uniform_candidates(
    n_starts: int, n_voxels: int, lower: np.ndarray, upper: np.ndarray, seed: int
) -> np.ndarray:
    """(n_starts, n_voxels, n_params) random log-uniform draws within [lower, upper].

    Same draw style `dce_pipeline._accel_multistart_refine` used (now removed),
    generalized to be assembled once and shared by every backend rather than
    being an accelerated-only rescue mechanism.
    """
    if n_starts <= 0:
        return np.zeros((0, n_voxels, lower.shape[0]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    log_lo = np.log(np.maximum(lower, 1e-30))
    log_hi = np.log(np.maximum(upper, lower + 1e-30))
    n_params = lower.shape[0]
    draws = np.exp(
        log_lo[None, None, :] + rng.random((n_starts, n_voxels, n_params)) * (log_hi - log_lo)[None, None, :]
    )
    return np.clip(draws, lower[None, None, :], upper[None, None, :])


def _clamp_to_bounds(candidates: np.ndarray, bounds_row: np.ndarray) -> np.ndarray:
    """Clamp a (n_voxels, n_params) candidate array into `bounds_row`'s [lo, hi] pairs.

    Applied once in `fit_with_multistart` so every backend -- python or
    accelerated -- sees only in-bounds starting points, instead of each
    runner having to (or, in the accelerated case, failing to) clamp its own
    `initial_parameters` before use.
    """
    n_params = candidates.shape[-1]
    lo = np.asarray([bounds_row[2 * j] for j in range(n_params)], dtype=np.float64)
    hi = np.asarray([bounds_row[2 * j + 1] for j in range(n_params)], dtype=np.float64)
    return np.clip(candidates, lo, hi)


def _e_space_bounds(ktrans_lo: float, ktrans_hi: float, fp_lo: float, fp_hi: float) -> Tuple[float, float]:
    """Map (Ktrans, Fp) bounds to extraction-fraction bounds E=Ktrans/Fp in (0, 1).

    Same formula `dce_pipeline._extraction_fraction_init_bounds` used (now
    removed) for its bounds half; kept separate from the per-candidate E
    initial-value clip, which each caller applies per voxel/candidate.
    """
    e_lo = min(max(ktrans_lo / max(fp_hi, 1e-12), 0.0), 1.0 - 1e-10)
    e_hi = min(max(ktrans_hi / max(fp_lo, 1e-12), e_lo + 1e-10), 1.0 - 1e-8)
    return float(e_lo), float(e_hi)


# Hardcoded canonical (per-minute) defaults -- already in the internal fit
# units, unlike caller-supplied prefs which follow the input timer's unit.
# Kept as a standalone constant so `_run_tissue_uptake_python` can merge the
# caller's *raw* overrides into these canonical values with correct
# unit scaling, instead of re-scaling the already-canonical defaults too.
_TISSUE_UPTAKE_DEFAULTS: Dict[str, Any] = {
    "lower_limit_ktrans": 1e-7,
    "upper_limit_ktrans": 2.0,
    "initial_value_ktrans": 2e-4,
    "lower_limit_fp": 1e-4,
    "upper_limit_fp": 100.0,
    "initial_value_fp": 0.2,
    "lower_limit_vp": 0.0,
    "upper_limit_vp": 1.0,
    "initial_value_vp": 0.02,
    "lower_limit_tp": 0.0,
    "upper_limit_tp": 1e6,
    "initial_value_tp": 0.05,
    "max_nfev": 2000,
    "tol_fun": 1e-12,
    "tol_x": 1e-6,
    "robust": "off",
    "multistart_starts": 4,
    "multistart_seed": 0,
}


def _tissue_uptake_settings(prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    settings: Dict[str, Any] = dict(_TISSUE_UPTAKE_DEFAULTS)
    if prefs:
        settings.update(prefs)
    return settings


def _tissue_uptake_bounds_row(settings: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(settings["lower_limit_ktrans"]),
            float(settings["upper_limit_ktrans"]),
            float(settings["lower_limit_fp"]),
            float(settings["upper_limit_fp"]),
            float(settings["lower_limit_vp"]),
            float(settings["upper_limit_vp"]),
        ],
        dtype=np.float64,
    )


def assemble_tissue_uptake_candidates(inputs: FitInputs) -> np.ndarray:
    """Fixed default + per-voxel linear-Patlak seed + N random log-uniform draws.

    Candidate space is [Ktrans, Fp, Vp] in output (raw, un-canonicalized)
    units -- shared by both backends even though they solve in different
    internal parameterizations (CPU: Ktrans/Fp/Tp canonical minutes;
    accelerated: E=Ktrans/Fp, vp, Fp); each backend's runner converts this
    physical/output space into whatever its own solver needs. The Ktrans/Fp
    linear-Patlak seed mirrors the one `dce_models.model_tissue_uptake_fit`
    has always tried; the rest of that function's 4 hand-tuned candidates are
    replaced by genuine random-search coverage, now shared by both backends
    for the first time (previously only the accelerated backend had any
    random multi-start, via the separate `_accel_multistart_refine`).
    """
    settings = inputs.prefs
    k_lo, k_hi = float(settings["lower_limit_ktrans"]), float(settings["upper_limit_ktrans"])
    fp_lo, fp_hi = float(settings["lower_limit_fp"]), float(settings["upper_limit_fp"])
    vp_lo, vp_hi = float(settings["lower_limit_vp"]), float(settings["upper_limit_vp"])
    k0 = float(settings["initial_value_ktrans"])
    fp0 = float(settings["initial_value_fp"])
    vp0 = float(settings["initial_value_vp"])
    n_voxels = inputs.n_voxels

    # The fixed default's Ktrans/Fp were historically expressed in canonical
    # (per-minute) units by dce_models.model_tissue_uptake_fit's own hardcoded
    # fallback; pre-divide by rate_in_to_min so the CPU runner's canonical
    # conversion (`* rate_in_to_min`) recovers that exact intended value
    # regardless of the input timer's unit. No-op whenever the timer is
    # already minutes-native (rate_in_to_min=1, true for every real pipeline
    # run and for the accelerated backend, which never applies this scaling).
    _, _, rate_in_to_min, _ = _canonical_time_context([float(v) for v in inputs.timer], settings)
    k0_raw = k0 / rate_in_to_min
    fp0_raw = fp0 / rate_in_to_min

    fixed = np.tile(np.array([k0_raw, fp0_raw, vp0], dtype=np.float64)[None, None, :], (1, n_voxels, 1))

    patlak_k = np.full(n_voxels, k0_raw, dtype=np.float64)
    patlak_fp = np.full(n_voxels, fp0_raw, dtype=np.float64)
    cp_vec = [float(v) for v in inputs.cp]
    t_vec = [float(v) for v in inputs.timer]
    for i in range(n_voxels):
        try:
            estimate = model_patlak_linear([float(v) for v in inputs.ct[:, i]], cp_vec, t_vec)
            k_guess = float(estimate[0])
            if math.isfinite(k_guess):
                k_guess = min(max(k_guess, k_lo), k_hi)
                patlak_k[i] = k_guess
                patlak_fp[i] = min(max(max(fp0_raw, k_guess * 1.25), fp_lo), fp_hi)
        except Exception:
            continue
    patlak = np.stack([patlak_k, patlak_fp, np.full(n_voxels, vp0, dtype=np.float64)], axis=-1)[None, :, :]

    n_random = int(settings.get("multistart_starts", 4))
    seed = int(settings.get("multistart_seed", 0))
    lower = np.array([k_lo, fp_lo, vp_lo], dtype=np.float64)
    upper = np.array([k_hi, fp_hi, vp_hi], dtype=np.float64)
    random_candidates = _log_uniform_candidates(n_random, n_voxels, lower, upper, seed)

    return np.concatenate([fixed, patlak, random_candidates], axis=0)


# See _TISSUE_UPTAKE_DEFAULTS: hardcoded canonical (per-minute) defaults, kept
# standalone so _run_2cxm_python can merge raw overrides in without
# re-scaling the already-canonical defaults.
_2CXM_DEFAULTS: Dict[str, Any] = {
    "lower_limit_ktrans": 1e-7,
    "upper_limit_ktrans": 2.0,
    "initial_value_ktrans": 2e-4,
    "lower_limit_ve": 0.02,
    "upper_limit_ve": 1.0,
    "initial_value_ve": 0.2,
    "lower_limit_vp": 1e-3,
    "upper_limit_vp": 1.0,
    "initial_value_vp": 0.02,
    "lower_limit_fp": 1e-4,
    "upper_limit_fp": 2.0,
    "initial_value_fp": 20.0 / 100.0,
    "max_nfev": 4000,
    "multistart_starts": 5,
    "multistart_seed": 0,
}


def _2cxm_settings(prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    settings: Dict[str, Any] = dict(_2CXM_DEFAULTS)
    if prefs:
        settings.update(prefs)
    return settings


def _2cxm_bounds_row(settings: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(settings["lower_limit_ktrans"]),
            float(settings["upper_limit_ktrans"]),
            float(settings["lower_limit_ve"]),
            float(settings["upper_limit_ve"]),
            float(settings["lower_limit_vp"]),
            float(settings["upper_limit_vp"]),
            float(settings["lower_limit_fp"]),
            float(settings["upper_limit_fp"]),
        ],
        dtype=np.float64,
    )


def assemble_2cxm_candidates(inputs: FitInputs) -> np.ndarray:
    """Fixed default + N random log-uniform draws.

    Candidate space is [Ktrans, ve, vp, Fp] in output (raw) units. Unlike
    patlak/tissue_uptake there is no closed-form seed for this model, so
    every candidate beyond the fixed prefs-default start is a random draw --
    this is the first multi-start of any kind CPU has had for 2cxm (it
    previously ran a single canonical `curve_fit` with no multistart at all);
    the accelerated backend already had random multi-start via
    `_accel_multistart_refine`, now replaced by this same shared mechanism.
    """
    settings = inputs.prefs
    lower = np.array(
        [
            float(settings["lower_limit_ktrans"]),
            float(settings["lower_limit_ve"]),
            float(settings["lower_limit_vp"]),
            float(settings["lower_limit_fp"]),
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            float(settings["upper_limit_ktrans"]),
            float(settings["upper_limit_ve"]),
            float(settings["upper_limit_vp"]),
            float(settings["upper_limit_fp"]),
        ],
        dtype=np.float64,
    )
    n_voxels = inputs.n_voxels
    # Ktrans/Fp defaults were historically canonical (per-minute); pre-divide by
    # rate_in_to_min so the CPU runner's canonical conversion recovers that
    # exact value regardless of timer unit (no-op when timer is minutes-native,
    # true for every real pipeline run and for the accelerated backend, which
    # never applies this scaling). See assemble_tissue_uptake_candidates.
    _, _, rate_in_to_min, _ = _canonical_time_context([float(v) for v in inputs.timer], settings)
    fixed_row = np.array(
        [
            float(settings["initial_value_ktrans"]) / rate_in_to_min,
            float(settings["initial_value_ve"]),
            float(settings["initial_value_vp"]),
            float(settings["initial_value_fp"]) / rate_in_to_min,
        ],
        dtype=np.float64,
    )
    fixed = np.tile(fixed_row[None, None, :], (1, n_voxels, 1))

    n_random = int(settings.get("multistart_starts", 5))
    seed = int(settings.get("multistart_seed", 0))
    random_candidates = _log_uniform_candidates(n_random, n_voxels, lower, upper, seed)
    return np.concatenate([fixed, random_candidates], axis=0)


def _run_scipy_per_voxel(
    inputs: FitInputs,
    initial_parameters: np.ndarray,
    cfit_fn,
    n_params: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-voxel scipy `least_squares` loop shared by every CPU/python model.

    ``cfit_fn(params_vec, cp_vec, t_vec) -> predicted Ct list`` wraps the
    model's forward curve function (e.g. `model_patlak_cfit`, `model_tofts_cfit`).
    Exceptions are caught per voxel: one bad voxel leaves that voxel's row NaN
    rather than failing the whole (now batched, no outer per-voxel loop in
    dce_pipeline.py) call. `extra[i]` is a `(ci_lo, ci_hi)` tuple (or `None` on
    failure) -- the same format every model's runner uses, so the top-level
    `fit_*_stage_d` output assembly doesn't need to know whether a given
    backend exposes a Jacobian. `initial_parameters` arrives already clamped
    to `inputs.bounds_row` by `fit_with_multistart`.
    """
    settings = inputs.prefs
    lb = [float(inputs.bounds_row[2 * j]) for j in range(n_params)]
    ub = [float(inputs.bounds_row[2 * j + 1]) for j in range(n_params)]
    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)
    cp_vec = [float(v) for v in inputs.cp]
    t_vec = [float(v) for v in inputs.timer]

    n_voxels = inputs.n_voxels
    params = np.full((n_voxels, n_params), np.nan, dtype=np.float64)
    chi = np.full(n_voxels, np.nan, dtype=np.float64)
    success = np.zeros(n_voxels, dtype=bool)
    extra = np.empty(n_voxels, dtype=object)

    for i in range(n_voxels):
        ct_vec = [float(v) for v in inputs.ct[:, i]]

        def residual(params_vec, ct_vec=ct_vec):
            pred = cfit_fn(params_vec, cp_vec, t_vec)
            return [pred[j] - ct_vec[j] for j in range(len(ct_vec))]

        x0 = [float(v) for v in initial_parameters[i]]
        try:
            fit = least_squares(residual, x0=x0, bounds=(lb, ub), **lsq_kwargs)
        except Exception:
            continue
        params[i, :] = fit.x
        chi[i] = float(sum(v * v for v in fit.fun))
        success[i] = True
        extra[i] = _ci_bounds_from_fit(fit)

    return params, success, chi, extra


def _run_patlak_python(
    inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _run_scipy_per_voxel(
        inputs,
        initial_parameters,
        cfit_fn=lambda p, cp_vec, t_vec: model_patlak_cfit(p[0], p[1], cp_vec, t_vec),
        n_params=2,
    )


def _run_tofts_python(
    inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _run_scipy_per_voxel(
        inputs,
        initial_parameters,
        cfit_fn=lambda p, cp_vec, t_vec: model_tofts_cfit(p[0], p[1], cp_vec, t_vec),
        n_params=2,
    )


def _run_ex_tofts_python(
    inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _run_scipy_per_voxel(
        inputs,
        initial_parameters,
        cfit_fn=lambda p, cp_vec, t_vec: model_extended_tofts_cfit(p[0], p[1], p[2], cp_vec, t_vec),
        n_params=3,
    )


_TISSUE_UPTAKE_RATE_KEYS = [
    "lower_limit_ktrans",
    "upper_limit_ktrans",
    "initial_value_ktrans",
    "lower_limit_fp",
    "upper_limit_fp",
    "initial_value_fp",
]
_TISSUE_UPTAKE_TIME_CONSTANT_KEYS = ["lower_limit_tp", "upper_limit_tp", "initial_value_tp"]


def _run_tissue_uptake_python(
    inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-voxel scipy fit in CPU's native (Ktrans, Fp, Tp) canonical-minutes space.

    Candidates arrive in shared output-space [Ktrans, Fp, Vp] (raw/un-
    canonicalized units); each is converted to a canonical (ktrans_min,
    fp_min, tp_min) starting point via the same vp=(Fp+PS)*Tp algebra
    `dce_models.model_tissue_uptake_fit` uses on its way out, run through
    exactly that function's forward model/objective, then converted back.
    Exceptions are caught per voxel/candidate (unlike the other models' shared
    per-voxel runner) since random draws can occasionally land somewhere the
    solver can't handle -- one bad random candidate should not sink a voxel
    that another candidate fits fine.
    """
    settings = inputs.prefs
    t_vec = [float(v) for v in inputs.timer]
    cp_vec = [float(v) for v in inputs.cp]
    timer_min, _, rate_in_to_min, rate_min_to_output = _canonical_time_context(t_vec, settings)
    canonical = _merge_prefs_in_canonical_units(
        _TISSUE_UPTAKE_DEFAULTS,
        inputs.raw_prefs,
        rate_keys=_TISSUE_UPTAKE_RATE_KEYS,
        time_constant_keys=_TISSUE_UPTAKE_TIME_CONSTANT_KEYS,
        rate_in_to_min=rate_in_to_min,
    )
    lb = [
        float(canonical["lower_limit_ktrans"]),
        float(canonical["lower_limit_fp"]),
        float(canonical["lower_limit_tp"]),
    ]
    ub = [
        float(canonical["upper_limit_ktrans"]),
        float(canonical["upper_limit_fp"]),
        float(canonical["upper_limit_tp"]),
    ]
    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)

    n_voxels = inputs.n_voxels
    params = np.full((n_voxels, 3), np.nan, dtype=np.float64)
    chi = np.full(n_voxels, np.nan, dtype=np.float64)
    success = np.zeros(n_voxels, dtype=bool)
    extra = np.empty(n_voxels, dtype=object)

    for i in range(n_voxels):
        ct_vec = [float(v) for v in inputs.ct[:, i]]
        ktrans0_raw, fp0_raw, vp0_raw = (float(v) for v in initial_parameters[i])
        ktrans0 = ktrans0_raw * rate_in_to_min
        fp0 = fp0_raw * rate_in_to_min
        ps0 = 1e8 if abs(fp0 - ktrans0) < 1e-12 else ktrans0 * fp0 / (fp0 - ktrans0)
        denom0 = fp0 + ps0
        tp0 = vp0_raw / denom0 if math.isfinite(denom0) and abs(denom0) > 1e-12 else float(canonical["initial_value_tp"])
        x0 = [
            min(max(ktrans0, lb[0]), ub[0]),
            min(max(fp0, lb[1]), ub[1]),
            min(max(tp0, lb[2]), ub[2]),
        ]

        def residual(p, ct_vec=ct_vec):
            pred = model_tissue_uptake_cfit(p[0], p[1], p[2], cp_vec, timer_min)
            return [pred[j] - ct_vec[j] for j in range(len(ct_vec))]

        try:
            fit = least_squares(residual, x0=x0, bounds=(lb, ub), **lsq_kwargs)
        except Exception:
            continue

        ktrans_fit, fp_fit, tp_fit = float(fit.x[0]), float(fit.x[1]), float(fit.x[2])
        ps_fit = 1e8 if abs(fp_fit - ktrans_fit) < 1e-12 else ktrans_fit * fp_fit / (fp_fit - ktrans_fit)
        vp_fit = (fp_fit + ps_fit) * tp_fit

        params[i, 0] = ktrans_fit * rate_min_to_output
        params[i, 1] = fp_fit * rate_min_to_output
        params[i, 2] = vp_fit
        chi[i] = float(sum(v * v for v in fit.fun))
        success[i] = True

        ci_lo, ci_hi = _ci_bounds_from_fit(fit)
        extra[i] = (
            [ci_lo[0] * rate_min_to_output, ci_lo[1] * rate_min_to_output, (fp_fit + ps_fit) * ci_lo[2]],
            [ci_hi[0] * rate_min_to_output, ci_hi[1] * rate_min_to_output, (fp_fit + ps_fit) * ci_hi[2]],
        )

    return params, success, chi, extra


_2CXM_RATE_KEYS = [
    "lower_limit_ktrans",
    "upper_limit_ktrans",
    "initial_value_ktrans",
    "lower_limit_fp",
    "upper_limit_fp",
    "initial_value_fp",
]


def _run_2cxm_python(
    inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-voxel canonical-units OSIPI curve_fit, reusing `_fit_2cxm_osipi_canonical`.

    Each candidate is run through the exact existing (validated) canonical
    fit function unchanged, just with its `initial_value_*` settings swapped
    for this candidate's starting point -- the safest way to add multistart
    to the model flagged as the most numerically fragile in this project,
    since none of its existing math is touched.
    """
    settings = inputs.prefs
    t_vec = [float(v) for v in inputs.timer]
    cp_vec = [float(v) for v in inputs.cp]
    timer_min, _, rate_in_to_min, rate_min_to_output = _canonical_time_context(t_vec, settings)
    canonical = _merge_prefs_in_canonical_units(
        _2CXM_DEFAULTS, inputs.raw_prefs, rate_keys=_2CXM_RATE_KEYS, rate_in_to_min=rate_in_to_min
    )

    n_voxels = inputs.n_voxels
    params = np.full((n_voxels, 4), np.nan, dtype=np.float64)
    chi = np.full(n_voxels, np.nan, dtype=np.float64)
    success = np.zeros(n_voxels, dtype=bool)
    extra = np.empty(n_voxels, dtype=object)

    for i in range(n_voxels):
        ct_vec = [float(v) for v in inputs.ct[:, i]]
        ktrans0_raw, ve0, vp0, fp0_raw = (float(v) for v in initial_parameters[i])
        candidate_settings = dict(canonical)
        candidate_settings["initial_value_ktrans"] = ktrans0_raw * rate_in_to_min
        candidate_settings["initial_value_ve"] = ve0
        candidate_settings["initial_value_vp"] = vp0
        candidate_settings["initial_value_fp"] = fp0_raw * rate_in_to_min

        try:
            quick = _fit_2cxm_osipi_canonical(ct_vec, cp_vec, timer_min, settings=candidate_settings)
        except Exception:
            continue
        if quick is None:
            continue

        for idx in (0, 3, 5, 6, 11, 12):
            quick[idx] = float(quick[idx]) * rate_min_to_output

        params[i, 0] = quick[0]
        params[i, 1] = quick[1]
        params[i, 2] = quick[2]
        params[i, 3] = quick[3]
        chi[i] = quick[4]
        success[i] = True
        extra[i] = (
            [quick[5], quick[7], quick[9], quick[11]],
            [quick[6], quick[8], quick[10], quick[12]],
        )

    return params, success, chi, extra


def _run_accelerated(
    backend: str,
    model_id_name: str,
    n_params: int,
    inputs: FitInputs,
    initial_parameters: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One `fit_constrained` call for the whole batch, shared by every accelerated model."""
    from dce_pipeline import _load_fit_module_for_acceleration  # local: breaks an import cycle

    fit_module = _load_fit_module_for_acceleration(backend)
    n_fits = inputs.n_voxels

    data = np.ascontiguousarray(np.asarray(inputs.ct.T, dtype=np.float32))
    timer_f32 = np.ascontiguousarray(np.asarray(inputs.timer, dtype=np.float32).reshape(-1))
    cp_f32 = np.ascontiguousarray(np.asarray(inputs.cp, dtype=np.float32).reshape(-1))
    user_info = np.ascontiguousarray(np.concatenate([timer_f32, cp_f32], axis=0), dtype=np.float32)

    try:
        model_id = int(getattr(fit_module.ModelID, model_id_name))
    except AttributeError as exc:
        raise RuntimeError(f"Acceleration backend does not expose ModelID.{model_id_name}") from exc

    constraint_types = np.ascontiguousarray(
        np.full((n_params,), int(fit_module.ConstraintType.LOWER_UPPER), dtype=np.int32)
    )
    constraints = np.ascontiguousarray(
        np.tile(np.asarray(inputs.bounds_row, dtype=np.float32)[None, :], (n_fits, 1))
    )
    tolerance = float(inputs.prefs.get("gpu_tolerance", 1e-6))
    max_iterations = int(inputs.prefs.get("gpu_max_n_iterations", 200))

    parameters, states, chi_squares, _, _ = fit_module.fit_constrained(
        data=data,
        weights=None,
        model_id=model_id,
        initial_parameters=np.ascontiguousarray(initial_parameters, dtype=np.float32),
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=tolerance,
        max_number_iterations=max_iterations,
        parameters_to_fit=None,
        estimator_id=int(fit_module.EstimatorID.LSE),
        user_info=user_info,
    )

    params = np.asarray(parameters, dtype=np.float64).reshape(n_fits, -1)
    chi = np.asarray(chi_squares, dtype=np.float64).reshape(-1)
    success = np.asarray(states, dtype=np.int32).reshape(-1) == 0
    failed = ~success
    if np.any(failed):
        params[failed, :] = np.nan
        chi[failed] = np.nan
    extra = np.full(n_fits, None, dtype=object)
    return params, success, chi, extra


def _run_patlak_accelerated(
    backend: str, inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _run_accelerated(backend, "PATLAK", 2, inputs, initial_parameters)


def _run_tofts_accelerated(
    backend: str, inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _run_accelerated(backend, "TOFTS", 2, inputs, initial_parameters)


def _run_ex_tofts_accelerated(
    backend: str, inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _run_accelerated(backend, "TOFTS_EXTENDED", 3, inputs, initial_parameters)


def _run_tissue_uptake_accelerated(
    backend: str, inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reparametrize shared [Ktrans, Fp, Vp] candidates into the kernel's [E, vp, Fp] space.

    E = Ktrans/Fp mirrors `dce_pipeline._extraction_fraction_init_bounds`
    (now removed); vp passes straight through unconverted (accelerated never
    applied a canonical-unit conversion, matching the prior implementation
    exactly).
    """
    bounds = inputs.bounds_row  # [k_lo, k_hi, fp_lo, fp_hi, vp_lo, vp_hi]
    e_lo, e_hi = _e_space_bounds(bounds[0], bounds[1], bounds[2], bounds[3])
    e_bounds = np.array([e_lo, e_hi, bounds[4], bounds[5], bounds[2], bounds[3]], dtype=np.float64)
    e_inputs = replace(inputs, bounds_row=e_bounds)

    ktrans0 = initial_parameters[:, 0]
    fp0 = np.maximum(initial_parameters[:, 1], 1e-12)
    vp0 = initial_parameters[:, 2]
    e0 = np.clip(ktrans0 / fp0, e_lo + 1e-10, e_hi - 1e-10)
    e_init = np.stack([e0, vp0, fp0], axis=-1)

    params, success, chi, extra = _run_accelerated(backend, "TISSUE_UPTAKE", 3, e_inputs, e_init)
    # Kernel params are [E, vp, Fp]; recover Ktrans = E * Fp.
    ktrans_out = params[:, 0] * params[:, 2]
    out_params = np.stack([ktrans_out, params[:, 2], params[:, 1]], axis=-1)
    return out_params, success, chi, extra


def _run_2cxm_accelerated(
    backend: str, inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reparametrize shared [Ktrans, ve, vp, Fp] candidates into the kernel's [E, ve, vp, Fp] space."""
    bounds = inputs.bounds_row  # [k_lo,k_hi, ve_lo,ve_hi, vp_lo,vp_hi, fp_lo,fp_hi]
    e_lo, e_hi = _e_space_bounds(bounds[0], bounds[1], bounds[6], bounds[7])
    e_bounds = np.array(
        [e_lo, e_hi, bounds[2], bounds[3], bounds[4], bounds[5], bounds[6], bounds[7]], dtype=np.float64
    )
    e_inputs = replace(inputs, bounds_row=e_bounds)

    ktrans0 = initial_parameters[:, 0]
    ve0 = initial_parameters[:, 1]
    vp0 = initial_parameters[:, 2]
    fp0 = np.maximum(initial_parameters[:, 3], 1e-12)
    e0 = np.clip(ktrans0 / fp0, e_lo + 1e-10, e_hi - 1e-10)
    e_init = np.stack([e0, ve0, vp0, fp0], axis=-1)

    params, success, chi, extra = _run_accelerated(backend, "TWO_COMPARTMENT_EXCHANGE", 4, e_inputs, e_init)
    # Kernel params are [E, ve, vp, Fp]; recover Ktrans = E * Fp.
    ktrans_out = params[:, 0] * params[:, 3]
    out_params = np.stack([ktrans_out, params[:, 1], params[:, 2], params[:, 3]], axis=-1)
    return out_params, success, chi, extra


_PYTHON_RUNNERS = {
    "patlak": _run_patlak_python,
    "tofts": _run_tofts_python,
    "ex_tofts": _run_ex_tofts_python,
    "tissue_uptake": _run_tissue_uptake_python,
    "2cxm": _run_2cxm_python,
}
_ACCELERATED_RUNNERS = {
    "patlak": _run_patlak_accelerated,
    "tofts": _run_tofts_accelerated,
    "ex_tofts": _run_ex_tofts_accelerated,
    "tissue_uptake": _run_tissue_uptake_accelerated,
    "2cxm": _run_2cxm_accelerated,
}


def fit_with_multistart(
    backend: str,
    model_name: str,
    inputs: FitInputs,
    candidates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Try every candidate start, keep the per-voxel best result by chi-square.

    Generalizes dce_models._best_fit_over_starts and
    dce_pipeline._accel_multistart_refine's "keep the lower SSE/chi-square"
    bookkeeping into one implementation that works whether a candidate is fit
    one voxel at a time (python) or the whole batch at once (cpufit/gpufit).

    Returns (params, success_mask, chi_square, extra) for the best candidate
    per voxel; `extra` is a `(ci_lo, ci_hi)` tuple or `None` (see
    `_run_scipy_per_voxel`/`_run_accelerated`).
    """
    runners = _PYTHON_RUNNERS if backend == "python" else _ACCELERATED_RUNNERS
    runner = runners.get(model_name)
    if runner is None:
        raise NotImplementedError(
            f"fit_with_multistart: model '{model_name}' is not yet migrated for backend '{backend}'"
        )

    n_starts, n_voxels, n_params = candidates.shape
    best_params = np.full((n_voxels, n_params), np.nan, dtype=np.float64)
    best_chi = np.full(n_voxels, np.inf, dtype=np.float64)
    best_success = np.zeros(n_voxels, dtype=bool)
    best_extra = np.empty(n_voxels, dtype=object)

    for s in range(n_starts):
        start = _clamp_to_bounds(candidates[s], inputs.bounds_row)
        if backend == "python":
            params, success, chi, extra = runner(inputs, start)
        else:
            params, success, chi, extra = runner(backend, inputs, start)
        with np.errstate(invalid="ignore"):
            improved = success & np.isfinite(chi) & (chi < best_chi)
        best_params[improved] = params[improved]
        best_chi[improved] = chi[improved]
        best_extra[improved] = extra[improved]
        best_success = best_success | improved

    best_chi = np.where(best_success, best_chi, np.nan)
    return best_params, best_success, best_chi, best_extra


def _validate_stage_d_inputs(
    ct: np.ndarray, cp: np.ndarray, timer: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Normalize/validate the (ct, cp, timer) triple shared by every fit_*_stage_d entry point.

    Returns (ct_arr, cp_arr, timer_arr, single_voxel); ct_arr is always 2-D
    (n_time, n_voxels), with single_voxel recording whether the caller passed
    a bare (n_time,) vector that should be unwrapped again on output.
    """
    ct_arr = np.asarray(ct, dtype=np.float64)
    single_voxel = ct_arr.ndim == 1
    if single_voxel:
        ct_arr = ct_arr[:, None]
    cp_arr = np.asarray(cp, dtype=np.float64).reshape(-1)
    timer_arr = np.asarray(timer, dtype=np.float64).reshape(-1)

    if not (ct_arr.shape[0] == cp_arr.shape[0] == timer_arr.shape[0]):
        raise ValueError(
            f"ct/cp/timer lengths differ: {ct_arr.shape[0]} / {cp_arr.shape[0]} / {timer_arr.shape[0]}"
        )
    if ct_arr.shape[0] == 0:
        raise ValueError("ct/cp/timer must be non-empty")
    return ct_arr, cp_arr, timer_arr, single_voxel


def _assemble_stage_d_output(
    n_voxels: int,
    n_params: int,
    params: np.ndarray,
    chi: np.ndarray,
    extra: np.ndarray,
    *,
    ci_fallback: str,
) -> np.ndarray:
    """Build the shared MATLAB-style output row every model uses.

    Column layout (row_len = 3*n_params + 1): point estimates, then sse, then
    ci_low/ci_high interleaved per parameter in the same order -- true for all
    five models' `MODEL_LAYOUTS` entries. `extra[i]` is a `(ci_lo, ci_hi)`
    tuple (real CIs, from either backend's runner) or `None` (no Jacobian
    available); `ci_fallback` picks what the CI columns fall back to in the
    `None` case: `"point_estimate"` (repeat the fitted value, used by every
    model but patlak) or `"sentinel"` (patlak's -1.0, its long-standing
    accelerated-backend convention).
    """
    row_len = 3 * n_params + 1
    out = np.full((n_voxels, row_len), np.nan, dtype=np.float64)
    for i in range(n_voxels):
        out[i, :n_params] = params[i, :]
        out[i, n_params] = float(chi[i])
        extra_i = extra[i]
        for j in range(n_params):
            lo_col, hi_col = n_params + 1 + 2 * j, n_params + 2 + 2 * j
            if extra_i is not None:
                ci_lo, ci_hi = extra_i
                out[i, lo_col], out[i, hi_col] = float(ci_lo[j]), float(ci_hi[j])
            elif ci_fallback == "sentinel":
                out[i, lo_col] = out[i, hi_col] = -1.0
            else:
                out[i, lo_col] = out[i, hi_col] = float(params[i, j])
    return out


@dataclass
class _ModelSpec:
    """Everything `_fit_stage_d_batch` needs to fit one model, gathered in one place."""

    settings_fn: Any
    bounds_fn: Any
    assemble_fn: Any
    n_params: int
    ci_fallback: str  # "point_estimate" or "sentinel" (patlak only, see _assemble_stage_d_output)


_MODEL_SPECS: Dict[str, _ModelSpec] = {
    "patlak": _ModelSpec(_patlak_settings, _patlak_bounds_row, assemble_patlak_candidates, 2, "sentinel"),
    "tofts": _ModelSpec(_tofts_settings, _tofts_bounds_row, assemble_tofts_candidates, 2, "point_estimate"),
    "ex_tofts": _ModelSpec(
        _ex_tofts_settings, _ex_tofts_bounds_row, assemble_ex_tofts_candidates, 3, "point_estimate"
    ),
    "tissue_uptake": _ModelSpec(
        _tissue_uptake_settings,
        _tissue_uptake_bounds_row,
        assemble_tissue_uptake_candidates,
        3,
        "point_estimate",
    ),
    "2cxm": _ModelSpec(_2cxm_settings, _2cxm_bounds_row, assemble_2cxm_candidates, 4, "point_estimate"),
}


def _fit_stage_d_batch(
    model_name: str,
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Optional[Dict[str, Any]],
    backend: str,
) -> np.ndarray:
    """Shared engine behind every `fit_*_stage_d` entry point.

    Assembles this model's settings/bounds/candidates via its `_ModelSpec`,
    runs them through `fit_with_multistart`, and maps the result to the
    model's MATLAB-style output row. One or many voxels, either backend --
    see the public `fit_*_stage_d` wrappers for the per-model contract.
    """
    _reject_algorithm_override(prefs, model_name)
    ct_arr, cp_arr, timer_arr, single_voxel = _validate_stage_d_inputs(ct, cp, timer)

    spec = _MODEL_SPECS[model_name]
    settings = spec.settings_fn(prefs)
    bounds_row = spec.bounds_fn(settings)
    inputs = FitInputs(
        ct=ct_arr, cp=cp_arr, timer=timer_arr, bounds_row=bounds_row, prefs=settings, raw_prefs=prefs
    )
    candidates = spec.assemble_fn(inputs)

    params, _success, chi, extra = fit_with_multistart(backend, model_name, inputs, candidates)
    out = _assemble_stage_d_output(ct_arr.shape[1], spec.n_params, params, chi, extra, ci_fallback=spec.ci_fallback)
    return out[0] if single_voxel else out


def fit_patlak_stage_d(
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Optional[Dict[str, Any]] = None,
    backend: str = "python",
) -> np.ndarray:
    """Stage-D patlak fit for one or many voxels, on any backend.

    Args:
      ct: (n_time,) for a single voxel, or (n_time, n_voxels) for a batch.
      cp, timer: (n_time,), shared across voxels.
      backend: "python" (scipy per-voxel) or an accelerated backend name
        accepted by dce_pipeline._load_fit_module_for_acceleration
        (e.g. "gpufit", "cpufit_cpu").

    Returns a MATLAB-style row: (7,) for a single voxel, (n_voxels, 7) for a
    batch, matching dce_pipeline.MODEL_LAYOUTS["patlak"]["param_names"]
    (Ktrans, vp, sse, ktrans_ci_low/high, vp_ci_low/high). On the accelerated
    backend (no Jacobian available), CI columns are -1.0 -- patlak's
    long-standing accelerated-backend sentinel, unlike every other model here.
    """
    return _fit_stage_d_batch("patlak", ct, cp, timer, prefs, backend)


def fit_tofts_stage_d(
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Optional[Dict[str, Any]] = None,
    backend: str = "python",
) -> np.ndarray:
    """Stage-D tofts fit for one or many voxels, on any backend.

    Single fixed starting candidate today (no multi-start, matching the prior
    implementation on both backends).

    Args:
      ct: (n_time,) for a single voxel, or (n_time, n_voxels) for a batch.
      cp, timer: (n_time,), shared across voxels.
      backend: "python" (scipy per-voxel) or an accelerated backend name
        accepted by dce_pipeline._load_fit_module_for_acceleration
        (e.g. "gpufit", "cpufit_cpu").

    Returns a MATLAB-style row: (7,) for a single voxel, (n_voxels, 7) for a
    batch, matching dce_pipeline.MODEL_LAYOUTS["tofts"]["param_names"]
    (Ktrans, ve, sse, ktrans_ci_low/high, ve_ci_low/high). On the accelerated
    backend (no Jacobian available), CI columns repeat the point estimate --
    the same convention the prior implementation used for this model.
    """
    return _fit_stage_d_batch("tofts", ct, cp, timer, prefs, backend)


def fit_ex_tofts_stage_d(
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Optional[Dict[str, Any]] = None,
    backend: str = "python",
) -> np.ndarray:
    """Stage-D ex_tofts fit for one or many voxels, on any backend.

    Fixed x1/x10/x100-on-Ktrans multi-start (the same 3 candidates
    `dce_models.model_extended_tofts_fit` has always tried on the CPU path).
    Unlike the tofts/patlak migrations, this is a real behavior change on the
    accelerated backend: it previously ran a single fixed start with no
    multistart at all, and now gets the same 3-candidate multistart the CPU
    path already had.

    Args:
      ct: (n_time,) for a single voxel, or (n_time, n_voxels) for a batch.
      cp, timer: (n_time,), shared across voxels.
      backend: "python" (scipy per-voxel) or an accelerated backend name
        accepted by dce_pipeline._load_fit_module_for_acceleration
        (e.g. "gpufit", "cpufit_cpu").

    Returns a MATLAB-style row: (10,) for a single voxel, (n_voxels, 10) for a
    batch, matching dce_pipeline.MODEL_LAYOUTS["ex_tofts"]["param_names"]
    (Ktrans, ve, vp, sse, ktrans_ci_low/high, ve_ci_low/high, vp_ci_low/high).
    On the accelerated backend (no Jacobian available), CI columns repeat the
    point estimate -- the same convention the prior implementation used for
    this model.
    """
    return _fit_stage_d_batch("ex_tofts", ct, cp, timer, prefs, backend)


def fit_tissue_uptake_stage_d(
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Optional[Dict[str, Any]] = None,
    backend: str = "python",
) -> np.ndarray:
    """Stage-D tissue_uptake fit for one or many voxels, on any backend.

    Fixed default + per-voxel linear-Patlak seed + N random log-uniform
    draws (see `assemble_tissue_uptake_candidates`). Real behavior change on
    both backends: CPU previously used 4 hand-tuned candidates plus a patlak
    seed; the accelerated backend previously used its own separate
    random-multistart mechanism (`_accel_multistart_refine`, now removed).
    Both now draw from the same shared candidate set.

    Args:
      ct: (n_time,) for a single voxel, or (n_time, n_voxels) for a batch.
      cp, timer: (n_time,), shared across voxels.
      backend: "python" (scipy per-voxel) or an accelerated backend name
        accepted by dce_pipeline._load_fit_module_for_acceleration
        (e.g. "gpufit", "cpufit_cpu").

    Returns a MATLAB-style row: (10,) for a single voxel, (n_voxels, 10) for a
    batch, matching dce_pipeline.MODEL_LAYOUTS["tissue_uptake"]["param_names"]
    (Ktrans, fp, vp, sse, ktrans_ci_low/high, fp_ci_low/high, vp_ci_low/high).
    On the accelerated backend (no Jacobian available), CI columns repeat the
    point estimate -- the same convention the prior implementation used.
    """
    return _fit_stage_d_batch("tissue_uptake", ct, cp, timer, prefs, backend)


def fit_2cxm_stage_d(
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Optional[Dict[str, Any]] = None,
    backend: str = "python",
) -> np.ndarray:
    """Stage-D 2cxm fit for one or many voxels, on any backend.

    Fixed default + N random log-uniform draws (see `assemble_2cxm_candidates`).
    Real behavior change on both backends: CPU previously had no multistart at
    all for this model (a single canonical `curve_fit` call); the accelerated
    backend previously used its own separate random-multistart mechanism
    (`_accel_multistart_refine`, now removed). Both now draw from the same
    shared candidate set.

    Args:
      ct: (n_time,) for a single voxel, or (n_time, n_voxels) for a batch.
      cp, timer: (n_time,), shared across voxels.
      backend: "python" (scipy per-voxel) or an accelerated backend name
        accepted by dce_pipeline._load_fit_module_for_acceleration
        (e.g. "gpufit", "cpufit_cpu").

    Returns a MATLAB-style row: (13,) for a single voxel, (n_voxels, 13) for a
    batch, matching dce_pipeline.MODEL_LAYOUTS["2cxm"]["param_names"] (Ktrans,
    ve, vp, fp, sse, ktrans_ci_low/high, ve_ci_low/high, vp_ci_low/high,
    fp_ci_low/high). On the accelerated backend (no Jacobian available), CI
    columns repeat the point estimate -- the same convention the prior
    implementation used.
    """
    return _fit_stage_d_batch("2cxm", ct, cp, timer, prefs, backend)
