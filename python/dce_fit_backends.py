"""Shared Stage-D fitting machinery: one place to assemble fit inputs and run
multi-start optimization, shared by the CPU/python and accelerated
(cpufit/gpufit) backends.

Wired up for patlak, tofts, and ex_tofts so far. The intent is that
tissue_uptake/2cxm migrate onto the same three pieces -- `FitInputs` + a
per-model candidate-start assembler + `fit_with_multistart` -- in a follow-up
pass, replacing dce_models._best_fit_over_starts and
dce_pipeline._accel_multistart_refine once every model has moved over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from dce_models import (
    _ci_bounds_from_fit,
    _least_squares_kwargs,
    model_extended_tofts_cfit,
    model_patlak_cfit,
    model_patlak_linear,
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
    """Per-voxel patlak candidate starts: linear-regression seed, then x10/x100.

    Mirrors the seeding `dce_models.model_patlak_fit` has always used (the
    closed-form linear Patlak estimate per voxel, falling back to the prefs
    default when non-finite) expanded into the same 3 fixed-multiplier rows
    it has always tried. Returns shape (3, n_voxels, 2), columns [ktrans, vp].
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

    return np.stack(
        [
            np.stack([base_k, base_vp], axis=-1),
            np.stack([base_k * 10.0, base_vp], axis=-1),
            np.stack([base_k * 100.0, base_vp], axis=-1),
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


def _run_scipy_per_voxel(
    inputs: FitInputs,
    initial_parameters: np.ndarray,
    cfit_fn,
    n_params: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-voxel scipy `least_squares` loop shared by every CPU/python model.

    ``cfit_fn(params_vec, cp_vec, t_vec) -> predicted Ct list`` wraps the
    model's forward curve function (e.g. `model_patlak_cfit`, `model_tofts_cfit`).
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
    success = np.ones(n_voxels, dtype=bool)
    extra = np.empty(n_voxels, dtype=object)

    for i in range(n_voxels):
        ct_vec = [float(v) for v in inputs.ct[:, i]]

        def residual(params_vec, ct_vec=ct_vec):
            pred = cfit_fn(params_vec, cp_vec, t_vec)
            return [pred[j] - ct_vec[j] for j in range(len(ct_vec))]

        x0 = [min(max(float(initial_parameters[i, j]), lb[j]), ub[j]) for j in range(n_params)]
        fit = least_squares(residual, x0=x0, bounds=(lb, ub), **lsq_kwargs)
        params[i, :] = fit.x
        chi[i] = float(sum(v * v for v in fit.fun))
        extra[i] = fit

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


_PYTHON_RUNNERS = {
    "patlak": _run_patlak_python,
    "tofts": _run_tofts_python,
    "ex_tofts": _run_ex_tofts_python,
}
_ACCELERATED_RUNNERS = {
    "patlak": _run_patlak_accelerated,
    "tofts": _run_tofts_accelerated,
    "ex_tofts": _run_ex_tofts_accelerated,
}


def run_backend_fit(
    backend: str,
    model_name: str,
    inputs: FitInputs,
    initial_parameters: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one fit attempt for every voxel from `initial_parameters`.

    Returns (params, success_mask, chi_square, extra) arrays, one row per
    voxel; `extra` carries a backend-specific per-voxel payload (the scipy
    ``OptimizeResult`` for ``backend="python"``, used for confidence-interval
    computation; ``None`` for accelerated backends, which don't expose a
    Jacobian). Backend-agnostic entry point: the caller assembles `inputs`
    once and passes the same object regardless of which backend is selected.
    """
    runners = _PYTHON_RUNNERS if backend == "python" else _ACCELERATED_RUNNERS
    runner = runners.get(model_name)
    if runner is None:
        raise NotImplementedError(
            f"run_backend_fit: model '{model_name}' is not yet migrated for backend '{backend}'"
        )
    if backend == "python":
        return runner(inputs, initial_parameters)
    return runner(backend, inputs, initial_parameters)


def fit_with_multistart(
    backend: str,
    model_name: str,
    inputs: FitInputs,
    candidates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Try every candidate start, keep the per-voxel best result by chi-square.

    Generalizes dce_models._best_fit_over_starts and
    dce_pipeline._accel_multistart_refine's "keep the lower SSE/chi-square"
    bookkeeping into one implementation that works whether `run_backend_fit`
    fits one voxel at a time (python) or the whole batch at once
    (cpufit/gpufit).
    """
    n_starts, n_voxels, n_params = candidates.shape
    best_params = np.full((n_voxels, n_params), np.nan, dtype=np.float64)
    best_chi = np.full(n_voxels, np.inf, dtype=np.float64)
    best_success = np.zeros(n_voxels, dtype=bool)
    best_extra = np.empty(n_voxels, dtype=object)

    for s in range(n_starts):
        params, success, chi, extra = run_backend_fit(backend, model_name, inputs, candidates[s])
        with np.errstate(invalid="ignore"):
            improved = success & np.isfinite(chi) & (chi < best_chi)
        best_params[improved] = params[improved]
        best_chi[improved] = chi[improved]
        best_extra[improved] = extra[improved]
        best_success = best_success | improved

    best_chi = np.where(best_success, best_chi, np.nan)
    return best_params, best_success, best_chi, best_extra


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
    (Ktrans, vp, sse, ktrans_ci_low/high, vp_ci_low/high).
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

    settings = _patlak_settings(prefs)
    bounds_row = _patlak_bounds_row(settings)
    inputs = FitInputs(ct=ct_arr, cp=cp_arr, timer=timer_arr, bounds_row=bounds_row, prefs=settings)
    candidates = assemble_patlak_candidates(inputs)

    params, _success, chi, extra = fit_with_multistart(backend, "patlak", inputs, candidates)

    n_voxels = ct_arr.shape[1]
    out = np.full((n_voxels, 7), np.nan, dtype=np.float64)
    for i in range(n_voxels):
        out[i, 0] = float(params[i, 0])
        out[i, 1] = float(params[i, 1])
        out[i, 2] = float(chi[i])
        fit_obj = extra[i]
        if fit_obj is not None:
            ci_lo, ci_hi = _ci_bounds_from_fit(fit_obj)
            out[i, 3], out[i, 4] = ci_lo[0], ci_hi[0]
            out[i, 5], out[i, 6] = ci_lo[1], ci_hi[1]
        else:
            out[i, 3] = out[i, 4] = out[i, 5] = out[i, 6] = -1.0

    return out[0] if single_voxel else out


def fit_tofts_stage_d(
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Optional[Dict[str, Any]] = None,
    backend: str = "python",
) -> np.ndarray:
    """Stage-D tofts fit for one or many voxels, on any backend.

    Single fixed starting candidate today (no multi-start, matching the prior
    implementation on both backends), run through the same
    assemble-candidates/`fit_with_multistart` machinery as patlak.

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

    settings = _tofts_settings(prefs)
    bounds_row = _tofts_bounds_row(settings)
    inputs = FitInputs(ct=ct_arr, cp=cp_arr, timer=timer_arr, bounds_row=bounds_row, prefs=settings)
    candidates = assemble_tofts_candidates(inputs)

    params, _success, chi, extra = fit_with_multistart(backend, "tofts", inputs, candidates)

    n_voxels = ct_arr.shape[1]
    out = np.full((n_voxels, 7), np.nan, dtype=np.float64)
    for i in range(n_voxels):
        out[i, 0] = float(params[i, 0])
        out[i, 1] = float(params[i, 1])
        out[i, 2] = float(chi[i])
        fit_obj = extra[i]
        if fit_obj is not None:
            ci_lo, ci_hi = _ci_bounds_from_fit(fit_obj)
            out[i, 3], out[i, 4] = ci_lo[0], ci_hi[0]
            out[i, 5], out[i, 6] = ci_lo[1], ci_hi[1]
        else:
            out[i, 3] = out[i, 4] = float(params[i, 0])
            out[i, 5] = out[i, 6] = float(params[i, 1])

    return out[0] if single_voxel else out


def fit_ex_tofts_stage_d(
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Optional[Dict[str, Any]] = None,
    backend: str = "python",
) -> np.ndarray:
    """Stage-D ex_tofts fit for one or many voxels, on any backend.

    Fixed x1/x10/x100-on-Ktrans multi-start (the same 3 candidates
    `dce_models.model_extended_tofts_fit` has always tried on the CPU path),
    run through the shared assemble-candidates/`fit_with_multistart` machinery.
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

    settings = _ex_tofts_settings(prefs)
    bounds_row = _ex_tofts_bounds_row(settings)
    inputs = FitInputs(ct=ct_arr, cp=cp_arr, timer=timer_arr, bounds_row=bounds_row, prefs=settings)
    candidates = assemble_ex_tofts_candidates(inputs)

    params, _success, chi, extra = fit_with_multistart(backend, "ex_tofts", inputs, candidates)

    n_voxels = ct_arr.shape[1]
    out = np.full((n_voxels, 10), np.nan, dtype=np.float64)
    for i in range(n_voxels):
        out[i, 0] = float(params[i, 0])
        out[i, 1] = float(params[i, 1])
        out[i, 2] = float(params[i, 2])
        out[i, 3] = float(chi[i])
        fit_obj = extra[i]
        if fit_obj is not None:
            ci_lo, ci_hi = _ci_bounds_from_fit(fit_obj)
            out[i, 4], out[i, 5] = ci_lo[0], ci_hi[0]
            out[i, 6], out[i, 7] = ci_lo[1], ci_hi[1]
            out[i, 8], out[i, 9] = ci_lo[2], ci_hi[2]
        else:
            out[i, 4] = out[i, 5] = float(params[i, 0])
            out[i, 6] = out[i, 7] = float(params[i, 1])
            out[i, 8] = out[i, 9] = float(params[i, 2])

    return out[0] if single_voxel else out
