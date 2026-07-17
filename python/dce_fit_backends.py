"""Shared Stage-D fitting machinery: one place to assemble fit inputs and run
multi-start optimization, shared by the CPU/python and accelerated
(cpufit/gpufit) backends.

Currently wired up for the patlak model only (pilot). The intent is that
tofts/ex_tofts/tissue_uptake/2cxm migrate onto the same three pieces --
`FitInputs` + a per-model candidate-start assembler + `fit_with_multistart` --
in a follow-up pass, replacing dce_models._best_fit_over_starts and
dce_pipeline._accel_multistart_refine once every model has moved over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from dce_models import _ci_bounds_from_fit, _least_squares_kwargs, model_patlak_cfit, model_patlak_linear


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


def _run_patlak_python(
    inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    settings = inputs.prefs
    lb = [float(inputs.bounds_row[0]), float(inputs.bounds_row[2])]
    ub = [float(inputs.bounds_row[1]), float(inputs.bounds_row[3])]
    lsq_kwargs = _least_squares_kwargs(settings, default_max_nfev=2000)
    cp_vec = [float(v) for v in inputs.cp]
    t_vec = [float(v) for v in inputs.timer]

    n_voxels = inputs.n_voxels
    params = np.full((n_voxels, 2), np.nan, dtype=np.float64)
    chi = np.full(n_voxels, np.nan, dtype=np.float64)
    success = np.ones(n_voxels, dtype=bool)
    extra = np.empty(n_voxels, dtype=object)

    for i in range(n_voxels):
        ct_vec = [float(v) for v in inputs.ct[:, i]]

        def residual(params_vec):
            pred = model_patlak_cfit(params_vec[0], params_vec[1], cp_vec, t_vec)
            return [pred[j] - ct_vec[j] for j in range(len(ct_vec))]

        x0 = [
            min(max(float(initial_parameters[i, 0]), lb[0]), ub[0]),
            min(max(float(initial_parameters[i, 1]), lb[1]), ub[1]),
        ]
        fit = least_squares(residual, x0=x0, bounds=(lb, ub), **lsq_kwargs)
        params[i, 0] = float(fit.x[0])
        params[i, 1] = float(fit.x[1])
        chi[i] = float(sum(v * v for v in fit.fun))
        extra[i] = fit

    return params, success, chi, extra


def _run_patlak_accelerated(
    backend: str, inputs: FitInputs, initial_parameters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from dce_pipeline import _load_fit_module_for_acceleration  # local: breaks an import cycle

    fit_module = _load_fit_module_for_acceleration(backend)
    n_fits = inputs.n_voxels

    data = np.ascontiguousarray(np.asarray(inputs.ct.T, dtype=np.float32))
    timer_f32 = np.ascontiguousarray(np.asarray(inputs.timer, dtype=np.float32).reshape(-1))
    cp_f32 = np.ascontiguousarray(np.asarray(inputs.cp, dtype=np.float32).reshape(-1))
    user_info = np.ascontiguousarray(np.concatenate([timer_f32, cp_f32], axis=0), dtype=np.float32)

    try:
        model_id = int(fit_module.ModelID.PATLAK)
    except AttributeError as exc:
        raise RuntimeError("Acceleration backend does not expose ModelID.PATLAK") from exc

    n_params = 2
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
    if model_name != "patlak":
        raise NotImplementedError(f"run_backend_fit: model '{model_name}' is not yet migrated")
    if backend == "python":
        return _run_patlak_python(inputs, initial_parameters)
    return _run_patlak_accelerated(backend, inputs, initial_parameters)


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
