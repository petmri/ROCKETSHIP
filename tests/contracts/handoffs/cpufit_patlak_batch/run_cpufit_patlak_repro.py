#!/usr/bin/env python3
"""Reproduce cpufit PATLAK drift observed in batch parity diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import _fit_stage_d_model  # noqa: E402


def _args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run cpufit PATLAK repro payloads.")
    p.add_argument(
        "--payload",
        action="append",
        type=Path,
        default=[],
        help="Payload .npz path. Can be provided multiple times.",
    )
    p.add_argument("--json", type=Path, default=None, help="Optional output JSON path.")
    p.add_argument(
        "--include-control",
        action="store_true",
        help="Include ses01 control payload when --payload is not supplied.",
    )
    p.add_argument(
        "--include-drift",
        action="store_true",
        help="Include ses02 drift payload when --payload is not supplied.",
    )
    p.add_argument("--sample-size", type=int, default=0, help="Optional cap on curve count per payload.")

    args = p.parse_args()
    if not args.payload:
        include_control = args.include_control or not args.include_drift
        include_drift = args.include_drift or not args.include_control
        if include_control:
            args.payload.append(here / "ses01_control_repro.npz")
        if include_drift:
            args.payload.append(here / "ses02_drift_repro.npz")
    return args


def _prefs(payload: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "lower_limit_ktrans": float(payload["prefs_lower_limit_ktrans"][0]),
        "upper_limit_ktrans": float(payload["prefs_upper_limit_ktrans"][0]),
        "initial_value_ktrans": float(payload["prefs_initial_value_ktrans"][0]),
        "lower_limit_vp": float(payload["prefs_lower_limit_vp"][0]),
        "upper_limit_vp": float(payload["prefs_upper_limit_vp"][0]),
        "initial_value_vp": float(payload["prefs_initial_value_vp"][0]),
        "tol_fun": float(payload["prefs_tol_fun"][0]),
        "tol_x": float(payload["prefs_tol_x"][0]),
        "max_nfev": int(payload["prefs_max_nfev"][0]),
        "gpu_tolerance": float(payload["prefs_gpu_tolerance"][0]),
        "gpu_max_n_iterations": int(payload["prefs_gpu_max_n_iterations"][0]),
        "max_iter": int(payload["prefs_max_nfev"][0]),
        "robust": "off",
    }


def _metrics(fit_values: np.ndarray, ref_values: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(fit_values) & np.isfinite(ref_values)
    x = np.asarray(fit_values[m], dtype=np.float64)
    y = np.asarray(ref_values[m], dtype=np.float64)
    if x.size < 2:
        return {"n": int(x.size), "corr": float("nan"), "slope": float("nan"), "ratio": float("nan"), "mae": float("nan")}
    a = np.vstack([y, np.ones_like(y)]).T
    slope, _ = np.linalg.lstsq(a, x, rcond=None)[0]
    corr = float(np.corrcoef(x, y)[0, 1])
    ratio = float(np.mean(x) / np.mean(y)) if abs(float(np.mean(y))) > 0 else float("nan")
    mae = float(np.mean(np.abs(x - y)))
    return {"n": int(x.size), "corr": corr, "slope": float(slope), "ratio": ratio, "mae": mae}


def _state_summary(states: np.ndarray) -> Dict[str, Any]:
    import pycpufit.cpufit as cf

    raw = cf.summarize_fit_states(states.astype(np.int32))
    return {
        "total_fits": int(raw.get("total_fits", 0)),
        "counts": {str(k): int(v) for k, v in raw.get("counts", {}).items()},
    }


def _run_payload(path: Path, sample_size: int) -> Dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    ct = np.asarray(payload["ct"], dtype=np.float64)
    cp = np.asarray(payload["cp_use"], dtype=np.float64).reshape(-1)
    timer = np.asarray(payload["timer_min"], dtype=np.float64).reshape(-1)
    matlab_k = np.asarray(payload["matlab_ktrans"], dtype=np.float64).reshape(-1)
    prefs = _prefs(payload)

    if sample_size > 0 and sample_size < ct.shape[1]:
        ct = ct[:, :sample_size]
        matlab_k = matlab_k[:sample_size]

    cpu = _fit_stage_d_model(
        model_name="patlak",
        ct=ct,
        cp_use=cp,
        timer=timer,
        prefs=prefs,
        r1o=None,
        relaxivity=3.4,
        fw=0.8,
        stlv_use=None,
        sttum=None,
        start_injection_min=float(timer[0]),
        sss=None,
        ssstum=None,
        acceleration_backend="none",
    )
    cpuf = _fit_stage_d_model(
        model_name="patlak",
        ct=ct,
        cp_use=cp,
        timer=timer,
        prefs=prefs,
        r1o=None,
        relaxivity=3.4,
        fw=0.8,
        stlv_use=None,
        sttum=None,
        start_injection_min=float(timer[0]),
        sss=None,
        ssstum=None,
        acceleration_backend="cpufit_cpu",
    )

    # Direct cpufit call for fit-state diagnostics.
    import pycpufit.cpufit as cf

    data = np.ascontiguousarray(ct.T.astype(np.float32))
    user_info = np.ascontiguousarray(np.concatenate([timer.astype(np.float32), cp.astype(np.float32)]), dtype=np.float32)
    init = np.ascontiguousarray(
        np.tile(
            np.array([[float(prefs["initial_value_ktrans"]), float(prefs["initial_value_vp"])]], dtype=np.float32),
            (data.shape[0], 1),
        )
    )
    constraints = np.ascontiguousarray(
        np.tile(
            np.array(
                [[
                    float(prefs["lower_limit_ktrans"]),
                    float(prefs["upper_limit_ktrans"]),
                    float(prefs["lower_limit_vp"]),
                    float(prefs["upper_limit_vp"]),
                ]],
                dtype=np.float32,
            ),
            (data.shape[0], 1),
        )
    )
    c_types = np.ascontiguousarray(np.full((2,), int(cf.ConstraintType.LOWER_UPPER), dtype=np.int32))
    _p, states, _chi, iterations, _elapsed = cf.fit_constrained(
        data=data,
        weights=None,
        model_id=int(cf.ModelID.PATLAK),
        initial_parameters=init,
        constraints=constraints,
        constraint_types=c_types,
        tolerance=float(prefs["gpu_tolerance"]),
        max_number_iterations=int(prefs["gpu_max_n_iterations"]),
        parameters_to_fit=None,
        estimator_id=int(cf.EstimatorID.LSE),
        user_info=user_info,
    )

    out = {
        "payload": str(path),
        "session": str(payload["session"][0]),
        "subject": str(payload["subject"][0]),
        "n_curves": int(ct.shape[1]),
        "cpu_vs_matlab": _metrics(np.asarray(cpu[:, 0], dtype=np.float64), matlab_k),
        "cpufit_vs_matlab": _metrics(np.asarray(cpuf[:, 0], dtype=np.float64), matlab_k),
        "cpufit_vs_cpu": _metrics(np.asarray(cpuf[:, 0], dtype=np.float64), np.asarray(cpu[:, 0], dtype=np.float64)),
        "cpufit_states": _state_summary(np.asarray(states, dtype=np.int32)),
        "cpufit_iterations": {
            "mean": float(np.mean(iterations)),
            "p95": float(np.quantile(iterations, 0.95)),
            "max": float(np.max(iterations)),
        },
    }
    return out


def _print_row(label: str, m: Dict[str, float]) -> None:
    print(
        f"  {label:<18} n={int(m['n']):>5d} "
        f"corr={m['corr']:>9.6f} slope={m['slope']:>9.6f} "
        f"ratio={m['ratio']:>9.6f} mae={m['mae']:>11.6g}"
    )


def main() -> int:
    args = _args()
    results: List[Dict[str, Any]] = []
    for payload in args.payload:
        payload = payload.expanduser().resolve()
        if not payload.exists():
            raise FileNotFoundError(f"Missing payload: {payload}")
        res = _run_payload(payload, args.sample_size)
        results.append(res)

        print(f"\n[{res['subject']} {res['session']}] {payload.name}")
        _print_row("CPU vs MATLAB", res["cpu_vs_matlab"])
        _print_row("CPUfit vs MATLAB", res["cpufit_vs_matlab"])
        _print_row("CPUfit vs CPU", res["cpufit_vs_cpu"])
        print(f"  cpufit states      {res['cpufit_states']['counts']}")
        print(
            "  cpufit iterations "
            f"mean={res['cpufit_iterations']['mean']:.2f} "
            f"p95={res['cpufit_iterations']['p95']:.2f} "
            f"max={res['cpufit_iterations']['max']:.0f}"
        )

    if args.json:
        out = args.json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
