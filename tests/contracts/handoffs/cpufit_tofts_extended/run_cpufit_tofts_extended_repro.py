#!/usr/bin/env python3
"""Reproduce pycpufit TOFTS_EXTENDED behavior on handoff payloads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BIDS_PAYLOAD = THIS_DIR / "bids_short_timer_repro.npz"
DEFAULT_OSIPI_PAYLOAD = THIS_DIR / "osipi_control_repro.npz"


def _find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "python" / "dce_models.py").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing python/dce_models.py")


REPO_ROOT = _find_repo_root(THIS_DIR)


def _load_payload(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        timer = np.asarray(payload["timer_min"], dtype=np.float64).reshape(-1)
        cp = np.asarray(payload["cp_mM"], dtype=np.float64).reshape(-1)
        ct = np.asarray(payload["ct_mM"], dtype=np.float64).reshape(-1)
        label = str(payload.get("label", path.stem))
        source = str(payload.get("source", ""))
    if not (timer.size == cp.size == ct.size):
        raise ValueError(
            f"Payload series lengths differ for {path}: timer={timer.size}, cp={cp.size}, ct={ct.size}"
        )
    return {
        "label": label,
        "source": source,
        "timer_min": timer,
        "cp_mM": cp,
        "ct_mM": ct,
    }


def _fit_cpufit_extended_tofts(
    timer_min: np.ndarray,
    cp_mM: np.ndarray,
    ct_mM: np.ndarray,
    *,
    time_scale: float,
    tolerance: float,
    max_iterations: int,
) -> Dict[str, Any]:
    try:
        import pycpufit.cpufit as cf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"pycpufit unavailable: {exc}") from exc

    data = np.ascontiguousarray(ct_mM.reshape(1, -1).astype(np.float32))
    timer_scaled = np.ascontiguousarray((timer_min * float(time_scale)).astype(np.float32))
    cp_float = np.ascontiguousarray(cp_mM.astype(np.float32))
    user_info = np.ascontiguousarray(np.concatenate([timer_scaled, cp_float]), dtype=np.float32)

    initial = np.ascontiguousarray(np.array([[2e-4, 0.2, 0.02]], dtype=np.float32))
    constraints = np.ascontiguousarray(np.array([[1e-7, 2.0, 0.02, 1.0, 1e-3, 1.0]], dtype=np.float32))
    constraint_types = np.ascontiguousarray(
        np.full((3,), int(cf.ConstraintType.LOWER_UPPER), dtype=np.int32)
    )

    parameters, states, chi_squares, n_iterations, elapsed_sec = cf.fit_constrained(
        data=data,
        weights=None,
        model_id=int(cf.ModelID.TOFTS_EXTENDED),
        initial_parameters=initial,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=float(tolerance),
        max_number_iterations=int(max_iterations),
        parameters_to_fit=None,
        estimator_id=int(cf.EstimatorID.LSE),
        user_info=user_info,
    )

    params = np.asarray(parameters, dtype=np.float64).reshape(1, -1)[0]
    state = int(np.asarray(states, dtype=np.int32).reshape(-1)[0])
    n_iter = int(np.asarray(n_iterations, dtype=np.int32).reshape(-1)[0])
    chi = float(np.asarray(chi_squares, dtype=np.float64).reshape(-1)[0])
    return {
        "state": state,
        "iterations": n_iter,
        "chi_square": chi,
        "params": {
            "ktrans_per_min": float(params[0]),
            "ve": float(params[1]),
            "vp": float(params[2]),
        },
        "params_all_finite": bool(np.all(np.isfinite(params))),
        "elapsed_sec": float(elapsed_sec),
    }


def _fit_cpu_extended_tofts(timer_min: np.ndarray, cp_mM: np.ndarray, ct_mM: np.ndarray) -> Dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from dce_models import model_extended_tofts_fit  # noqa: E402

    fit = np.asarray(
        model_extended_tofts_fit(
            ct=ct_mM.tolist(),
            cp=cp_mM.tolist(),
            timer=timer_min.tolist(),
        ),
        dtype=np.float64,
    ).reshape(-1)
    core = fit[:3]
    return {
        "params": {
            "ktrans_per_min": float(core[0]),
            "ve": float(core[1]),
            "vp": float(core[2]),
        },
        "sse": float(fit[3]) if fit.size > 3 else float("nan"),
        "params_all_finite": bool(np.all(np.isfinite(core))),
    }


def _run_case(
    path: Path,
    *,
    time_scale: float,
    tolerance: float,
    max_iterations: int,
) -> Dict[str, Any]:
    payload = _load_payload(path)
    timer_min = payload["timer_min"]
    cp_mM = payload["cp_mM"]
    ct_mM = payload["ct_mM"]

    cpufit = _fit_cpufit_extended_tofts(
        timer_min=timer_min,
        cp_mM=cp_mM,
        ct_mM=ct_mM,
        time_scale=time_scale,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    cpu = _fit_cpu_extended_tofts(timer_min=timer_min, cp_mM=cp_mM, ct_mM=ct_mM)
    return {
        "payload_path": str(path),
        "label": payload["label"],
        "source": payload["source"],
        "timer_min_span": float(timer_min[-1] - timer_min[0]) if timer_min.size > 0 else 0.0,
        "timer_min_n": int(timer_min.size),
        "cp_min": float(np.min(cp_mM)),
        "cp_max": float(np.max(cp_mM)),
        "ct_min": float(np.min(ct_mM)),
        "ct_max": float(np.max(ct_mM)),
        "cpufit": cpufit,
        "cpu_reference": cpu,
    }


def _print_human(result: Dict[str, Any]) -> None:
    print(f"payload: {result['payload_path']}")
    print(f"label: {result['label']}")
    print(f"source: {result['source']}")
    print(
        f"samples={result['timer_min_n']} timer_span_min={result['timer_min_span']:.9f} "
        f"cp_range=[{result['cp_min']:.6g},{result['cp_max']:.6g}] "
        f"ct_range=[{result['ct_min']:.6g},{result['ct_max']:.6g}]"
    )
    cfit = result["cpufit"]
    cpu = result["cpu_reference"]
    print(
        "cpufit:"
        f" state={cfit['state']} iterations={cfit['iterations']} "
        f"params_finite={cfit['params_all_finite']} chi={cfit['chi_square']:.6g} "
        f"params={cfit['params']}"
    )
    print(
        "cpu_ref:"
        f" params_finite={cpu['params_all_finite']} sse={cpu['sse']:.6g} "
        f"params={cpu['params']}"
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=DEFAULT_BIDS_PAYLOAD)
    parser.add_argument("--control-payload", type=Path, default=DEFAULT_OSIPI_PAYLOAD)
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    payloads = [Path(args.payload).expanduser().resolve()]
    if not bool(args.skip_control):
        payloads.append(Path(args.control_payload).expanduser().resolve())

    try:
        results = [
            _run_case(
                path=path,
                time_scale=float(args.time_scale),
                tolerance=float(args.tolerance),
                max_iterations=int(args.max_iterations),
            )
            for path in payloads
        ]
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output = {
        "time_scale": float(args.time_scale),
        "tolerance": float(args.tolerance),
        "max_iterations": int(args.max_iterations),
        "results": results,
    }
    if bool(args.json):
        print(json.dumps(output, indent=2))
    else:
        for idx, result in enumerate(results):
            if idx > 0:
                print("")
            _print_human(result)

    # Non-zero exit if any case returns non-finite cpu params or cpufit execution error state.
    for result in results:
        cpufit_state = int(result["cpufit"]["state"])
        cpu_ok = bool(result["cpu_reference"]["params_all_finite"])
        if not cpu_ok:
            return 3
        if cpufit_state < 0:
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
