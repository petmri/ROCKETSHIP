"""Stage-D diagnostics for batch MATLAB-vs-Python parity investigation.

This runner isolates:
1) Backend contribution on identical Stage-B arrays (CPU vs CPUfit).
2) MATLAB-vs-Python single-curve/ROI contract agreement for Patlak.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List

import nibabel as nib
import numpy as np
from scipy.io import loadmat, savemat


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_models import model_patlak_fit, model_patlak_linear  # noqa: E402
from dce_pipeline import DcePipelineConfig, _fit_stage_d_model, _stage_d_fit_prefs  # noqa: E402


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage-D batch parity diagnostics.")
    p.add_argument(
        "--python-cleanref-root",
        type=Path,
        default=REPO_ROOT / "RUNNER_DATA/derivatives/dceprep-python-batch-cleanref",
        help="Python clean-reference derivative root.",
    )
    p.add_argument(
        "--matlab-cleanref-root",
        type=Path,
        default=REPO_ROOT / "RUNNER_DATA/derivatives/dceprep-matlab-cleanref",
        help="MATLAB clean-reference derivative root.",
    )
    p.add_argument("--subject", default="sub-1101743")
    p.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02"])
    p.add_argument("--sample-size", type=int, default=2000, help="Sample voxel count for backend isolation.")
    p.add_argument(
        "--contract-size",
        type=int,
        default=128,
        help="Sample curve count for MATLAB single-curve contract check.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-matlab-contract",
        action="store_true",
        help="Skip MATLAB single-curve/ROI contract checks.",
    )
    p.add_argument("--matlab-bin", default="/opt/homebrew/bin/matlab")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: out/batch_stage_d_diagnostics_YYYYMMDD.json).",
    )
    return p.parse_args()


def _session_prefix(subject: str, session: str) -> str:
    return f"{subject}_{session}"


def _provenance_path(py_root: Path, subject: str, session: str) -> Path:
    prefix = _session_prefix(subject, session)
    return py_root / subject / session / "reports" / f"{prefix}_desc-provenance.json"


def _matlab_ktrans_map(mat_root: Path, subject: str, session: str) -> Path:
    dce_dir = mat_root / subject / session / "dce"
    for candidate in (
        dce_dir / "dce_patlak_fit_Ktrans.nii",
        dce_dir / "dce_patlak_fit_Ktrans.nii.gz",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing canonical MATLAB patlak Ktrans map under {dce_dir}")


def _regression_metrics(fit_values: np.ndarray, ref_values: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(fit_values) & np.isfinite(ref_values)
    x = np.asarray(fit_values[mask], dtype=np.float64)
    y = np.asarray(ref_values[mask], dtype=np.float64)
    if x.size < 2:
        return {"n": int(x.size), "corr": float("nan"), "slope": float("nan"), "mean_ratio": float("nan"), "mae": float("nan")}

    a = np.vstack([y, np.ones_like(y)]).T
    slope, _intercept = np.linalg.lstsq(a, x, rcond=None)[0]
    corr = np.corrcoef(x, y)[0, 1]
    mean_ref = float(np.mean(y))
    return {
        "n": int(x.size),
        "corr": float(corr),
        "slope": float(slope),
        "mean_ratio": float(np.mean(x) / mean_ref) if abs(mean_ref) > 0.0 else float("nan"),
        "mae": float(np.mean(np.abs(x - y))),
    }


def _fit_patlak_subset(
    ct_subset: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Dict[str, Any],
    acceleration_backend: str,
) -> np.ndarray:
    fit = _fit_stage_d_model(
        model_name="patlak",
        ct=ct_subset,
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
        acceleration_backend=acceleration_backend,
    )
    return np.asarray(fit, dtype=np.float64)


def _run_matlab_patlak_contract(
    *,
    matlab_bin: str,
    curves: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Dict[str, Any],
    init_k: np.ndarray,
    init_vp: np.ndarray,
) -> np.ndarray:
    if not Path(matlab_bin).exists():
        raise FileNotFoundError(f"MATLAB binary not found: {matlab_bin}")

    prefs_mat = {
        "lower_limit_ktrans": float(prefs["lower_limit_ktrans"]),
        "upper_limit_ktrans": float(prefs["upper_limit_ktrans"]),
        "initial_value_ktrans": float(prefs["initial_value_ktrans"]),
        "lower_limit_vp": float(prefs["lower_limit_vp"]),
        "upper_limit_vp": float(prefs["upper_limit_vp"]),
        "initial_value_vp": float(prefs["initial_value_vp"]),
        "TolFun": float(prefs["tol_fun"]),
        "TolX": float(prefs["tol_x"]),
        "MaxIter": int(prefs["max_iter"]),
        "MaxFunEvals": int(prefs["max_nfev"]),
        "Robust": str(prefs["robust"]),
    }

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        in_mat = tmp / "contract_in.mat"
        out_mat = tmp / "contract_out.mat"
        script = tmp / "run_contract.m"

        savemat(
            in_mat,
            {
                "Ct": np.asarray(curves, dtype=np.float64),
                "Cp_row": np.asarray(cp, dtype=np.float64).reshape(1, -1),
                "timer_col": np.asarray(timer, dtype=np.float64).reshape(-1, 1),
                "init_k": np.asarray(init_k, dtype=np.float64).reshape(-1, 1),
                "init_vp": np.asarray(init_vp, dtype=np.float64).reshape(-1, 1),
                "prefs_base": prefs_mat,
            },
        )

        script.write_text(
            "\n".join(
                [
                    f"load('{in_mat.as_posix()}');",
                    f"addpath('{REPO_ROOT.as_posix()}');",
                    f"addpath('{(REPO_ROOT / 'dce').as_posix()}');",
                    "n = size(Ct,2);",
                    "out_nl = zeros(n,7);",
                    "prefs = prefs_base;",
                    "for i=1:n",
                    "  prefs.initial_value_ktrans = init_k(i,1);",
                    "  prefs.initial_value_vp = init_vp(i,1);",
                    "  [fit,~] = model_patlak(Ct(:,i), Cp_row, timer_col, prefs);",
                    "  out_nl(i,:) = fit;",
                    "end",
                    f"save('{out_mat.as_posix()}','out_nl');",
                ]
            ),
            encoding="utf-8",
        )

        proc = subprocess.run(
            [matlab_bin, "-batch", f"run('{script.as_posix()}')"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            msg = "\n".join(
                [
                    "MATLAB contract execution failed.",
                    f"return_code={proc.returncode}",
                    f"stdout_tail={proc.stdout[-2000:]}",
                    f"stderr_tail={proc.stderr[-2000:]}",
                ]
            )
            raise RuntimeError(msg)

        out = loadmat(out_mat)
        return np.asarray(out["out_nl"], dtype=np.float64)


def _print_metric_row(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label:<24} n={int(metrics.get('n', 0)):>6d} "
        f"corr={metrics.get('corr', float('nan')):>10.6f} "
        f"slope={metrics.get('slope', float('nan')):>10.6f} "
        f"ratio={metrics.get('mean_ratio', float('nan')):>10.6f} "
        f"mae={metrics.get('mae', float('nan')):>12.6g}"
    )


def _run_for_session(args: argparse.Namespace, session: str, rng: np.random.Generator) -> Dict[str, Any]:
    prov_path = _provenance_path(args.python_cleanref_root, args.subject, session)
    if not prov_path.exists():
        raise FileNotFoundError(f"Missing provenance file: {prov_path}")

    cfg_dict = json.loads(prov_path.read_text(encoding="utf-8"))["config"]
    cfg = DcePipelineConfig.from_dict(cfg_dict)
    prefs = _stage_d_fit_prefs(cfg)

    checkpoint_dir = Path(cfg_dict["checkpoint_dir"])
    b_arrays = np.load(checkpoint_dir / "b_out_arrays.npz", allow_pickle=True)
    ct = np.asarray(b_arrays["Ct"], dtype=np.float64)
    cp = np.asarray(b_arrays["Cp_use"], dtype=np.float64).reshape(-1)
    timer = np.asarray(b_arrays["timer"], dtype=np.float64).reshape(-1)
    tumind = np.asarray(b_arrays["tumind"], dtype=np.int64).reshape(-1)

    sample_n = min(int(args.sample_size), int(ct.shape[1]))
    sample_idx = np.sort(rng.choice(ct.shape[1], size=sample_n, replace=False))
    ct_sample = ct[:, sample_idx]

    mat_map_path = _matlab_ktrans_map(args.matlab_cleanref_root, args.subject, session)
    mat_map_flat = np.asarray(nib.load(str(mat_map_path)).get_fdata(), dtype=np.float64).reshape(-1, order="F")
    mat_sample = mat_map_flat[tumind[sample_idx]]

    cpu_fit = _fit_patlak_subset(ct_sample, cp, timer, prefs, "none")
    cpu_vs_mat = _regression_metrics(cpu_fit[:, 0], mat_sample)

    cpufit_metrics: Dict[str, float] | None = None
    cpufit_vs_cpu: Dict[str, float] | None = None
    cpufit_available = True
    cpufit_error = None
    try:
        cpufit_fit = _fit_patlak_subset(ct_sample, cp, timer, prefs, "cpufit_cpu")
        cpufit_metrics = _regression_metrics(cpufit_fit[:, 0], mat_sample)
        cpufit_vs_cpu = _regression_metrics(cpufit_fit[:, 0], cpu_fit[:, 0])
    except Exception as exc:
        cpufit_available = False
        cpufit_error = str(exc)
        cpufit_fit = None

    print(f"\n[{args.subject} {session}] Stage-D backend isolation")
    _print_metric_row("CPU vs MATLAB", cpu_vs_mat)
    if cpufit_available and cpufit_metrics is not None and cpufit_vs_cpu is not None:
        _print_metric_row("CPUfit vs MATLAB", cpufit_metrics)
        _print_metric_row("CPUfit vs CPU", cpufit_vs_cpu)
    else:
        print(f"{'CPUfit':<24} unavailable ({cpufit_error})")

    out: Dict[str, Any] = {
        "session": session,
        "subject": args.subject,
        "matlab_ktrans_map": str(mat_map_path),
        "sample_size": int(sample_n),
        "backend_isolation": {
            "cpu_vs_matlab": cpu_vs_mat,
            "cpufit_available": cpufit_available,
            "cpufit_error": cpufit_error,
            "cpufit_vs_matlab": cpufit_metrics,
            "cpufit_vs_cpu": cpufit_vs_cpu,
        },
    }

    if args.skip_matlab_contract:
        return out

    contract_n = min(int(args.contract_size), int(ct.shape[1]))
    contract_idx = np.sort(rng.choice(ct.shape[1], size=contract_n, replace=False))
    contract_curves = ct[:, contract_idx]
    init_contract = np.asarray(
        [model_patlak_linear(contract_curves[:, i], cp, timer) for i in range(contract_curves.shape[1])],
        dtype=np.float64,
    )
    py_contract = np.asarray(
        [model_patlak_fit(contract_curves[:, i], cp, timer, prefs) for i in range(contract_curves.shape[1])],
        dtype=np.float64,
    )
    mat_contract = _run_matlab_patlak_contract(
        matlab_bin=args.matlab_bin,
        curves=contract_curves,
        cp=cp,
        timer=timer,
        prefs=prefs,
        init_k=init_contract[:, 0],
        init_vp=init_contract[:, 1],
    )

    contract_metrics = {
        "ktrans": _regression_metrics(py_contract[:, 0], mat_contract[:, 0]),
        "vp": _regression_metrics(py_contract[:, 1], mat_contract[:, 1]),
        "sse": _regression_metrics(py_contract[:, 2], mat_contract[:, 2]),
    }

    # ROI-like contract curves (whole mask + random subsets)
    roi_curve_all = np.mean(ct, axis=1)
    subset_sizes = [max(100, int(ct.shape[1] * 0.10)), max(100, int(ct.shape[1] * 0.01))]
    roi_curves: List[np.ndarray] = [roi_curve_all]
    roi_labels = ["roi_all", "roi_10pct", "roi_1pct"]
    for size in subset_sizes:
        pick = np.sort(rng.choice(ct.shape[1], size=min(size, ct.shape[1]), replace=False))
        roi_curves.append(np.mean(ct[:, pick], axis=1))
    roi_matrix = np.stack(roi_curves, axis=1)
    roi_init = np.asarray([model_patlak_linear(roi_matrix[:, i], cp, timer) for i in range(roi_matrix.shape[1])], dtype=np.float64)
    py_roi = np.asarray([model_patlak_fit(roi_matrix[:, i], cp, timer, prefs) for i in range(roi_matrix.shape[1])], dtype=np.float64)
    mat_roi = _run_matlab_patlak_contract(
        matlab_bin=args.matlab_bin,
        curves=roi_matrix,
        cp=cp,
        timer=timer,
        prefs=prefs,
        init_k=roi_init[:, 0],
        init_vp=roi_init[:, 1],
    )

    roi_absdiff = np.abs(py_roi[:, :3] - mat_roi[:, :3])
    roi_contract = {
        roi_labels[i]: {
            "matlab": [float(v) for v in mat_roi[i, :3]],
            "python": [float(v) for v in py_roi[i, :3]],
            "abs_diff": [float(v) for v in roi_absdiff[i, :3]],
        }
        for i in range(len(roi_labels))
    }

    print(f"[{args.subject} {session}] MATLAB contract (single-curve sample)")
    _print_metric_row("Python vs MATLAB Ktrans", contract_metrics["ktrans"])
    _print_metric_row("Python vs MATLAB vp", contract_metrics["vp"])
    _print_metric_row("Python vs MATLAB sse", contract_metrics["sse"])

    out["matlab_contract"] = {
        "contract_size": int(contract_n),
        "single_curve_metrics": contract_metrics,
        "roi_examples": roi_contract,
    }
    return out


def main() -> int:
    args = _args()
    rng = np.random.default_rng(int(args.seed))

    results: Dict[str, Any] = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "subject": args.subject,
        "sessions": args.sessions,
        "python_cleanref_root": str(args.python_cleanref_root),
        "matlab_cleanref_root": str(args.matlab_cleanref_root),
        "sample_size": int(args.sample_size),
        "contract_size": int(args.contract_size),
        "skip_matlab_contract": bool(args.skip_matlab_contract),
        "by_session": [],
    }

    for session in args.sessions:
        results["by_session"].append(_run_for_session(args, session, rng))

    out_path = args.output_json
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d")
        out_path = REPO_ROOT / "out" / f"batch_stage_d_diagnostics_{stamp}.json"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote diagnostics: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
