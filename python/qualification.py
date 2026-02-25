"""Dataset-level Python workflow qualification utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from bids_discovery import BidsSession, discover_bids_sessions
from dce_pipeline import DcePipelineConfig, run_dce_pipeline
from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline


DEFAULT_DCE_MODELS = ("tofts", "ex_tofts", "patlak")
PRIMARY_MODEL_PARAMS = {
    "tofts": ("Ktrans", "ve"),
    "ex_tofts": ("Ktrans", "ve", "vp"),
    "patlak": ("Ktrans", "vp"),
}
PRIMARY_MODEL_MIN_NONZERO_FRACTION = 1e-4
PRIMARY_MODEL_MIN_FINITE_NONZERO_RATIO = 0.90


@dataclass
class QualificationRunConfig:
    bids_root: Path
    output_root: Path
    backend: str = "auto"
    dce_models: Iterable[str] = DEFAULT_DCE_MODELS
    run_t1: bool = True
    run_dce: bool = True
    write_postfit_arrays: bool = True


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_one(parent: Path, pattern: str) -> Optional[Path]:
    matches = sorted(parent.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _load_array(path: Path) -> np.ndarray:
    suffixes = tuple(path.suffixes)
    if suffixes and suffixes[-1] == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for NIfTI map qualification") from exc
    image = nib.load(str(path))
    return np.asarray(image.get_fdata(), dtype=np.float64)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _map_sanity_stats(path: Path) -> Dict[str, Any]:
    arr = _load_array(path)
    finite = np.isfinite(arr)
    nonzero = arr != 0.0
    finite_nonzero = finite & nonzero
    out = {
        "path": str(path),
        "shape": list(arr.shape),
        "finite_fraction": float(np.mean(finite)),
        "nonzero_fraction": float(np.mean(nonzero)),
        "finite_nonzero_fraction": float(np.mean(finite_nonzero)),
    }
    if np.any(finite_nonzero):
        vals = arr[finite_nonzero]
        out["mean_nonzero"] = float(np.mean(vals))
        out["p95_nonzero"] = float(np.percentile(vals, 95.0))
    else:
        out["mean_nonzero"] = float("nan")
        out["p95_nonzero"] = float("nan")
    return out


def _finite_nonzero_ratio(stats: Dict[str, Any]) -> float:
    nonzero_fraction = _safe_float(stats.get("nonzero_fraction"), default=0.0)
    finite_nonzero_fraction = _safe_float(stats.get("finite_nonzero_fraction"), default=0.0)
    if nonzero_fraction <= 0.0:
        return 1.0
    ratio = finite_nonzero_fraction / nonzero_fraction
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return float(ratio)


def _primary_model_map_blockers(map_stats: Dict[str, Dict[str, Any]], requested_models: Iterable[str]) -> List[str]:
    blockers: List[str] = []
    selected = [str(name).strip().lower() for name in requested_models]
    for model_name in selected:
        expected_params = PRIMARY_MODEL_PARAMS.get(model_name)
        if expected_params is None:
            continue
        for param_name in expected_params:
            stat_key = f"{model_name}:{param_name}"
            stats = map_stats.get(stat_key)
            if stats is None:
                blockers.append(f"Missing primary-model map output for {stat_key}")
                continue

            nonzero_fraction = _safe_float(stats.get("nonzero_fraction"), default=0.0)
            if nonzero_fraction <= PRIMARY_MODEL_MIN_NONZERO_FRACTION:
                continue

            finite_nonzero_ratio = _finite_nonzero_ratio(stats)
            if finite_nonzero_ratio < PRIMARY_MODEL_MIN_FINITE_NONZERO_RATIO:
                blockers.append(
                    "Primary-model map has excessive non-finite fitted voxels for "
                    f"{stat_key}: finite_nonzero_ratio={finite_nonzero_ratio:.3f} "
                    f"(threshold={PRIMARY_MODEL_MIN_FINITE_NONZERO_RATIO:.3f}, "
                    f"nonzero_fraction={nonzero_fraction:.3f})"
                )
    return blockers


def _parse_vfa_metadata(raw_anat_dir: Path) -> tuple[List[float], Optional[float], List[Path]]:
    json_files = sorted(raw_anat_dir.glob("*_flip-*_VFA.json"))
    if not json_files:
        return [], None, []

    rows: List[tuple[float, Optional[float], Path]] = []
    for path in json_files:
        payload = _read_json(path)
        if "FlipAngle" not in payload:
            continue
        fa = float(payload["FlipAngle"])
        tr_ms: Optional[float] = None
        if "RepetitionTime" in payload:
            tr_ms = float(payload["RepetitionTime"]) * 1000.0
        rows.append((fa, tr_ms, path))
    rows.sort(key=lambda row: row[0])

    flip_angles = [float(row[0]) for row in rows]
    tr_values = [row[1] for row in rows if row[1] is not None]
    tr_ms = float(tr_values[0]) if tr_values else None
    return flip_angles, tr_ms, [row[2] for row in rows]


def _infer_vfa_frame_count(path: Path) -> int:
    arr = _load_array(path)
    if arr.ndim == 4:
        return int(arr.shape[-1])
    if arr.ndim == 3:
        return int(arr.shape[-1])
    if arr.ndim == 2:
        return 1
    raise ValueError(f"Unsupported VFA rank for frame counting: shape={arr.shape}")


def _session_t1_inputs(session: BidsSession) -> Dict[str, Any]:
    anat_deriv = session.derivatives_path / "anat"
    anat_raw = session.rawdata_path / "anat"
    vfa_deriv = _find_one(anat_deriv, "*desc-bfczunified_VFA.nii*")
    if vfa_deriv is None:
        raise FileNotFoundError(f"Missing derivative VFA file under {anat_deriv}")
    flip_angles, tr_ms, sidecars = _parse_vfa_metadata(anat_raw)
    return {
        "vfa_deriv": vfa_deriv,
        "flip_angles": flip_angles,
        "tr_ms": tr_ms,
        "raw_vfa_jsons": sidecars,
    }


def _session_dce_inputs(session: BidsSession) -> Dict[str, Path]:
    dce_deriv = session.derivatives_path / "dce"
    anat_deriv = session.derivatives_path / "anat"

    dynamic = _find_one(dce_deriv, "*desc-bfcz_DCE.nii*")
    if dynamic is None:
        dynamic = _find_one(dce_deriv, "*DCE.nii*")
    aif = _find_one(dce_deriv, "*desc-AIF_T1map.nii*")
    roi = _find_one(anat_deriv, "*desc-brain_mask.nii*")
    t1map = _find_one(anat_deriv, "*space-DCEref_T1map.nii*")

    missing = []
    if dynamic is None:
        missing.append("dynamic")
    if aif is None:
        missing.append("aif (*desc-AIF_T1map.nii*)")
    if roi is None:
        missing.append("roi")
    if t1map is None:
        missing.append("t1map")
    if missing:
        raise FileNotFoundError(f"Missing DCE derivative inputs ({', '.join(missing)}) for {session.id}")

    return {
        "dynamic": dynamic,
        "aif": aif,
        "roi": roi,
        "t1map": t1map,
    }


def _default_model_flags(selected_models: Iterable[str]) -> Dict[str, int]:
    keys = ("tofts", "ex_tofts", "patlak", "tissue_uptake", "two_cxm", "fxr", "auc", "nested", "FXL_rr")
    selected = {str(m).strip().lower() for m in selected_models}
    return {key: int(key in selected) for key in keys}


def _run_t1_for_session(session: BidsSession, output_dir: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    blockers: List[str] = []
    warnings: List[str] = []

    try:
        inputs = _session_t1_inputs(session)
        frame_count = _infer_vfa_frame_count(Path(inputs["vfa_deriv"]))
        flip_angles = list(inputs["flip_angles"])
        if flip_angles and len(flip_angles) != frame_count:
            if len(flip_angles) > frame_count:
                warnings.append(
                    f"Trimmed VFA flip-angle metadata from {len(flip_angles)} to {frame_count} to match derivative frames"
                )
                flip_angles = flip_angles[:frame_count]
            else:
                blockers.append(
                    f"VFA flip-angle metadata length {len(flip_angles)} is less than derivative frame count {frame_count}"
                )

        payload = {
            "output_dir": str(output_dir),
            "vfa_files": [str(inputs["vfa_deriv"])],
            "fit_type": "t1_fa_fit",
            "output_basename": "T1_map",
            "output_label": session.id,
            "rsquared_threshold": 0.6,
            "write_r_squared": True,
            "write_rho_map": True,
        }
        if flip_angles:
            payload["flip_angles_deg"] = flip_angles
        if inputs["tr_ms"] is not None:
            payload["tr_ms"] = float(inputs["tr_ms"])
        if blockers:
            return {
                "status": "failed",
                "duration_sec": float(time.perf_counter() - started),
                "blockers": blockers,
                "warnings": warnings,
            }
        summary = run_parametric_t1_pipeline(ParametricT1Config.from_dict(payload))
    except Exception as exc:
        return {
            "status": "error",
            "duration_sec": float(time.perf_counter() - started),
            "blockers": [f"T1 run failed: {type(exc).__name__}: {exc}"],
            "warnings": [],
        }

    outputs = summary.get("outputs", {})
    t1_map_path = Path(str(outputs.get("t1_map_path")))
    rsq_map_path = Path(str(outputs.get("rsquared_map_path")))
    rho_map_path = Path(str(outputs.get("rho_map_path")))
    if not t1_map_path.exists():
        blockers.append(f"Missing T1 map output: {t1_map_path}")
    if not rsq_map_path.exists():
        blockers.append(f"Missing Rsquared output: {rsq_map_path}")
    if not rho_map_path.exists():
        blockers.append(f"Missing rho output: {rho_map_path}")

    t1_map = _load_array(t1_map_path)
    valid_t1 = np.isfinite(t1_map) & (t1_map > 0.0)
    valid_count = int(np.count_nonzero(valid_t1))
    if valid_count <= 0:
        blockers.append("No positive finite T1 voxels found")

    metrics = dict(summary.get("metrics", {}))
    valid_fits = int(metrics.get("valid_fits", 0))
    if valid_fits <= 0:
        blockers.append("T1 metrics reported zero valid_fits")
    r2_mean = _safe_float(metrics.get("r2_mean"))
    if not np.isfinite(r2_mean):
        warnings.append("T1 r2_mean is not finite")

    return {
        "status": "ok" if not blockers else "failed",
        "duration_sec": float(time.perf_counter() - started),
        "summary_path": str(summary.get("meta", {}).get("summary_path", "")),
        "blockers": blockers,
        "warnings": warnings,
        "metrics": {
            "valid_fits": valid_fits,
            "valid_positive_t1_voxels": valid_count,
            "r2_mean": r2_mean,
        },
        "outputs": outputs,
    }


def _run_dce_for_session(
    session: BidsSession, output_dir: Path, *, backend: str, models: Iterable[str], write_postfit_arrays: bool
) -> Dict[str, Any]:
    started = time.perf_counter()
    blockers: List[str] = []
    warnings: List[str] = []

    try:
        inputs = _session_dce_inputs(session)
        config = DcePipelineConfig.from_dict(
            {
                "subject_source_path": str(session.rawdata_path),
                "subject_tp_path": str(session.derivatives_path),
                "output_dir": str(output_dir),
                "checkpoint_dir": str(output_dir / "checkpoints"),
                "backend": str(backend),
                "write_xls": True,
                "aif_mode": "auto",
                "dynamic_files": [str(inputs["dynamic"])],
                "aif_files": [str(inputs["aif"])],
                "roi_files": [str(inputs["roi"])],
                "t1map_files": [str(inputs["t1map"])],
                # Use ROI as a fallback noise mask for qualification robustness on synthetic fixtures
                # that have near-zero background variance in the default corner patch.
                "noise_files": [str(inputs["roi"])],
                "drift_files": [],
                "model_flags": _default_model_flags(models),
                "stage_overrides": {
                    "stage_a_mode": "real",
                    "stage_b_mode": "real",
                    "stage_d_mode": "real",
                    "aif_curve_mode": "raw",
                    "write_param_maps": True,
                    "write_postfit_arrays": bool(write_postfit_arrays),
                },
            }
        )
        summary = run_dce_pipeline(config)
    except Exception as exc:
        return {
            "status": "error",
            "duration_sec": float(time.perf_counter() - started),
            "blockers": [f"DCE run failed: {type(exc).__name__}: {exc}"],
            "warnings": [],
        }

    stage_d = dict(summary.get("stages", {}).get("D", {}))
    model_outputs = dict(stage_d.get("model_outputs", {}))
    models_run = [str(v) for v in stage_d.get("models_run", [])]
    requested_models = [str(m).strip().lower() for m in models]
    for model in requested_models:
        if model not in models_run:
            blockers.append(f"Requested model did not run: {model}")

    map_stats: Dict[str, Dict[str, Any]] = {}
    postfit_stats: Dict[str, Dict[str, Any]] = {}
    for model_name, output in model_outputs.items():
        if not isinstance(output, dict):
            continue

        map_paths = dict(output.get("map_paths", {}))
        if not map_paths:
            blockers.append(f"No parameter maps written for model {model_name}")
        for param, raw_path in map_paths.items():
            p = Path(str(raw_path))
            if not p.exists():
                blockers.append(f"Missing map output for {model_name}:{param}: {p}")
                continue
            map_stats[f"{model_name}:{param}"] = _map_sanity_stats(p)

        postfit_path_raw = output.get("postfit_arrays_path")
        if write_postfit_arrays:
            if not isinstance(postfit_path_raw, str) or not postfit_path_raw:
                blockers.append(f"Missing postfit arrays output for model {model_name}")
                continue
            postfit_path = Path(postfit_path_raw)
            if not postfit_path.exists():
                blockers.append(f"Missing postfit arrays file for model {model_name}: {postfit_path}")
                continue
            with np.load(postfit_path) as payload:
                if "voxel_results" not in payload:
                    blockers.append(f"postfit arrays missing voxel_results for model {model_name}")
                    continue
                voxel_results = np.asarray(payload["voxel_results"], dtype=np.float64)
                finite_ratio = float(np.mean(np.isfinite(voxel_results)))
                postfit_stats[model_name] = {
                    "voxel_results_shape": list(voxel_results.shape),
                    "voxel_results_finite_fraction": finite_ratio,
                }
                if finite_ratio < 0.5:
                    blockers.append(f"Model {model_name} has severe non-finite output fraction ({finite_ratio:.3f})")
                elif finite_ratio < 0.95:
                    warnings.append(f"Model {model_name} has elevated non-finite output fraction ({finite_ratio:.3f})")

    blockers.extend(_primary_model_map_blockers(map_stats, requested_models))

    summary_path = Path(str(summary.get("meta", {}).get("summary_path", "")))
    if not summary_path.exists():
        blockers.append(f"Missing DCE summary JSON: {summary_path}")

    return {
        "status": "ok" if not blockers else "failed",
        "duration_sec": float(time.perf_counter() - started),
        "summary_path": str(summary_path),
        "blockers": blockers,
        "warnings": warnings,
        "stage_d": {
            "backend_used": stage_d.get("backend_used"),
            "models_run": models_run,
            "models_skipped": stage_d.get("models_skipped", []),
        },
        "map_stats": map_stats,
        "postfit_stats": postfit_stats,
    }


def _write_markdown_report(summary: Dict[str, Any], target: Path) -> None:
    lines: List[str] = []
    lines.append("# Python Qualification Summary")
    lines.append("")
    lines.append(f"- bids_root: `{summary['meta']['bids_root']}`")
    lines.append(f"- output_root: `{summary['meta']['output_root']}`")
    lines.append(f"- backend: `{summary['meta']['backend']}`")
    lines.append(f"- sessions_discovered: `{summary['meta']['sessions_discovered']}`")
    lines.append(f"- sessions_passed: `{summary['meta']['sessions_passed']}`")
    lines.append(f"- sessions_failed: `{summary['meta']['sessions_failed']}`")
    lines.append(f"- blocker_count: `{summary['meta']['blocker_count']}`")
    lines.append(f"- warning_count: `{summary['meta']['warning_count']}`")
    lines.append("")
    lines.append("| session | status | t1 | dce | blockers | warnings |")
    lines.append("|---|---|---|---|---:|---:|")
    for run in summary["sessions"]:
        t1_status = run.get("t1", {}).get("status", "skipped")
        dce_status = run.get("dce", {}).get("status", "skipped")
        lines.append(
            f"| {run['session_id']} | {run['status']} | {t1_status} | {dce_status} | "
            f"{len(run.get('blockers', []))} | {len(run.get('warnings', []))} |"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bids_qualification(config: QualificationRunConfig) -> Dict[str, Any]:
    bids_root = Path(config.bids_root).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sessions = discover_bids_sessions(bids_root)
    if not sessions:
        raise RuntimeError(f"No BIDS sessions discovered under {bids_root}")

    session_reports: List[Dict[str, Any]] = []
    blocker_count = 0
    warning_count = 0
    passed = 0
    failed = 0

    for session in sessions:
        session_start = time.perf_counter()
        session_dir = output_root / session.id
        run: Dict[str, Any] = {
            "session_id": session.id,
            "subject": session.subject,
            "session": session.session,
            "rawdata_path": str(session.rawdata_path),
            "derivatives_path": str(session.derivatives_path),
            "t1": {"status": "skipped"},
            "dce": {"status": "skipped"},
            "blockers": [],
            "warnings": [],
        }

        if config.run_t1:
            t1_report = _run_t1_for_session(session, session_dir / "t1")
            run["t1"] = t1_report
            run["blockers"].extend(t1_report.get("blockers", []))
            run["warnings"].extend(t1_report.get("warnings", []))

        if config.run_dce:
            dce_report = _run_dce_for_session(
                session,
                session_dir / "dce",
                backend=str(config.backend),
                models=config.dce_models,
                write_postfit_arrays=bool(config.write_postfit_arrays),
            )
            run["dce"] = dce_report
            run["blockers"].extend(dce_report.get("blockers", []))
            run["warnings"].extend(dce_report.get("warnings", []))

        run["duration_sec"] = float(time.perf_counter() - session_start)
        run["status"] = "passed" if not run["blockers"] else "failed"
        blocker_count += len(run["blockers"])
        warning_count += len(run["warnings"])
        if run["status"] == "passed":
            passed += 1
        else:
            failed += 1
        session_reports.append(run)

    summary: Dict[str, Any] = {
        "meta": {
            "pipeline": "python_qualification",
            "status": "ok" if failed == 0 else "failed",
            "bids_root": str(bids_root),
            "output_root": str(output_root),
            "backend": str(config.backend),
            "sessions_discovered": len(sessions),
            "sessions_passed": passed,
            "sessions_failed": failed,
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "run_t1": bool(config.run_t1),
            "run_dce": bool(config.run_dce),
            "dce_models": [str(m).strip().lower() for m in config.dce_models],
        },
        "sessions": session_reports,
    }
    summary_path = output_root / "qualification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["meta"]["summary_json_path"] = str(summary_path)

    markdown_path = output_root / "qualification_summary.md"
    _write_markdown_report(summary, markdown_path)
    summary["meta"]["summary_markdown_path"] = str(markdown_path)
    return summary
