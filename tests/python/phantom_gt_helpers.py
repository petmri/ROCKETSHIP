"""Helpers for synthetic-phantom ground-truth reliability checks on BIDS_test datasets."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import BidsSession, discover_bids_sessions  # noqa: E402
from dce_pipeline import DcePipelineConfig, run_dce_pipeline  # noqa: E402
from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline  # noqa: E402
from qualification import _default_model_flags, _load_array, _session_t1_inputs  # noqa: E402


# Default gating/training set for the current tolerance profile. Diagnostic cases (for
# example sub-08phantom) are intentionally excluded until tolerances are recalibrated.
PHANTOM_SUBJECTS = ("sub-05phantom", "sub-06phantom", "sub-07phantom")
PHANTOM_LABELS = {
    1: "muscle_fat",
    2: "brain",
    3: "vessel",
}
PHANTOM_GT_TOLERANCES_PATH = REPO_ROOT / "tests" / "data" / "BIDS_test" / "phantom_gt_mae_tolerances.json"


def discover_phantom_gt_sessions(bids_root: Path, subjects: Optional[Iterable[str]] = None) -> List[BidsSession]:
    selected = {str(s).strip() for s in (subjects or PHANTOM_SUBJECTS)}
    out: List[BidsSession] = []
    for session in discover_bids_sessions(Path(bids_root)):
        if session.subject not in selected:
            continue
        if (session.rawdata_path / "gt").is_dir():
            out.append(session)
    return out


def _find_one(parent: Path, pattern: str) -> Optional[Path]:
    matches = sorted(parent.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _gt_file(
    gt_dir: Path,
    canonical_pattern: str,
    *,
    legacy_pattern: Optional[str] = None,
    required: bool = True,
) -> Optional[Path]:
    path = _find_one(gt_dir, canonical_pattern)
    if path is None and legacy_pattern:
        path = _find_one(gt_dir, legacy_pattern)
    if path is None and required:
        raise FileNotFoundError(f"Missing GT file {canonical_pattern} under {gt_dir}")
    return path


def phantom_gt_inputs(session: BidsSession) -> Dict[str, Path]:
    gt_dir = session.rawdata_path / "gt"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing gt directory for phantom session: {gt_dir}")

    out: Dict[str, Path] = {}
    out["gt_dir"] = gt_dir
    out["seg"] = _gt_file(gt_dir, "*desc-gtTissueClass_mask.nii*", legacy_pattern="*desc-segmentation.nii*")  # type: ignore[assignment]
    out["aif_mask"] = _gt_file(gt_dir, "*desc-gtAIFMask_mask.nii*", required=False)  # type: ignore[assignment]
    out["aif_timeseries"] = _gt_file(
        gt_dir, "*desc-gtAIF_timeseries.txt", legacy_pattern="*_aif_ground_truth.txt", required=False
    )  # type: ignore[assignment]
    out["t1"] = _gt_file(gt_dir, "*desc-gtT1_T1map.nii*", legacy_pattern="*_t1_true.nii*")  # type: ignore[assignment]
    out["ktrans"] = _gt_file(gt_dir, "*desc-gtKtrans_map.nii*", legacy_pattern="*_ktrans.nii*")  # type: ignore[assignment]
    out["ve"] = _gt_file(gt_dir, "*desc-gtVe_map.nii*", legacy_pattern="*_ve.nii*")  # type: ignore[assignment]
    out["vp"] = _gt_file(gt_dir, "*desc-gtVp_map.nii*", legacy_pattern="*_vp.nii*")  # type: ignore[assignment]
    out["fp"] = _gt_file(gt_dir, "*desc-gtFp_map.nii*", legacy_pattern="*_fp.nii*", required=False)  # type: ignore[assignment]
    return out


def _infer_vfa_frame_count(path: Path) -> int:
    arr = _load_array(path)
    if arr.ndim >= 3:
        return int(arr.shape[-1])
    return 1


def _run_t1_for_phantom_session(session: BidsSession, output_dir: Path) -> Dict[str, Any]:
    inputs = _session_t1_inputs(session)
    frame_count = _infer_vfa_frame_count(Path(inputs["vfa_deriv"]))
    flip_angles = list(inputs["flip_angles"])
    if flip_angles and len(flip_angles) > frame_count:
        flip_angles = flip_angles[:frame_count]

    payload: Dict[str, Any] = {
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
        payload["flip_angles_deg"] = [float(v) for v in flip_angles]
    if inputs["tr_ms"] is not None:
        payload["tr_ms"] = float(inputs["tr_ms"])
    return run_parametric_t1_pipeline(ParametricT1Config.from_dict(payload))


def _run_dce_for_phantom_session(
    session: BidsSession,
    output_dir: Path,
    *,
    backend: str,
    models: Iterable[str],
    t1_map_path: Path,
    steady_state_end_1b: Optional[int] = None,
) -> Dict[str, Any]:
    dce_deriv = session.derivatives_path / "dce"
    anat_deriv = session.derivatives_path / "anat"
    dynamic = _find_one(dce_deriv, "*desc-bfcz_DCE.nii*") or _find_one(dce_deriv, "*DCE.nii*")
    aif = _find_one(dce_deriv, "*desc-AIF_T1map.nii*")
    roi = _find_one(anat_deriv, "*desc-brain_mask.nii*")

    missing: List[str] = []
    if dynamic is None:
        missing.append("dynamic")
    if aif is None:
        missing.append("aif")
    if roi is None:
        missing.append("roi")
    if missing:
        raise FileNotFoundError(f"Missing derivative DCE inputs ({', '.join(missing)}) for {session.id}")

    stage_overrides: Dict[str, Any] = {
        "stage_a_mode": "real",
        "stage_b_mode": "real",
        "stage_d_mode": "real",
        "aif_curve_mode": "raw",
        "write_param_maps": True,
        "write_postfit_arrays": False,
    }
    if steady_state_end_1b is not None and int(steady_state_end_1b) >= 1:
        # Phantom-only diagnostic alignment: use generator-provided baseline_images so
        # Stage-A baseline matches the GT AIF header. This is not the general real-data
        # solution; TODO is to port MATLAB baseline auto-detection into the pipeline.
        stage_overrides["steady_state_start"] = 1
        stage_overrides["steady_state_end"] = int(steady_state_end_1b)

    cfg = DcePipelineConfig.from_dict(
        {
            "subject_source_path": str(session.rawdata_path),
            "subject_tp_path": str(session.derivatives_path),
            "output_dir": str(output_dir),
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "backend": str(backend),
            "write_xls": False,
            "aif_mode": "auto",
            "dynamic_files": [str(dynamic)],
            "aif_files": [str(aif)],
            "roi_files": [str(roi)],
            "t1map_files": [str(t1_map_path)],
            # Synthetic phantom fixtures can have near-zero background noise; use ROI for deterministic noise estimate.
            "noise_files": [str(roi)],
            "drift_files": [],
            "model_flags": _default_model_flags(models),
            "stage_overrides": stage_overrides,
        }
    )
    return run_dce_pipeline(cfg)


def _region_mae_stats(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    mask_use = np.asarray(mask, dtype=bool)
    finite = mask_use & np.isfinite(pred) & np.isfinite(truth)
    n = int(np.count_nonzero(finite))
    if n <= 0:
        return None
    pred_vals = pred[finite]
    truth_vals = truth[finite]
    err = pred_vals - truth_vals
    abs_err = np.abs(err)
    truth_abs = np.abs(truth_vals)
    truth_mean = float(np.mean(truth_vals))
    truth_mean_abs = float(np.mean(truth_abs))
    truth_median = float(np.median(truth_vals))
    truth_median_abs = float(np.median(truth_abs))
    pred_median = float(np.median(pred_vals))
    out: Dict[str, float] = {
        "n": float(n),
        "mae": float(np.mean(abs_err)),
        "bias": float(np.mean(err)),
        "max_abs": float(np.max(abs_err)),
        "p95_abs": float(np.percentile(abs_err, 95.0)),
        "pred_median": pred_median,
        "gt_mean": truth_mean,
        "gt_mean_abs": truth_mean_abs,
        "gt_median": truth_median,
        "gt_median_abs": truth_median_abs,
        "gt_p95_abs": float(np.percentile(truth_abs, 95.0)),
    }
    if truth_mean_abs > 1e-12:
        out["mae_rel_gt_mean_abs"] = float(out["mae"] / truth_mean_abs)
        out["mae_pct_gt_mean_abs"] = float(out["mae_rel_gt_mean_abs"] * 100.0)
        out["bias_rel_gt_mean_abs"] = float(out["bias"] / truth_mean_abs)
    if truth_median_abs > 1e-12:
        out["mae_rel_gt_median_abs"] = float(out["mae"] / truth_median_abs)
        out["mae_pct_gt_median_abs"] = float(out["mae_rel_gt_median_abs"] * 100.0)
        out["bias_rel_gt_median_abs"] = float(out["bias"] / truth_median_abs)
    if abs(truth_median) > 1e-12:
        out["median_bias"] = float(pred_median - truth_median)
        out["median_bias_pct_gt_median"] = float((out["median_bias"] / truth_median) * 100.0)
    return out


def _parse_gt_aif_timeseries(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    baseline_images: Optional[int] = None
    tres_min: Optional[float] = None
    relaxivity_per_mM_per_s: Optional[float] = None
    aif_t1_pre_ms: Optional[float] = None
    tr_ms: Optional[float] = None
    fa_deg: Optional[float] = None
    aif_concentration_kind: Optional[str] = None
    plasma_correction_applied: Optional[bool] = None
    recommended_rocketship_hematocrit: Optional[float] = None
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        if baseline_images is None:
            m = re.search(r"baseline_images=(\d+)", line)
            if m:
                baseline_images = int(m.group(1))
        if tres_min is None:
            m = re.search(r"tres_min=([0-9.+-eE]+)", line)
            if m:
                tres_min = float(m.group(1))
        if relaxivity_per_mM_per_s is None:
            m = re.search(r"relaxivity_per_mM_per_s=([0-9.+-eE]+)", line)
            if m:
                relaxivity_per_mM_per_s = float(m.group(1))
        if aif_t1_pre_ms is None:
            m = re.search(r"aif_t1_pre_ms=([0-9.+-eE]+)", line)
            if m:
                aif_t1_pre_ms = float(m.group(1))
        if tr_ms is None:
            m = re.search(r"tr_ms=([0-9.+-eE]+)", line)
            if m:
                tr_ms = float(m.group(1))
        if fa_deg is None:
            m = re.search(r"fa_deg=([0-9.+-eE]+)", line)
            if m:
                fa_deg = float(m.group(1))
        if aif_concentration_kind is None:
            m = re.search(r"aif_concentration_kind=([A-Za-z_]+)", line)
            if m:
                aif_concentration_kind = str(m.group(1)).strip().lower()
        if plasma_correction_applied is None:
            m = re.search(r"plasma_correction_applied=([A-Za-z0-9._+-]+)", line)
            if m:
                token = str(m.group(1)).strip().lower()
                if token in {"1", "true", "yes"}:
                    plasma_correction_applied = True
                elif token in {"0", "false", "no"}:
                    plasma_correction_applied = False
        if recommended_rocketship_hematocrit is None:
            m = re.search(r"recommended_rocketship_hematocrit=([0-9.+-eE]+)", line)
            if m:
                recommended_rocketship_hematocrit = float(m.group(1))

    json_sidecar = path.with_suffix(".json")
    if json_sidecar.exists():
        try:
            payload = json.loads(json_sidecar.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if baseline_images is None and payload.get("BaselineImages") is not None:
            baseline_images = int(payload["BaselineImages"])
        if tres_min is None:
            if payload.get("TemporalResolutionMin") is not None:
                tres_min = float(payload["TemporalResolutionMin"])
            elif payload.get("TemporalResolution") is not None:
                tres_min = float(payload["TemporalResolution"]) / 60.0
        if relaxivity_per_mM_per_s is None and payload.get("Relaxivity_per_mM_per_s") is not None:
            relaxivity_per_mM_per_s = float(payload["Relaxivity_per_mM_per_s"])
        if aif_t1_pre_ms is None and payload.get("AIFPrecontrastT1_ms") is not None:
            aif_t1_pre_ms = float(payload["AIFPrecontrastT1_ms"])
        if tr_ms is None and payload.get("RepetitionTime") is not None:
            tr_ms = float(payload["RepetitionTime"]) * 1000.0
        if fa_deg is None and payload.get("FlipAngle") is not None:
            fa_deg = float(payload["FlipAngle"])
        if aif_concentration_kind is None and payload.get("AIFConcentrationKind") is not None:
            aif_concentration_kind = str(payload["AIFConcentrationKind"]).strip().lower()
        if plasma_correction_applied is None and payload.get("PlasmaCorrectionApplied") is not None:
            plasma_correction_applied = bool(payload["PlasmaCorrectionApplied"])
        if recommended_rocketship_hematocrit is None and payload.get("RecommendedROCKETSHIPHematocrit") is not None:
            recommended_rocketship_hematocrit = float(payload["RecommendedROCKETSHIPHematocrit"])
    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"GT AIF file must have at least 3 columns (frame,time,cp): {path}")
    frame_idx = np.asarray(data[:, 0], dtype=np.int64)
    time_min = np.asarray(data[:, 1], dtype=np.float64)
    cp_mM = np.asarray(data[:, 2], dtype=np.float64)
    si_norm = np.asarray(data[:, 3], dtype=np.float64) if data.shape[1] >= 4 else None
    return {
        "path": str(path),
        "frame_idx": frame_idx,
        "time_min": time_min,
        "cp_mM": cp_mM,
        "si_norm": si_norm,
        "baseline_images": baseline_images,
        "tres_min": tres_min,
        "tres_sec": (float(tres_min) * 60.0) if tres_min is not None else None,
        "relaxivity_per_mM_per_s": relaxivity_per_mM_per_s,
        "aif_t1_pre_ms": aif_t1_pre_ms,
        "tr_ms": tr_ms,
        "fa_deg": fa_deg,
        "aif_concentration_kind": aif_concentration_kind,
        "plasma_correction_applied": plasma_correction_applied,
        "recommended_rocketship_hematocrit": recommended_rocketship_hematocrit,
        "json_sidecar_path": str(json_sidecar) if json_sidecar.exists() else None,
    }


def _curve_dt_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return {"dt_median": float("nan"), "dt_mean": float("nan")}
    diff = np.diff(arr)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return {"dt_median": float("nan"), "dt_mean": float("nan")}
    return {"dt_median": float(np.median(finite)), "dt_mean": float(np.mean(finite))}


def _aif_comparison_from_artifacts(
    *,
    gt_aif_path: Optional[Path],
    checkpoint_dir: Path,
) -> Optional[Dict[str, Any]]:
    if gt_aif_path is None or not gt_aif_path.exists():
        return None

    b_npz = checkpoint_dir / "b_out_arrays.npz"
    a_npz = checkpoint_dir / "a_out_arrays.npz"
    a_json = checkpoint_dir / "a_out.json"
    if not b_npz.exists() or not a_npz.exists():
        return None

    gt = _parse_gt_aif_timeseries(gt_aif_path)

    a_meta: Dict[str, Any] = {}
    if a_json.exists():
        try:
            a_meta = json.loads(a_json.read_text(encoding="utf-8"))
        except Exception:
            a_meta = {}

    with np.load(b_npz) as payload:
        if "Cp_use" not in payload or "timer" not in payload:
            return None
        cp_fit = np.asarray(payload["Cp_use"], dtype=np.float64).reshape(-1)
        timer_fit = np.asarray(payload["timer"], dtype=np.float64).reshape(-1)
    with np.load(a_npz) as payload:
        cp_stage_a = np.asarray(payload["Cp"], dtype=np.float64) if "Cp" in payload else None
        delta_r1_lv = np.asarray(payload["deltaR1LV"], dtype=np.float64) if "deltaR1LV" in payload else None

    timer_gt = np.asarray(gt["time_min"], dtype=np.float64).reshape(-1)
    cp_gt = np.asarray(gt["cp_mM"], dtype=np.float64).reshape(-1)
    n_compare = int(min(timer_gt.size, timer_fit.size, cp_gt.size, cp_fit.size))
    if n_compare <= 0:
        return None

    timer_gt_cmp = timer_gt[:n_compare]
    timer_fit_cmp = timer_fit[:n_compare]
    cp_gt_cmp = cp_gt[:n_compare]
    cp_fit_cmp = cp_fit[:n_compare]

    timer_offset_min = float(np.median(timer_fit_cmp - timer_gt_cmp))
    timer_resid_sec = (timer_fit_cmp - (timer_gt_cmp + timer_offset_min)) * 60.0
    timer_mae_after_offset_sec = float(np.mean(np.abs(timer_resid_sec)))
    gt_dt = _curve_dt_stats(timer_gt_cmp)
    fit_dt = _curve_dt_stats(timer_fit_cmp)

    relaxivity = None
    hematocrit = None
    if a_meta.get("relaxivity") is not None:
        try:
            relaxivity = float(a_meta.get("relaxivity"))
        except Exception:
            relaxivity = None
    if a_meta.get("hematocrit") is not None:
        try:
            hematocrit = float(a_meta.get("hematocrit"))
        except Exception:
            hematocrit = None

    def _curve_metrics(pred_vals: Optional[np.ndarray], truth_vals: np.ndarray) -> Optional[Dict[str, Any]]:
        if pred_vals is None:
            return None
        pred_arr = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
        truth_arr = np.asarray(truth_vals, dtype=np.float64).reshape(-1)
        n_cmp = int(min(pred_arr.size, truth_arr.size))
        if n_cmp <= 0:
            return None
        pred_cmp = pred_arr[:n_cmp]
        truth_cmp = truth_arr[:n_cmp]
        stats = _region_mae_stats(pred_cmp, truth_cmp, np.ones((n_cmp,), dtype=bool))
        if stats is None:
            return None
        out = _numeric_copy(stats)
        corr = float("nan")
        if n_cmp >= 2:
            x = pred_cmp - float(np.mean(pred_cmp))
            y = truth_cmp - float(np.mean(truth_cmp))
            denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
            if denom > 0.0 and np.isfinite(denom):
                corr = float(np.sum(x * y) / denom)
        peak_gt = float(np.max(truth_cmp)) if np.any(np.isfinite(truth_cmp)) else float("nan")
        peak_fit = float(np.max(pred_cmp)) if np.any(np.isfinite(pred_cmp)) else float("nan")
        peak_ratio = (
            float(peak_fit / peak_gt)
            if np.isfinite(peak_gt) and abs(peak_gt) > 1e-12 and np.isfinite(peak_fit)
            else float("nan")
        )
        out["correlation"] = corr
        out["peak_gt_mM"] = peak_gt
        out["peak_fit_mM"] = peak_fit
        out["peak_ratio_fit_over_gt"] = peak_ratio
        return out

    def _roi_mean_time_curve(arr: Optional[np.ndarray], n_time: int) -> Optional[np.ndarray]:
        if arr is None:
            return None
        data = np.asarray(arr, dtype=np.float64)
        if data.ndim == 1:
            return np.asarray(data[:n_time], dtype=np.float64)
        if data.ndim >= 2 and data.shape[0] >= n_time:
            axes = tuple(range(1, data.ndim))
            return np.asarray(np.mean(data[:n_time, ...], axis=axes), dtype=np.float64)
        flat = data.reshape(-1)
        if flat.size >= n_time:
            return np.asarray(flat[:n_time], dtype=np.float64)
        return None

    cp_use_metrics = _curve_metrics(cp_fit_cmp, cp_gt_cmp)
    if cp_use_metrics is None:
        return None
    cp_stage_a_curve = _roi_mean_time_curve(cp_stage_a, n_compare)
    cp_stage_a_metrics = _curve_metrics(cp_stage_a_curve, cp_gt_cmp)

    cb_stage_a = None
    if cp_stage_a_curve is not None and hematocrit is not None and np.isfinite(hematocrit):
        cb_stage_a = np.asarray(cp_stage_a_curve * (1.0 - float(hematocrit)), dtype=np.float64)
    elif delta_r1_lv is not None and relaxivity is not None and np.isfinite(relaxivity) and relaxivity > 0.0:
        delta_curve = _roi_mean_time_curve(delta_r1_lv, n_compare)
        if delta_curve is not None:
            cb_stage_a = np.asarray(delta_curve / float(relaxivity), dtype=np.float64)
    cb_stage_a_metrics = _curve_metrics(cb_stage_a, cp_gt_cmp)

    steady_state_time = a_meta.get("steady_state_time")
    fit_baseline_n: Optional[int] = None
    fit_baseline_mean: Optional[float] = None
    if isinstance(steady_state_time, list) and len(steady_state_time) == 2:
        try:
            ss_start = int(steady_state_time[0])
            ss_end = int(steady_state_time[1])
            if ss_end >= ss_start >= 1:
                fit_baseline_n = int(ss_end - ss_start + 1)
                fit_baseline_mean = float(np.mean(cp_fit[ss_start - 1 : ss_end]))
        except Exception:
            pass

    gt_baseline_n = gt.get("baseline_images")
    gt_baseline_mean = None
    fit_baseline_mean_at_gt_n = None
    if isinstance(gt_baseline_n, int) and gt_baseline_n > 0:
        n0 = min(int(gt_baseline_n), cp_gt.size, cp_fit.size)
        if n0 > 0:
            gt_baseline_mean = float(np.mean(cp_gt[:n0]))
            fit_baseline_mean_at_gt_n = float(np.mean(cp_fit[:n0]))

    cp_peak_gt = cp_use_metrics.get("peak_gt_mM")
    cp_peak_fit = cp_use_metrics.get("peak_fit_mM")
    cp_peak_ratio = cp_use_metrics.get("peak_ratio_fit_over_gt")

    return {
        "gt": {
            "path": str(gt_aif_path),
            "n": int(cp_gt.size),
            "baseline_images": gt_baseline_n,
            "tres_min": gt.get("tres_min"),
            "tres_sec": gt.get("tres_sec"),
            "cp_peak_mM": float(cp_peak_gt) if cp_peak_gt is not None else None,
            "cp_median_mM": float(np.median(cp_gt_cmp)),
            "cp_baseline_mean_mM": gt_baseline_mean,
        },
        "fit": {
            "n": int(cp_fit.size),
            "time_resolution_min": float(a_meta.get("time_resolution_min")) if a_meta.get("time_resolution_min") is not None else None,
            "steady_state_time": steady_state_time,
            "steady_state_n": fit_baseline_n,
            "cp_peak_mM": float(cp_peak_fit) if cp_peak_fit is not None else None,
            "cp_median_mM": float(np.median(cp_fit_cmp)),
            "cp_baseline_mean_mM": fit_baseline_mean,
            "cp_baseline_mean_mM_at_gt_baseline_n": fit_baseline_mean_at_gt_n,
            "relaxivity": relaxivity,
            "hematocrit": hematocrit,
        },
        "compare": {
            "n_compare": n_compare,
            "timer_offset_min": timer_offset_min,
            "timer_offset_sec": float(timer_offset_min * 60.0),
            "timer_mae_after_offset_sec": timer_mae_after_offset_sec,
            "timer_dt_gt_sec": float(gt_dt["dt_median"] * 60.0) if np.isfinite(gt_dt["dt_median"]) else None,
            "timer_dt_fit_sec": float(fit_dt["dt_median"] * 60.0) if np.isfinite(fit_dt["dt_median"]) else None,
            "timer_dt_ratio_fit_over_gt": (
                float(fit_dt["dt_median"] / gt_dt["dt_median"])
                if np.isfinite(gt_dt["dt_median"]) and abs(gt_dt["dt_median"]) > 1e-12 and np.isfinite(fit_dt["dt_median"])
                else None
            ),
            "cp_correlation": cp_use_metrics.get("correlation"),
            "cp_peak_ratio_fit_over_gt": cp_peak_ratio,
            "cp": cp_use_metrics,
            "cp_use": cp_use_metrics,
            "cp_stage_a": cp_stage_a_metrics,
            "cb_stage_a": cb_stage_a_metrics,
        },
        "artifacts": {
            "b_out_arrays_npz": str(b_npz),
            "a_out_arrays_npz": str(a_npz),
            "a_out_json": str(a_json) if a_json.exists() else None,
        },
    }


def _numeric_copy(stats: Dict[str, float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in stats.items():
        if value is None:
            out[key] = None
            continue
        if key == "n":
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def run_phantom_gt_session_compare(
    session: BidsSession,
    *,
    output_dir: Path,
    backend: str,
    dce_models: Iterable[str] = ("tofts", "ex_tofts", "patlak"),
) -> Dict[str, Any]:
    gt = phantom_gt_inputs(session)
    gt_seg = np.asarray(_load_array(gt["seg"]), dtype=np.float64)
    gt_t1 = np.asarray(_load_array(gt["t1"]), dtype=np.float64)
    gt_maps = {
        "Ktrans": np.asarray(_load_array(gt["ktrans"]), dtype=np.float64),
        "ve": np.asarray(_load_array(gt["ve"]), dtype=np.float64),
        "vp": np.asarray(_load_array(gt["vp"]), dtype=np.float64),
    }
    if gt.get("fp") is not None:
        gt_maps["fp"] = np.asarray(_load_array(gt["fp"]), dtype=np.float64)
    gt_aif_meta: Optional[Dict[str, Any]] = None
    gt_baseline_images: Optional[int] = None
    if gt.get("aif_timeseries") is not None:
        gt_aif_meta = _parse_gt_aif_timeseries(Path(gt["aif_timeseries"]))
        raw_baseline = gt_aif_meta.get("baseline_images")
        if isinstance(raw_baseline, int) and raw_baseline >= 1:
            gt_baseline_images = int(raw_baseline)

    t1_summary = _run_t1_for_phantom_session(session, output_dir / "t1")
    t1_map_path = Path(str(t1_summary.get("outputs", {}).get("t1_map_path", "")))
    if not t1_map_path.exists():
        raise FileNotFoundError(f"Phantom T1 map missing after T1 run: {t1_map_path}")
    pred_t1 = np.asarray(_load_array(t1_map_path), dtype=np.float64)

    dce_summary = _run_dce_for_phantom_session(
        session,
        output_dir / "dce",
        backend=backend,
        models=dce_models,
        t1_map_path=t1_map_path,
        steady_state_end_1b=gt_baseline_images,
    )
    stage_d = dict(dce_summary.get("stages", {}).get("D", {}))

    result: Dict[str, Any] = {
        "session_id": session.id,
        "subject": session.subject,
        "backend_requested": str(backend),
        "backend_used": str(stage_d.get("backend_used", "")),
        "aif": {},
        "t1": {},
        "dce": {},
        "artifacts": {
            "t1_summary_json": str(t1_summary.get("meta", {}).get("summary_path", "")),
            "dce_summary_json": str(dce_summary.get("meta", {}).get("summary_path", "")),
        },
    }

    checkpoint_dir = Path(output_dir) / "dce" / "checkpoints"
    aif_compare = _aif_comparison_from_artifacts(gt_aif_path=gt.get("aif_timeseries"), checkpoint_dir=checkpoint_dir)
    if aif_compare is not None:
        if gt_baseline_images is not None:
            aif_compare.setdefault("fit", {})
            aif_compare["fit"]["requested_steady_state_end"] = int(gt_baseline_images)
        result["aif"] = aif_compare

    for label_value, region_name in PHANTOM_LABELS.items():
        region_mask = gt_seg == float(label_value)
        stats = _region_mae_stats(pred_t1, gt_t1, region_mask)
        if stats is not None:
            result["t1"][region_name] = _numeric_copy(stats)

    model_outputs = dict(stage_d.get("model_outputs", {}))
    for model_name in dce_models:
        model_dict = model_outputs.get(model_name)
        if not isinstance(model_dict, dict):
            continue
        map_paths = dict(model_dict.get("map_paths", {}))
        model_metrics: Dict[str, Any] = {}
        preferred_order = ("Ktrans", "ve", "vp", "fp")
        param_names: List[str] = []
        for key in preferred_order:
            if key in map_paths and key in gt_maps:
                param_names.append(key)
        for key in sorted(map_paths.keys()):
            if key in gt_maps and key not in param_names:
                param_names.append(key)
        for param_name in param_names:
            map_path_str = map_paths.get(param_name)
            if not map_path_str:
                continue
            pred_map = np.asarray(_load_array(Path(str(map_path_str))), dtype=np.float64)
            truth_map = gt_maps[param_name]
            region_metrics: Dict[str, Any] = {}
            for label_value, region_name in PHANTOM_LABELS.items():
                region_mask = (gt_seg == float(label_value)) & (pred_map != 0.0)
                stats = _region_mae_stats(pred_map, truth_map, region_mask)
                if stats is not None:
                    region_metrics[region_name] = _numeric_copy(stats)
            if region_metrics:
                model_metrics[param_name] = region_metrics
        if model_metrics:
            result["dce"][str(model_name)] = model_metrics

    return result


def run_phantom_gt_bids_summary(
    *,
    bids_root: Path,
    output_root: Path,
    backend: str,
    subjects: Optional[Iterable[str]] = None,
    dce_models: Iterable[str] = ("tofts", "ex_tofts", "patlak"),
) -> Dict[str, Any]:
    sessions = discover_phantom_gt_sessions(bids_root, subjects=subjects)
    if not sessions:
        raise RuntimeError(f"No phantom GT sessions found under {Path(bids_root).resolve()}")

    runs: List[Dict[str, Any]] = []
    for session in sessions:
        runs.append(
            run_phantom_gt_session_compare(
                session,
                output_dir=Path(output_root) / session.id,
                backend=backend,
                dce_models=dce_models,
            )
        )

    return {
        "meta": {
            "bids_root": str(Path(bids_root).expanduser().resolve()),
            "output_root": str(Path(output_root).expanduser().resolve()),
            "backend_requested": str(backend),
            "sessions": [s.id for s in sessions],
            "dce_models": [str(m) for m in dce_models],
        },
        "sessions": runs,
    }


def load_phantom_gt_tolerances(path: Optional[Path] = None) -> Dict[str, Any]:
    target = Path(path or PHANTOM_GT_TOLERANCES_PATH)
    return json.loads(target.read_text(encoding="utf-8"))


def _ceil_step(value: float, step: float) -> float:
    if step <= 0:
        return round(float(value), 6)
    return round(float(np.ceil(float(value) / step) * step), 6)


def _rounded_tolerance(value: float, *, metric_kind: str) -> float:
    v = max(0.0, float(value))
    if metric_kind == "t1":
        if v < 100.0:
            return _ceil_step(v, 10.0)
        if v < 1000.0:
            return _ceil_step(v, 25.0)
        return _ceil_step(v, 50.0)
    # DCE parameters are mostly in /min or fraction scale.
    if v < 0.05:
        return _ceil_step(v, 0.005)
    if v < 0.5:
        return _ceil_step(v, 0.01)
    if v < 2.0:
        return _ceil_step(v, 0.05)
    return _ceil_step(v, 0.1)


def build_phantom_gt_tolerance_profile(
    summary_by_backend: Dict[str, Dict[str, Any]],
    *,
    margin_factor_dce: float = 1.25,
    margin_factor_t1: float = 1.20,
    dce_additive_margin: float = 0.01,
    t1_additive_margin_ms: float = 50.0,
) -> Dict[str, Any]:
    observed_t1: Dict[str, float] = {}
    observed_dce: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for backend_key, summary in summary_by_backend.items():
        sessions = list(summary.get("sessions", []))
        for session in sessions:
            for region_name, stats in dict(session.get("t1", {})).items():
                mae = float(stats.get("mae", float("nan")))
                if np.isfinite(mae):
                    observed_t1[region_name] = max(observed_t1.get(region_name, 0.0), mae)
            backend_used = str(session.get("backend_used") or backend_key)
            observed_dce.setdefault(backend_used, {})
            for model_name, model_metrics in dict(session.get("dce", {})).items():
                model_store = observed_dce[backend_used].setdefault(str(model_name), {})
                for param_name, region_metrics in dict(model_metrics).items():
                    param_store = model_store.setdefault(str(param_name), {})
                    for region_name, stats in dict(region_metrics).items():
                        mae = float(stats.get("mae", float("nan")))
                        if np.isfinite(mae):
                            param_store[region_name] = max(param_store.get(region_name, 0.0), mae)

    t1_tol: Dict[str, float] = {}
    for region_name, max_mae in observed_t1.items():
        raw = (max_mae * float(margin_factor_t1)) + float(t1_additive_margin_ms)
        t1_tol[region_name] = _rounded_tolerance(raw, metric_kind="t1")

    dce_tol: Dict[str, Any] = {}
    for backend_used, models in observed_dce.items():
        dce_tol[backend_used] = {}
        for model_name, params in models.items():
            dce_tol[backend_used][model_name] = {}
            for param_name, regions in params.items():
                dce_tol[backend_used][model_name][param_name] = {}
                for region_name, max_mae in regions.items():
                    raw = (float(max_mae) * float(margin_factor_dce)) + float(dce_additive_margin)
                    dce_tol[backend_used][model_name][param_name][region_name] = _rounded_tolerance(
                        raw, metric_kind="dce"
                    )

    return {
        "version": 1,
        "gate_ready": False,
        "metric": "voxelwise_region_mae",
        "label_map": {str(k): v for k, v in PHANTOM_LABELS.items()},
        "backend_aliases": {
            "auto": "cpufit_cpu",
        },
        "tolerances": {
            "t1_ms": t1_tol,
            "dce": dce_tol,
        },
        "observed_max_mae": {
            "t1_ms": observed_t1,
            "dce": observed_dce,
        },
        "provenance": {
            "source": "phantom_gt_explore",
            "source_sessions": [str(s) for s in PHANTOM_SUBJECTS],
            "margin_factor_dce": float(margin_factor_dce),
            "margin_factor_t1": float(margin_factor_t1),
            "dce_additive_margin": float(dce_additive_margin),
            "t1_additive_margin_ms": float(t1_additive_margin_ms),
            "notes": [
                "This profile is provisional and not a merge gate yet (performance triage in progress).",
                "Tolerances are model- and region-specific because model assumptions intentionally bias some regions.",
                "Initial profile derived from local ROCKETSHIP CPU and cpufit_cpu (backend=auto) runs on phantom sessions.",
                "MATLAB calibration can be added later without changing test code structure.",
            ],
        },
    }
