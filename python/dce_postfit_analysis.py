"""Core post-fit analysis helpers for DCE Part E workflow.

This module ports the non-GUI statistical core from MATLAB `fitting_analysis.m`
for f-test and Akaike-based model comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
from typing import Any, Dict, List, Literal, Sequence

import numpy as np
from scipy import stats


Region = Literal["voxel", "roi"]


_MODEL_META: Dict[str, Dict[str, int]] = {
    # sse_col is 1-based, matching MATLAB column indexing in fitting_analysis.m.
    "aif": {"fp": 2, "sse_col": 3},
    "tofts": {"fp": 2, "sse_col": 3},
    "aif_vp": {"fp": 3, "sse_col": 4},
    "ex_tofts": {"fp": 3, "sse_col": 4},
    "fxr": {"fp": 3, "sse_col": 4},
    "patlak": {"fp": 2, "sse_col": 3},
}


@dataclass
class DceFitStats:
    """Minimal fit payload needed by post-fit analysis."""

    model_name: str
    timer: np.ndarray
    fitting_results: np.ndarray | None = None
    roi_results: np.ndarray | None = None
    roi_names: List[str] | None = None
    tumind_1based: np.ndarray | None = None
    dimensions: Sequence[int] | None = None


def is_valid_model(model_name: str) -> bool:
    return str(model_name).strip().lower() in _MODEL_META


def _to_2d_float(array: np.ndarray | None, label: str) -> np.ndarray:
    if array is None:
        raise ValueError(f"{label} is required")
    out = np.asarray(array, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {out.shape}")
    return out


def get_sse_and_fp(stats_in: DceFitStats, region: Region) -> tuple[np.ndarray, int, int]:
    """Return SSE vector, free-parameter count, and time-point count."""
    model_name = str(stats_in.model_name).strip().lower()
    if model_name not in _MODEL_META:
        raise ValueError(f"model '{stats_in.model_name}' is not supported for post-fit comparison")

    meta = _MODEL_META[model_name]
    sse_index = int(meta["sse_col"]) - 1
    fp = int(meta["fp"])

    if region == "voxel":
        results = _to_2d_float(stats_in.fitting_results, "fitting_results")
    else:
        results = _to_2d_float(stats_in.roi_results, "roi_results")

    if results.shape[1] <= sse_index:
        raise ValueError(
            f"{region} results for model '{stats_in.model_name}' do not contain required SSE column {sse_index + 1}"
        )

    timer = np.asarray(stats_in.timer, dtype=np.float64).reshape(-1)
    n = int(timer.size)
    if n <= 0:
        raise ValueError("timer must contain at least one entry")

    sse = results[:, sse_index]
    return sse, fp, n


def compute_ftest(lower_model: DceFitStats, higher_model: DceFitStats, region: Region) -> Dict[str, np.ndarray | float]:
    """Compute per-region f-test p-values comparing nested lower vs higher model."""
    sse_lower, fp_lower, n_lower = get_sse_and_fp(lower_model, region)
    sse_higher, fp_higher, n_higher = get_sse_and_fp(higher_model, region)

    if n_lower != n_higher:
        raise ValueError("models have different timer lengths")
    if fp_lower >= fp_higher:
        raise ValueError("lower model must have fewer free parameters than higher model")
    if sse_lower.shape != sse_higher.shape:
        raise ValueError(f"result lengths differ: {sse_lower.shape} vs {sse_higher.shape}")

    df1 = float(fp_higher - fp_lower)
    df2 = float(n_higher - fp_higher)
    if df2 <= 0.0:
        raise ValueError("invalid degrees of freedom: n - fp_higher must be positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        f_stat = ((sse_lower - sse_higher) / df1) / (sse_higher / df2)
    p_values = stats.f.sf(f_stat, df1, df2)

    invalid = (~np.isfinite(f_stat)) | (~np.isfinite(p_values))
    p_values = np.asarray(p_values, dtype=np.float64)
    p_values[invalid] = np.nan
    f_stat = np.asarray(f_stat, dtype=np.float64)
    f_stat[invalid] = np.nan

    return {
        "p_values": p_values,
        "f_stat": f_stat,
        "mean_p": float(np.nanmean(p_values)) if np.any(np.isfinite(p_values)) else float("nan"),
    }


def compute_aic(models: Sequence[DceFitStats], region: Region) -> Dict[str, np.ndarray | List[str]]:
    """Compute AIC and relative likelihood across a model set."""
    if len(models) < 2:
        raise ValueError("at least two models are required for AIC comparison")

    sse_list: List[np.ndarray] = []
    fp_list: List[int] = []
    n_ref: int | None = None
    length_ref: int | None = None
    model_names: List[str] = []

    for model in models:
        sse, fp, n = get_sse_and_fp(model, region)
        if n_ref is None:
            n_ref = int(n)
        elif n != n_ref:
            raise ValueError("all models must share the same timer length")
        if length_ref is None:
            length_ref = int(sse.shape[0])
        elif int(sse.shape[0]) != length_ref:
            raise ValueError("all models must have the same number of fitted regions")
        sse_list.append(np.asarray(sse, dtype=np.float64))
        fp_list.append(int(fp))
        model_names.append(str(model.model_name))

    assert n_ref is not None
    sse_matrix = np.stack(sse_list, axis=1)  # [n_region, n_models]
    fp_array = np.asarray(fp_list, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        aic = (float(n_ref) * np.log(sse_matrix / float(n_ref))) + (2.0 * fp_array[None, :])

    aic_finite = np.where(np.isfinite(aic), aic, np.inf)
    sorted_idx = np.argsort(aic_finite, axis=1)
    best_idx = sorted_idx[:, 0]
    second_idx = sorted_idx[:, 1]

    row_idx = np.arange(aic.shape[0])
    best_aic = aic_finite[row_idx, best_idx]
    second_aic = aic_finite[row_idx, second_idx]

    with np.errstate(over="ignore", invalid="ignore"):
        relative_likelihood = np.exp((best_aic[:, None] - aic_finite) / 2.0)
        relative_likelihood_best_vs_second = np.exp((best_aic - second_aic) / 2.0)

    best_model_names = [model_names[int(i)] for i in best_idx]
    second_model_names = [model_names[int(i)] for i in second_idx]

    return {
        "model_names": np.asarray(model_names, dtype=object),
        "aic": aic,
        "relative_likelihood": relative_likelihood,
        "best_model_names": np.asarray(best_model_names, dtype=object),
        "second_model_names": np.asarray(second_model_names, dtype=object),
        "relative_likelihood_best_vs_second": relative_likelihood_best_vs_second,
    }


def write_ftest_roi_csv(
    output_path: Path,
    roi_names: Sequence[str],
    p_values: np.ndarray,
    sse_higher: np.ndarray,
    sse_lower: np.ndarray,
    *,
    higher_model_name: str,
    lower_model_name: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ROI", "p_value", f"Residual_{higher_model_name}", f"Residual_{lower_model_name}"])
        for roi, p_val, sse_hi, sse_lo in zip(roi_names, p_values, sse_higher, sse_lower):
            writer.writerow([str(roi), float(p_val), float(sse_hi), float(sse_lo)])
    return output_path


def write_aic_roi_csv(
    output_path: Path,
    roi_names: Sequence[str],
    aic_result: Dict[str, np.ndarray | List[str]],
) -> Path:
    model_names = [str(v) for v in np.asarray(aic_result["model_names"]).tolist()]
    aic = np.asarray(aic_result["aic"], dtype=np.float64)
    best_names = [str(v) for v in np.asarray(aic_result["best_model_names"]).tolist()]
    second_names = [str(v) for v in np.asarray(aic_result["second_model_names"]).tolist()]
    like12 = np.asarray(aic_result["relative_likelihood_best_vs_second"], dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["ROI", "best_model", "second_model", "best_vs_second_relative_likelihood"]
            + [f"AIC_{name}" for name in model_names]
        )
        for idx, roi in enumerate(roi_names):
            writer.writerow(
                [str(roi), best_names[idx], second_names[idx], float(like12[idx])]
                + [float(v) for v in aic[idx, :]]
            )
    return output_path


def voxel_values_to_volume(values: np.ndarray, tumind_1based: np.ndarray, shape: Sequence[int], fill_value: float = -1.0) -> np.ndarray:
    """Map vectorized voxel stats into a dense volume using MATLAB-style 1-based indices."""
    out = np.full(tuple(int(v) for v in shape), float(fill_value), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    tumind = np.asarray(tumind_1based, dtype=np.int64).reshape(-1) - 1

    if values.shape[0] != tumind.shape[0]:
        raise ValueError(f"values length {values.shape[0]} != tumind length {tumind.shape[0]}")
    n_voxels = int(math.prod(out.shape))
    if np.any(tumind < 0) or np.any(tumind >= n_voxels):
        raise ValueError("tumind contains out-of-range voxel indices")

    out_flat = out.reshape(-1)
    out_flat[tumind] = values
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Type {type(value)} is not JSON serializable")


def _ensure_roi_names(names: Sequence[str] | None, n_rows: int) -> List[str]:
    if names is not None and len(names) == n_rows:
        return [str(v) for v in names]
    return [f"roi_{idx + 1:03d}" for idx in range(n_rows)]


def run_ftest_analysis(
    lower_model: DceFitStats,
    higher_model: DceFitStats,
    *,
    region: Region,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run f-test analysis and write reproducible output artifacts."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_ftest(lower_model, higher_model, region)
    payload: Dict[str, Any] = {
        "analysis": "ftest",
        "region": region,
        "lower_model": str(lower_model.model_name),
        "higher_model": str(higher_model.model_name),
        "mean_p": float(result["mean_p"]),
    }

    p_values = np.asarray(result["p_values"], dtype=np.float64)
    p_vector_path = output_dir / f"ftest_{region}_p_values.npy"
    np.save(p_vector_path, p_values)
    payload["p_values_vector_path"] = str(p_vector_path)

    if region == "roi":
        sse_lower, _, _ = get_sse_and_fp(lower_model, "roi")
        sse_higher, _, _ = get_sse_and_fp(higher_model, "roi")
        roi_names = _ensure_roi_names(higher_model.roi_names or lower_model.roi_names, p_values.shape[0])
        roi_csv = write_ftest_roi_csv(
            output_dir / "ftest_roi_summary.csv",
            roi_names,
            p_values,
            sse_higher,
            sse_lower,
            higher_model_name=str(higher_model.model_name),
            lower_model_name=str(lower_model.model_name),
        )
        payload["roi_csv_path"] = str(roi_csv)
    else:
        if higher_model.tumind_1based is not None and higher_model.dimensions is not None:
            p_volume = voxel_values_to_volume(p_values, higher_model.tumind_1based, higher_model.dimensions, fill_value=-1.0)
            p_volume_path = output_dir / "ftest_voxel_p_map.npy"
            np.save(p_volume_path, p_volume)
            payload["voxel_map_path"] = str(p_volume_path)

    summary_path = output_dir / f"ftest_{region}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    payload["summary_json_path"] = str(summary_path)
    return payload


def run_aic_analysis(
    models: Sequence[DceFitStats],
    *,
    region: Region,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run AIC analysis and write reproducible output artifacts."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_aic(models, region)
    payload: Dict[str, Any] = {
        "analysis": "aic",
        "region": region,
        "model_names": [str(v) for v in np.asarray(result["model_names"]).tolist()],
    }

    aic = np.asarray(result["aic"], dtype=np.float64)
    like = np.asarray(result["relative_likelihood"], dtype=np.float64)
    like12 = np.asarray(result["relative_likelihood_best_vs_second"], dtype=np.float64)

    aic_path = output_dir / f"aic_{region}_matrix.npy"
    like_path = output_dir / f"aic_{region}_relative_likelihood.npy"
    like12_path = output_dir / f"aic_{region}_best_vs_second.npy"
    np.save(aic_path, aic)
    np.save(like_path, like)
    np.save(like12_path, like12)
    payload["aic_matrix_path"] = str(aic_path)
    payload["relative_likelihood_path"] = str(like_path)
    payload["best_vs_second_path"] = str(like12_path)

    if region == "roi":
        roi_names = _ensure_roi_names(models[0].roi_names, aic.shape[0])
        roi_csv = write_aic_roi_csv(output_dir / "aic_roi_summary.csv", roi_names, result)
        payload["roi_csv_path"] = str(roi_csv)
    else:
        best_names = np.asarray(result["best_model_names"], dtype=object)
        model_names = [str(v) for v in np.asarray(result["model_names"], dtype=object).tolist()]
        name_to_idx = {name: idx + 1 for idx, name in enumerate(model_names)}
        best_index = np.asarray([name_to_idx[str(v)] for v in best_names], dtype=np.float64)
        best_index_path = output_dir / "aic_voxel_best_model_index.npy"
        np.save(best_index_path, best_index)
        payload["best_model_index_path"] = str(best_index_path)

        first = models[0]
        if first.tumind_1based is not None and first.dimensions is not None:
            best_index_vol = voxel_values_to_volume(best_index, first.tumind_1based, first.dimensions, fill_value=-1.0)
            like12_vol = voxel_values_to_volume(like12, first.tumind_1based, first.dimensions, fill_value=-1.0)
            best_index_vol_path = output_dir / "aic_voxel_best_model_index_map.npy"
            like12_vol_path = output_dir / "aic_voxel_best_vs_second_map.npy"
            np.save(best_index_vol_path, best_index_vol)
            np.save(like12_vol_path, like12_vol)
            payload["best_model_index_map_path"] = str(best_index_vol_path)
            payload["best_vs_second_map_path"] = str(like12_vol_path)

    summary_path = output_dir / f"aic_{region}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    payload["summary_json_path"] = str(summary_path)
    return payload
