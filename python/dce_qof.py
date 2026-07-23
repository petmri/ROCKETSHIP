"""Per-voxel quality-of-fit (QoF) maps from DCE post-fit arrays.

Consumes a Stage-D `*_postfit_arrays.npz` (written when `write_postfit_arrays=true`) plus, when
available, the sibling Stage-B checkpoint `checkpoints/b_out.json`, and produces per-voxel
**reduced χ²** and **noise-σ** maps using estimator B (`dce_sigma`). Reduced χ² is the first QoF
signal in `docs/project-management/projects/batch-parity/sigma_estimators.md`.

σ is estimated from `ct_voxel_mM` (per-voxel `C(t)`) by the successive-difference method, with the
bolus wash-in excised using the injection timing in `b_out.json`. Reduced χ² is then
`SSE / (σ²·(N−p))`, with SSE and `p` taken from the fit results (mirroring
`dce_postfit_analysis._MODEL_META`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dce_sigma import bolus_exclude_window, reduced_chi_square, sigma_successive_difference


# Free-parameter count `p` and 1-based SSE column per model. The first six mirror
# `dce_postfit_analysis._MODEL_META` (there keyed `fp`/`sse_col`); the last two extend it to the
# remaining pipeline models. `p` is the fitted-parameter count used in the N−p denominator.
_MODEL_PARAMS: Dict[str, Dict[str, int]] = {
    "aif": {"p": 2, "sse_col": 3},
    "tofts": {"p": 2, "sse_col": 3},
    "aif_vp": {"p": 3, "sse_col": 4},
    "ex_tofts": {"p": 3, "sse_col": 4},
    "fxr": {"p": 3, "sse_col": 4},
    "patlak": {"p": 2, "sse_col": 3},
    "tissue_uptake": {"p": 3, "sse_col": 4},
    "2cxm": {"p": 4, "sse_col": 5},
}


def bolus_window_from_checkpoint(
    checkpoint_path: Path,
    n_frames: int,
    timer_min0: float = 0.0,
    *,
    margin_frames: int = 2,
) -> Optional[Tuple[int, int]]:
    """Derive the wash-in exclude window from a Stage-B `b_out.json`.

    Uses the injection *minutes* (`start_injection_min`/`end_injection_min`) divided by
    `time_resolution_min`, not `start_time_index` (that is the analysis-window bound, not the
    bolus onset). Returns `None` if the checkpoint is missing or lacks timing.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    meta = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    dt = meta.get("time_resolution_min")
    start_min = meta.get("start_injection_min")
    end_min = meta.get("end_injection_min")
    if dt is None or start_min is None or not float(dt) > 0:
        return None
    onset = (float(start_min) - float(timer_min0)) / float(dt)
    if end_min is None:
        duration = 0.0
    else:
        duration = max(0.0, (float(end_min) - float(start_min)) / float(dt))
    return bolus_exclude_window(onset, duration, n_frames, margin_frames=margin_frames)


def compute_qof(
    npz_path: Path,
    *,
    robust: bool = True,
    margin_frames: int = 2,
    checkpoint_path: Optional[Path] = None,
    chi2_reliable_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute per-voxel σ (estimator B) and reduced χ² from a post-fit NPZ.

    Returns a dict with `model_name`, per-voxel `sigma`/`chi2v`/`sse`, `tumind`, `dims`, the
    `window` used, `n_obs`/`p`, and (if `chi2_reliable_max` given) a boolean `reliable` mask.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        model_name = str(np.asarray(data["model_name"]).reshape(-1)[0])
        ct = np.asarray(data["ct_voxel_mM"], dtype=np.float64)  # (T, V)
        timer = np.asarray(data["timer_min"], dtype=np.float64).reshape(-1)
        voxel_results = np.asarray(data["voxel_results"], dtype=np.float64)  # (V, ncols)
        tumind = np.asarray(data["tumind_0based"], dtype=np.int64).reshape(-1)
        dims = (
            tuple(int(v) for v in np.asarray(data["dimensions_xyz"]).reshape(-1).tolist())
            if "dimensions_xyz" in data
            else None
        )

    if model_name not in _MODEL_PARAMS:
        raise ValueError(f"unknown model '{model_name}'; known: {sorted(_MODEL_PARAMS)}")
    p = _MODEL_PARAMS[model_name]["p"]
    sse_col = _MODEL_PARAMS[model_name]["sse_col"]
    if sse_col - 1 >= voxel_results.shape[1]:
        raise ValueError(
            f"voxel_results has {voxel_results.shape[1]} cols; SSE col {sse_col} out of range"
        )
    sse = voxel_results[:, sse_col - 1]
    n_obs = int(ct.shape[0])

    if checkpoint_path is None:
        checkpoint_path = npz_path.parent / "checkpoints" / "b_out.json"
    window = bolus_window_from_checkpoint(
        checkpoint_path, n_obs, timer_min0=float(timer[0]) if timer.size else 0.0,
        margin_frames=margin_frames,
    )

    sigma = np.asarray(sigma_successive_difference(ct, exclude=window, robust=robust), dtype=np.float64)
    chi2v = np.asarray(reduced_chi_square(sse, sigma, n_obs=n_obs, n_params=p), dtype=np.float64)

    result: Dict[str, Any] = {
        "model_name": model_name,
        "sigma": sigma,
        "chi2v": chi2v,
        "sse": np.asarray(sse, dtype=np.float64),
        "tumind": tumind,
        "dims": dims,
        "window": window,
        "n_obs": n_obs,
        "p": p,
    }
    if chi2_reliable_max is not None:
        result["reliable"] = np.isfinite(chi2v) & (chi2v <= float(chi2_reliable_max))
    return result


def reliable_mask(
    chi2v: np.ndarray,
    *,
    percentile: Optional[float] = None,
    chi2_max: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Boolean "reliable" mask from reduced χ² plus the threshold used.

    Works on a per-voxel vector or a reconstructed volume (NaN-aware). Exactly one of
    `percentile` (computes the threshold as that percentile of the finite χ²) or `chi2_max`
    (an explicit cutoff) must be given. Reliable ⟺ finite χ² ≤ threshold.
    """
    arr = np.asarray(chi2v, dtype=np.float64)
    finite = np.isfinite(arr)
    if chi2_max is None:
        if percentile is None:
            raise ValueError("provide either percentile or chi2_max")
        chi2_max = float(np.percentile(arr[finite], percentile)) if finite.any() else float("nan")
    mask = finite & (arr <= chi2_max)
    return mask, float(chi2_max)


def qof_volumes(npz_path: Path, **kwargs: Any) -> Dict[str, Any]:
    """Compute QoF and return σ / reduced-χ² as reconstructed spatial volumes.

    `kwargs` are forwarded to :func:`compute_qof`. Volumes are NaN outside the ROI. Also returns
    the raw `compute_qof` result under `"res"` and the spatial `"dims"`.
    """
    res = compute_qof(Path(npz_path), **kwargs)
    dims = res["dims"]
    if dims is None:
        raise ValueError("post-fit NPZ lacks dimensions_xyz; cannot reconstruct volumes")
    return {
        "chi2nu": _reconstruct_volume(res["chi2v"], res["tumind"], dims),
        "sigma": _reconstruct_volume(res["sigma"], res["tumind"], dims),
        "dims": dims,
        "res": res,
    }


def _reconstruct_volume(values: np.ndarray, tumind: np.ndarray, dims: Tuple[int, ...]) -> np.ndarray:
    """Scatter a per-voxel vector into a spatial volume (Fortran order, NaN outside the ROI)."""
    coords = np.unravel_index(np.asarray(tumind, dtype=np.int64), dims, order="F")
    volume = np.full(dims, np.nan, dtype=np.float32)
    volume[coords] = np.asarray(values, dtype=np.float32)
    return volume


def write_qof_maps(
    npz_path: Path,
    out_dir: Path,
    *,
    reference_nifti: Optional[Path] = None,
    robust: bool = True,
    margin_frames: int = 2,
    checkpoint_path: Optional[Path] = None,
    chi2_reliable_max: Optional[float] = None,
    chi2_reliable_percentile: Optional[float] = None,
) -> Dict[str, str]:
    """Compute QoF and write `*_qof_{sigma,chi2nu}[,_reliable]` maps next to the fit outputs.

    Writes NIfTI when `reference_nifti` (a 3-D map to borrow affine/header from) is given, else
    `.npy`. A `qof_reliable` layer is added when `chi2_reliable_percentile` (e.g. 95.0, keep the
    best 95%) or `chi2_reliable_max` is given. Returns the written paths keyed by map name.
    """
    import nibabel as nib  # local import: only needed when writing NIfTI

    res = compute_qof(
        npz_path, robust=robust, margin_frames=margin_frames, checkpoint_path=checkpoint_path,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(npz_path).name.replace("_postfit_arrays.npz", "")

    layers = {"qof_sigma": res["sigma"], "qof_chi2nu": res["chi2v"]}
    if chi2_reliable_percentile is not None or chi2_reliable_max is not None:
        mask, _ = reliable_mask(
            res["chi2v"], percentile=chi2_reliable_percentile, chi2_max=chi2_reliable_max
        )
        layers["qof_reliable"] = mask.astype(np.float32)

    paths: Dict[str, str] = {}
    dims = res["dims"]
    ref = None
    if reference_nifti is not None and Path(reference_nifti).exists():
        ref_img = nib.load(str(reference_nifti))
        ref = (ref_img.affine, ref_img.header)

    for name, values in layers.items():
        out_base = out_dir / f"{stem}_{name}"
        if dims is not None and ref is not None:
            volume = _reconstruct_volume(values, res["tumind"], dims)
            header = ref[1].copy()
            header.set_data_dtype(np.float32)
            out_path = out_base.with_suffix(".nii.gz")
            nib.save(nib.Nifti1Image(volume, ref[0], header), str(out_path))
        elif dims is not None:
            out_path = out_base.with_suffix(".npy")
            np.save(out_path, _reconstruct_volume(values, res["tumind"], dims))
        else:
            out_path = out_base.with_suffix(".npy")
            np.save(out_path, np.asarray(values))
        paths[name] = str(out_path)
    return paths
