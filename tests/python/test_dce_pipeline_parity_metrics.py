"""Dataset-backed DCE parity checks (MATLAB map vs Python map)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np
import pytest
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _apply_model_specific_prefs,
    _stage_d_fit_prefs,
    run_dce_pipeline,
)
import dce_config  # noqa: E402
import dce_qof  # noqa: E402

# Fit budgets come from the shipped defaults file, not a literal here. A hardcoded 50 in this
# config silently decoupled the gate from the shipped configuration: `dce/dce_preferences.txt`
# and `python/dce_defaults.json` both moved to 200, so the suite was grading Python-at-50
# against a MATLAB reference built at 200. Env vars stay as a sweep escape hatch.
_PARITY_DEFAULTS = dce_config.load_defaults()

MULTI_MODEL_PARITY_SPECS = {
    "tofts": {"params": ["Ktrans", "ve"]},
    "ex_tofts": {"params": ["Ktrans", "ve", "vp"]},
    "patlak": {"params": ["Ktrans", "vp"]},
    "tissue_uptake": {"params": ["Ktrans", "fp", "vp"]},
    "2cxm": {"params": ["Ktrans", "ve", "vp", "fp"]},
}


def _parity_log(message: str) -> None:
    print(f"[PARITY] {message}", flush=True)


DOWNSAMPLE_SUBJECT = "sub-10bbbdownsample"
DOWNSAMPLE_SESSION = "ses-01"


def _dataset_paths(root: Path) -> dict:
    """Resolve DCE parity inputs within the BIDS_test sub-10bbbdownsample subject.

    ``root`` is the BIDS dataset root (``tests/data/BIDS_test``). MATLAB reference
    maps live under the ``derivatives/matlabref`` pipeline tree, keeping their
    original ``Dyn-1_*`` filenames.
    """
    raw = root / "rawdata" / DOWNSAMPLE_SUBJECT / DOWNSAMPLE_SESSION
    der = root / "derivatives" / DOWNSAMPLE_SUBJECT / DOWNSAMPLE_SESSION
    matlabref = root / "derivatives" / "matlabref" / DOWNSAMPLE_SUBJECT / DOWNSAMPLE_SESSION / "dce"
    stem = f"{DOWNSAMPLE_SUBJECT}_{DOWNSAMPLE_SESSION}"
    return {
        "root": root,
        "processed": der,
        "matlabref": matlabref,
        "dynamic": raw / "dce" / f"{stem}_DCE.nii",
        "aif": der / "dce" / f"{stem}_label-AIF_mask.nii",
        "roi": der / "anat" / f"{stem}_label-brain_mask.nii",
        "roi_gm": der / "anat" / f"{stem}_label-GM_mask.nii",
        "roi_wm": der / "anat" / f"{stem}_label-WM_mask.nii",
        "t1map": der / "anat" / f"{stem}_space-DCEref_T1map.nii",
        "noise": der / "anat" / f"{stem}_label-noise_mask.nii",
        "matlab_tofts_ktrans": matlabref / "Dyn-1_tofts_fit_Ktrans.nii",
        "matlab_tofts_ve": matlabref / "Dyn-1_tofts_fit_ve.nii",
    }


def _matlab_map_path(paths: dict, model_name: str, param: str) -> Path:
    return Path(paths["matlabref"]) / f"Dyn-1_{model_name}_fit_{param}.nii"


def _model_flags(models: list[str]) -> dict[str, int]:
    flags = {
        "tofts": 0,
        "ex_tofts": 0,
        "patlak": 0,
        "tissue_uptake": 0,
        "two_cxm": 0,
        "fxr": 0,
        "auc": 0,
        "nested": 0,
        "FXL_rr": 0,
    }
    for model in models:
        if model == "2cxm":
            flags["two_cxm"] = 1
        elif model in flags:
            flags[model] = 1
    return flags


def _default_downsample_root() -> Path:
    return REPO_ROOT / "tests/data" / "BIDS_test"


def _make_config(
    paths: dict,
    out_dir: Path,
    *,
    backend: str,
    models: list[str],
    roi_path: Path | None = None,
) -> DcePipelineConfig:
    roi_use = paths["roi"] if roi_path is None else roi_path
    stage_overrides = {
        "rootname": "Dyn-1",
        "stage_a_mode": "real",
        "stage_b_mode": "real",
        "stage_d_mode": "real",
        "tr_ms": 8.29,
        "fa_deg": 15.0,
        "time_resolution_sec": 15.84,
        # Must match A_make_R1maps_func.m, which calls find_end_ss_tv.
        "steady_state_auto_method": "tv",
        "auto_find_injection": 1,
        "relaxivity": 3.6,
        "hematocrit": 0.42,
        "snr_filter": 5.0,
        "time_smoothing": "none",
        "time_smoothing_window": 0,
        "voxel_MaxFunEvals": int(
            os.environ.get("ROCKETSHIP_PARITY_VOXEL_MAXFUNEVALS")
            or _PARITY_DEFAULTS.default_for("voxel_MaxFunEvals")
        ),
        "voxel_MaxIter": int(
            os.environ.get("ROCKETSHIP_PARITY_VOXEL_MAXITER")
            or _PARITY_DEFAULTS.default_for("voxel_MaxIter")
        ),
    }

    numeric_override_keys = {
        "ROCKETSHIP_PARITY_VOXEL_LOWER_LIMIT_KTRANS": "voxel_lower_limit_ktrans",
        "ROCKETSHIP_PARITY_VOXEL_UPPER_LIMIT_KTRANS": "voxel_upper_limit_ktrans",
        "ROCKETSHIP_PARITY_VOXEL_INITIAL_VALUE_KTRANS": "voxel_initial_value_ktrans",
        "ROCKETSHIP_PARITY_VOXEL_LOWER_LIMIT_VE": "voxel_lower_limit_ve",
        "ROCKETSHIP_PARITY_VOXEL_UPPER_LIMIT_VE": "voxel_upper_limit_ve",
        "ROCKETSHIP_PARITY_VOXEL_INITIAL_VALUE_VE": "voxel_initial_value_ve",
        "ROCKETSHIP_PARITY_VOXEL_LOWER_LIMIT_VP": "voxel_lower_limit_vp",
        "ROCKETSHIP_PARITY_VOXEL_UPPER_LIMIT_VP": "voxel_upper_limit_vp",
        "ROCKETSHIP_PARITY_VOXEL_INITIAL_VALUE_VP": "voxel_initial_value_vp",
        "ROCKETSHIP_PARITY_VOXEL_LOWER_LIMIT_FP": "voxel_lower_limit_fp",
        "ROCKETSHIP_PARITY_VOXEL_UPPER_LIMIT_FP": "voxel_upper_limit_fp",
        "ROCKETSHIP_PARITY_VOXEL_INITIAL_VALUE_FP": "voxel_initial_value_fp",
        "ROCKETSHIP_PARITY_VOXEL_LOWER_LIMIT_TP": "voxel_lower_limit_tp",
        "ROCKETSHIP_PARITY_VOXEL_UPPER_LIMIT_TP": "voxel_upper_limit_tp",
        "ROCKETSHIP_PARITY_VOXEL_INITIAL_VALUE_TP": "voxel_initial_value_tp",
        "ROCKETSHIP_PARITY_GPU_TOLERANCE": "gpu_tolerance",
        "ROCKETSHIP_PARITY_GPU_MAX_N_ITERATIONS": "gpu_max_n_iterations",
        # Model-specific tuning (applies only to the named model in pipeline).
        "ROCKETSHIP_PARITY_2CXM_LOWER_LIMIT_KTRANS": "voxel_lower_limit_ktrans_2cxm",
        "ROCKETSHIP_PARITY_2CXM_UPPER_LIMIT_KTRANS": "voxel_upper_limit_ktrans_2cxm",
        "ROCKETSHIP_PARITY_2CXM_INITIAL_VALUE_KTRANS": "voxel_initial_value_ktrans_2cxm",
        "ROCKETSHIP_PARITY_2CXM_LOWER_LIMIT_VE": "voxel_lower_limit_ve_2cxm",
        "ROCKETSHIP_PARITY_2CXM_UPPER_LIMIT_VE": "voxel_upper_limit_ve_2cxm",
        "ROCKETSHIP_PARITY_2CXM_INITIAL_VALUE_VE": "voxel_initial_value_ve_2cxm",
        "ROCKETSHIP_PARITY_2CXM_LOWER_LIMIT_VP": "voxel_lower_limit_vp_2cxm",
        "ROCKETSHIP_PARITY_2CXM_UPPER_LIMIT_VP": "voxel_upper_limit_vp_2cxm",
        "ROCKETSHIP_PARITY_2CXM_INITIAL_VALUE_VP": "voxel_initial_value_vp_2cxm",
        "ROCKETSHIP_PARITY_2CXM_LOWER_LIMIT_FP": "voxel_lower_limit_fp_2cxm",
        "ROCKETSHIP_PARITY_2CXM_UPPER_LIMIT_FP": "voxel_upper_limit_fp_2cxm",
        "ROCKETSHIP_PARITY_2CXM_INITIAL_VALUE_FP": "voxel_initial_value_fp_2cxm",
        "ROCKETSHIP_PARITY_2CXM_MAXFUNEVALS": "voxel_MaxFunEvals_2cxm",
        "ROCKETSHIP_PARITY_2CXM_MAXITER": "voxel_MaxIter_2cxm",
        "ROCKETSHIP_PARITY_TISSUE_LOWER_LIMIT_KTRANS": "voxel_lower_limit_ktrans_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_UPPER_LIMIT_KTRANS": "voxel_upper_limit_ktrans_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_INITIAL_VALUE_KTRANS": "voxel_initial_value_ktrans_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_LOWER_LIMIT_VP": "voxel_lower_limit_vp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_UPPER_LIMIT_VP": "voxel_upper_limit_vp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_INITIAL_VALUE_VP": "voxel_initial_value_vp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_LOWER_LIMIT_FP": "voxel_lower_limit_fp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_UPPER_LIMIT_FP": "voxel_upper_limit_fp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_INITIAL_VALUE_FP": "voxel_initial_value_fp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_LOWER_LIMIT_TP": "voxel_lower_limit_tp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_UPPER_LIMIT_TP": "voxel_upper_limit_tp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_INITIAL_VALUE_TP": "voxel_initial_value_tp_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_MAXFUNEVALS": "voxel_MaxFunEvals_tissue_uptake",
        "ROCKETSHIP_PARITY_TISSUE_MAXITER": "voxel_MaxIter_tissue_uptake",
    }
    for env_key, override_key in numeric_override_keys.items():
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
            if value.is_integer():
                value = int(value)
            stage_overrides[override_key] = value
        except ValueError:
            pass

    text_override_keys = {
        "ROCKETSHIP_PARITY_2CXM_ROBUST": "voxel_Robust_2cxm",
        "ROCKETSHIP_PARITY_TISSUE_ROBUST": "voxel_Robust_tissue_uptake",
    }
    for env_key, override_key in text_override_keys.items():
        raw = os.environ.get(env_key, "").strip()
        if raw:
            stage_overrides[override_key] = raw

    return DcePipelineConfig(
        subject_source_path=paths["root"],
        subject_tp_path=paths["processed"],
        output_dir=out_dir,
        backend=backend,
        aif_mode="fitted",
        checkpoint_dir=out_dir / "checkpoints",
        write_xls=True,
        dynamic_files=[paths["dynamic"]],
        aif_files=[paths["aif"]],
        roi_files=[roi_use],
        t1map_files=[paths["t1map"]],
        noise_files=[paths["noise"]],
        model_flags=_model_flags(models),
        stage_overrides=stage_overrides,
    )


def _make_tofts_post_8ef4988_config(paths: dict, out_dir: Path, *, backend: str) -> DcePipelineConfig:
    # _make_config's defaults already auto-detect steady-state end + injection timing
    # (the "post-8ef4988 timing policy"), which is now all this tofts-only runtime-parity
    # comparison needs: the Stage-B AIF fit always holds t_base_end at the resolved baseline
    # end and always fits the upslope duration, so there is no timing method left to layer on.
    return _make_config(paths, out_dir, backend=backend, models=["tofts"])


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(np.squeeze(nib.load(str(path)).get_fdata()), dtype=np.float64)


def _maybe_load_nifti(path: Path) -> np.ndarray | None:
    return _load_nifti(path) if Path(path).exists() else None


def _load_param_ci(py_dir: Path, paths: dict, model_name: str, param: str) -> dict | None:
    """Load MATLAB + Python CI (low/high) maps for a parameter, or None if any is absent.

    CI maps use the lowercase parameter base (e.g. `ktrans_ci_low`, `ve_ci_low`). Returns the
    kwargs dict consumed by `_ci_metrics` (reported-only; tolerant of missing maps).
    """
    base = param.lower()
    m_lo = _maybe_load_nifti(_matlab_map_path(paths, model_name, f"{base}_ci_low"))
    m_hi = _maybe_load_nifti(_matlab_map_path(paths, model_name, f"{base}_ci_high"))
    p_lo = _maybe_load_nifti(py_dir / f"Dyn-1_{model_name}_fit_{base}_ci_low.nii.gz")
    p_hi = _maybe_load_nifti(py_dir / f"Dyn-1_{model_name}_fit_{base}_ci_high.nii.gz")
    if all(a is not None for a in (m_lo, m_hi, p_lo, p_hi)):
        return {"matlab_ci_low": m_lo, "matlab_ci_high": m_hi, "py_ci_low": p_lo, "py_ci_high": p_hi}
    return None


def _write_union_roi_mask(reference_roi_path: Path, member_paths: list[Path], dst_roi_path: Path) -> None:
    """Write a binary ROI = union of `member_paths` (each voxel>0), using the reference header.

    The pipeline only fits voxels inside the ROI it is given, so the run ROI must cover every
    region we later evaluate (sparse brain + dense GM + dense WM); otherwise GM/WM voxels are
    never fit and their maps read as background.
    """
    ref_img = nib.load(str(reference_roi_path))
    shape = np.squeeze(ref_img.get_fdata()).shape
    union = np.zeros(shape, dtype=bool)
    for member in member_paths:
        data = np.asarray(np.squeeze(nib.load(str(member)).get_fdata()), dtype=np.float64)
        union |= data > 0
    if not union.any():
        raise AssertionError("Union ROI mask has no voxels")
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(union.astype(np.float32), ref_img.affine, header), str(dst_roi_path))


def _write_sparse_roi_mask(src_roi_path: Path, dst_roi_path: Path, stride: int) -> None:
    roi_img = nib.load(str(src_roi_path))
    roi_data = np.asarray(np.squeeze(roi_img.get_fdata()), dtype=np.float32)
    src_mask = roi_data > 0
    src_flat = src_mask.reshape(-1, order="F")
    indices = np.flatnonzero(src_flat)
    if indices.size == 0:
        raise AssertionError(f"ROI mask has no voxels: {src_roi_path}")

    stride_use = max(1, int(stride))
    keep = indices[::stride_use]
    if keep.size < min(64, indices.size):
        keep = indices

    subset_flat = np.zeros_like(src_flat, dtype=np.float32)
    subset_flat[keep] = 1.0
    subset = subset_flat.reshape(src_mask.shape, order="F")

    header = roi_img.header.copy()
    header.set_data_dtype(np.float32)
    subset_img = nib.Nifti1Image(subset, roi_img.affine, header)
    nib.save(subset_img, str(dst_roi_path))


def _metrics(
    py_map: np.ndarray,
    matlab_map: np.ndarray,
    roi_mask: np.ndarray,
    extra_mask: np.ndarray | None = None,
) -> dict:
    mask = np.isfinite(py_map) & np.isfinite(matlab_map) & (roi_mask > 0)
    if extra_mask is not None:
        mask = mask & np.asarray(extra_mask, dtype=bool)
    x = py_map[mask]
    y = matlab_map[mask]
    if x.size < 2:
        raise AssertionError("Too few voxels for parity metrics")
    diff = x - y
    mse = float(np.mean(diff * diff))
    # Rank (Spearman), not Pearson: Pearson is a sum-of-products statistic, so a single
    # high-leverage voxel (e.g. a non-identifiable fit pinned at a parameter bound) can
    # collapse it even though every other voxel agrees closely -- observed on real fixtures
    # (patlak brain corr ~-0.007 from one degenerate-seed voxel out of 237; see
    # docs/project-management/projects/archived/batch-parity/batch_parity.md, "Tabled" section, and
    # the same fix already applied to tests/contracts/check_matlabref_map_drift.py). A
    # genuine algorithm change still collapses Spearman to ~0/negative.
    corr = float(spearmanr(x, y).correlation) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    # Scatter is gated on `nrmse`, RMSE over the *reference* RMS, not on absolute RMSE. These
    # parameters span two orders of magnitude across models (reference RMS runs 0.0012 for
    # patlak Ktrans to 0.058 for ex_tofts ve), so one absolute bound cannot be honest for all
    # of them: an rmse_max of 0.02 -- the tightest value the tofts numbers allowed -- would have
    # admitted a 1737% error on patlak Ktrans. Normalizing makes a proportional error read as
    # itself, and matches how test_backend_equivalence.py measures the same kind of scatter.
    rmse = float(np.sqrt(mse))
    ref_rms = float(np.sqrt(np.mean(y * y)))
    return {
        "n": int(x.size),
        "corr": corr,
        "mse": mse,
        "rmse": rmse,
        "ref_rms": ref_rms,
        "nrmse": (rmse / ref_rms) if ref_rms > 0 else float("nan"),
    }


def _ci_metrics(
    py_map: np.ndarray,
    matlab_map: np.ndarray,
    roi_mask: np.ndarray,
    *,
    matlab_ci_low: np.ndarray,
    matlab_ci_high: np.ndarray,
    py_ci_low: np.ndarray | None = None,
    py_ci_high: np.ndarray | None = None,
    extra_mask: np.ndarray | None = None,
) -> dict:
    """Confidence-interval-aware parity metrics (reported-only, never gated).

    Both MATLAB and Python report a 95% CI, so CI widths are directly comparable.
    Returns the CI-normalized absolute difference (median + p95) using the MATLAB CI
    as the denominator, and the proportion of voxels falling outside the other side's CI.

    Zero-width intervals are excluded from every field here (see below); `n_zero_ci_width`
    and `n_zero_py_ci_width` report how many there were, so a degenerate reference shows up
    as a count rather than as fake agreement or fake disagreement.

    HISTORY (resolved 2026-07-23): MATLAB's CI maps on the sub-10bbbdownsample fixture were
    zero-width for every voxel and every model, which made these metrics non-functional. The
    cause was that commit a9d78b6 regenerated the baseline on a GPU machine, and the gpufit
    path zero-pads CI columns (FXLfit_generic.m) -- only the CPU fit()/confint() path produces
    real intervals. Fixed by regenerating with force_cpu=1; the generator now refuses to run
    otherwise. The maps carry real widths today and these fields are live.

    Still reported-only, never gated: the fields are diagnostics, and no threshold for them
    has been calibrated.
    """
    mask = (
        np.isfinite(py_map)
        & np.isfinite(matlab_map)
        & np.isfinite(matlab_ci_low)
        & np.isfinite(matlab_ci_high)
        & (roi_mask > 0)
    )
    if extra_mask is not None:
        mask = mask & np.asarray(extra_mask, dtype=bool)
    x = py_map[mask]
    y = matlab_map[mask]
    lo = matlab_ci_low[mask]
    hi = matlab_ci_high[mask]
    width = hi - lo
    positive_width = width > 0
    out: dict = {
        "n": int(x.size),
        "n_zero_ci_width": int(np.count_nonzero(~positive_width)),
    }
    abs_diff = np.abs(x - y)
    if np.any(positive_width):
        norm = abs_diff[positive_width] / width[positive_width]
        out["ci_norm_absdiff_median"] = float(np.median(norm))
        out["ci_norm_absdiff_p95"] = float(np.percentile(norm, 95.0))
    else:
        out["ci_norm_absdiff_median"] = float("nan")
        out["ci_norm_absdiff_p95"] = float("nan")
    # Restrict to positive-width intervals, as ci_norm_absdiff already does. A zero-width
    # "interval" makes almost any value trivially "outside" it, which turns a degenerate CI
    # into a fake ~1.0 disagreement rather than the missing datum it actually is.
    if np.any(positive_width):
        out["prop_py_outside_matlab_ci"] = float(
            np.mean((x[positive_width] < lo[positive_width]) | (x[positive_width] > hi[positive_width]))
        )
    else:
        out["prop_py_outside_matlab_ci"] = float("nan")
    if py_ci_low is not None and py_ci_high is not None:
        plo = py_ci_low[mask]
        phi = py_ci_high[mask]
        py_positive_width = (phi - plo) > 0
        out["n_zero_py_ci_width"] = int(np.count_nonzero(~py_positive_width))
        if np.any(py_positive_width):
            out["prop_matlab_outside_py_ci"] = float(
                np.mean((y[py_positive_width] < plo[py_positive_width]) | (y[py_positive_width] > phi[py_positive_width]))
            )
        else:
            out["prop_matlab_outside_py_ci"] = float("nan")
    return out


def _write_parity_summary(summary_dir: Path | None, file_name: str, payload: dict) -> None:
    if summary_dir is None:
        return
    out_path = summary_dir / file_name
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _parity_log(f"summary_json={out_path}")


def _load_roi_xls_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = ""

    if "\t" in text and "\n" in text:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise AssertionError(f"ROI table appears empty: {path}")
        header = [cell.strip() for cell in lines[0].split("\t")]
        rows = [[cell.strip() for cell in line.split("\t")] for line in lines[1:]]
        return header, rows

    try:
        import xlrd  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/config check
        pytest.skip(f"xlrd is required to read binary .xls ROI tables ({exc})")

    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    header = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
    rows: list[list[str]] = []
    for r in range(1, sheet.nrows):
        row: list[str] = []
        for c in range(sheet.ncols):
            value = sheet.cell_value(r, c)
            if isinstance(value, float):
                row.append(f"{value:.17g}")
            else:
                row.append(str(value).strip())
        rows.append(row)
    return header, rows


def _normalized_roi_header(header: list[str]) -> list[str]:
    out: list[str] = []
    for cell in header:
        text = str(cell).strip()
        if text.lower() == "residual":
            out.append("SSE")
        else:
            out.append(text)
    return out


def _canonical_roi_token(name: str) -> str:
    """Reduce an ROI label to its tissue token so BIDS-style mask filenames align with
    the MATLAB reference's ``T1_*_roi`` names (e.g. ``...label-brain_mask`` -> ``brain``)."""
    text = str(name).strip().lower()
    for token in ("brain", "gm", "wm", "aif", "noise"):
        if token in text:
            return token
    return text


# One absolute limit for every model and every column of the ROI-xls gate. It replaced four
# hand-tuned per-model limits (tofts 0.03, ex_tofts/patlak 0.01, tissue_uptake 0.05) that had
# drifted into 20-770x headroom as the Stage-B AIF work landed. Re-measured 2026-07-29, worst
# column per model: tofts 0.001440 (Ktrans 95% high), ex_tofts 0.000013, patlak 0.000013,
# tissue_uptake 0.002235 (excluding Fp, below) -- so this single value is ~7x the worst case
# and is a real tightening for three of the four.
ROI_XLS_MAX_ABS_ERR = 0.01

# Columns excluded from the ROI-xls gate, per model. This is the suite's one remaining
# hand-curated exclusion, and it is the same finding that keeps tissue_uptake out of the
# voxelwise gated set (UNGATED_MODEL_REASONS above), not a second independent exception.
#
# tissue_uptake's Fp (plasma flow) is not reliably estimable on this fixture: at 15.84 s frames
# the bolus rise occupies a single sample, and Fp is determined almost entirely by that leading
# edge. It is also the parameter most exposed to how the AIF's peak is fitted, which is exactly
# the thing the data cannot pin down (the peak has leverage 1 in the biexponential model -- see
# docs/project-management/projects/archived/batch-parity/aif_fitting_parity.md). Gating on Fp
# measures the fixture's temporal resolution, not the port's correctness.
#
# Re-measured 2026-07-29 and still precisely scoped: Fp (0.073345) and its two CI columns
# (0.055581, 0.091109) are the *only* tissue_uptake columns above the limit; the worst of all
# the others is Vp 95% high at 0.002235.
ROI_XLS_EXCLUDED_COLUMNS = {
    "tissue_uptake": ("fp", "fp 95% low", "fp 95% high"),
}


def _compare_roi_table_against_reference(
    *,
    model_name: str,
    py_path: Path,
    ref_path: Path,
    max_abs_err: float,
) -> dict:
    py_header_raw, py_rows = _load_roi_xls_rows(py_path)
    ref_header_raw, ref_rows = _load_roi_xls_rows(ref_path)
    py_header = _normalized_roi_header(py_header_raw)
    ref_header = _normalized_roi_header(ref_header_raw)

    assert py_header == ref_header, (
        f"{model_name}: ROI XLS header mismatch\n"
        f"python={py_header}\n"
        f"ref={ref_header}"
    )
    assert len(py_rows) == len(ref_rows), (
        f"{model_name}: ROI XLS row-count mismatch: python={len(py_rows)} ref={len(ref_rows)}"
    )
    assert len(py_rows) > 0, f"{model_name}: ROI XLS has no data rows"

    excluded = {c.strip().lower() for c in ROI_XLS_EXCLUDED_COLUMNS.get(model_name, ())}
    value_names = [str(c).strip().lower() for c in py_header[2:]]
    keep_mask = np.asarray([name not in excluded for name in value_names], dtype=bool)
    dropped = sorted(excluded.intersection(value_names))
    if dropped:
        _parity_log(f"{model_name}_roi_xls: excluding column(s) {dropped} from the gate")

    abs_errors: list[float] = []
    for row_idx, (py_row, ref_row) in enumerate(zip(py_rows, ref_rows)):
        assert len(py_row) == len(ref_row), (
            f"{model_name}: row length mismatch at row {row_idx + 1}: "
            f"python={len(py_row)} ref={len(ref_row)}"
        )

        py_roi_name = str(py_row[1]).strip()
        ref_roi_name = str(ref_row[1]).strip()
        assert _canonical_roi_token(py_roi_name) == _canonical_roi_token(ref_roi_name), (
            f"{model_name}: ROI tissue mismatch at row {row_idx + 1}: "
            f"python={py_roi_name!r} ref={ref_roi_name!r}"
        )

        py_vals = np.asarray([float(v) for v in py_row[2:]], dtype=np.float64)[keep_mask]
        ref_vals = np.asarray([float(v) for v in ref_row[2:]], dtype=np.float64)[keep_mask]
        both_nan = np.isnan(py_vals) & np.isnan(ref_vals)
        both_finite = np.isfinite(py_vals) & np.isfinite(ref_vals)
        valid = both_nan | both_finite
        assert bool(np.all(valid)), (
            f"{model_name}: non-matching NaN/finite state in ROI row {row_idx + 1}"
        )

        if np.any(both_finite):
            diff = np.abs(py_vals[both_finite] - ref_vals[both_finite])
            abs_errors.extend(diff.tolist())

    max_err = float(np.max(abs_errors)) if abs_errors else 0.0
    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    summary = (
        f"{model_name}_roi_xls: rows={len(py_rows)}, mae={mae:.6f}, "
        f"max_abs_err={max_err:.6f}"
    )
    _parity_log(summary)
    assert max_err <= float(max_abs_err), f"{summary} (max_abs_err_limit={max_abs_err})"
    return {"rows": int(len(py_rows)), "mae": mae, "max_abs_err": max_err}


# Standard suite: gated Python-vs-MATLAB parity on Tofts, Patlak and ex-Tofts. Runs by default
# (no flag). `--parity-suite=allmodels` additionally runs tissue_uptake/2cxm as reported-only
# diagnostics.
STANDARD_PARITY_MODELS = ["tofts", "patlak", "ex_tofts"]
ALLMODELS_EXTRA = ["tissue_uptake", "2cxm"]

# Gate policy, reviewed 2026-07-29, extended 2026-08-12. One rule applied uniformly: every
# parameter of every gated model is gated, over every region, against one threshold pair, after
# one identifiability filter. No per-model, per-parameter or per-region exceptions.
#
# Since 2026-08-12 the gate is three-part, matching tests/python/test_backend_equivalence.py:
#   1. Parameter agreement on the identifiable subset -- voxels where neither side pinned a
#      bound and neither settled in a different basin of an equally good fit.
#   2. SSE agreement over *every* voxel, pinned and tied included. Python may choose a
#      different optimum; it may not choose a worse one.
#   3. Bound-hit accounting, reported per parameter, plus the collapse guard below.
# Part 2 is what makes part 1's exclusions safe: nothing leaves the comparison entirely.
GATED_MODELS = {"tofts", "patlak", "ex_tofts"}

# The two models that stay reported-only, and the measurement that puts them there. Both were
# re-measured with the identifiability filter below already applied, so neither is a
# bound-pinning artifact -- these are properties of the fixture, not of the port.
UNGATED_MODEL_REASONS = {
    "tissue_uptake": (
        "Fp is not identifiable at this fixture's 15.84 s frames: the bolus rise occupies a "
        "single sample and Fp is set almost entirely by that leading edge. Filtered corr is "
        "0.08 (WM) / 0.18 (GM). Ktrans and vp inherit it -- the model fits E=Ktrans/Fp, so an "
        "undetermined Fp propagates into both. Same root cause as the ROI-xls Fp exclusion "
        "(aif_fitting_parity.md S8)."
    ),
    "2cxm": (
        "The identifiable subset collapses: 0/57 GM, 9/119 WM and 26/222 brain voxels survive "
        "the bound filter, i.e. almost every 2cxm fit pins a parameter. There is not enough "
        "determined signal on this fixture to gate on."
    ),
}

# Identifiability filter. A voxel is comparable for a model only if NEITHER side left ANY of
# that model's compared parameters sitting on a bound: against a bound the objective is flat, so
# two optimizers stop at different points of one plateau and the disagreement measures the
# constraint rather than the port. Identical rule to tests/python/test_backend_equivalence.py.
#
# This replaced two hand-rolled masks that each covered one corner of it -- `ktrans_upper_exclude`
# (Ktrans near its 2.0 ceiling, applied only to ex_tofts/2cxm) and `ve_ktrans_min` (Ktrans at its
# 1e-7 floor, applied only to ve). The partial masks were why ex_tofts was ungateable: with the
# full filter its worst check goes 0.807 -> 0.9998 (Ktrans/GM/cpu) and 0.765 -> 0.9998
# (Ktrans/GM/auto). It also removes padding -- 60 of tofts' 229 brain voxels sat on ve's 0.02
# floor and agreed trivially (corr 0.9952), inflating tofts Ktrans/brain from a true 0.9616.
BOUND_REL_TOL = 1e-6
# Below this the filter has eaten the comparison and a "pass" would mean nothing, so gated checks
# fail rather than pass quietly. Observed minimum on gated models is 0.578 (ex_tofts/brain).
IDENTIFIABLE_FRACTION_MIN = 0.25

# Local-minimum ties. Bound-pinning above catches a flat objective pressed against a *constraint*;
# this catches the same flatness in the *interior*, where two optimizers settle in different basins
# of an equally good (or better) fit. Requiring agreement there would demand bug-compatibility with
# MATLAB's choice of local minimum, which is not what the port owes.
#
# Real case that motivated it (sub-10bbbdownsample, ex_tofts, GM voxel (23,18)): MATLAB Ktrans
# 0.021699 against Python 0.030951 -- 43% apart -- while Python's SSE is 2.4% *lower* and its value
# sits inside MATLAB's own 95% CI [0.006894, 0.036503]. One such voxel in a 39-voxel region drove
# nrmse to 0.175 (gate 0.25) while contributing 100.0% of the squared error; every other GM voxel
# agreed to 0.000000. It only became visible once `voxel_MaxFunEvals` 50 -> 200 stopped MATLAB
# truncating that fit with ve pinned on its 0.02 floor, which had been hiding it in the bound mask.
#
# This exclusion is only sound because SSE agreement (below) is gated over *every* voxel, including
# the ones excluded here -- so a port defect that produces genuinely worse fits still fails, and
# `IDENTIFIABLE_FRACTION_MIN` still fails a check whose comparable subset has collapsed.
SSE_TIE_REL_TOL = 1e-3
# A tie means "different basin", not "same basin, slightly different stopping point", so the
# parameter difference has to be large before flatness is the explanation. At 1e-3 this mask
# swallowed the accelerated comparison whole -- 302 of tofts' auto-vs-matlab voxels qualified on
# sub-percent jitter and the identifiable fraction collapsed to 0.042, which the collapse guard
# correctly failed. The motivating voxel differs by 43%.
TIE_PARAM_REL_TOL = 0.25

# SSE agreement, gated over all voxels. This is the load-bearing half of the arrangement above:
# the port may pick a different basin, but must never fit the data *worse* than MATLAB.
#
# Thresholds are per-backend. `cpu` is the reference implementation and is held tight. `auto`
# carries the accelerator's documented speed/accuracy tradeoff -- cpufit stops on `gpu_tolerance`
# and so leaves some voxels marginally short of MATLAB's optimum -- and its looser bound matches
# tests/python/test_backend_equivalence.py, which accepts the same tradeoff cpu-vs-cpufit at
# SSE_REL_MEDIAN_MAX = 1e-3.
#
# Pearson correlation is deliberately NOT part of this gate. On a heavy-tailed SSE distribution it
# is dominated by single voxels: ex_tofts/brain/cpu reads corr 0.9975 while its relative median is
# 5.9e-8 and *no* voxel fits worse. That is the same reason check_matlabref_map_drift.py gates on
# rank correlation instead. Relative median and worse-fraction are both robust, and both say
# directly what this gate exists to say.
SSE_REL_MEDIAN_MAX = {"cpu": 1e-6, "auto": 1e-3}   # observed <= 6.0e-8 (cpu) / 8.2e-5 (auto)
SSE_WORSE_FRAC_MAX = {"cpu": 0.02, "auto": 0.30}   # observed 0.000 (cpu) / 0.193 (auto)


def _at_bound(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = max(hi - lo, 1e-12)
    return (np.abs(values - lo) <= BOUND_REL_TOL * span) | (np.abs(values - hi) <= BOUND_REL_TOL * span)


def _identifiable_mask(
    params: list[str],
    py_maps: dict[str, np.ndarray],
    matlab_maps: dict[str, np.ndarray],
    prefs: dict,
) -> tuple[np.ndarray, dict[str, int]]:
    """Voxels where neither side pinned any compared parameter at one of its bounds.

    Bounds are read from the pipeline's own fit prefs so the filter tracks the fitter instead of
    a second copy of the numbers. For the three gated models those are exactly the values in
    `dce/dce_preferences.txt`, so the same thresholds legitimately apply to the MATLAB maps.
    (`2cxm` and `tissue_uptake` carry Python-side model-specific overrides that MATLAB has no
    equivalent for -- e.g. ve floor 0.05 vs 0.02 -- so the filter is slightly conservative on
    those two. Both are reported-only, so this affects a diagnostic, not a gate.)
    """
    pinned = np.zeros(np.shape(matlab_maps[params[0]]), dtype=bool)
    per_param: dict[str, int] = {}
    for param in params:
        key = param.lower()
        lo = float(prefs[f"lower_limit_{key}"])
        hi = float(prefs[f"upper_limit_{key}"])
        hit = _at_bound(py_maps[param], lo, hi) | _at_bound(matlab_maps[param], lo, hi)
        per_param[param] = int(np.count_nonzero(hit))
        pinned |= hit
    return ~pinned, per_param


def _local_minimum_tie_mask(
    params: list[str],
    lhs_maps: dict[str, np.ndarray],
    rhs_maps: dict[str, np.ndarray],
    lhs_sse: np.ndarray,
    rhs_sse: np.ndarray,
) -> np.ndarray:
    """Voxels where `lhs` reached an equal-or-better fit than `rhs` at different parameters.

    Both conditions are required. Equal SSE with equal parameters is the ordinary agreeing case
    and stays in the comparison; worse SSE stays in too, and fails, which is the point.
    """
    not_worse = np.asarray(lhs_sse) <= np.asarray(rhs_sse) * (1.0 + SSE_TIE_REL_TOL)
    differs = np.zeros(np.shape(rhs_maps[params[0]]), dtype=bool)
    for param in params:
        a = np.asarray(lhs_maps[param], dtype=np.float64)
        b = np.asarray(rhs_maps[param], dtype=np.float64)
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
        differs |= np.abs(a - b) > TIE_PARAM_REL_TOL * scale
    return np.asarray(not_worse, dtype=bool) & differs


def _sse_agreement(lhs_sse: np.ndarray, rhs_sse: np.ndarray, mask: np.ndarray) -> dict:
    """Fit-quality agreement over every voxel in `mask`, pinned and tied ones included."""
    ok = np.asarray(mask, dtype=bool) & np.isfinite(lhs_sse) & np.isfinite(rhs_sse) & (rhs_sse > 0)
    a = np.asarray(rhs_sse, dtype=np.float64)[ok]
    b = np.asarray(lhs_sse, dtype=np.float64)[ok]
    if a.size < 2:
        return {"n": int(a.size), "corr": float("nan"), "rel_median": float("nan"),
                "worse_frac": float("nan")}
    rel = np.abs(a - b) / a
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
    return {
        "n": int(a.size),
        "corr": corr,
        "rel_median": float(np.median(rel)),
        # Fraction where the left side fits materially worse than the right.
        "worse_frac": float(np.mean(b > a * (1.0 + SSE_TIE_REL_TOL))),
    }


# QoF filtering (sigma_estimators.md): exclude voxels whose Python-CPU reduced χ² exceeds this
# ABSOLUTE cutoff (residuals > τ× the estimated noise variance), per model, using the CPU (reference)
# backend's χ² for every check. τ=6.0 was calibrated on sub-10bbbdownsample: χ² positively tracks
# cross-backend divergence (Spearman ≈0.29), τ=6 removes the ~8% clearly-anomalous tail (χ²≫median~1.7,
# up to ~500) while retaining ~92%, and — unlike a percentile (p95 ranged 5.8–8.5 across ROI sizes) —
# is stable regardless of how many voxels are evaluated. Set env ROCKETSHIP_PARITY_QOF_CHI2_MAX<=0 to
# disable.
QOF_CHI2_MAX = float(os.environ.get("ROCKETSHIP_PARITY_QOF_CHI2_MAX", "6.0"))


@pytest.mark.parity
@pytest.mark.integration
def test_bbb_p19_region_parity(
    parity_suite: set[str],
    parity_dataset_root: str,
    parity_roi_stride: int,
    parity_summary_dir: Path | None,
    parity_thresholds: dict,
) -> None:
    root = Path(parity_dataset_root) if parity_dataset_root else _default_downsample_root()
    paths = _dataset_paths(root)

    models = [m for m in STANDARD_PARITY_MODELS if m in MULTI_MODEL_PARITY_SPECS]
    if "allmodels" in parity_suite:
        models += [m for m in ALLMODELS_EXTRA if m in MULTI_MODEL_PARITY_SPECS]

    # Default-on: skip gracefully (not fail) if this environment lacks the fixture assets.
    required_assets = [paths["roi"], paths["roi_gm"], paths["roi_wm"]] + [
        _matlab_map_path(paths, model_name, param)
        for model_name in models
        for param in MULTI_MODEL_PARITY_SPECS[model_name]["params"]
    ]
    missing = [str(p) for p in required_assets if not Path(p).exists()]
    if missing:
        pytest.skip(f"parity fixture assets missing ({len(missing)}); first: {missing[0]}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_cpu = tmp_path / "python_out_cpu"
        out_auto = tmp_path / "python_out_auto"
        sparse_roi_path = tmp_path / "roi_sparse.nii.gz"
        roi_stride = parity_roi_stride
        _parity_log(
            "starting multi-model backend parity: "
            f"root={paths['root']} roi_stride={roi_stride} models={models}"
        )
        _write_sparse_roi_mask(paths["roi"], sparse_roi_path, roi_stride)
        # Run ROI must cover every region we evaluate so GM/WM voxels are actually fit.
        run_roi_path = tmp_path / "roi_run_union.nii.gz"
        _write_union_roi_mask(
            paths["roi"], [sparse_roi_path, paths["roi_gm"], paths["roi_wm"]], run_roi_path
        )

        cpu_cfg = _make_config(paths, out_cpu, backend="cpu", models=models, roi_path=run_roi_path)
        # QoF filtering reads the per-voxel post-fit arrays (C(t) + SSE) from the CPU reference run.
        cpu_cfg.stage_overrides = {**cpu_cfg.stage_overrides, "write_postfit_arrays": True}
        cpu_result = run_dce_pipeline(cpu_cfg)
        auto_result = run_dce_pipeline(
            _make_config(paths, out_auto, backend="auto", models=models, roi_path=run_roi_path)
        )
        assert cpu_result["meta"]["status"] == "ok"
        assert auto_result["meta"]["status"] == "ok"

        # Evaluation regions. Brain is the sparse whole-brain diagnostic; GM/WM are dense
        # curated tissue. Gated checks run across all three regions.
        regions = {
            "brain": _load_nifti(sparse_roi_path) > 0,
            "gm": _load_nifti(paths["roi_gm"]) > 0,
            "wm": _load_nifti(paths["roi_wm"]) > 0,
        }

        # One pair for every gated check, whatever the model, parameter or region.
        gate_corr_min = float(parity_thresholds["gate_corr_min"])
        gate_nrmse_max = float(parity_thresholds["gate_nrmse_max"])

        # Gated scope: every parameter of every model in GATED_MODELS, Python-vs-MATLAB, over
        # all three regions. Backend consistency (auto-vs-cpu) stays reported -- CI installs no
        # accelerator, so `auto` resolves to the same code path as `cpu` there and gating it
        # would assert that a function equals itself. tests/python/test_backend_equivalence.py
        # is the real cpu-vs-cpufit/gpufit gate, and CI runs it with the backends installed.
        failures: list[str] = []
        checks: list[dict] = []

        def run_check(
            lhs: np.ndarray,
            rhs: np.ndarray,
            *,
            label: str,
            region_mask: np.ndarray,
            gated: bool,
            identifiable: np.ndarray | None = None,
            ci: dict | None = None,
            qof_mask: np.ndarray | None = None,
        ) -> None:
            finite = np.isfinite(lhs) & np.isfinite(rhs)
            combined = np.asarray(region_mask, dtype=bool)
            n_pre_qof = int(np.count_nonzero(finite & combined))
            if qof_mask is not None:
                combined = combined & np.asarray(qof_mask, dtype=bool)
            n_post_qof = int(np.count_nonzero(finite & combined))
            if identifiable is not None:
                combined = combined & np.asarray(identifiable, dtype=bool)
            n_valid = int(np.count_nonzero(finite & combined))
            frac_identifiable = (n_valid / n_post_qof) if n_post_qof else 0.0
            check_rec: dict = {
                "label": label,
                "gated": bool(gated),
                "corr_min": float(gate_corr_min),
                "nrmse_max": float(gate_nrmse_max),
                "valid_voxels": n_valid,
                "qof_filtered": bool(qof_mask is not None),
                "qof_excluded_voxels": int(n_pre_qof - n_post_qof) if qof_mask is not None else 0,
                "bound_pinned_excluded_voxels": int(n_post_qof - n_valid),
                "identifiable_fraction": float(frac_identifiable),
            }
            # A gated check whose identifiable subset has collapsed is verifying nothing, the
            # same silent hole the <2-voxel branch below covers -- just reached by degrees
            # rather than all at once (e.g. every fit drifting onto a bound).
            if gated and n_valid >= 2 and frac_identifiable < IDENTIFIABLE_FRACTION_MIN:
                msg = (
                    f"only {frac_identifiable:.3f} of QoF-passing voxels are identifiable "
                    f"({n_valid}/{n_post_qof}, min {IDENTIFIABLE_FRACTION_MIN}); "
                    "the gate has nothing determined left to compare"
                )
                check_rec["status"] = "collapsed"
                check_rec["error"] = msg
                failures.append(f"{label}: {msg}")
                _parity_log(f"{label}: FAILED (gated, identifiable fraction {frac_identifiable:.3f})")
                checks.append(check_rec)
                return
            if n_valid < 2:
                # A gated check with nothing to compare is a silent hole, not a pass.
                if gated:
                    check_rec["status"] = "collapsed"
                    msg = f"gated parity check has only {n_valid} valid voxels (mask collapse); parity not verified"
                    check_rec["error"] = msg
                    failures.append(f"{label}: {msg}")
                    _parity_log(f"{label}: FAILED (gated, mask collapse valid_voxels={n_valid})")
                else:
                    check_rec["status"] = "skipped"
                    _parity_log(f"{label}: skipped (valid_voxels={n_valid})")
                checks.append(check_rec)
                return

            metrics = _metrics(lhs, rhs, combined)
            check_rec["metrics"] = metrics
            summary = (
                f"{label}: n={metrics['n']}, corr={metrics['corr']:.6f}, "
                f"nrmse={metrics['nrmse']:.6f} (rmse={metrics['rmse']:.6f}, "
                f"ref_rms={metrics['ref_rms']:.6f}), ident={frac_identifiable:.3f}"
            )
            if ci is not None:
                ci_metrics = _ci_metrics(lhs, rhs, combined, **ci)
                check_rec["ci_metrics"] = ci_metrics
                summary += (
                    f", ci_norm_absdiff_p95={ci_metrics['ci_norm_absdiff_p95']:.4f}"
                    f", prop_out={ci_metrics['prop_py_outside_matlab_ci']:.4f}"
                )
            _parity_log(summary)

            if not gated:
                check_rec["status"] = "reported"
                checks.append(check_rec)
                return

            # A non-finite nrmse means the reference is degenerate (all-zero over the mask), so
            # there is nothing to normalize against and nothing verified -- fail, do not pass.
            ok = (
                metrics["corr"] >= gate_corr_min
                and np.isfinite(metrics["nrmse"])
                and metrics["nrmse"] <= gate_nrmse_max
            )
            if ok:
                check_rec["status"] = "pass"
            else:
                err = f"{summary} (corr_min={gate_corr_min}, nrmse_max={gate_nrmse_max})"
                check_rec["status"] = "failed"
                check_rec["error"] = err
                failures.append(f"{label}: {err}")
                _parity_log(f"{label}: FAILED (gated)")
            checks.append(check_rec)

        # Per-model QoF reliable masks from the CPU run's reduced χ² (absolute cutoff QOF_CHI2_MAX;
        # see sigma_estimators.md). The same mask filters every check for that model (cpu/auto vs
        # matlab, auto vs cpu).
        qof_masks: dict[str, np.ndarray | None] = {}
        qof_records: list[dict] = []
        for model_name in models:
            npz = out_cpu / f"Dyn-1_{model_name}_fit_postfit_arrays.npz"
            if QOF_CHI2_MAX <= 0.0 or not npz.exists():
                qof_masks[model_name] = None
                continue
            # shrink_sigma: eBayes-moderate σ² + prior-predictive clamp before χ²_ν, so
            # motion-inflated σ (which would otherwise suppress χ²_ν) can't hide a bad fit.
            chi2_vol = np.squeeze(dce_qof.qof_volumes(npz, shrink_sigma=True)["chi2nu"])
            mask_vol, tau = dce_qof.reliable_mask(chi2_vol, chi2_max=QOF_CHI2_MAX)
            qof_masks[model_name] = mask_vol
            n_fit = int(np.isfinite(chi2_vol).sum())
            n_excl = n_fit - int(mask_vol.sum())
            qof_records.append({
                "model": model_name,
                "chi2_max": float(tau),
                "fit_voxels": n_fit,
                "excluded_voxels": int(n_excl),
            })
            _parity_log(
                f"QoF {model_name}: chi2_nu<={tau:g} excludes {n_excl}/{n_fit} fitted voxels"
            )

        # Bounds come from the same prefs the fitter ran with, per model.
        base_prefs = _stage_d_fit_prefs(cpu_cfg)
        ident_records: list[dict] = []

        for model_name in models:
            params = list(MULTI_MODEL_PARITY_SPECS[model_name]["params"])
            gated_model = model_name in GATED_MODELS
            _parity_log(f"model={model_name}: running checks (gated={gated_model})")
            if not gated_model:
                _parity_log(f"  reported-only: {UNGATED_MODEL_REASONS[model_name]}")
            qof_mask_m = qof_masks.get(model_name)
            prefs_m = _apply_model_specific_prefs(base_prefs, model_name)

            matlab_maps = {p: _load_nifti(_matlab_map_path(paths, model_name, p)) for p in params}
            py_maps = {
                "cpu": {p: _load_nifti(out_cpu / f"Dyn-1_{model_name}_fit_{p}.nii.gz") for p in params},
                "auto": {p: _load_nifti(out_auto / f"Dyn-1_{model_name}_fit_{p}.nii.gz") for p in params},
            }

            matlab_sse = _load_nifti(_matlab_map_path(paths, model_name, "sse"))
            py_sse = {
                b: _load_nifti(out / f"Dyn-1_{model_name}_fit_sse.nii.gz")
                for b, out in (("cpu", out_cpu), ("auto", out_auto))
            }

            # One identifiable mask per (model, comparison) -- shared by every parameter and
            # region of that comparison, so a voxel is never in-scope for ve and out for Ktrans.
            ident: dict[str, np.ndarray] = {}
            for backend in ("cpu", "auto"):
                mask, pinned_counts = _identifiable_mask(
                    params, py_maps[backend], matlab_maps, prefs_m
                )
                tie = _local_minimum_tie_mask(
                    params, py_maps[backend], matlab_maps, py_sse[backend], matlab_sse
                )
                ident[f"{backend}_vs_matlab"] = mask & ~tie
                ident_records.append({
                    "model": model_name,
                    "comparison": f"{backend}_vs_matlab",
                    "pinned_voxels_per_param": pinned_counts,
                    "local_minimum_tie_voxels": int(np.count_nonzero(tie & mask)),
                })
                _parity_log(
                    f"  bound-pinned ({backend} vs matlab, either side): "
                    + ", ".join(f"{p}={n}" for p, n in pinned_counts.items())
                    + f"; local-minimum ties (equal-or-better SSE, different params): "
                    f"{int(np.count_nonzero(tie & mask))}"
                )
            tie_auto_cpu = _local_minimum_tie_mask(
                params, py_maps["auto"], py_maps["cpu"], py_sse["auto"], py_sse["cpu"]
            )
            ident["auto_vs_cpu"], _ = _identifiable_mask(
                params, py_maps["auto"], py_maps["cpu"], prefs_m
            )
            ident["auto_vs_cpu"] = ident["auto_vs_cpu"] & ~tie_auto_cpu

            # Reported-only CI diagnostics; loaded once per parameter, not once per region.
            ci_maps = {
                p: (_load_param_ci(out_cpu, paths, model_name, p),
                    _load_param_ci(out_auto, paths, model_name, p))
                for p in params
            }

            for region_name, region_mask in regions.items():
                # Part 2: fit quality over EVERY voxel in the region -- bound-pinned and
                # tied ones included. This is what licenses dropping tied voxels from the
                # parameter checks: the port may land in a different basin, but a basin that
                # fits the data worse still fails here.
                for backend in ("cpu", "auto"):
                    scope = np.asarray(region_mask, dtype=bool)
                    if qof_mask_m is not None:
                        scope = scope & np.asarray(qof_mask_m, dtype=bool)
                    sse_m = _sse_agreement(py_sse[backend], matlab_sse, scope)
                    label = f"{model_name}_sse_{region_name}_{backend}_vs_matlab"
                    rel_max = SSE_REL_MEDIAN_MAX[backend]
                    worse_max = SSE_WORSE_FRAC_MAX[backend]
                    ok = (
                        sse_m["n"] >= 2
                        and np.isfinite(sse_m["rel_median"]) and sse_m["rel_median"] <= rel_max
                        and np.isfinite(sse_m["worse_frac"]) and sse_m["worse_frac"] <= worse_max
                    )
                    summary = (
                        f"{label}: n={sse_m['n']}, rel_median={sse_m['rel_median']:.3e}, "
                        f"worse_frac={sse_m['worse_frac']:.4f}"
                    )
                    _parity_log(summary)
                    checks.append({
                        "label": label, "gated": bool(gated_model), "status": "pass" if ok else "failed",
                        "metrics": sse_m, "check": "sse_agreement",
                        "sse_rel_median_max": rel_max,
                        "sse_worse_frac_max": worse_max,
                    })
                    if gated_model and not ok:
                        err = f"{summary} (rel_median_max={rel_max}, worse_frac_max={worse_max})"
                        failures.append(f"{label}: {err}")
                        _parity_log(f"{label}: FAILED (gated)")

                for param in params:
                    matlab_map = matlab_maps[param]
                    ci_cpu, ci_auto = ci_maps[param]
                    run_check(
                        py_maps["cpu"][param], matlab_map,
                        label=f"{model_name}_{param}_{region_name}_cpu_vs_matlab",
                        region_mask=region_mask, gated=gated_model,
                        identifiable=ident["cpu_vs_matlab"], ci=ci_cpu, qof_mask=qof_mask_m,
                    )
                    run_check(
                        py_maps["auto"][param], matlab_map,
                        label=f"{model_name}_{param}_{region_name}_auto_vs_matlab",
                        region_mask=region_mask, gated=gated_model,
                        identifiable=ident["auto_vs_matlab"], ci=ci_auto, qof_mask=qof_mask_m,
                    )
                    run_check(
                        py_maps["auto"][param], py_maps["cpu"][param],
                        label=f"{model_name}_{param}_{region_name}_auto_vs_cpu",
                        region_mask=region_mask, gated=False,
                        identifiable=ident["auto_vs_cpu"], qof_mask=qof_mask_m,
                    )

        _parity_log("completed multi-model backend parity")
        summary_payload = {
            "suite": "multi-model",
            "dataset_root": str(paths["root"]),
            "roi_stride": int(roi_stride),
            "gated_models": sorted(GATED_MODELS),
            "ungated_model_reasons": UNGATED_MODEL_REASONS,
            "gate_corr_min": gate_corr_min,
            "gate_nrmse_max": gate_nrmse_max,
            "regions": sorted(regions.keys()),
            "qof_chi2_max": QOF_CHI2_MAX,
            "qof_filtering": qof_records,
            "identifiability_filtering": ident_records,
            "identifiable_fraction_min": IDENTIFIABLE_FRACTION_MIN,
            "sse_tie_rel_tol": SSE_TIE_REL_TOL,
            "tie_param_rel_tol": TIE_PARAM_REL_TOL,
            "sse_rel_median_max": SSE_REL_MEDIAN_MAX,
            "sse_worse_frac_max": SSE_WORSE_FRAC_MAX,
            "gated_failures": failures,
            "checks": checks,
        }
        _write_parity_summary(parity_summary_dir, "parity_multi_model_summary.json", summary_payload)
        # Anti-vacuous guard: at least one gated check must have actually compared data.
        gated_pass = sum(1 for c in checks if c.get("gated") and c.get("status") == "pass")
        if gated_pass == 0:
            failures.append(
                "no gated parity checks passed with data; suite verified nothing "
                "(check pipeline outputs / masks / gated-model configuration)"
            )
        if failures:
            failure_text = "\n\n".join(failures)
            pytest.fail(
                "multi-model parity checks failed; see details below:\n"
                f"{failure_text}"
            )


# ROI-summary (.xls) table parity: MATLAB averages each parameter's concentration curve over the
# whole-brain ROI and fits once (average-then-fit). We reproduce that with the pipeline's ROI-only
# mode (fit_voxels=0) — it skips the per-voxel fit entirely, so this is fast (a few seconds) and
# matches MATLAB exactly. Voxelwise map parity is covered separately by test_bbb_p19_region_parity.
@pytest.mark.parity
@pytest.mark.integration
def test_bbb_p19_roi_xls_parity(
    parity_dataset_root: str,
    parity_summary_dir: Path | None,
) -> None:
    root = Path(parity_dataset_root) if parity_dataset_root else _default_downsample_root()
    paths = _dataset_paths(root)
    models = ["tofts", "ex_tofts", "patlak", "tissue_uptake"]

    ref_xls_paths = {
        m: Path(paths["matlabref"]) / f"Dyn-1_{m}_fit_rois.xls" for m in models
    }
    missing = [str(p) for p in ref_xls_paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"MATLAB ROI-xls baselines missing ({len(missing)}); first: {missing[0]}")

    failures: list[str] = []
    roi_checks: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "python_out_cpu"
        # ROI-only mode: whole-brain average-then-fit (roi_files[0] = brain), no voxelwise fit.
        cfg = _make_config(paths, out_dir, backend="cpu", models=models)
        cfg.stage_overrides = {**cfg.stage_overrides, "fit_voxels": 0}
        result = run_dce_pipeline(cfg)
        assert result["meta"]["status"] == "ok"

        for model_name in models:
            limit = ROI_XLS_MAX_ABS_ERR
            py_xls = out_dir / f"Dyn-1_{model_name}_fit_rois.xls"
            label = f"{model_name}_roi_xls_cpu_vs_matlab"
            rec: dict = {"label": label, "max_abs_err_limit": limit}
            if not py_xls.exists():
                rec["status"] = "failed"
                rec["error"] = f"missing python ROI xls output ({py_xls})"
                failures.append(f"{label}: {rec['error']}")
                roi_checks.append(rec)
                continue
            try:
                rec["metrics"] = _compare_roi_table_against_reference(
                    model_name=model_name, py_path=py_xls, ref_path=ref_xls_paths[model_name], max_abs_err=limit
                )
                rec["status"] = "pass"
                _parity_log(f"{label}: pass (max_abs_err<={limit})")
            except AssertionError as exc:
                rec["status"] = "failed"
                rec["error"] = str(exc)
                failures.append(f"{label}: {exc}")
                _parity_log(f"{label}: FAILED")
            roi_checks.append(rec)

    _write_parity_summary(
        parity_summary_dir,
        "parity_roi_xls_summary.json",
        {"suite": "roi-xls", "dataset_root": str(paths["root"]), "roi_checks": roi_checks},
    )
    if failures:
        pytest.fail("ROI-xls parity checks failed:\n" + "\n\n".join(failures))
