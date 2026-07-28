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

from dce_pipeline import DcePipelineConfig, run_dce_pipeline  # noqa: E402
import dce_qof  # noqa: E402

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
        "aif": der / "dce" / f"{stem}_desc-AIFroi_mask.nii",
        "roi": der / "anat" / f"{stem}_desc-brain_mask.nii",
        "roi_gm": der / "anat" / f"{stem}_desc-GMroi_mask.nii",
        "roi_wm": der / "anat" / f"{stem}_desc-WMroi_mask.nii",
        "t1map": der / "anat" / f"{stem}_space-DCEref_T1map.nii",
        "noise": der / "anat" / f"{stem}_desc-noise_mask.nii",
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
        "aif_curve_mode": "fitted",
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
        "voxel_MaxFunEvals": int(os.environ.get("ROCKETSHIP_PARITY_VOXEL_MAXFUNEVALS", "50")),
        "voxel_MaxIter": int(os.environ.get("ROCKETSHIP_PARITY_VOXEL_MAXITER", "50")),
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
    return {
        "n": int(x.size),
        "corr": corr,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
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
    the MATLAB reference's ``T1_*_roi`` names (e.g. ``...desc-brain_mask`` -> ``brain``)."""
    text = str(name).strip().lower()
    for token in ("brain", "gm", "wm", "aif", "noise"):
        if token in text:
            return token
    return text


# Columns excluded from the ROI-xls gate, per model.
#
# tissue_uptake's Fp (plasma flow) is not reliably estimable on this fixture: at 15.84 s frames
# the bolus rise occupies a single sample, and Fp is determined almost entirely by that leading
# edge. It is also the parameter most exposed to how the AIF's peak is fitted, which is exactly
# the thing the data cannot pin down (the peak has leverage 1 in the biexponential model -- see
# docs/project-management/projects/archived/batch-parity/aif_fitting_parity.md). Python and MATLAB have
# never agreed on it here; every other tissue_uptake column agrees to <0.004. Gating on Fp
# measures the fixture's temporal resolution, not the port's correctness.
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


# Standard suite: gated Python-vs-MATLAB parity on Tofts & Patlak Ktrans. Runs by default
# (no flag). `--parity-suite=allmodels` additionally runs ex_tofts/tissue_uptake/2cxm as
# reported-only diagnostics.
STANDARD_PARITY_MODELS = ["tofts", "patlak"]
ALLMODELS_EXTRA = ["ex_tofts", "tissue_uptake", "2cxm"]

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

        ktrans_corr_min = float(parity_thresholds["model_ktrans_corr_min"])
        ktrans_mse_max = float(parity_thresholds["model_ktrans_mse_max"])
        param_corr_min = float(parity_thresholds["model_param_corr_min"])
        param_mse_max = float(parity_thresholds["model_param_mse_max"])
        ve_ktrans_min = float(parity_thresholds["ve_ktrans_min"])
        ktrans_upper_exclude = float(parity_thresholds["ktrans_upper_exclude"])

        # Gated scope: only Tofts & Patlak, Ktrans parameter, Python-vs-MATLAB. Everything
        # else (other params, other models, backend-consistency) is reported, never gated.
        gated_models = {"tofts", "patlak"}

        failures: list[str] = []
        checks: list[dict] = []

        def run_check(
            lhs: np.ndarray,
            rhs: np.ndarray,
            *,
            label: str,
            region_mask: np.ndarray,
            corr_min: float,
            mse_max: float,
            gated: bool,
            extra_mask: np.ndarray | None = None,
            ci: dict | None = None,
            qof_mask: np.ndarray | None = None,
        ) -> None:
            combined = np.asarray(region_mask, dtype=bool)
            if extra_mask is not None:
                combined = combined & np.asarray(extra_mask, dtype=bool)
            n_pre_qof = int(np.count_nonzero(np.isfinite(lhs) & np.isfinite(rhs) & combined))
            if qof_mask is not None:
                combined = combined & np.asarray(qof_mask, dtype=bool)
            n_valid = int(np.count_nonzero(np.isfinite(lhs) & np.isfinite(rhs) & combined))
            check_rec: dict = {
                "label": label,
                "gated": bool(gated),
                "corr_min": float(corr_min),
                "mse_max": float(mse_max),
                "valid_voxels": n_valid,
                "qof_filtered": bool(qof_mask is not None),
                "qof_excluded_voxels": int(n_pre_qof - n_valid) if qof_mask is not None else 0,
            }
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
            summary = f"{label}: n={metrics['n']}, corr={metrics['corr']:.6f}, rmse={metrics['rmse']:.6f}"
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

            rmse_max = float(np.sqrt(mse_max))
            if metrics["corr"] >= corr_min and metrics["rmse"] <= rmse_max:
                check_rec["status"] = "pass"
            else:
                err = f"{summary} (corr_min={corr_min}, rmse_max={rmse_max:.6f})"
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

        for model_name in models:
            spec = MULTI_MODEL_PARITY_SPECS[model_name]
            _parity_log(f"model={model_name}: running checks")
            gated_model = model_name in gated_models
            qof_mask_m = qof_masks.get(model_name)
            py_cpu_ktrans = _load_nifti(out_cpu / f"Dyn-1_{model_name}_fit_Ktrans.nii.gz")
            py_auto_ktrans = _load_nifti(out_auto / f"Dyn-1_{model_name}_fit_Ktrans.nii.gz")
            matlab_ktrans = _load_nifti(_matlab_map_path(paths, model_name, "Ktrans"))

            # Ktrans CI maps (reported-only; tolerant of absence).
            ci_cpu = _load_param_ci(out_cpu, paths, model_name, "Ktrans")
            ci_auto = _load_param_ci(out_auto, paths, model_name, "Ktrans")

            # Ktrans upper-bound exclusion for the unstable (reported-only) models.
            if model_name in {"ex_tofts", "2cxm"}:
                cpu_excl = (matlab_ktrans < ktrans_upper_exclude) & (py_cpu_ktrans < ktrans_upper_exclude)
                auto_excl = (matlab_ktrans < ktrans_upper_exclude) & (py_auto_ktrans < ktrans_upper_exclude)
                cpu_auto_excl = (py_cpu_ktrans < ktrans_upper_exclude) & (py_auto_ktrans < ktrans_upper_exclude)
            else:
                cpu_excl = auto_excl = cpu_auto_excl = None

            for region_name, region_mask in regions.items():
                # Every gated model gates on every region. The former tofts+GM exception (GM
                # reported-only, on the grounds that tofts Ktrans was non-identifiable there)
                # was retired 2026-07-28: the Stage-B AIF fix (aif_fitting_parity.md S11) lifted
                # tofts_ktrans_gm to corr 0.980 (cpu) / 0.992 (auto) against a 0.95 floor, so the
                # hand-curated exception no longer describes anything real.
                ktrans_gated = gated_model
                run_check(
                    py_cpu_ktrans, matlab_ktrans,
                    label=f"{model_name}_ktrans_{region_name}_cpu_vs_matlab",
                    region_mask=region_mask, corr_min=ktrans_corr_min, mse_max=ktrans_mse_max,
                    gated=ktrans_gated, extra_mask=cpu_excl, ci=ci_cpu, qof_mask=qof_mask_m,
                )
                run_check(
                    py_auto_ktrans, matlab_ktrans,
                    label=f"{model_name}_ktrans_{region_name}_auto_vs_matlab",
                    region_mask=region_mask, corr_min=ktrans_corr_min, mse_max=ktrans_mse_max,
                    gated=ktrans_gated, extra_mask=auto_excl, ci=ci_auto, qof_mask=qof_mask_m,
                )
                run_check(
                    py_auto_ktrans, py_cpu_ktrans,
                    label=f"{model_name}_ktrans_{region_name}_auto_vs_cpu",
                    region_mask=region_mask, corr_min=ktrans_corr_min, mse_max=ktrans_mse_max,
                    gated=False, extra_mask=cpu_auto_excl, qof_mask=qof_mask_m,
                )

                for param in spec["params"]:
                    if param == "Ktrans":
                        continue
                    py_cpu_map = _load_nifti(out_cpu / f"Dyn-1_{model_name}_fit_{param}.nii.gz")
                    py_auto_map = _load_nifti(out_auto / f"Dyn-1_{model_name}_fit_{param}.nii.gz")
                    matlab_map = _load_nifti(_matlab_map_path(paths, model_name, param))
                    ci_cpu_p = _load_param_ci(out_cpu, paths, model_name, param)
                    ci_auto_p = _load_param_ci(out_auto, paths, model_name, param)
                    if param.lower() == "ve":
                        cpu_pm = (py_cpu_ktrans > ve_ktrans_min) & (matlab_ktrans > ve_ktrans_min)
                        auto_pm = (py_auto_ktrans > ve_ktrans_min) & (matlab_ktrans > ve_ktrans_min)
                        cpu_auto_pm = (py_cpu_ktrans > ve_ktrans_min) & (py_auto_ktrans > ve_ktrans_min)
                    else:
                        cpu_pm = auto_pm = cpu_auto_pm = None
                    run_check(
                        py_cpu_map, matlab_map,
                        label=f"{model_name}_{param}_{region_name}_cpu_vs_matlab",
                        region_mask=region_mask, corr_min=param_corr_min, mse_max=param_mse_max,
                        gated=False, extra_mask=cpu_pm, ci=ci_cpu_p, qof_mask=qof_mask_m,
                    )
                    run_check(
                        py_auto_map, matlab_map,
                        label=f"{model_name}_{param}_{region_name}_auto_vs_matlab",
                        region_mask=region_mask, corr_min=param_corr_min, mse_max=param_mse_max,
                        gated=False, extra_mask=auto_pm, ci=ci_auto_p, qof_mask=qof_mask_m,
                    )
                    run_check(
                        py_auto_map, py_cpu_map,
                        label=f"{model_name}_{param}_{region_name}_auto_vs_cpu",
                        region_mask=region_mask, corr_min=param_corr_min, mse_max=param_mse_max,
                        gated=False, extra_mask=cpu_auto_pm, qof_mask=qof_mask_m,
                    )

        _parity_log("completed multi-model backend parity")
        summary_payload = {
            "suite": "multi-model",
            "dataset_root": str(paths["root"]),
            "roi_stride": int(roi_stride),
            "gated_models": sorted(gated_models),
            "regions": sorted(regions.keys()),
            "qof_chi2_max": QOF_CHI2_MAX,
            "qof_filtering": qof_records,
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
    roi_abs_err_limits = {"tofts": 0.03, "ex_tofts": 0.01, "patlak": 0.01, "tissue_uptake": 0.05}

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
            limit = float(roi_abs_err_limits[model_name])
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
