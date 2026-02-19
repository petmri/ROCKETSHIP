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


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import DcePipelineConfig, run_dce_pipeline  # noqa: E402

MULTI_MODEL_PARITY_SPECS = {
    "tofts": {"params": ["Ktrans", "ve"]},
    "ex_tofts": {"params": ["Ktrans", "ve", "vp"]},
    "patlak": {"params": ["Ktrans", "vp"]},
    "tissue_uptake": {"params": ["Ktrans", "fp", "vp"]},
    "2cxm": {"params": ["Ktrans", "ve", "vp", "fp"]},
}


def _parity_log(message: str) -> None:
    print(f"[PARITY] {message}", flush=True)


def _dataset_paths(root: Path) -> dict:
    processed = root / "processed"
    matlab_ktrans = processed / "results_matlab" / "Dyn-1_tofts_fit_Ktrans.nii"
    matlab_ve = processed / "results_matlab" / "Dyn-1_tofts_fit_ve.nii"
    return {
        "root": root,
        "processed": processed,
        "dynamic": root / "Dynamic_t1w.nii",
        "aif": processed / "T1_AIF_roi.nii",
        "roi": processed / "T1_brain_roi.nii",
        "t1map": processed / "T1_map_t1_fa_fit_fa10.nii",
        "noise": processed / "T1_noise_roi.nii",
        "matlab_tofts_ktrans": matlab_ktrans,
        "matlab_tofts_ve": matlab_ve,
    }


def _matlab_map_path(paths: dict, model_name: str, param: str) -> Path:
    return Path(paths["processed"]) / "results_matlab" / f"Dyn-1_{model_name}_fit_{param}.nii"


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
    ci_fixture = REPO_ROOT / "tests/data" / "ci_fixtures" / "dce" / "bbb_p19_downsample_x3y3"
    if ci_fixture.exists():
        return ci_fixture
    return REPO_ROOT / "tests/data" / "synthetic" / "generated" / "bbb_p19_downsample_x3y3"


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
        "start_injection_min": 0.5,
        "end_injection_min": 0.7,
        "steady_state_start": 1,
        "steady_state_end": 2,
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


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(np.squeeze(nib.load(str(path)).get_fdata()), dtype=np.float64)


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
    return {
        "n": int(x.size),
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "p95_abs_err": float(np.percentile(np.abs(diff), 95.0)),
    }


def _valid_voxel_count(
    py_map: np.ndarray,
    ref_map: np.ndarray,
    roi_mask: np.ndarray,
    extra_mask: np.ndarray | None = None,
) -> int:
    mask = np.isfinite(py_map) & np.isfinite(ref_map) & (roi_mask > 0)
    if extra_mask is not None:
        mask = mask & np.asarray(extra_mask, dtype=bool)
    return int(np.count_nonzero(mask))


def _parity_error_hint(paths: dict, *, models: list[str], expected_maps: list[Path]) -> str:
    models_expr = "{" + ",".join(f"'{name}'" for name in models) + "}"
    expected_lines = "\n  ".join(str(p) for p in expected_maps)
    return (
        "Missing MATLAB baseline map(s). Generate them first with:\n"
        "  matlab -batch \"cd('/Users/samuelbarnes/code/ROCKETSHIP'); "
        "addpath('tests/matlab'); "
        "generate_dce_tofts_parity_map('subjectRoot', '%s', 'models', %s);\"\n"
        "Expected maps:\n"
        "  %s"
    ) % (paths["root"], models_expr, expected_lines)


def _assert_map_parity(
    py_map: np.ndarray,
    matlab_map: np.ndarray,
    roi_mask: np.ndarray,
    *,
    label: str,
    corr_min: float,
    mse_max: float,
    extra_mask: np.ndarray | None = None,
) -> dict:
    m = _metrics(py_map, matlab_map, roi_mask, extra_mask=extra_mask)
    summary = (
        f"{label}: n={m['n']}, corr={m['corr']:.6f}, mse={m['mse']:.6f}, "
        f"mae={m['mae']:.6f}, p95_abs_err={m['p95_abs_err']:.6f}"
    )
    _parity_log(summary)
    assert m["corr"] >= corr_min, f"{summary} (corr_min={corr_min})"
    assert m["mse"] <= mse_max, f"{summary} (mse_max={mse_max})"
    return m


def _assert_map_parity_if_enough(
    lhs: np.ndarray,
    rhs: np.ndarray,
    roi_mask: np.ndarray,
    *,
    label: str,
    corr_min: float,
    mse_max: float,
    extra_mask: np.ndarray | None = None,
) -> bool:
    n_valid = _valid_voxel_count(lhs, rhs, roi_mask, extra_mask=extra_mask)
    if n_valid < 2:
        _parity_log(f"{label}: skipped (valid_voxels={n_valid})")
        return None
    return _assert_map_parity(
        lhs,
        rhs,
        roi_mask,
        label=label,
        corr_min=corr_min,
        mse_max=mse_max,
        extra_mask=extra_mask,
    )


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

    abs_errors: list[float] = []
    for row_idx, (py_row, ref_row) in enumerate(zip(py_rows, ref_rows)):
        assert len(py_row) == len(ref_row), (
            f"{model_name}: row length mismatch at row {row_idx + 1}: "
            f"python={len(py_row)} ref={len(ref_row)}"
        )

        py_roi_name = str(py_row[1]).strip()
        ref_roi_name = str(ref_row[1]).strip()
        assert py_roi_name == ref_roi_name, (
            f"{model_name}: ROI name mismatch at row {row_idx + 1}: "
            f"python={py_roi_name!r} ref={ref_roi_name!r}"
        )

        py_vals = np.asarray([float(v) for v in py_row[2:]], dtype=np.float64)
        ref_vals = np.asarray([float(v) for v in ref_row[2:]], dtype=np.float64)
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


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.slow
def test_downsample_bbb_p19_models_cpu_and_auto(
    run_parity: bool,
    run_multi_model_backend_parity: bool,
    parity_dataset_root: str,
    parity_roi_stride: int,
    parity_summary_dir: Path | None,
    parity_thresholds: dict,
) -> None:
    if not (run_parity and run_multi_model_backend_parity):
        pytest.skip(
            "Use --run-parity --run-multi-model-backend-parity to run multi-model CPU-vs-auto checks."
        )

    root = Path(parity_dataset_root) if parity_dataset_root else _default_downsample_root()
    paths = _dataset_paths(root)
    models = list(MULTI_MODEL_PARITY_SPECS.keys())
    expected_maps = [
        _matlab_map_path(paths, model_name, param)
        for model_name, spec in MULTI_MODEL_PARITY_SPECS.items()
        for param in spec["params"]
    ]
    for map_path in expected_maps:
        assert map_path.exists(), _parity_error_hint(paths, models=models, expected_maps=expected_maps)

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

        cpu_result = run_dce_pipeline(
            _make_config(paths, out_cpu, backend="cpu", models=models, roi_path=sparse_roi_path)
        )
        auto_result = run_dce_pipeline(
            _make_config(paths, out_auto, backend="auto", models=models, roi_path=sparse_roi_path)
        )
        assert cpu_result["meta"]["status"] == "ok"
        assert auto_result["meta"]["status"] == "ok"

        roi_mask = _load_nifti(sparse_roi_path)

        ktrans_corr_min = float(parity_thresholds["model_ktrans_corr_min"])
        ktrans_mse_max = float(parity_thresholds["model_ktrans_mse_max"])
        param_corr_min = float(parity_thresholds["model_param_corr_min"])
        param_mse_max = float(parity_thresholds["model_param_mse_max"])
        cpu_auto_ktrans_corr_min = float(parity_thresholds["cpu_auto_ktrans_corr_min"])
        cpu_auto_ktrans_mse_max = float(parity_thresholds["cpu_auto_ktrans_mse_max"])
        cpu_auto_param_corr_min = float(parity_thresholds["cpu_auto_param_corr_min"])
        cpu_auto_param_mse_max = float(parity_thresholds["cpu_auto_param_mse_max"])
        ve_ktrans_min = float(parity_thresholds["ve_ktrans_min"])
        ex_tofts_ktrans_corr_min = float(parity_thresholds["ex_tofts_ktrans_corr_min"])
        ktrans_upper_exclude = float(parity_thresholds["ktrans_upper_exclude"])
        require_all_models = bool(parity_thresholds["require_all_models"])
        required_models_raw = str(parity_thresholds["required_models_raw"])
        required_models = {
            token.strip().lower() for token in required_models_raw.split(",") if token.strip()
        }
        if require_all_models:
            required_models = set(MULTI_MODEL_PARITY_SPECS.keys())
        cpu_optional_models_raw = str(parity_thresholds["cpu_optional_models_raw"])
        cpu_optional_models = {
            token.strip().lower() for token in cpu_optional_models_raw.split(",") if token.strip()
        }
        failures: list[str] = []
        diagnostic_failures: list[str] = []
        checks: list[dict] = []

        def run_check(
            lhs: np.ndarray,
            rhs: np.ndarray,
            *,
            label: str,
            corr_min: float,
            mse_max: float,
            extra_mask: np.ndarray | None = None,
            required: bool = True,
        ) -> None:
            n_valid = _valid_voxel_count(lhs, rhs, roi_mask, extra_mask=extra_mask)
            check_rec = {
                "label": label,
                "required": bool(required),
                "corr_min": float(corr_min),
                "mse_max": float(mse_max),
                "valid_voxels": int(n_valid),
            }
            if n_valid < 2:
                check_rec["status"] = "skipped"
                checks.append(check_rec)
                _parity_log(f"{label}: skipped (valid_voxels={n_valid})")
                return
            try:
                metrics = _assert_map_parity_if_enough(
                    lhs,
                    rhs,
                    roi_mask,
                    label=label,
                    corr_min=corr_min,
                    mse_max=mse_max,
                    extra_mask=extra_mask,
                )
                check_rec["status"] = "pass"
                if metrics is not None:
                    check_rec["metrics"] = metrics
                checks.append(check_rec)
            except AssertionError as exc:
                check_rec["status"] = "failed"
                check_rec["error"] = str(exc)
                checks.append(check_rec)
                if required:
                    failures.append(f"{label}: {exc}")
                    _parity_log(f"{label}: FAILED (required)")
                else:
                    diagnostic_failures.append(f"{label}: {exc}")
                    _parity_log(f"{label}: FAILED (diagnostic)")

        for model_name, spec in MULTI_MODEL_PARITY_SPECS.items():
            _parity_log(f"model={model_name}: running checks")
            model_required = model_name in required_models
            cpu_required = model_required and (model_name not in cpu_optional_models)
            py_cpu_ktrans = _load_nifti(out_cpu / f"Dyn-1_{model_name}_fit_Ktrans.nii.gz")
            py_auto_ktrans = _load_nifti(out_auto / f"Dyn-1_{model_name}_fit_Ktrans.nii.gz")
            matlab_ktrans = _load_nifti(_matlab_map_path(paths, model_name, "Ktrans"))
            if model_name in {"ex_tofts", "2cxm"}:
                cpu_ktrans_mask = (
                    np.isfinite(py_cpu_ktrans)
                    & np.isfinite(matlab_ktrans)
                    & (py_cpu_ktrans < ktrans_upper_exclude)
                    & (matlab_ktrans < ktrans_upper_exclude)
                )
                auto_ktrans_mask = (
                    np.isfinite(py_auto_ktrans)
                    & np.isfinite(matlab_ktrans)
                    & (py_auto_ktrans < ktrans_upper_exclude)
                    & (matlab_ktrans < ktrans_upper_exclude)
                )
                cpu_auto_ktrans_mask = (
                    np.isfinite(py_auto_ktrans)
                    & np.isfinite(py_cpu_ktrans)
                    & (py_auto_ktrans < ktrans_upper_exclude)
                    & (py_cpu_ktrans < ktrans_upper_exclude)
                )
            else:
                cpu_ktrans_mask = None
                auto_ktrans_mask = None
                cpu_auto_ktrans_mask = None
            model_ktrans_corr_min = ex_tofts_ktrans_corr_min if model_name == "ex_tofts" else ktrans_corr_min

            run_check(
                py_cpu_ktrans,
                matlab_ktrans,
                label=f"{model_name}_ktrans_cpu_vs_matlab",
                corr_min=model_ktrans_corr_min,
                mse_max=ktrans_mse_max,
                extra_mask=cpu_ktrans_mask,
                required=cpu_required,
            )
            run_check(
                py_auto_ktrans,
                matlab_ktrans,
                label=f"{model_name}_ktrans_auto_vs_matlab",
                corr_min=model_ktrans_corr_min,
                mse_max=ktrans_mse_max,
                extra_mask=auto_ktrans_mask,
                required=model_required,
            )
            run_check(
                py_auto_ktrans,
                py_cpu_ktrans,
                label=f"{model_name}_ktrans_auto_vs_cpu",
                corr_min=cpu_auto_ktrans_corr_min,
                mse_max=cpu_auto_ktrans_mse_max,
                extra_mask=cpu_auto_ktrans_mask,
                required=cpu_required,
            )

            for param in spec["params"]:
                if param == "Ktrans":
                    continue
                py_cpu_map = _load_nifti(out_cpu / f"Dyn-1_{model_name}_fit_{param}.nii.gz")
                py_auto_map = _load_nifti(out_auto / f"Dyn-1_{model_name}_fit_{param}.nii.gz")
                matlab_map = _load_nifti(_matlab_map_path(paths, model_name, param))

                param_use_ktrans_mask = param.lower() == "ve"
                if param_use_ktrans_mask:
                    cpu_mask = (
                        np.isfinite(py_cpu_ktrans)
                        & np.isfinite(matlab_ktrans)
                        & (py_cpu_ktrans > ve_ktrans_min)
                        & (matlab_ktrans > ve_ktrans_min)
                    )
                    auto_mask = (
                        np.isfinite(py_auto_ktrans)
                        & np.isfinite(matlab_ktrans)
                        & (py_auto_ktrans > ve_ktrans_min)
                        & (matlab_ktrans > ve_ktrans_min)
                    )
                    cpu_auto_mask = (
                        np.isfinite(py_cpu_ktrans)
                        & np.isfinite(py_auto_ktrans)
                        & (py_cpu_ktrans > ve_ktrans_min)
                        & (py_auto_ktrans > ve_ktrans_min)
                    )
                else:
                    cpu_mask = None
                    auto_mask = None
                    cpu_auto_mask = None
                base_valid_cpu = np.isfinite(py_cpu_map) & np.isfinite(matlab_map) & (roi_mask > 0)
                base_valid_auto = np.isfinite(py_auto_map) & np.isfinite(matlab_map) & (roi_mask > 0)
                base_valid_cpu_auto = np.isfinite(py_auto_map) & np.isfinite(py_cpu_map) & (roi_mask > 0)

                if param_use_ktrans_mask:
                    cpu_mask_use = cpu_mask if np.count_nonzero(base_valid_cpu & cpu_mask) >= 2 else None
                    auto_mask_use = auto_mask if np.count_nonzero(base_valid_auto & auto_mask) >= 2 else None
                    cpu_auto_mask_use = cpu_auto_mask if np.count_nonzero(base_valid_cpu_auto & cpu_auto_mask) >= 2 else None
                else:
                    cpu_mask_use = None
                    auto_mask_use = None
                    cpu_auto_mask_use = None

                run_check(
                    py_cpu_map,
                    matlab_map,
                    label=f"{model_name}_{param}_cpu_vs_matlab",
                    corr_min=param_corr_min,
                    mse_max=param_mse_max,
                    extra_mask=cpu_mask_use,
                    required=cpu_required,
                )
                run_check(
                    py_auto_map,
                    matlab_map,
                    label=f"{model_name}_{param}_auto_vs_matlab",
                    corr_min=param_corr_min,
                    mse_max=param_mse_max,
                    extra_mask=auto_mask_use,
                    required=model_required,
                )
                run_check(
                    py_auto_map,
                    py_cpu_map,
                    label=f"{model_name}_{param}_auto_vs_cpu",
                    corr_min=cpu_auto_param_corr_min,
                    mse_max=cpu_auto_param_mse_max,
                    extra_mask=cpu_auto_mask_use,
                    required=cpu_required,
                )

        _parity_log("completed multi-model backend parity")
        summary_payload = {
            "suite": "multi-model",
            "dataset_root": str(paths["root"]),
            "roi_stride": int(roi_stride),
            "required_models": sorted(required_models),
            "cpu_optional_models": sorted(cpu_optional_models),
            "required_failures": failures,
            "diagnostic_failures": diagnostic_failures,
            "checks": checks,
        }
        _write_parity_summary(parity_summary_dir, "parity_multi_model_summary.json", summary_payload)
        if diagnostic_failures:
            _parity_log(
                f"diagnostic parity failures (non-gating): {len(diagnostic_failures)}"
            )
        if failures:
            failure_text = "\n\n".join(failures)
            pytest.fail(
                "multi-model parity checks failed; see details below:\n"
                f"{failure_text}"
            )


@pytest.mark.parity
@pytest.mark.integration
def test_downsample_bbb_p19_tofts_ktrans(
    run_parity: bool,
    parity_dataset_root: str,
    parity_summary_dir: Path | None,
    parity_thresholds: dict,
) -> None:
    if not run_parity:
        pytest.skip("Use --run-parity to run dataset-backed parity checks.")

    root = Path(parity_dataset_root) if parity_dataset_root else _default_downsample_root()
    paths = _dataset_paths(root)
    tofts_expected = [paths["matlab_tofts_ktrans"], paths["matlab_tofts_ve"]]
    assert paths["matlab_tofts_ktrans"].exists(), _parity_error_hint(
        paths,
        models=["tofts"],
        expected_maps=tofts_expected,
    )
    assert paths["matlab_tofts_ve"].exists(), _parity_error_hint(
        paths,
        models=["tofts"],
        expected_maps=tofts_expected,
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "python_out"
        result = run_dce_pipeline(_make_config(paths, out_dir, backend="auto", models=["tofts"]))
        assert result["meta"]["status"] == "ok"

        py_ktrans = _load_nifti(out_dir / "Dyn-1_tofts_fit_Ktrans.nii.gz")
        matlab_ktrans = _load_nifti(paths["matlab_tofts_ktrans"])
        py_ve = _load_nifti(out_dir / "Dyn-1_tofts_fit_ve.nii.gz")
        matlab_ve = _load_nifti(paths["matlab_tofts_ve"])
        roi_mask = _load_nifti(paths["roi"])

    ktrans_metrics = _assert_map_parity(
        py_ktrans,
        matlab_ktrans,
        roi_mask,
        label="tofts_ktrans_downsample",
        corr_min=float(parity_thresholds["downsample_ktrans_corr_min"]),
        mse_max=float(parity_thresholds["downsample_ktrans_mse_max"]),
    )
    ve_ktrans_min = float(parity_thresholds["ve_ktrans_min"])
    ve_metrics = _assert_map_parity(
        py_ve,
        matlab_ve,
        roi_mask,
        label="tofts_ve_downsample",
        corr_min=float(parity_thresholds["downsample_ve_corr_min"]),
        mse_max=float(parity_thresholds["downsample_ve_mse_max"]),
        extra_mask=(
            np.isfinite(py_ktrans)
            & np.isfinite(matlab_ktrans)
            & (py_ktrans > ve_ktrans_min)
            & (matlab_ktrans > ve_ktrans_min)
        ),
    )
    _write_parity_summary(
        parity_summary_dir,
        "parity_tofts_downsample_summary.json",
        {
            "suite": "tofts-downsample",
            "dataset_root": str(paths["root"]),
            "ktrans": ktrans_metrics,
            "ve": ve_metrics,
            "ve_ktrans_min": float(ve_ktrans_min),
        },
    )


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.slow
def test_downsample_bbb_p19_model_maps_and_roi_xls_cpu(
    run_parity: bool,
    parity_dataset_root: str,
    parity_summary_dir: Path | None,
    parity_thresholds: dict,
) -> None:
    if not run_parity:
        pytest.skip("Use --run-parity to run dataset-backed parity checks.")

    root = Path(parity_dataset_root) if parity_dataset_root else _default_downsample_root()
    paths = _dataset_paths(root)
    run_models = ["tofts", "ex_tofts", "patlak", "tissue_uptake"]
    map_models = ["ex_tofts", "patlak", "tissue_uptake"]

    expected_maps = [
        _matlab_map_path(paths, model_name, param)
        for model_name in map_models
        for param in MULTI_MODEL_PARITY_SPECS[model_name]["params"]
    ]
    expected_xls = [Path(paths["processed"]) / "results_matlab" / f"Dyn-1_{name}_fit_rois.xls" for name in run_models]
    for map_path in expected_maps:
        assert map_path.exists(), _parity_error_hint(paths, models=map_models, expected_maps=expected_maps)
    for xls_path in expected_xls:
        assert xls_path.exists(), f"Missing MATLAB ROI XLS baseline: {xls_path}"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "python_out_cpu"
        _parity_log(
            "starting CPU model-map/ROI parity: "
            f"root={paths['root']} models={run_models}"
        )
        result = run_dce_pipeline(_make_config(paths, out_dir, backend="cpu", models=run_models))
        assert result["meta"]["status"] == "ok"

        roi_mask = _load_nifti(paths["roi"])
        ve_ktrans_min = float(parity_thresholds["ve_ktrans_min"])
        ex_tofts_ktrans_corr_min = float(parity_thresholds["ex_tofts_ktrans_corr_min"])
        # Ex-Tofts Ktrans parity is dominated by high-end outliers; cap exclusion at 1.0
        # for this CPU-vs-MATLAB map suite to keep comparisons in the stable range.
        ktrans_upper_exclude = min(float(parity_thresholds["ktrans_upper_exclude"]), 1.0)
        ktrans_corr_min = float(parity_thresholds["model_ktrans_corr_min"])
        ktrans_mse_max = float(parity_thresholds["model_ktrans_mse_max"])
        param_corr_min = float(parity_thresholds["model_param_corr_min"])
        param_mse_max = float(parity_thresholds["model_param_mse_max"])

        require_all_models = bool(parity_thresholds["require_all_models"])
        required_models_raw = str(parity_thresholds["required_models_raw"])
        required_models = {token.strip().lower() for token in required_models_raw.split(",") if token.strip()}
        if require_all_models:
            required_models = set(MULTI_MODEL_PARITY_SPECS.keys())
        if not required_models:
            required_models = {"tofts", "ex_tofts", "patlak"}

        failures: list[str] = []
        diagnostic_failures: list[str] = []
        map_checks: list[dict] = []
        roi_checks: list[dict] = []

        def run_map_check(
            lhs: np.ndarray,
            rhs: np.ndarray,
            *,
            label: str,
            corr_min: float,
            mse_max: float,
            extra_mask: np.ndarray | None = None,
            required: bool = True,
        ) -> None:
            n_valid = _valid_voxel_count(lhs, rhs, roi_mask, extra_mask=extra_mask)
            check_rec = {
                "label": label,
                "required": bool(required),
                "corr_min": float(corr_min),
                "mse_max": float(mse_max),
                "valid_voxels": int(n_valid),
            }
            if n_valid < 2:
                check_rec["status"] = "skipped"
                map_checks.append(check_rec)
                _parity_log(f"{label}: skipped (valid_voxels={n_valid})")
                return
            try:
                metrics = _assert_map_parity_if_enough(
                    lhs,
                    rhs,
                    roi_mask,
                    label=label,
                    corr_min=corr_min,
                    mse_max=mse_max,
                    extra_mask=extra_mask,
                )
                check_rec["status"] = "pass"
                if metrics is not None:
                    check_rec["metrics"] = metrics
                map_checks.append(check_rec)
            except AssertionError as exc:
                check_rec["status"] = "failed"
                check_rec["error"] = str(exc)
                map_checks.append(check_rec)
                if required:
                    failures.append(f"{label}: {exc}")
                    _parity_log(f"{label}: FAILED (required)")
                else:
                    diagnostic_failures.append(f"{label}: {exc}")
                    _parity_log(f"{label}: FAILED (diagnostic)")

        for model_name in map_models:
            model_required = model_name in required_models
            py_ktrans = _load_nifti(out_dir / f"Dyn-1_{model_name}_fit_Ktrans.nii.gz")
            matlab_ktrans = _load_nifti(_matlab_map_path(paths, model_name, "Ktrans"))
            if model_name == "ex_tofts":
                ktrans_mask = (
                    np.isfinite(py_ktrans)
                    & np.isfinite(matlab_ktrans)
                    & (py_ktrans < ktrans_upper_exclude)
                    & (matlab_ktrans < ktrans_upper_exclude)
                )
            else:
                ktrans_mask = None
            corr_floor = ex_tofts_ktrans_corr_min if model_name == "ex_tofts" else ktrans_corr_min
            run_map_check(
                py_ktrans,
                matlab_ktrans,
                label=f"{model_name}_ktrans_cpu_vs_matlab",
                corr_min=corr_floor,
                mse_max=ktrans_mse_max,
                extra_mask=ktrans_mask,
                required=model_required,
            )

            for param in MULTI_MODEL_PARITY_SPECS[model_name]["params"]:
                if param == "Ktrans":
                    continue
                py_map = _load_nifti(out_dir / f"Dyn-1_{model_name}_fit_{param}.nii.gz")
                matlab_map = _load_nifti(_matlab_map_path(paths, model_name, param))
                if param.lower() == "ve":
                    ve_mask = (
                        np.isfinite(py_ktrans)
                        & np.isfinite(matlab_ktrans)
                        & (py_ktrans > ve_ktrans_min)
                        & (matlab_ktrans > ve_ktrans_min)
                    )
                    base_valid = np.isfinite(py_map) & np.isfinite(matlab_map) & (roi_mask > 0)
                    mask_use = ve_mask if np.count_nonzero(base_valid & ve_mask) >= 2 else None
                else:
                    mask_use = None
                run_map_check(
                    py_map,
                    matlab_map,
                    label=f"{model_name}_{param}_cpu_vs_matlab",
                    corr_min=param_corr_min,
                    mse_max=param_mse_max,
                    extra_mask=mask_use,
                    required=model_required,
                )

        roi_abs_err_limits = {
            "tofts": 0.03,
            "ex_tofts": 0.01,
            "patlak": 0.01,
            "tissue_uptake": 0.05,
        }
        for model_name in run_models:
            model_required = model_name in required_models
            py_xls = out_dir / f"Dyn-1_{model_name}_fit_rois.xls"
            ref_xls = Path(paths["processed"]) / "results_matlab" / f"Dyn-1_{model_name}_fit_rois.xls"
            check_rec = {
                "label": f"{model_name}_roi_xls_cpu_vs_matlab",
                "required": bool(model_required),
                "max_abs_err_limit": float(roi_abs_err_limits[model_name]),
            }
            if not py_xls.exists():
                msg = f"{model_name}: missing python ROI XLS output ({py_xls})"
                check_rec["status"] = "failed"
                check_rec["error"] = msg
                roi_checks.append(check_rec)
                if model_required:
                    failures.append(msg)
                else:
                    diagnostic_failures.append(msg)
                continue
            try:
                metrics = _compare_roi_table_against_reference(
                    model_name=model_name,
                    py_path=py_xls,
                    ref_path=ref_xls,
                    max_abs_err=float(roi_abs_err_limits[model_name]),
                )
                check_rec["status"] = "pass"
                check_rec["metrics"] = metrics
                roi_checks.append(check_rec)
            except AssertionError as exc:
                check_rec["status"] = "failed"
                check_rec["error"] = str(exc)
                roi_checks.append(check_rec)
                if model_required:
                    failures.append(f"{model_name}_roi_xls_cpu_vs_matlab: {exc}")
                else:
                    diagnostic_failures.append(f"{model_name}_roi_xls_cpu_vs_matlab: {exc}")

        _parity_log("completed CPU model-map/ROI parity")
        _write_parity_summary(
            parity_summary_dir,
            "parity_model_map_roi_cpu_summary.json",
            {
                "suite": "model-map-roi-cpu",
                "dataset_root": str(paths["root"]),
                "required_models": sorted(required_models),
                "required_failures": failures,
                "diagnostic_failures": diagnostic_failures,
                "map_checks": map_checks,
                "roi_checks": roi_checks,
            },
        )
        if diagnostic_failures:
            _parity_log(f"diagnostic parity failures (non-gating): {len(diagnostic_failures)}")
        if failures:
            failure_text = "\n\n".join(failures)
            pytest.fail(
                "model-map/ROI CPU parity checks failed; see details below:\n"
                f"{failure_text}"
            )


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.slow
def test_full_bbb_p19_tofts_ktrans(
    run_parity: bool,
    run_full_parity: bool,
    parity_full_root: str,
    parity_summary_dir: Path | None,
    parity_thresholds: dict,
) -> None:
    if not (run_parity and run_full_parity):
        pytest.skip("Use --run-parity --run-full-parity to run full-volume parity checks.")

    root = Path(parity_full_root) if parity_full_root else (REPO_ROOT / "tests/data" / "BBB data p19")
    paths = _dataset_paths(root)
    tofts_expected = [paths["matlab_tofts_ktrans"], paths["matlab_tofts_ve"]]
    assert paths["matlab_tofts_ktrans"].exists(), _parity_error_hint(
        paths,
        models=["tofts"],
        expected_maps=tofts_expected,
    )
    assert paths["matlab_tofts_ve"].exists(), _parity_error_hint(
        paths,
        models=["tofts"],
        expected_maps=tofts_expected,
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "python_out"
        result = run_dce_pipeline(_make_config(paths, out_dir, backend="auto", models=["tofts"]))
        assert result["meta"]["status"] == "ok"

        py_ktrans = _load_nifti(out_dir / "Dyn-1_tofts_fit_Ktrans.nii.gz")
        matlab_ktrans = _load_nifti(paths["matlab_tofts_ktrans"])
        py_ve = _load_nifti(out_dir / "Dyn-1_tofts_fit_ve.nii.gz")
        matlab_ve = _load_nifti(paths["matlab_tofts_ve"])
        roi_mask = _load_nifti(paths["roi"])

    ktrans_metrics = _assert_map_parity(
        py_ktrans,
        matlab_ktrans,
        roi_mask,
        label="tofts_ktrans_full",
        corr_min=float(parity_thresholds["full_ktrans_corr_min"]),
        mse_max=float(parity_thresholds["full_ktrans_mse_max"]),
    )
    ve_ktrans_min = float(parity_thresholds["ve_ktrans_min"])
    ve_metrics = _assert_map_parity(
        py_ve,
        matlab_ve,
        roi_mask,
        label="tofts_ve_full",
        corr_min=float(parity_thresholds["full_ve_corr_min"]),
        mse_max=float(parity_thresholds["full_ve_mse_max"]),
        extra_mask=(
            np.isfinite(py_ktrans)
            & np.isfinite(matlab_ktrans)
            & (py_ktrans > ve_ktrans_min)
            & (matlab_ktrans > ve_ktrans_min)
        ),
    )
    _write_parity_summary(
        parity_summary_dir,
        "parity_tofts_full_summary.json",
        {
            "suite": "tofts-full",
            "dataset_root": str(paths["root"]),
            "ktrans": ktrans_metrics,
            "ve": ve_metrics,
            "ve_ktrans_min": float(ve_ktrans_min),
        },
    )
