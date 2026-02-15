"""Dataset-backed DCE parity checks (MATLAB map vs Python map)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import DcePipelineConfig, run_dce_pipeline  # noqa: E402


RUN_PARITY = os.environ.get("ROCKETSHIP_RUN_PIPELINE_PARITY", "0") == "1"
RUN_FULL = os.environ.get("ROCKETSHIP_RUN_FULL_VOLUME_PARITY", "0") == "1"


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


def _default_downsample_root() -> Path:
    ci_fixture = REPO_ROOT / "test_data" / "ci_fixtures" / "dce" / "bbb_p19_downsample_x3y3"
    if ci_fixture.exists():
        return ci_fixture
    return REPO_ROOT / "test_data" / "synthetic" / "generated" / "bbb_p19_downsample_x3y3"


def _make_config(paths: dict, out_dir: Path) -> DcePipelineConfig:
    return DcePipelineConfig(
        subject_source_path=paths["root"],
        subject_tp_path=paths["processed"],
        output_dir=out_dir,
        backend="cpu",
        checkpoint_dir=out_dir / "checkpoints",
        write_xls=True,
        dynamic_files=[paths["dynamic"]],
        aif_files=[paths["aif"]],
        roi_files=[paths["roi"]],
        t1map_files=[paths["t1map"]],
        noise_files=[paths["noise"]],
        model_flags={
            "tofts": 1,
            "ex_tofts": 0,
            "patlak": 0,
            "tissue_uptake": 0,
            "two_cxm": 0,
            "fxr": 0,
            "auc": 0,
            "nested": 0,
            "FXL_rr": 0,
        },
        stage_overrides={
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
            "voxel_MaxFunEvals": 50,
        },
    )


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(np.squeeze(nib.load(str(path)).get_fdata()), dtype=np.float64)


def _metrics(py_map: np.ndarray, matlab_map: np.ndarray, roi_mask: np.ndarray) -> dict:
    mask = np.isfinite(py_map) & np.isfinite(matlab_map) & (roi_mask > 0)
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


def _parity_error_hint(paths: dict) -> str:
    return (
        "Missing MATLAB baseline map(s). Generate them first with:\n"
        "  matlab -batch \"cd('/Users/samuelbarnes/code/ROCKETSHIP'); "
        "addpath('tests/matlab'); "
        "generate_dce_tofts_parity_map('subjectRoot', '%s');\"\n"
        "Expected maps:\n"
        "  %s\n"
        "  %s"
    ) % (paths["root"], paths["matlab_tofts_ktrans"], paths["matlab_tofts_ve"])


def _assert_map_parity(
    testcase: unittest.TestCase,
    py_map: np.ndarray,
    matlab_map: np.ndarray,
    roi_mask: np.ndarray,
    *,
    label: str,
    corr_min: float,
    mse_max: float,
) -> None:
    m = _metrics(py_map, matlab_map, roi_mask)
    summary = (
        f"{label}: n={m['n']}, corr={m['corr']:.6f}, mse={m['mse']:.6f}, "
        f"mae={m['mae']:.6f}, p95_abs_err={m['p95_abs_err']:.6f}"
    )
    testcase.assertGreaterEqual(m["corr"], corr_min, f"{summary} (corr_min={corr_min})")
    testcase.assertLessEqual(m["mse"], mse_max, f"{summary} (mse_max={mse_max})")


class TestDcePipelineParityMetrics(unittest.TestCase):
    @unittest.skipUnless(RUN_PARITY, "Set ROCKETSHIP_RUN_PIPELINE_PARITY=1 to run dataset parity checks.")
    def test_downsample_bbb_p19_tofts_ktrans(self) -> None:
        root = Path(
            os.environ.get(
                "ROCKETSHIP_BBB_DOWNSAMPLED_ROOT",
                str(_default_downsample_root()),
            )
        )
        paths = _dataset_paths(root)
        self.assertTrue(paths["matlab_tofts_ktrans"].exists(), _parity_error_hint(paths))
        self.assertTrue(paths["matlab_tofts_ve"].exists(), _parity_error_hint(paths))

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "python_out"
            result = run_dce_pipeline(_make_config(paths, out_dir))
            self.assertEqual(result["meta"]["status"], "ok")

            py_ktrans = _load_nifti(out_dir / "Dyn-1_tofts_fit_Ktrans.nii.gz")
            matlab_ktrans = _load_nifti(paths["matlab_tofts_ktrans"])
            py_ve = _load_nifti(out_dir / "Dyn-1_tofts_fit_ve.nii.gz")
            matlab_ve = _load_nifti(paths["matlab_tofts_ve"])
            roi_mask = _load_nifti(paths["roi"])

        _assert_map_parity(
            self,
            py_ktrans,
            matlab_ktrans,
            roi_mask,
            label="tofts_ktrans_downsample",
            corr_min=float(os.environ.get("ROCKETSHIP_PARITY_DOWNSAMPLED_KTRANS_CORR_MIN", "0.99")),
            mse_max=float(os.environ.get("ROCKETSHIP_PARITY_DOWNSAMPLED_KTRANS_MSE_MAX", "0.001")),
        )
        _assert_map_parity(
            self,
            py_ve,
            matlab_ve,
            roi_mask,
            label="tofts_ve_downsample",
            corr_min=float(os.environ.get("ROCKETSHIP_PARITY_DOWNSAMPLED_VE_CORR_MIN", "0.97")),
            mse_max=float(os.environ.get("ROCKETSHIP_PARITY_DOWNSAMPLED_VE_MSE_MAX", "0.002")),
        )

    @unittest.skipUnless(
        RUN_PARITY and RUN_FULL,
        "Set ROCKETSHIP_RUN_PIPELINE_PARITY=1 and ROCKETSHIP_RUN_FULL_VOLUME_PARITY=1 for full-volume parity.",
    )
    def test_full_bbb_p19_tofts_ktrans(self) -> None:
        root = Path(
            os.environ.get(
                "ROCKETSHIP_BBB_FULL_ROOT",
                str(REPO_ROOT / "test_data" / "BBB data p19"),
            )
        )
        paths = _dataset_paths(root)
        self.assertTrue(paths["matlab_tofts_ktrans"].exists(), _parity_error_hint(paths))
        self.assertTrue(paths["matlab_tofts_ve"].exists(), _parity_error_hint(paths))

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "python_out"
            result = run_dce_pipeline(_make_config(paths, out_dir))
            self.assertEqual(result["meta"]["status"], "ok")

            py_ktrans = _load_nifti(out_dir / "Dyn-1_tofts_fit_Ktrans.nii.gz")
            matlab_ktrans = _load_nifti(paths["matlab_tofts_ktrans"])
            py_ve = _load_nifti(out_dir / "Dyn-1_tofts_fit_ve.nii.gz")
            matlab_ve = _load_nifti(paths["matlab_tofts_ve"])
            roi_mask = _load_nifti(paths["roi"])

        _assert_map_parity(
            self,
            py_ktrans,
            matlab_ktrans,
            roi_mask,
            label="tofts_ktrans_full",
            corr_min=float(os.environ.get("ROCKETSHIP_PARITY_FULL_KTRANS_CORR_MIN", "0.99")),
            mse_max=float(os.environ.get("ROCKETSHIP_PARITY_FULL_KTRANS_MSE_MAX", "0.001")),
        )
        _assert_map_parity(
            self,
            py_ve,
            matlab_ve,
            roi_mask,
            label="tofts_ve_full",
            corr_min=float(os.environ.get("ROCKETSHIP_PARITY_FULL_VE_CORR_MIN", "0.97")),
            mse_max=float(os.environ.get("ROCKETSHIP_PARITY_FULL_VE_MSE_MAX", "0.002")),
        )


if __name__ == "__main__":
    unittest.main()
