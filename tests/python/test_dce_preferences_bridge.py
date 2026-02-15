"""Tests for MATLAB dce_preferences.txt bridging into Python DCE pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _fit_aif_biexp,
    _stage_d_fit_prefs,
    run_dce_pipeline,
)


def _write_prefs(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_config(tmp: Path, prefs_path: Path, extra_overrides: dict | None = None, *, backend: str = "auto") -> DcePipelineConfig:
    subject_source = tmp / "subject_source"
    subject_tp = tmp / "subject_tp"
    output_dir = tmp / "output"

    dynamic = subject_tp / "dynamic.nii.gz"
    aif = subject_tp / "aif_mask.nii.gz"
    t1map = subject_tp / "t1map.nii.gz"
    dynamic.parent.mkdir(parents=True, exist_ok=True)
    dynamic.write_text("")
    aif.write_text("")
    t1map.write_text("")

    overrides = {
        "stage_a_mode": "scaffold",
        "stage_b_mode": "scaffold",
        "stage_d_mode": "scaffold",
        "dce_preferences_path": str(prefs_path),
    }
    if extra_overrides:
        overrides.update(extra_overrides)

    return DcePipelineConfig(
        subject_source_path=subject_source,
        subject_tp_path=subject_tp,
        output_dir=output_dir,
        backend=backend,
        checkpoint_dir=tmp / "checkpoints",
        write_xls=False,
        dynamic_files=[dynamic],
        aif_files=[aif],
        roi_files=[],
        t1map_files=[t1map],
        noise_files=[],
        drift_files=[],
        model_flags={"tofts": 1},
        stage_overrides=overrides,
    )


class TestDcePreferencesBridge(unittest.TestCase):
    def test_stage_d_preferences_loaded_with_expression_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            prefs = _write_prefs(
                tmp / "dce_preferences.txt",
                [
                    "voxel_lower_limit_ktrans = 10^-5",
                    "voxel_upper_limit_ve = 0.73",
                    "voxel_MaxFunEvals = 77",
                    "fxr_fw = 0.66",
                ],
            )
            config = _make_config(tmp, prefs)
            fit_prefs = _stage_d_fit_prefs(config)

            self.assertAlmostEqual(float(fit_prefs["lower_limit_ktrans"]), 1e-5, places=12)
            self.assertAlmostEqual(float(fit_prefs["upper_limit_ve"]), 0.73, places=12)
            self.assertEqual(int(fit_prefs["max_nfev"]), 77)
            self.assertAlmostEqual(float(fit_prefs["fxr_fw"]), 0.66, places=12)

    def test_stage_overrides_take_precedence_over_preferences_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            prefs = _write_prefs(tmp / "dce_preferences.txt", ["voxel_MaxFunEvals = 77"])
            config = _make_config(tmp, prefs, {"voxel_MaxFunEvals": 123})
            fit_prefs = _stage_d_fit_prefs(config)
            self.assertEqual(int(fit_prefs["max_nfev"]), 123)

    @patch("dce_pipeline.is_gpufit_available", return_value=True)
    def test_force_cpu_preference_overrides_backend_auto(self, _gpufit_mock: object) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            prefs = _write_prefs(tmp / "dce_preferences.txt", ["force_cpu = 1"])
            config = _make_config(tmp, prefs, backend="auto")
            result = run_dce_pipeline(config)
            self.assertEqual(result["stages"]["D"]["selected_backend"], "cpu")

    @patch("scipy.optimize.curve_fit")
    def test_aif_advanced_preferences_flow_into_fit_kwargs(self, curve_fit_mock: object) -> None:
        curve_fit_mock.return_value = (np.array([1.0, 1.0, 1.0, 0.1]), np.eye(4))
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            prefs = _write_prefs(
                tmp / "dce_preferences.txt",
                [
                    "aif_MaxFunEvals = 123",
                    "aif_TolFun = 10^-8",
                    "aif_TolX = 10^-9",
                    "aif_Robust = Bisquare",
                ],
            )
            config = _make_config(tmp, prefs)
            timer = np.linspace(0.0, 1.0, num=8, dtype=np.float64)
            curve = np.array([0.0, 0.0, 0.4, 1.2, 0.8, 0.6, 0.4, 0.3], dtype=np.float64)

            _fit_aif_biexp(
                config=config,
                timer=timer,
                curve=curve,
                start_injection_min=timer[2],
                end_injection_min=timer[4],
                fitting_au=False,
            )

            kwargs = curve_fit_mock.call_args.kwargs
            self.assertEqual(kwargs["method"], "trf")
            self.assertEqual(int(kwargs["max_nfev"]), 123)
            self.assertAlmostEqual(float(kwargs["ftol"]), 1e-8, places=12)
            self.assertAlmostEqual(float(kwargs["xtol"]), 1e-9, places=12)
            self.assertEqual(kwargs["loss"], "cauchy")


if __name__ == "__main__":
    unittest.main()

