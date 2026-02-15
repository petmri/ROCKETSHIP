"""Unit tests for DCE in-memory CLI pipeline scaffold."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import DcePipelineConfig, run_dce_pipeline, _run_stage_b_real, _run_stage_d_real  # noqa: E402


def _make_config(tmp_dir: Path) -> DcePipelineConfig:
    subject_source = tmp_dir / "subject_source"
    subject_tp = tmp_dir / "subject_tp"
    output = tmp_dir / "output"

    dynamic_file = subject_tp / "dynamic.nii.gz"
    aif_file = subject_tp / "aif_mask.nii.gz"
    t1map_file = subject_tp / "t1map.nii.gz"

    dynamic_file.parent.mkdir(parents=True, exist_ok=True)
    dynamic_file.write_text("")
    aif_file.write_text("")
    t1map_file.write_text("")

    return DcePipelineConfig(
        subject_source_path=subject_source,
        subject_tp_path=subject_tp,
        output_dir=output,
        backend="auto",
        checkpoint_dir=tmp_dir / "checkpoints",
        write_xls=True,
        aif_mode="auto",
        dynamic_files=[dynamic_file],
        aif_files=[aif_file],
        roi_files=[],
        t1map_files=[t1map_file],
        noise_files=[],
        drift_files=[],
        model_flags={"tofts": 1, "patlak": 1},
        stage_overrides={"stage_a_mode": "scaffold"},
    )


def _make_stage_a_payload() -> dict:
    n_time = 24
    n_tum = 7
    n_aif = 5
    timer = np.arange(n_time, dtype=np.float64) * 0.1

    cp_base = np.zeros(n_time, dtype=np.float64)
    cp_base[4:] = 1.3 * np.exp(-0.5 * (timer[4:] - timer[4])) + 0.5 * np.exp(-0.05 * (timer[4:] - timer[4]))
    cp_base[4:7] += np.linspace(0.2, 0.5, 3)

    cp = np.stack([cp_base * (1.0 + 0.02 * j) for j in range(n_aif)], axis=1)
    ct = np.stack([0.7 * cp_base * (1.0 + 0.03 * j) for j in range(n_tum)], axis=1)
    stlv = cp * 120.0 + 95.0
    sttum = ct * 100.0 + 90.0

    r1_toi = ct * 0.3 + 0.9
    r1_lv = cp * 0.25 + 1.0
    delta_r1_lv = r1_lv - np.mean(r1_lv[:2, :], axis=0)[np.newaxis, :]
    delta_r1_toi = r1_toi - np.mean(r1_toi[:2, :], axis=0)[np.newaxis, :]

    return {
        "stage": "A",
        "status": "ok",
        "impl": "real",
        "image_shape": [n_tum, 1, 1],
        "quant": True,
        "rootname": "python_dce",
        "time_resolution_min": 0.1,
        "start_injection": 5,
        "end_injection": 7,
        "arrays": {
            "Cp": cp,
            "Ct": ct,
            "Stlv": stlv,
            "Sttum": sttum,
            "Sss": np.mean(stlv[:2, :], axis=0),
            "Ssstum": np.mean(sttum[:2, :], axis=0),
            "R1tLV": r1_lv,
            "R1tTOI": r1_toi,
            "deltaR1LV": delta_r1_lv,
            "deltaR1TOI": delta_r1_toi,
            "T1TUM": np.full(n_tum, 1250.0),
            "tumind": np.arange(n_tum),
            "timer": timer,
        },
    }


class TestDcePipeline(unittest.TestCase):
    def test_pipeline_emits_progress_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            events: list[dict] = []
            run_dce_pipeline(config, event_callback=events.append)

            event_types = [str(e.get("type", "")) for e in events]
            self.assertIn("run_start", event_types)
            self.assertIn("run_done", event_types)
            self.assertIn("stage_start", event_types)
            self.assertIn("stage_done", event_types)

            stage_start = [e for e in events if e.get("type") == "stage_start"]
            stage_done = [e for e in events if e.get("type") == "stage_done"]
            self.assertEqual([e.get("stage") for e in stage_start], ["A", "B", "D"])
            self.assertEqual([e.get("stage") for e in stage_done], ["A", "B", "D"])

    def test_pipeline_writes_summary_and_stage_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            result = run_dce_pipeline(config)

            summary_path = Path(result["meta"]["summary_path"])
            self.assertTrue(summary_path.exists(), "Expected summary json to be written")

            for stage in ("a_out.json", "b_out.json", "d_out.json"):
                self.assertTrue((config.checkpoint_dir / stage).exists(), f"Missing checkpoint file: {stage}")

            summary_payload = json.loads(summary_path.read_text())
            self.assertIn("stages", summary_payload)
            self.assertIn("A", summary_payload["stages"])
            self.assertIn("B", summary_payload["stages"])
            self.assertIn("D", summary_payload["stages"])
            self.assertEqual(summary_payload["stages"]["B"]["impl"], "scaffold")

    def test_pipeline_rejects_imagej_roi_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            bad_roi = config.subject_tp_path / "bad.roi"
            bad_roi.write_text("roi")
            config.roi_files = [bad_roi]

            with self.assertRaisesRegex(ValueError, r"ImageJ ROI"):
                run_dce_pipeline(config)

    def test_backend_auto_falls_back_to_cpu_when_gpufit_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.backend = "auto"

            with patch(
                "dce_pipeline.probe_acceleration_backend",
                return_value={
                    "backend": "none",
                    "reason": "test_no_backend",
                    "cuda_available": False,
                    "pygpufit_imported": False,
                    "pycpufit_imported": False,
                    "pygpufit_error": None,
                    "pycpufit_error": None,
                },
            ):
                result = run_dce_pipeline(config)
            self.assertEqual(result["stages"]["D"]["selected_backend"], "cpu")

    def test_backend_gpufit_raises_when_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.backend = "gpufit"

            with patch(
                "dce_pipeline.probe_acceleration_backend",
                return_value={
                    "backend": "none",
                    "reason": "test_no_backend",
                    "cuda_available": False,
                    "pygpufit_imported": False,
                    "pycpufit_imported": False,
                    "pygpufit_error": "missing",
                    "pycpufit_error": "missing",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, r"GPUfit backend requested"):
                    run_dce_pipeline(config)

    def test_backend_auto_uses_cpufit_when_cuda_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.backend = "auto"

            with patch(
                "dce_pipeline.probe_acceleration_backend",
                return_value={
                    "backend": "cpufit_cpu",
                    "reason": "test_cpufit_only",
                    "cuda_available": False,
                    "pygpufit_imported": True,
                    "pycpufit_imported": True,
                    "pygpufit_error": None,
                    "pycpufit_error": None,
                },
            ):
                result = run_dce_pipeline(config)
            self.assertEqual(result["stages"]["D"]["selected_backend"], "gpufit")
            self.assertEqual(result["stages"]["D"]["acceleration_backend"], "cpufit_cpu")

    def test_backend_auto_prefers_gpufit_cuda_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.backend = "auto"

            with patch(
                "dce_pipeline.probe_acceleration_backend",
                return_value={
                    "backend": "gpufit_cuda",
                    "reason": "test_cuda_available",
                    "cuda_available": True,
                    "pygpufit_imported": True,
                    "pycpufit_imported": True,
                    "pygpufit_error": None,
                    "pycpufit_error": None,
                },
            ):
                result = run_dce_pipeline(config)
            self.assertEqual(result["stages"]["D"]["selected_backend"], "gpufit")
            self.assertEqual(result["stages"]["D"]["acceleration_backend"], "gpufit_cuda")

    def test_stage_a_real_mode_runs_with_mocked_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            roi_file = config.subject_tp_path / "roi_mask.nii.gz"
            roi_file.write_text("")
            config.roi_files = [roi_file]
            config.stage_overrides = {
                "stage_a_mode": "real",
                "stage_b_mode": "scaffold",
                "tr_ms": 5.0,
                "fa_deg": 15.0,
                "time_resolution_sec": 5.0,
                "steady_state_start": 1,
                "steady_state_end": 2,
                "snr_filter": 0.0,
            }

            rng = np.random.default_rng(42)
            dynamic = rng.normal(loc=100.0, scale=2.0, size=(6, 6, 2, 8))
            dynamic[..., 3:] += 15.0

            aif_mask = np.zeros((6, 6, 2), dtype=float)
            aif_mask[1:3, 1:3, :] = 1.0
            roi_mask = np.zeros((6, 6, 2), dtype=float)
            roi_mask[3:5, 3:5, :] = 1.0
            t1map = np.full((6, 6, 2), 1300.0, dtype=float)

            def fake_load(path: Path):
                p = str(path)
                if p.endswith("dynamic.nii.gz"):
                    return dynamic
                if p.endswith("aif_mask.nii.gz"):
                    return aif_mask
                if p.endswith("roi_mask.nii.gz"):
                    return roi_mask
                if p.endswith("t1map.nii.gz"):
                    return t1map
                raise AssertionError(f"Unexpected path {path}")

            with patch("dce_pipeline._load_nifti_data", side_effect=fake_load):
                with patch(
                    "dce_pipeline._save_stage_a_qc_figures",
                    return_value={"timecurves_png": "/tmp/a.png", "roi_overview_png": "/tmp/b.png"},
                ):
                    with patch(
                        "dce_pipeline._clean_ab",
                        side_effect=lambda ab, t1_vals, idx_vals, threshold_fraction: (ab, t1_vals, idx_vals),
                    ):
                        with patch(
                            "dce_pipeline._clean_r1",
                            side_effect=lambda r1_vals, t1_vals, idx_vals, threshold_fraction: (
                                r1_vals,
                                t1_vals,
                                idx_vals,
                            ),
                        ):
                            result = run_dce_pipeline(config)

            self.assertEqual(result["stages"]["A"]["impl"], "real")
            self.assertEqual(result["stages"]["A"]["status"], "ok")
            self.assertIn("array_shapes", result["stages"]["A"])
            self.assertIn("Cp", result["stages"]["A"]["array_shapes"])
            self.assertIn("Ct", result["stages"]["A"]["array_shapes"])

    def test_stage_b_real_raw_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {"stage_b_mode": "real", "aif_curve_mode": "raw"}
            stage_a = _make_stage_a_payload()

            result = _run_stage_b_real(config, stage_a)
            self.assertEqual(result["impl"], "real")
            self.assertEqual(result["aif_name"], "raw")
            self.assertIn("Cp_use", result["arrays"])
            self.assertEqual(result["arrays"]["Cp_use"].shape[0], result["arrays"]["timer"].shape[0])

    def test_stage_b_real_fitted_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_b_mode": "real",
                "aif_curve_mode": "fitted",
                "start_time_min": 0.0,
                "end_time_min": 0.0,
                "aif_MaxFunEvals": 4000,
            }
            stage_a = _make_stage_a_payload()

            result = _run_stage_b_real(config, stage_a)
            self.assertEqual(result["impl"], "real")
            self.assertEqual(result["aif_name"], "fitted")
            self.assertIn("fit_params_cp", result)
            self.assertEqual(result["arrays"]["Cp_use"].shape, result["arrays"]["CpROI"].shape)

    def test_stage_b_real_imported_mode_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            stage_a = _make_stage_a_payload()
            timer = stage_a["arrays"]["timer"]
            cp_use = np.mean(stage_a["arrays"]["Cp"], axis=1) * 0.95
            stlv_use = np.mean(stage_a["arrays"]["Stlv"], axis=1) * 1.01

            imported_path = Path(tmp) / "imported_aif.npz"
            np.savez_compressed(
                imported_path,
                Cp_use=cp_use,
                Stlv_use=stlv_use,
                timer=timer,
                start_injection=timer[4],
            )
            config.imported_aif_path = imported_path
            config.aif_mode = "imported"
            config.stage_overrides = {"stage_b_mode": "real"}

            result = _run_stage_b_real(config, stage_a)
            self.assertEqual(result["impl"], "real")
            self.assertEqual(result["aif_name"], "imported")
            self.assertIn("Cp_use", result["arrays"])
            self.assertEqual(result["arrays"]["Cp_use"].shape[0], timer.shape[0])

    def test_stage_d_real_generates_maps_and_xls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            roi_path = config.subject_tp_path / "roi_mask_for_d.nii.gz"
            roi_path.write_text("")
            config.roi_files = [roi_path]
            config.stage_overrides = {
                "stage_b_mode": "real",
                "stage_d_mode": "real",
                "aif_curve_mode": "raw",
                "write_param_maps": True,
            }
            config.model_flags = {"tofts": 1, "patlak": 1}

            stage_a = _make_stage_a_payload()
            stage_b = _run_stage_b_real(config, stage_a)

            roi_mask = np.zeros((7, 1, 1), dtype=float)
            roi_mask[:4, 0, 0] = 1.0

            def fake_load(path: Path):
                if str(path).endswith("roi_mask_for_d.nii.gz"):
                    return roi_mask
                raise AssertionError(f"Unexpected path {path}")

            with patch("dce_pipeline._load_nifti_data", side_effect=fake_load):
                stage_d = _run_stage_d_real(config, stage_a, stage_b)

            self.assertEqual(stage_d["impl"], "real")
            self.assertIn("tofts", stage_d["models_run"])
            self.assertIn("patlak", stage_d["models_run"])

            tofts_out = stage_d["model_outputs"]["tofts"]
            self.assertTrue(tofts_out["map_paths"])
            for _, out_path in tofts_out["map_paths"].items():
                self.assertTrue(Path(out_path).exists())
            self.assertTrue(tofts_out["xls_path"])
            self.assertTrue(Path(tofts_out["xls_path"]).exists())

    def test_pipeline_end_to_end_real_abd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            roi_file = config.subject_tp_path / "roi_mask.nii.gz"
            roi_file.write_text("")
            config.roi_files = [roi_file]
            config.model_flags = {"tofts": 1, "patlak": 1}
            config.stage_overrides = {
                "stage_a_mode": "real",
                "stage_b_mode": "real",
                "stage_d_mode": "real",
                "aif_curve_mode": "raw",
                "tr_ms": 5.0,
                "fa_deg": 15.0,
                "time_resolution_sec": 5.0,
                "steady_state_start": 1,
                "steady_state_end": 2,
                "snr_filter": 0.0,
                "write_param_maps": True,
            }

            rng = np.random.default_rng(42)
            dynamic = rng.normal(loc=100.0, scale=2.0, size=(6, 6, 2, 8))
            dynamic[..., 3:] += 15.0

            aif_mask = np.zeros((6, 6, 2), dtype=float)
            aif_mask[1:3, 1:3, :] = 1.0
            roi_mask = np.zeros((6, 6, 2), dtype=float)
            roi_mask[3:5, 3:5, :] = 1.0
            t1map = np.full((6, 6, 2), 1300.0, dtype=float)

            def fake_load(path: Path):
                p = str(path)
                if p.endswith("dynamic.nii.gz"):
                    return dynamic
                if p.endswith("aif_mask.nii.gz"):
                    return aif_mask
                if p.endswith("roi_mask.nii.gz"):
                    return roi_mask
                if p.endswith("t1map.nii.gz"):
                    return t1map
                raise AssertionError(f"Unexpected path {path}")

            with patch("dce_pipeline._load_nifti_data", side_effect=fake_load):
                with patch(
                    "dce_pipeline._save_stage_a_qc_figures",
                    return_value={"timecurves_png": "/tmp/a.png", "roi_overview_png": "/tmp/b.png"},
                ):
                    with patch(
                        "dce_pipeline._save_stage_b_qc_figure",
                        return_value={"aif_fitting_png": "/tmp/c.png"},
                    ):
                        with patch(
                            "dce_pipeline._clean_ab",
                            side_effect=lambda ab, t1_vals, idx_vals, threshold_fraction: (ab, t1_vals, idx_vals),
                        ):
                            with patch(
                                "dce_pipeline._clean_r1",
                                side_effect=lambda r1_vals, t1_vals, idx_vals, threshold_fraction: (
                                    r1_vals,
                                    t1_vals,
                                    idx_vals,
                                ),
                            ):
                                result = run_dce_pipeline(config)

            self.assertEqual(result["stages"]["A"]["impl"], "real")
            self.assertEqual(result["stages"]["B"]["impl"], "real")
            self.assertEqual(result["stages"]["D"]["impl"], "real")
            self.assertIn("tofts", result["stages"]["D"]["models_run"])
            self.assertIn("patlak", result["stages"]["D"]["models_run"])


if __name__ == "__main__":
    unittest.main()
