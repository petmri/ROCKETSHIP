"""Unit tests for DCE in-memory CLI pipeline scaffold."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    MODEL_LAYOUTS,
    _accelerated_output_has_usable_primary_params,
    _fit_stage_d_model,
    _fit_stage_d_model_accelerated,
    _glr_baseline_end,
    _legacy_sobel_baseline_end,
    _piecewise_constant_baseline_end,
    _resolve_baseline_window,
    _resolve_timepoint_window,
    _resolve_dynamic_metadata,
    _tv_baseline_end,
    run_dce_pipeline,
    _run_stage_a_real,
    _run_stage_b_real,
    _run_stage_d_real,
)


class _FakeAccelModule:
    class ModelID:
        PATLAK = 17
        TOFTS = 18
        TOFTS_EXTENDED = 19
        TISSUE_UPTAKE = 20
        TWO_COMPARTMENT_EXCHANGE = 21

    class ConstraintType:
        LOWER_UPPER = 3

    class EstimatorID:
        LSE = 0

    last_call = None

    @classmethod
    def fit_constrained(
        cls,
        *,
        data,
        weights,
        model_id,
        initial_parameters,
        constraints,
        constraint_types,
        tolerance,
        max_number_iterations,
        parameters_to_fit,
        estimator_id,
        user_info,
    ):
        del weights, tolerance, max_number_iterations, parameters_to_fit, estimator_id, user_info
        cls.last_call = {
            "model_id": int(model_id),
            "initial_parameters": np.asarray(initial_parameters, dtype=np.float32),
            "constraints": np.asarray(constraints, dtype=np.float32),
            "constraint_types": np.asarray(constraint_types, dtype=np.int32),
        }
        n_fits = int(np.asarray(data).shape[0])
        n_params = int(np.asarray(initial_parameters).shape[1])
        params = np.tile(np.arange(1, n_params + 1, dtype=np.float32), (n_fits, 1))
        states = np.zeros((n_fits,), dtype=np.int32)
        chi = np.full((n_fits,), 0.1, dtype=np.float32)
        n_iterations = np.ones((n_fits,), dtype=np.int32)
        exec_time = np.zeros((n_fits,), dtype=np.float32)
        return params, states, chi, n_iterations, exec_time


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


@pytest.mark.integration
class TestDcePipeline:
    def test_pipeline_emits_progress_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            events: list[dict] = []
            run_dce_pipeline(config, event_callback=events.append)

            event_types = [str(e.get("type", "")) for e in events]
            assert "run_start" in event_types
            assert "run_done" in event_types
            assert "stage_start" in event_types
            assert "stage_done" in event_types

            stage_start = [e for e in events if e.get("type") == "stage_start"]
            stage_done = [e for e in events if e.get("type") == "stage_done"]
            assert [e.get("stage") for e in stage_start] == ["A", "B", "D"]
            assert [e.get("stage") for e in stage_done] == ["A", "B", "D"]

    def test_pipeline_writes_summary_and_stage_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            result = run_dce_pipeline(config)

            summary_path = Path(result["meta"]["summary_path"])
            assert summary_path.exists(), "Expected summary json to be written"

            for stage in ("a_out.json", "b_out.json", "d_out.json"):
                assert (config.checkpoint_dir / stage).exists(), f"Missing checkpoint file: {stage}"

            summary_payload = json.loads(summary_path.read_text())
            assert "stages" in summary_payload
            assert "A" in summary_payload["stages"]
            assert "B" in summary_payload["stages"]
            assert "D" in summary_payload["stages"]
            assert summary_payload["stages"]["B"]["impl"] == "scaffold"

    def test_pipeline_rejects_imagej_roi_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            bad_roi = config.subject_tp_path / "bad.roi"
            bad_roi.write_text("roi")
            config.roi_files = [bad_roi]

            with pytest.raises(ValueError, match=r"ImageJ ROI"):
                run_dce_pipeline(config)

    def test_pipeline_rejects_reused_roi_as_aif_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.roi_files = [config.aif_files[0]]

            with pytest.raises(ValueError, match=r"AIF mask must be a dedicated vascular ROI"):
                run_dce_pipeline(config)

    def test_dynamic_metadata_requires_complete_manual_tuple_when_json_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_a_mode": "real",
                "tr_ms": 5.0,
                "fa_deg": 15.0,
            }
            with pytest.raises(ValueError, match=r"DCE metadata JSON not found"):
                _resolve_dynamic_metadata(config, n_timepoints=8)

    def test_dynamic_metadata_rejects_partial_manual_override_when_json_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            dynamic_text = str(config.dynamic_files[0])
            sidecar = Path(dynamic_text[:-7] + ".json")
            sidecar.write_text(json.dumps({"RepetitionTime": 0.005, "FlipAngle": 17}))
            config.stage_overrides = {
                "stage_a_mode": "real",
                "tr_ms": 5.0,
            }
            with pytest.raises(ValueError, match=r"Partial manual DCE metadata override is not allowed"):
                _resolve_dynamic_metadata(config, n_timepoints=8)

    def test_dynamic_metadata_json_requires_frame_spacing_when_missing_in_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_a_mode": "real",
                "use_dce_preferences": False,
            }
            dynamic_text = str(config.dynamic_files[0])
            sidecar = Path(dynamic_text[:-7] + ".json")
            sidecar.write_text(json.dumps({"RepetitionTime": 0.005, "FlipAngle": 17}))
            with pytest.raises(ValueError, match=r"Unable to determine DCE frame spacing"):
                _resolve_dynamic_metadata(config, n_timepoints=8)

    def test_dynamic_metadata_reads_relaxivity_and_hematocrit_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            dynamic_text = str(config.dynamic_files[0])
            sidecar = Path(dynamic_text[:-7] + ".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "RepetitionTime": 0.0085,
                        "TemporalResolution": 12.5,
                        "FlipAngle": 17.0,
                        "SyntheticPhantom": {
                            "Relaxivity_per_mM_per_s": 5.25,
                            "RecommendedROCKETSHIPHematocrit": 0.0,
                            "AIFConcentrationKind": "plasma",
                        },
                    }
                )
            )

            out = _resolve_dynamic_metadata(config, n_timepoints=8)
            assert out["tr_ms"] == pytest.approx(8.5)
            assert out["time_resolution_sec"] == pytest.approx(12.5)
            assert out["fa_deg"] == pytest.approx(17.0)
            assert out["relaxivity"] == pytest.approx(5.25)
            assert out["hematocrit"] == pytest.approx(0.0)
            assert out["aif_concentration_kind"] == "plasma"
            assert "SyntheticPhantom.Relaxivity_per_mM_per_s" in str(out["metadata_sources"].get("relaxivity", ""))
            assert "SyntheticPhantom.RecommendedROCKETSHIPHematocrit" in str(
                out["metadata_sources"].get("hematocrit", "")
            )

    def test_dynamic_metadata_time_resolution_from_repetition_time_with_rte_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            dynamic_text = str(config.dynamic_files[0])
            sidecar = Path(dynamic_text[:-7] + ".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "RepetitionTimeExcitation": 0.008,
                        "RepetitionTime": 15.3239,
                        "FlipAngle": 17.0,
                    }
                )
            )

            out = _resolve_dynamic_metadata(config, n_timepoints=64)
            assert out["tr_ms"] == pytest.approx(8.0)
            assert out["time_resolution_sec"] == pytest.approx(15.3239)
            assert "RepetitionTime@RepetitionTimeExcitation" in str(out["metadata_sources"]["time_resolution_sec"])

    def test_dynamic_metadata_time_resolution_from_acquisition_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            dynamic_text = str(config.dynamic_files[0])
            sidecar = Path(dynamic_text[:-7] + ".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "RepetitionTime": 0.008,
                        "AcquisitionDuration": 12.8,
                        "FlipAngle": 15.0,
                    }
                )
            )

            out = _resolve_dynamic_metadata(config, n_timepoints=64)
            assert out["tr_ms"] == pytest.approx(8.0)
            assert out["time_resolution_sec"] == pytest.approx(12.8)
            assert str(out["metadata_sources"]["time_resolution_sec"]).endswith(".AcquisitionDuration")

    def test_dynamic_metadata_time_resolution_from_trigger_delay_time_uses_raw_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            dynamic_text = str(config.dynamic_files[0])
            sidecar = Path(dynamic_text[:-7] + ".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "RepetitionTime": 0.008,
                        "TriggerDelayTime": 64000.0,
                        "FlipAngle": 15.0,
                    }
                )
            )

            out = _resolve_dynamic_metadata(config, n_timepoints=5, n_timepoints_raw=64)
            assert out["tr_ms"] == pytest.approx(8.0)
            assert out["time_resolution_sec"] == pytest.approx(1.0)
            assert "TriggerDelayTime/64/1000" in str(out["metadata_sources"]["time_resolution_sec"])

    def test_legacy_sobel_baseline_end_detects_pre_rise_frames(self) -> None:
        n_time = 40
        t = np.arange(n_time, dtype=np.float64)
        mean_curve = np.zeros(n_time, dtype=np.float64)
        mean_curve[:8] = 100.0
        mean_curve[8:14] = np.linspace(100.0, 150.0, 6)
        mean_curve[14:] = 150.0 + 15.0 * np.exp(-(t[14:] - t[14]) / 8.0)
        stlv = np.tile(mean_curve[:, np.newaxis], (1, 6))

        out = _legacy_sobel_baseline_end(stlv)

        assert out["method"] == "legacy_sobel"
        assert 2 <= int(out["end_ss_1b"]) <= 10
        assert int(out["global_edge_index_1b"]) >= int(out["end_ss_1b"])

    def test_piecewise_constant_baseline_end_returns_local_min_before_transition(self) -> None:
        n_time = 36
        t = np.arange(n_time, dtype=np.float64)
        mean_curve = np.full(n_time, 90.0, dtype=np.float64)
        mean_curve[5:10] = np.array([89.7, 89.4, 89.0, 89.3, 89.8], dtype=np.float64)
        mean_curve[10:] += 70.0
        mean_curve[10:] += 8.0 * np.exp(-(t[10:] - t[10]) / 4.0)
        stlv = np.tile(mean_curve[:, np.newaxis], (1, 4))

        out = _piecewise_constant_baseline_end(stlv)

        assert out["method"] == "piecewise_constant"
        assert 6 <= int(out["end_ss_1b"]) <= 10
        assert int(out["transition_index_1b"]) >= int(out["end_ss_1b"])
        assert int(out["local_max_index_1b"]) >= int(out["transition_index_1b"])

    def test_piecewise_constant_baseline_end_flat_baseline_tracks_forward_within_delta(self) -> None:
        n_time = 30
        mean_curve = np.full(n_time, 100.0, dtype=np.float64)
        mean_curve[8:] = 160.0
        stlv = np.tile(mean_curve[:, np.newaxis], (1, 5))

        out = _piecewise_constant_baseline_end(stlv)

        assert out["method"] == "piecewise_constant"
        # Without the flat-baseline forward step this tends to collapse to frame 1.
        assert int(out["end_ss_1b"]) > 1
        assert int(out["end_ss_1b"]) <= int(out["transition_index_1b"])
        assert out["flat_forward_delta_fraction"] == pytest.approx(0.01)
        assert out["flat_forward_delta_abs"] > 0.0

    def test_glr_baseline_end_detects_step_change(self) -> None:
        n_time = 40
        mean_curve = np.full(n_time, 100.0, dtype=np.float64)
        mean_curve[:8] += np.array([0.0, 0.2, -0.1, 0.1, -0.2, 0.15, -0.05, 0.0], dtype=np.float64)
        mean_curve[8:] += np.linspace(25.0, 15.0, n_time - 8)
        stlv = np.tile(mean_curve[:, np.newaxis], (1, 6))

        out = _glr_baseline_end(stlv)

        assert out["method"] == "glr"
        assert out["mode"] in {"glr_score", "early_jump"}
        assert 6 <= int(out["end_ss_1b"]) <= 10

    def test_tv_baseline_end_detects_step_change(self) -> None:
        n_time = 48
        mean_curve = np.full(n_time, 95.0, dtype=np.float64)
        mean_curve[:10] += np.array([0.0, -0.1, 0.15, -0.05, 0.08, -0.06, 0.1, -0.02, 0.04, 0.0], dtype=np.float64)
        mean_curve[10:] += 18.0 + 10.0 * np.exp(-np.arange(n_time - 10, dtype=np.float64) / 8.0)
        stlv = np.tile(mean_curve[:, np.newaxis], (1, 5))

        out = _tv_baseline_end(stlv)

        assert out["method"] == "tv"
        assert out["mode"] == "tv_jump"
        assert 8 <= int(out["end_ss_1b"]) <= 12
        assert out["lambda_tv"] >= 0.0
        assert 0.0 <= out["strength"] <= 1.0

    def test_resolve_baseline_window_uses_selected_auto_method_when_end_not_manual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_a_mode": "scaffold",
                "steady_state_auto_method": "piecewise_constant",
                "steady_state_start": 1,
            }
            mean_curve = np.full(24, 100.0, dtype=np.float64)
            mean_curve[4:7] = np.array([99.5, 99.0, 99.3], dtype=np.float64)
            mean_curve[7:] = 140.0
            stlv = np.tile(mean_curve[:, np.newaxis], (1, 3))

            ss_start, ss_end, info = _resolve_baseline_window(config, n_timepoints=24, stlv=stlv)

            assert ss_start == 0
            assert 4 <= ss_end <= 8
            assert info["method_requested"] == "piecewise_constant"
            assert info["method_used"] == "piecewise_constant"
            assert info["source"] == "steady_state_auto_method:piecewise_constant"

    def test_resolve_baseline_window_manual_end_overrides_auto_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_a_mode": "scaffold",
                "steady_state_start": 1,
                "steady_state_end": 3,
                "steady_state_auto_method": "legacy_sobel",
            }
            stlv = np.full((12, 2), 100.0, dtype=np.float64)

            ss_start, ss_end, info = _resolve_baseline_window(config, n_timepoints=12, stlv=stlv)

            assert (ss_start, ss_end) == (0, 3)
            assert info["method_requested"] == "legacy_sobel"
            assert info["method_used"] == "manual"
            assert info["source"] == "steady_state_end"

    def test_resolve_baseline_window_accepts_glr_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_a_mode": "scaffold",
                "steady_state_auto_method": "find_end_ss_edge",
            }
            mean_curve = np.full(20, 100.0, dtype=np.float64)
            mean_curve[5:] += 20.0
            stlv = np.tile(mean_curve[:, np.newaxis], (1, 4))

            _, ss_end, info = _resolve_baseline_window(config, n_timepoints=20, stlv=stlv)

            assert 4 <= ss_end <= 7
            assert info["method_requested"] == "glr"
            assert info["method_used"] == "glr"

    def test_resolve_baseline_window_defaults_to_legacy_sobel_when_no_options_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_a_mode": "scaffold",
                "use_dce_preferences": False,
            }
            mean_curve = np.full(24, 100.0, dtype=np.float64)
            mean_curve[8:12] = np.linspace(100.0, 140.0, 4)
            mean_curve[12:] = 140.0
            stlv = np.tile(mean_curve[:, np.newaxis], (1, 4))

            ss_start, ss_end, info = _resolve_baseline_window(config, n_timepoints=24, stlv=stlv)

            assert ss_start == 0
            assert 1 <= ss_end <= 12
            assert info["method_requested"] == "none"
            assert info["method_used"] == "legacy_sobel"
            assert info["source"] == "default_auto_method:legacy_sobel"

    def test_resolve_timepoint_window_defaults_to_full_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {"stage_a_mode": "scaffold"}

            start_0b, end_0b_exclusive, info = _resolve_timepoint_window(config, n_timepoints=12)

            assert (start_0b, end_0b_exclusive) == (0, 12)
            assert info["source"] == "default_full_range"
            assert int(info["n_timepoints_output"]) == 12

    def test_resolve_timepoint_window_uses_start_t_end_t(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {"stage_a_mode": "scaffold", "start_t": 3, "end_t": 8}

            start_0b, end_0b_exclusive, info = _resolve_timepoint_window(config, n_timepoints=12)

            assert (start_0b, end_0b_exclusive) == (2, 8)
            assert info["source"] == "start_t/end_t"
            assert int(info["n_timepoints_output"]) == 6

    def test_resolve_timepoint_window_treats_nonpositive_end_t_as_full_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {"stage_a_mode": "scaffold", "start_t": 4, "end_t": 0}

            start_0b, end_0b_exclusive, info = _resolve_timepoint_window(config, n_timepoints=10)

            assert (start_0b, end_0b_exclusive) == (3, 10)
            assert info["source"] == "start_t/end_t"
            assert int(info["n_timepoints_output"]) == 7

    def test_resolve_timepoint_window_rejects_inverted_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {"stage_a_mode": "scaffold", "start_t": 9, "end_t": 2}

            with pytest.raises(ValueError, match=r"start_t=9 must be <= end_t=2"):
                _resolve_timepoint_window(config, n_timepoints=12)

    def test_validate_accepts_import_aif_path_alias_for_imported_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.aif_mode = "imported"
            config.imported_aif_path = None
            config.stage_overrides = {
                "stage_a_mode": "scaffold",
                "stage_b_mode": "scaffold",
                "stage_d_mode": "scaffold",
                "import_aif_path": str(Path(tmp) / "imported_alias.npz"),
            }
            config.validate()

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
            assert result["stages"]["D"]["selected_backend"] == "cpu"

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
                with pytest.raises(RuntimeError, match=r"GPUfit backend requested"):
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
            assert result["stages"]["D"]["selected_backend"] == "gpufit"
            assert result["stages"]["D"]["acceleration_backend"] == "cpufit_cpu"

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
            assert result["stages"]["D"]["selected_backend"] == "gpufit"
            assert result["stages"]["D"]["acceleration_backend"] == "gpufit_cuda"

    def test_stage_d_gpu_failure_falls_back_to_cpufit_before_cpu(self) -> None:
        ct = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [1.2, 2.2, 3.2],
                [1.3, 2.3, 3.3],
                [1.4, 2.4, 3.4],
            ],
            dtype=np.float64,
        )
        cp = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64)
        timer = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        calls: list[str] = []

        def fake_accel(**kwargs):
            backend = str(kwargs["acceleration_backend"])
            calls.append(backend)
            if backend == "gpufit_cuda":
                raise RuntimeError("the provided PTX was compiled with an unsupported toolchain")
            if backend == "cpufit_cpu":
                return np.full((ct.shape[1], 7), 42.0, dtype=np.float64)
            return None

        with patch("dce_pipeline._cpufit_import_available", return_value=True):
            with patch("dce_pipeline._fit_stage_d_model_accelerated", side_effect=fake_accel):
                out = _fit_stage_d_model(
                    model_name="tofts",
                    ct=ct,
                    cp_use=cp,
                    timer=timer,
                    prefs={},
                    r1o=None,
                    relaxivity=3.6,
                    fw=0.8,
                    stlv_use=None,
                    sttum=None,
                    start_injection_min=0.0,
                    sss=None,
                    ssstum=None,
                    acceleration_backend="gpufit_cuda",
                )

        assert calls == ["gpufit_cuda", "cpufit_cpu"]
        assert out.shape == (ct.shape[1], 7)
        assert np.allclose(out, 42.0)

    def test_stage_d_gpu_failure_without_cpufit_falls_back_to_cpu(self) -> None:
        ct = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [1.2, 2.2, 3.2],
                [1.3, 2.3, 3.3],
                [1.4, 2.4, 3.4],
            ],
            dtype=np.float64,
        )
        cp = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64)
        timer = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        accel_calls: list[str] = []
        cpu_calls: list[int] = []

        def fake_accel(**kwargs):
            accel_calls.append(str(kwargs["acceleration_backend"]))
            raise RuntimeError("the provided PTX was compiled with an unsupported toolchain")

        def fake_cpu_fit(*args, **kwargs):
            del args, kwargs
            cpu_calls.append(1)
            return np.asarray([0.2, 0.3, 0.4, 0.2, 0.2, 0.3, 0.3], dtype=np.float64)

        with patch("dce_pipeline._cpufit_import_available", return_value=False):
            with patch("dce_pipeline._fit_stage_d_model_accelerated", side_effect=fake_accel):
                with patch("dce_pipeline._fit_model_curve", side_effect=fake_cpu_fit):
                    out = _fit_stage_d_model(
                        model_name="tofts",
                        ct=ct,
                        cp_use=cp,
                        timer=timer,
                        prefs={},
                        r1o=None,
                        relaxivity=3.6,
                        fw=0.8,
                        stlv_use=None,
                        sttum=None,
                        start_injection_min=0.0,
                        sss=None,
                        ssstum=None,
                        acceleration_backend="gpufit_cuda",
                    )

        assert accel_calls == ["gpufit_cuda"]
        assert len(cpu_calls) == ct.shape[1]
        assert out.shape == (ct.shape[1], 7)
        assert np.allclose(out[:, 0], 0.2)
        assert np.allclose(out[:, 1], 0.3)

    def test_stage_d_cpufit_failure_falls_back_to_gpufit(self) -> None:
        ct = np.asarray(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
                [1.4, 2.4],
            ],
            dtype=np.float64,
        )
        cp = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64)
        timer = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        calls: list[str] = []

        def fake_accel(**kwargs):
            backend = str(kwargs["acceleration_backend"])
            calls.append(backend)
            if backend == "cpufit_cpu":
                raise RuntimeError("status = -1, message = unknown model ID")
            if backend == "gpufit_cpu_fallback":
                return np.full((ct.shape[1], 10), 9.0, dtype=np.float64)
            return None

        with patch("dce_pipeline._gpufit_import_available", return_value=True):
            with patch("dce_pipeline._fit_stage_d_model_accelerated", side_effect=fake_accel):
                out = _fit_stage_d_model(
                    model_name="ex_tofts",
                    ct=ct,
                    cp_use=cp,
                    timer=timer,
                    prefs={},
                    r1o=None,
                    relaxivity=3.6,
                    fw=0.8,
                    stlv_use=None,
                    sttum=None,
                    start_injection_min=0.0,
                    sss=None,
                    ssstum=None,
                    acceleration_backend="cpufit_cpu",
                )

        assert calls == ["cpufit_cpu", "gpufit_cpu_fallback"]
        assert out.shape == (ct.shape[1], 10)
        assert np.allclose(out, 9.0)

    def test_stage_d_nonfinite_accelerated_output_falls_back_to_cpu(self) -> None:
        ct = np.asarray(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
            ],
            dtype=np.float64,
        )
        cp = np.asarray([0.8, 0.9, 1.0, 1.1], dtype=np.float64)
        timer = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)

        row_len = len(MODEL_LAYOUTS["ex_tofts"]["param_names"])
        accel_nan = np.full((ct.shape[1], row_len), np.nan, dtype=np.float64)
        cpu_row = np.asarray([0.2, 0.3, 0.1, 0.05, 0.2, 0.2, 0.3, 0.3, 0.1, 0.1], dtype=np.float64)
        cpu_calls: list[int] = []

        def fake_cpu_fit(*args, **kwargs):
            del args, kwargs
            cpu_calls.append(1)
            return cpu_row

        with patch("dce_pipeline._gpufit_import_available", return_value=False):
            with patch("dce_pipeline._fit_stage_d_model_accelerated", return_value=accel_nan):
                with patch("dce_pipeline._fit_model_curve", side_effect=fake_cpu_fit):
                    out = _fit_stage_d_model(
                        model_name="ex_tofts",
                        ct=ct,
                        cp_use=cp,
                        timer=timer,
                        prefs={},
                        r1o=None,
                        relaxivity=3.6,
                        fw=0.8,
                        stlv_use=None,
                        sttum=None,
                        start_injection_min=0.0,
                        sss=None,
                        ssstum=None,
                        acceleration_backend="cpufit_cpu",
                    )

        assert len(cpu_calls) == ct.shape[1]
        assert out.shape == (ct.shape[1], row_len)
        assert np.all(np.isfinite(out[:, :3]))
        assert np.allclose(out[0, :], cpu_row)

    def test_stage_d_nonfinite_accelerated_output_tries_fallback_acceleration(self) -> None:
        ct = np.asarray(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
            ],
            dtype=np.float64,
        )
        cp = np.asarray([0.8, 0.9, 1.0, 1.1], dtype=np.float64)
        timer = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)

        row_len = len(MODEL_LAYOUTS["ex_tofts"]["param_names"])
        accel_nan = np.full((ct.shape[1], row_len), np.nan, dtype=np.float64)
        accel_ok = np.full((ct.shape[1], row_len), 7.0, dtype=np.float64)
        calls: list[str] = []

        def fake_accel(**kwargs):
            backend = str(kwargs["acceleration_backend"])
            calls.append(backend)
            if backend == "cpufit_cpu":
                return accel_nan
            if backend == "gpufit_cpu_fallback":
                return accel_ok
            return None

        with patch("dce_pipeline._gpufit_import_available", return_value=True):
            with patch("dce_pipeline._fit_stage_d_model_accelerated", side_effect=fake_accel):
                out = _fit_stage_d_model(
                    model_name="ex_tofts",
                    ct=ct,
                    cp_use=cp,
                    timer=timer,
                    prefs={},
                    r1o=None,
                    relaxivity=3.6,
                    fw=0.8,
                    stlv_use=None,
                    sttum=None,
                    start_injection_min=0.0,
                    sss=None,
                    ssstum=None,
                    acceleration_backend="cpufit_cpu",
                )

        assert calls == ["cpufit_cpu", "gpufit_cpu_fallback"]
        assert out.shape == (ct.shape[1], row_len)
        assert np.allclose(out, 7.0)

    def test_accelerated_output_usable_primary_params_helper(self) -> None:
        good = np.asarray([[0.2, 0.3, 0.1], [np.nan, np.nan, np.nan]], dtype=np.float64)
        bad = np.asarray([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float64)
        assert _accelerated_output_has_usable_primary_params("ex_tofts", good)
        assert not _accelerated_output_has_usable_primary_params("ex_tofts", bad)

    @pytest.mark.parametrize(
        "model_name,model_id,expected_init,expected_bounds,expected_row0",
        [
            (
                "tofts",
                _FakeAccelModule.ModelID.TOFTS,
                [0.11, 0.22],
                [0.01, 1.01, 0.02, 1.02],
                [1.0, 2.0, 0.1, 1.0, 1.0, 2.0, 2.0],
            ),
            (
                "ex_tofts",
                _FakeAccelModule.ModelID.TOFTS_EXTENDED,
                [0.11, 0.22, 0.33],
                [0.01, 1.01, 0.02, 1.02, 0.03, 1.03],
                [1.0, 2.0, 3.0, 0.1, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            ),
            (
                "patlak",
                _FakeAccelModule.ModelID.PATLAK,
                [0.11, 0.33],
                [0.01, 1.01, 0.03, 1.03],
                [1.0, 2.0, 0.1, -1.0, -1.0, -1.0, -1.0],
            ),
            (
                "tissue_uptake",
                _FakeAccelModule.ModelID.TISSUE_UPTAKE,
                [0.11, 0.33, 0.44],
                [0.01, 1.01, 0.03, 1.03, 0.04, 1.04],
                [1.0, 3.0, 2.0, 0.1, 1.0, 1.0, 3.0, 3.0, 2.0, 2.0],
            ),
            (
                "2cxm",
                _FakeAccelModule.ModelID.TWO_COMPARTMENT_EXCHANGE,
                [0.11, 0.22, 0.33, 0.44],
                [0.01, 1.01, 0.02, 1.02, 0.03, 1.03, 0.04, 1.04],
                [1.0, 2.0, 3.0, 4.0, 0.1, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            ),
        ],
    )
    def test_stage_d_accelerated_outputs_for_supported_models(
        self,
        model_name: str,
        model_id: int,
        expected_init: list[float],
        expected_bounds: list[float],
        expected_row0: list[float],
    ) -> None:
        ct = np.asarray(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
                [1.4, 2.4],
            ],
            dtype=np.float64,
        )
        cp = np.asarray([0.7, 0.8, 0.9, 1.0, 1.1], dtype=np.float64)
        timer = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        prefs = {
            "initial_value_ktrans": 0.11,
            "initial_value_ve": 0.22,
            "initial_value_vp": 0.33,
            "initial_value_fp": 0.44,
            "lower_limit_ktrans": 0.01,
            "upper_limit_ktrans": 1.01,
            "lower_limit_ve": 0.02,
            "upper_limit_ve": 1.02,
            "lower_limit_vp": 0.03,
            "upper_limit_vp": 1.03,
            "lower_limit_fp": 0.04,
            "upper_limit_fp": 1.04,
            "gpu_tolerance": 1e-6,
            "gpu_max_n_iterations": 64,
        }

        _FakeAccelModule.last_call = None
        with patch("dce_pipeline._load_fit_module_for_acceleration", return_value=_FakeAccelModule):
            out = _fit_stage_d_model_accelerated(
                model_name=model_name,
                ct=ct,
                cp_use=cp,
                timer=timer,
                prefs=prefs,
                acceleration_backend="cpufit_cpu",
            )

        assert out is not None, f"{model_name}: expected accelerated output"
        assert out.shape == (ct.shape[1], len(MODEL_LAYOUTS[model_name]["param_names"])), (
            f"{model_name}: unexpected output shape"
        )
        assert np.allclose(np.asarray(out)[0, :], np.asarray(expected_row0, dtype=np.float64)), (
            f"{model_name}: unexpected output row"
        )
        assert _FakeAccelModule.last_call is not None, f"{model_name}: accelerator call missing"
        assert _FakeAccelModule.last_call["model_id"] == model_id
        assert np.allclose(_FakeAccelModule.last_call["initial_parameters"][0, :], np.asarray(expected_init))
        assert np.allclose(_FakeAccelModule.last_call["constraints"][0, :], np.asarray(expected_bounds))

    @pytest.mark.parametrize("model_name", ["ex_tofts", "tissue_uptake", "2cxm"])
    def test_stage_d_uses_acceleration_for_new_models(self, model_name: str) -> None:
        ct = np.asarray(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
            ],
            dtype=np.float64,
        )
        cp = np.asarray([0.8, 0.9, 1.0, 1.1], dtype=np.float64)
        timer = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)

        row_len = len(MODEL_LAYOUTS[model_name]["param_names"])
        sentinel = np.full((ct.shape[1], row_len), 77.0, dtype=np.float64)
        with patch("dce_pipeline._fit_stage_d_model_accelerated", return_value=sentinel) as mocked_accel:
            out = _fit_stage_d_model(
                model_name=model_name,
                ct=ct,
                cp_use=cp,
                timer=timer,
                prefs={},
                r1o=None,
                relaxivity=3.6,
                fw=0.8,
                stlv_use=None,
                sttum=None,
                start_injection_min=0.0,
                sss=None,
                ssstum=None,
                acceleration_backend="cpufit_cpu",
            )
        mocked_accel.assert_called_once()
        assert out.shape == (ct.shape[1], row_len), f"{model_name}: unexpected output shape"
        assert np.allclose(out, 77.0), f"{model_name}: unexpected output values"

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

            assert result["stages"]["A"]["impl"] == "real"
            assert result["stages"]["A"]["status"] == "ok"
            assert "array_shapes" in result["stages"]["A"]
            assert "Cp" in result["stages"]["A"]["array_shapes"]
            assert "Ct" in result["stages"]["A"]["array_shapes"]
            stage_a = result["stages"]["A"]
            assert float(stage_a["start_injection"]) == pytest.approx(float(stage_a["steady_state_time"][1]))
            assert float(stage_a["end_injection"]) >= float(stage_a["start_injection"])

    def test_stage_a_real_applies_start_t_end_t_time_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            roi_file = config.subject_tp_path / "roi_mask.nii.gz"
            roi_file.write_text("")
            config.roi_files = [roi_file]
            config.stage_overrides = {
                "stage_a_mode": "real",
                "tr_ms": 5.0,
                "fa_deg": 15.0,
                "time_resolution_sec": 6.0,
                "steady_state_start": 1,
                "steady_state_end": 2,
                "snr_filter": 0.0,
                "start_t": 3,
                "end_t": 7,
            }

            dynamic = np.ones((6, 6, 1, 10), dtype=float) * 100.0
            dynamic[..., 3:] += 10.0
            aif_mask = np.zeros((6, 6, 1), dtype=float)
            aif_mask[1:3, 1:3, :] = 1.0
            roi_mask = np.zeros((6, 6, 1), dtype=float)
            roi_mask[3:5, 3:5, :] = 1.0
            t1map = np.full((6, 6, 1), 1300.0, dtype=float)

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
                            stage_a = _run_stage_a_real(config)

            assert stage_a["status"] == "ok"
            assert stage_a["timepoint_window"]["start_1b"] == 3
            assert stage_a["timepoint_window"]["end_1b"] == 7
            assert stage_a["timepoint_window"]["n_timepoints_input"] == 10
            assert stage_a["timepoint_window"]["n_timepoints_output"] == 5
            cp = np.asarray(stage_a["arrays"]["Cp"], dtype=np.float64)
            timer = np.asarray(stage_a["arrays"]["timer"], dtype=np.float64)
            assert cp.shape[0] == 5
            assert timer.shape[0] == 5

    def test_stage_a_real_rejects_identical_aif_and_roi_masks(self) -> None:
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

            dynamic = np.ones((4, 4, 1, 6), dtype=float) * 100.0
            same_mask = np.zeros((4, 4, 1), dtype=float)
            same_mask[1:3, 1:3, :] = 1.0
            t1map = np.full((4, 4, 1), 1300.0, dtype=float)

            def fake_load(path: Path):
                p = str(path)
                if p.endswith("dynamic.nii.gz"):
                    return dynamic
                if p.endswith("aif_mask.nii.gz"):
                    return same_mask
                if p.endswith("roi_mask.nii.gz"):
                    return same_mask
                if p.endswith("t1map.nii.gz"):
                    return t1map
                raise AssertionError(f"Unexpected path {path}")

            with patch("dce_pipeline._load_nifti_data", side_effect=fake_load):
                with pytest.raises(ValueError, match=r"AIF mask must be a dedicated vascular ROI"):
                    run_dce_pipeline(config)

    def test_stage_b_real_raw_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {"stage_b_mode": "real", "aif_curve_mode": "raw"}
            stage_a = _make_stage_a_payload()

            result = _run_stage_b_real(config, stage_a)
            assert result["impl"] == "real"
            assert result["aif_name"] == "raw"
            assert "Cp_use" in result["arrays"]
            assert result["arrays"]["Cp_use"].shape[0] == result["arrays"]["timer"].shape[0]

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
            assert result["impl"] == "real"
            assert result["aif_name"] == "fitted"
            assert "fit_params_cp" in result
            assert result["fit_params_cp"].shape == (6,)
            assert result["fit_params_stlv"].shape == (6,)
            assert float(result["fit_t0_exp_cp"]) > float(result["fit_t_base_end_cp"])
            assert float(result["fit_t0_exp_stlv"]) > float(result["fit_t_base_end_stlv"])
            assert result["arrays"]["Cp_use"].shape == result["arrays"]["CpROI"].shape

    def test_stage_b_real_prefers_stage_a_auto_injection_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_b_mode": "real",
                "aif_curve_mode": "raw",
            }
            stage_a = _make_stage_a_payload()
            stage_a["start_injection_min_auto"] = 0.62
            stage_a["end_injection_min_auto"] = 0.94

            result = _run_stage_b_real(config, stage_a)
            assert float(result["start_injection_min"]) == pytest.approx(0.62)
            assert float(result["end_injection_min"]) == pytest.approx(0.94)

    def test_stage_b_real_auto_find_injection_overrides_manual_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.stage_overrides = {
                "stage_b_mode": "real",
                "aif_curve_mode": "raw",
                "auto_find_injection": 1,
                "start_injection_min": 0.20,
                "end_injection_min": 0.30,
            }
            stage_a = _make_stage_a_payload()
            stage_a["start_injection_min_auto"] = 0.61
            stage_a["end_injection_min_auto"] = 0.96

            result = _run_stage_b_real(config, stage_a)
            assert float(result["start_injection_min"]) == pytest.approx(0.61)
            assert float(result["end_injection_min"]) == pytest.approx(0.96)

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
            assert result["impl"] == "real"
            assert result["aif_name"] == "imported"
            assert "Cp_use" in result["arrays"]
            assert result["arrays"]["Cp_use"].shape[0] == timer.shape[0]

    def test_stage_b_real_imported_mode_script_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            stage_a = _make_stage_a_payload()
            timer = stage_a["arrays"]["timer"]
            cp_use = np.mean(stage_a["arrays"]["Cp"], axis=1) * 1.02
            stlv_use = np.mean(stage_a["arrays"]["Stlv"], axis=1) * 0.98

            imported_path = Path(tmp) / "imported_aif_alias.npz"
            np.savez_compressed(
                imported_path,
                Cp_use=cp_use,
                Stlv_use=stlv_use,
                timer=timer,
                start_injection=timer[4],
            )

            # Exercise script-style aliases without using top-level imported_aif_path.
            config.imported_aif_path = None
            config.aif_mode = "auto"
            config.stage_overrides = {
                "stage_b_mode": "real",
                "aif_type": 3,
                "import_aif_path": str(imported_path),
            }

            result = _run_stage_b_real(config, stage_a)
            assert result["impl"] == "real"
            assert result["aif_mode"] == "imported"
            assert result["aif_name"] == "imported"
            assert np.allclose(result["arrays"]["Cp_use"], cp_use)

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

            assert stage_d["impl"] == "real"
            assert "tofts" in stage_d["models_run"]
            assert "patlak" in stage_d["models_run"]

            tofts_out = stage_d["model_outputs"]["tofts"]
            assert tofts_out["map_paths"]
            for _, out_path in tofts_out["map_paths"].items():
                assert Path(out_path).exists()
            assert tofts_out["xls_path"]
            assert Path(tofts_out["xls_path"]).exists()

    def test_stage_d_real_optional_postfit_array_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            roi_path = config.subject_tp_path / "roi_mask_for_part_e.nii.gz"
            roi_path.write_text("")
            config.roi_files = [roi_path]
            config.stage_overrides = {
                "stage_b_mode": "real",
                "stage_d_mode": "real",
                "aif_curve_mode": "raw",
                "write_param_maps": True,
                "write_postfit_arrays": True,
            }
            config.model_flags = {"tofts": 1}

            stage_a = _make_stage_a_payload()
            stage_b = _run_stage_b_real(config, stage_a)

            roi_mask = np.zeros((7, 1, 1), dtype=float)
            roi_mask[:4, 0, 0] = 1.0

            def fake_load(path: Path):
                if str(path).endswith("roi_mask_for_part_e.nii.gz"):
                    return roi_mask
                raise AssertionError(f"Unexpected path {path}")

            with patch("dce_pipeline._load_nifti_data", side_effect=fake_load):
                stage_d = _run_stage_d_real(config, stage_a, stage_b)

            tofts_out = stage_d["model_outputs"]["tofts"]
            postfit_path = Path(tofts_out["postfit_arrays_path"])
            assert postfit_path.exists()

            with np.load(postfit_path) as payload:
                assert "timer_min" in payload
                assert "cp_mM" in payload
                assert "ct_voxel_mM" in payload
                assert "voxel_results" in payload
                assert "voxel_residuals" in payload
                assert "tumind_1based" in payload
                assert "dimensions_xyz" in payload
                assert "roi_ct_mM" in payload
                assert "roi_results" in payload
                assert "roi_residuals" in payload
                assert "roi_names" in payload

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

            assert result["stages"]["A"]["impl"] == "real"
            assert result["stages"]["B"]["impl"] == "real"
            assert result["stages"]["D"]["impl"] == "real"
            assert "tofts" in result["stages"]["D"]["models_run"]
            assert "patlak" in result["stages"]["D"]["models_run"]
