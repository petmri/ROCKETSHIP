"""Contract tests for Python DCE CLI pipeline outputs/events (non-parity)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import dce_cli  # noqa: E402
import dce_config  # noqa: E402


TINY_CONFIG = REPO_ROOT / "tests" / "python" / "dce_run_tiny.json"


def _abs_path(path_text: str) -> str:
    """Anchor to the tiny config's own directory, the way the CLI would anchor it.

    This copy is rewritten into a temp directory, so every relative path has to be made
    absolute against where it was authored before it moves -- including the one inside
    stage_overrides, which the cwd used to resolve by accident.
    """
    path = Path(path_text)
    if not path.is_absolute():
        path = (TINY_CONFIG.parent / path).resolve()
    return str(path)


def _make_temp_config(tmp_dir: Path) -> Path:
    payload = json.loads(TINY_CONFIG.read_text(encoding="utf-8"))
    payload["subject_source_path"] = _abs_path(str(payload["subject_source_path"]))
    payload["subject_tp_path"] = _abs_path(str(payload["subject_tp_path"]))
    payload["stage_overrides"] = dce_config.resolve_override_paths(
        payload.get("stage_overrides", {}), TINY_CONFIG.parent
    )
    payload["output_dir"] = str((tmp_dir / "out").resolve())
    payload["checkpoint_dir"] = str((tmp_dir / "checkpoints").resolve())
    payload["backend"] = "cpu"

    for key in ("dynamic_files", "aif_files", "roi_files", "t1map_files", "noise_files", "drift_files"):
        values = payload.get(key, [])
        payload[key] = [_abs_path(str(v)) for v in values]

    config_path = tmp_dir / "dce_contract_config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


@pytest.mark.integration
def test_cli_pipeline_writes_event_log_and_summary_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        config_path = _make_temp_config(tmp_dir)
        event_log_path = tmp_dir / "events.jsonl"

        rc = dce_cli.main(
            [
                "--config",
                str(config_path),
                "--events",
                "off",
                "--event-log",
                str(event_log_path),
            ]
        )
        assert rc == 0
        assert event_log_path.exists(), "Expected JSONL event log to be written"

        events = [
            json.loads(line)
            for line in event_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(events) > 0
        event_types = [str(e.get("type", "")) for e in events]

        assert "cli_config" in event_types
        assert "run_start" in event_types
        assert "run_done" in event_types
        assert "stage_start" in event_types
        assert "stage_done" in event_types
        assert "artifact_written" in event_types
        assert "model_start" in event_types
        assert "model_done" in event_types

        stage_start = [str(e.get("stage", "")) for e in events if e.get("type") == "stage_start"]
        stage_done = [str(e.get("stage", "")) for e in events if e.get("type") == "stage_done"]
        assert stage_start == ["A", "B", "D"]
        assert stage_done == ["A", "B", "D"]

        for event in events:
            if event.get("type") != "cli_config":
                assert "timestamp_utc" in event

        summary_path = tmp_dir / "out" / "dce_pipeline_run.json"
        assert summary_path.exists(), "Expected run summary file"
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

        assert "meta" in summary_payload
        assert "stages" in summary_payload
        assert summary_payload["meta"]["status"] == "ok"
        assert "A" in summary_payload["stages"]
        assert "B" in summary_payload["stages"]
        assert "D" in summary_payload["stages"]

        stage_d = summary_payload["stages"]["D"]
        for key in (
            "selected_backend",
            "acceleration_backend",
            "backend_reason",
            "backend_used",
            "models_run",
            "model_outputs",
        ):
            assert key in stage_d
        assert stage_d["selected_backend"] == "cpu"
        assert stage_d["acceleration_backend"] == "none"
        assert "tofts" in stage_d["models_run"]

        tofts_out = stage_d["model_outputs"]["tofts"]
        assert "map_paths" in tofts_out
        assert bool(tofts_out["map_paths"])
        for map_path in tofts_out["map_paths"].values():
            assert Path(map_path).exists()
        assert tofts_out["xls_path"]
        assert Path(tofts_out["xls_path"]).exists()


@pytest.mark.integration
def test_cli_pipeline_runs_on_uneven_volume_timing() -> None:
    """A sidecar `VolumeTiming` becomes the Stage-A/B timer, cropped by `start_t` and rebased to 0.

    The MATLAB side of the same contract is tests/matlab/unit/TestDceVolumeTiming.m.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        config_path = _make_temp_config(tmp_dir)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        meta_path = Path(payload["stage_overrides"]["dce_metadata_path"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # 18 frames: 20 s apart for the first five, 10 s apart after, so the auto-found
        # injection frames straddle the change in spacing.
        volume_timing = [20.0 * i if i < 5 else 80.0 + 10.0 * (i - 4) for i in range(18)]
        meta["VolumeTiming"] = volume_timing
        meta["NumberOfAverages"] = 3
        (tmp_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        payload["stage_overrides"]["dce_metadata_path"] = str(tmp_dir / "meta.json")
        payload["stage_overrides"]["start_t"] = 2
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        assert dce_cli.main(["--config", str(config_path), "--events", "off"]) == 0

        expected_timer = (np.asarray(volume_timing[1:]) - volume_timing[1]) / 60.0
        checkpoints = tmp_dir / "checkpoints"
        stage_a = json.loads((checkpoints / "a_out.json").read_text(encoding="utf-8"))
        with np.load(checkpoints / "a_out_arrays.npz") as arrays:
            np.testing.assert_allclose(arrays["timer"], expected_timer)
        with np.load(checkpoints / "b_out_arrays.npz") as arrays:
            np.testing.assert_allclose(arrays["timer"], expected_timer)

        # Median step of the cropped window (three 20 s steps, thirteen 10 s), not scaled by
        # NumberOfAverages.
        assert stage_a["time_resolution_min"] == pytest.approx(10.0 / 60.0)
        frames = np.arange(expected_timer.size, dtype=np.float64)
        for key in ("start_injection", "end_injection"):
            assert stage_a[f"{key}_min_auto"] == pytest.approx(
                np.interp(stage_a[key] - 1.0, frames, expected_timer)
            )
