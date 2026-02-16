"""Contract tests for Python DCE CLI pipeline outputs/events (non-parity)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import dce_cli  # noqa: E402


def _abs_path(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def _make_temp_config(tmp_dir: Path) -> Path:
    payload = json.loads((REPO_ROOT / "python" / "dce_default.json").read_text(encoding="utf-8"))
    payload["subject_source_path"] = _abs_path(str(payload["subject_source_path"]))
    payload["subject_tp_path"] = _abs_path(str(payload["subject_tp_path"]))
    payload["output_dir"] = str((tmp_dir / "out").resolve())
    payload["checkpoint_dir"] = str((tmp_dir / "checkpoints").resolve())
    payload["backend"] = "cpu"

    for key in ("dynamic_files", "aif_files", "roi_files", "t1map_files", "noise_files", "drift_files"):
        values = payload.get(key, [])
        payload[key] = [_abs_path(str(v)) for v in values]

    config_path = tmp_dir / "dce_contract_config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


class TestDcePipelineContracts(unittest.TestCase):
    def test_cli_pipeline_writes_event_log_and_summary_contract(self) -> None:
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
            self.assertEqual(rc, 0)
            self.assertTrue(event_log_path.exists(), "Expected JSONL event log to be written")

            events = [
                json.loads(line)
                for line in event_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(events), 0)
            event_types = [str(e.get("type", "")) for e in events]

            self.assertIn("cli_config", event_types)
            self.assertIn("run_start", event_types)
            self.assertIn("run_done", event_types)
            self.assertIn("stage_start", event_types)
            self.assertIn("stage_done", event_types)
            self.assertIn("artifact_written", event_types)
            self.assertIn("model_start", event_types)
            self.assertIn("model_done", event_types)

            stage_start = [str(e.get("stage", "")) for e in events if e.get("type") == "stage_start"]
            stage_done = [str(e.get("stage", "")) for e in events if e.get("type") == "stage_done"]
            self.assertEqual(stage_start, ["A", "B", "D"])
            self.assertEqual(stage_done, ["A", "B", "D"])

            for event in events:
                if event.get("type") != "cli_config":
                    self.assertIn("timestamp_utc", event)

            summary_path = tmp_dir / "out" / "dce_pipeline_run.json"
            self.assertTrue(summary_path.exists(), "Expected run summary file")
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertIn("meta", summary_payload)
            self.assertIn("stages", summary_payload)
            self.assertEqual(summary_payload["meta"]["status"], "ok")
            self.assertIn("A", summary_payload["stages"])
            self.assertIn("B", summary_payload["stages"])
            self.assertIn("D", summary_payload["stages"])

            stage_d = summary_payload["stages"]["D"]
            for key in (
                "selected_backend",
                "acceleration_backend",
                "backend_reason",
                "backend_used",
                "models_run",
                "model_outputs",
            ):
                self.assertIn(key, stage_d)
            self.assertEqual(stage_d["selected_backend"], "cpu")
            self.assertEqual(stage_d["acceleration_backend"], "none")
            self.assertIn("tofts", stage_d["models_run"])

            tofts_out = stage_d["model_outputs"]["tofts"]
            self.assertIn("map_paths", tofts_out)
            self.assertTrue(bool(tofts_out["map_paths"]))
            for map_path in tofts_out["map_paths"].values():
                self.assertTrue(Path(map_path).exists())
            self.assertTrue(tofts_out["xls_path"])
            self.assertTrue(Path(tofts_out["xls_path"]).exists())


if __name__ == "__main__":
    unittest.main()
