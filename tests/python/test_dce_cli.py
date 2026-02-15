"""Unit tests for Python DCE CLI argument merging."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import dce_cli  # noqa: E402


class TestDceCli(unittest.TestCase):
    def test_main_applies_dce_preferences_and_set_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_path = tmp / "config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "subject_source_path": str(tmp),
                        "subject_tp_path": str(tmp),
                        "output_dir": str(tmp / "out"),
                        "dynamic_files": [str(tmp / "dyn.nii.gz")],
                        "aif_files": [str(tmp / "aif.nii.gz")],
                        "t1map_files": [str(tmp / "t1.nii.gz")],
                        "stage_overrides": {"existing_key": "existing_value"},
                    }
                )
            )
            dce_pref = tmp / "dce_preferences.txt"
            dce_pref.write_text("voxel_MaxFunEvals = 50\n")

            with patch("dce_cli.DcePipelineConfig.from_dict", return_value=object()) as from_dict_mock:
                with patch("dce_cli.run_dce_pipeline", return_value={"meta": {"status": "ok"}}):
                    with patch("builtins.print"):
                        rc = dce_cli.main(
                            [
                                "--config",
                                str(cfg_path),
                                "--dce-preferences",
                                str(dce_pref),
                                "--set",
                                "voxel_MaxFunEvals=123",
                                "--set",
                                "blood_t1_ms=1600",
                            ]
                        )

            self.assertEqual(rc, 0)
            payload = from_dict_mock.call_args.args[0]
            self.assertIn("stage_overrides", payload)
            self.assertEqual(payload["stage_overrides"]["existing_key"], "existing_value")
            self.assertEqual(payload["stage_overrides"]["voxel_MaxFunEvals"], "123")
            self.assertEqual(payload["stage_overrides"]["blood_t1_ms"], "1600")
            self.assertEqual(payload["stage_overrides"]["dce_preferences_path"], str(dce_pref.resolve()))

    def test_parse_set_overrides_rejects_invalid_entries(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected KEY=VALUE"):
            dce_cli._parse_set_overrides(["bad_entry"])  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
