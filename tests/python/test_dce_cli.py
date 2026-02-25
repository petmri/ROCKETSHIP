"""Unit tests for Python DCE CLI argument merging."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import dce_cli  # noqa: E402


@pytest.mark.unit
def test_main_applies_dce_preferences_and_set_overrides() -> None:
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

        assert rc == 0
        payload = from_dict_mock.call_args.args[0]
        assert "stage_overrides" in payload
        assert payload["stage_overrides"]["existing_key"] == "existing_value"
        assert payload["stage_overrides"]["voxel_MaxFunEvals"] == "123"
        assert payload["stage_overrides"]["blood_t1_ms"] == "1600"
        assert payload["stage_overrides"]["dce_preferences_path"] == str(dce_pref.resolve())


@pytest.mark.unit
def test_parse_set_overrides_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        dce_cli._parse_set_overrides(["bad_entry"])  # pylint: disable=protected-access
