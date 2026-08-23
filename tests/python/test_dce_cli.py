"""Unit tests for Python DCE CLI argument merging."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import cli_overrides  # noqa: E402
import dce_cli  # noqa: E402


@pytest.mark.unit
def test_main_applies_set_overrides() -> None:
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
        # The CLI validates before it creates anything, so the stub needs validate() too.
        stub_config = SimpleNamespace(output_dir=(tmp / "out").resolve(), validate=lambda: None)
        with patch("dce_cli.DcePipelineConfig.from_dict", return_value=stub_config) as from_dict_mock:
            with patch("dce_cli.run_dce_pipeline", return_value={"meta": {"status": "ok"}}):
                with patch("builtins.print"):
                    rc = dce_cli.main(
                        [
                            "--config",
                            str(cfg_path),
                            "--set",
                            "voxel_MaxFunEvals=123",
                            "--set",
                            "blood_t1_ms=1600",
                            "--set",
                            "write_param_maps=false",
                            "--set",
                            "rootname=Dyn-1",
                        ]
                    )

        assert rc == 0
        payload = from_dict_mock.call_args.args[0]
        assert "stage_overrides" in payload
        assert payload["stage_overrides"]["existing_key"] == "existing_value"
        assert payload["stage_overrides"]["voxel_MaxFunEvals"] == 123
        assert payload["stage_overrides"]["blood_t1_ms"] == 1600
        # A --set boolean has to arrive as a real bool. Kept as the string "false" it is
        # truthy, so asking for no parameter maps used to write them anyway.
        assert payload["stage_overrides"]["write_param_maps"] is False
        # Bare words are not JSON and must survive as the text they were typed as.
        assert payload["stage_overrides"]["rootname"] == "Dyn-1"


@pytest.mark.unit
def test_parse_set_overrides_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        cli_overrides.parse_set_overrides(["bad_entry"])
