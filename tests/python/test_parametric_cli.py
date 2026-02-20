"""Unit tests for Python parametric T1 CLI argument merging."""

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

import parametric_cli  # noqa: E402


@pytest.mark.unit
def test_main_applies_overrides_and_set_values() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        cfg_path = tmp / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "output_dir": "out/default",
                    "vfa_files": ["tests/data/BIDS_test/rawdata/sub-01/ses-01/anat/sub-01_ses-01_flip-01_VFA.nii.gz"],
                }
            ),
            encoding="utf-8",
        )

        mock_config = SimpleNamespace(output_dir=(tmp / "out").resolve())
        with patch("parametric_cli.ParametricT1Config.from_dict", return_value=mock_config) as from_dict_mock:
            with patch("parametric_cli.run_parametric_t1_pipeline", return_value={"meta": {"status": "ok"}}):
                with patch("builtins.print"):
                    rc = parametric_cli.main(
                        [
                            "--config",
                            str(cfg_path),
                            "--output-dir",
                            str(tmp / "out"),
                            "--tr-ms",
                            "9.5",
                            "--rsquared-threshold",
                            "0.75",
                            "--set",
                            "write_rho_map=true",
                            "--set",
                            "output_label=cli_override",
                        ]
                    )

        assert rc == 0
        payload = from_dict_mock.call_args.args[0]
        assert payload["output_dir"] == str((tmp / "out").resolve())
        assert payload["tr_ms"] == pytest.approx(9.5)
        assert payload["rsquared_threshold"] == pytest.approx(0.75)
        assert payload["write_rho_map"] is True
        assert payload["output_label"] == "cli_override"
        assert from_dict_mock.call_args.kwargs["base_dir"] == cfg_path.parent.resolve()


@pytest.mark.unit
def test_parse_set_overrides_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        parametric_cli._parse_set_overrides(["bad_entry"])  # pylint: disable=protected-access
