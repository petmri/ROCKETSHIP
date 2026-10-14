"""Where a relative path in a config file points.

The DCE and parametric interfaces used to disagree. Parametric anchored relative paths to
the config file's own directory; DCE anchored them to the process cwd, so the shipped
example config only ran from the repository root and the DCE GUI applied a third rule
(repo root) on top. These tests pin the one rule both sides now follow.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

import dce_cli  # noqa: E402
import dce_config  # noqa: E402
import parametric_pipeline  # noqa: E402
from dce_pipeline import DcePipelineConfig  # noqa: E402


def _payload(**overrides):
    """A config whose every path is relative, so the anchor decides where it lands."""
    payload = {
        "subject_source_path": "raw/sub-01",
        "subject_tp_path": "deriv/sub-01",
        "output_dir": "out/run",
        "checkpoint_dir": "out/run/checkpoints",
        "dynamic_files": ["deriv/sub-01/dce/dyn.nii.gz"],
        "aif_files": ["deriv/sub-01/anat/aif.nii.gz"],
        "roi_files": ["deriv/sub-01/anat/roi.nii.gz"],
        "t1map_files": ["raw/sub-01/anat/t1.nii.gz"],
        "stage_overrides": {"dce_metadata_path": "raw/sub-01/dce/meta.json"},
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_relative_paths_anchor_to_the_config_directory(tmp_path) -> None:
    config = DcePipelineConfig.from_dict(_payload(), base_dir=tmp_path)

    assert config.subject_source_path == (tmp_path / "raw/sub-01").resolve()
    assert config.output_dir == (tmp_path / "out/run").resolve()
    assert config.checkpoint_dir == (tmp_path / "out/run/checkpoints").resolve()
    assert config.dynamic_files == [(tmp_path / "deriv/sub-01/dce/dyn.nii.gz").resolve()]
    assert config.t1map_files == [(tmp_path / "raw/sub-01/anat/t1.nii.gz").resolve()]


@pytest.mark.unit
def test_path_valued_overrides_use_the_same_anchor(tmp_path) -> None:
    """A config that found its images relative to itself and its metadata relative to the
    cwd would be worse than either rule applied consistently."""

    config = DcePipelineConfig.from_dict(_payload(), base_dir=tmp_path)

    assert config.stage_overrides["dce_metadata_path"] == str(
        (tmp_path / "raw/sub-01/dce/meta.json").resolve()
    )


@pytest.mark.unit
def test_non_path_overrides_are_left_alone(tmp_path) -> None:
    payload = _payload()
    payload["stage_overrides"] = {"rootname": "Dyn-1", "start_t": 3, "time_smoothing": "none"}

    config = DcePipelineConfig.from_dict(payload, base_dir=tmp_path)

    assert config.stage_overrides == {"rootname": "Dyn-1", "start_t": 3, "time_smoothing": "none"}


@pytest.mark.unit
def test_absolute_paths_ignore_the_anchor(tmp_path) -> None:
    absolute = tmp_path / "elsewhere" / "dyn.nii.gz"
    payload = _payload(dynamic_files=[str(absolute)])

    config = DcePipelineConfig.from_dict(payload, base_dir=tmp_path / "config_dir")

    assert config.dynamic_files == [absolute.resolve()]


@pytest.mark.unit
def test_without_an_anchor_the_cwd_still_applies(tmp_path, monkeypatch) -> None:
    """Callers that build the payload themselves already hold absolute paths and pass no
    base_dir; they must keep working unchanged."""

    monkeypatch.chdir(tmp_path)
    config = DcePipelineConfig.from_dict(_payload())

    assert config.dynamic_files == [(tmp_path / "deriv/sub-01/dce/dyn.nii.gz").resolve()]


@pytest.mark.unit
def test_dce_and_parametric_resolve_a_relative_path_identically(tmp_path) -> None:
    """The finding this module exists for: the two interfaces disagreed on the same input."""

    config_dir = tmp_path / "study" / "configs"
    config_dir.mkdir(parents=True)
    relative = "../data/sub-01/anat/t1.nii.gz"

    dce = DcePipelineConfig.from_dict(_payload(t1map_files=[relative]), base_dir=config_dir)
    parametric = parametric_pipeline._resolve_path(relative, config_dir)

    assert dce.t1map_files == [parametric]


@pytest.mark.unit
def test_every_path_valued_key_is_a_real_preference() -> None:
    """The list is a classification of keys in dce_defaults.json, so it can drift out of it.

    A key renamed there and not here would silently stop being anchored -- the same class of
    drift as a retired key left behind in a hardcoded default block.
    """
    defaults = dce_config.load_defaults()
    for key in dce_config.PATH_VALUED_KEYS:
        assert defaults.knows(key), f"{key} is not a key dce_defaults.json declares"


@pytest.mark.unit
def test_cli_runs_a_relative_config_from_an_unrelated_directory(tmp_path, monkeypatch) -> None:
    """The user-visible bug: the shipped example config only ran from the repo root."""

    config_dir = tmp_path / "study"
    config_dir.mkdir()
    config_path = config_dir / "run.json"
    config_path.write_text(json.dumps(_payload()), encoding="utf-8")

    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with patch("dce_cli.run_dce_pipeline", return_value={"meta": {"status": "ok"}}):
        with patch("builtins.print"):
            rc = dce_cli.main(["--config", str(config_path), "--events", "off"])

    assert rc == 0
    assert (config_dir / "out" / "run").is_dir()
    assert not (elsewhere / "out").exists()


@pytest.mark.unit
def test_a_set_path_is_relative_to_where_it_was_typed(tmp_path, monkeypatch) -> None:
    """--set is typed at a shell prompt, so it follows the cwd like any other CLI path,
    while the config file's own paths follow the config."""

    config_dir = tmp_path / "study"
    config_dir.mkdir()
    config_path = config_dir / "run.json"
    config_path.write_text(json.dumps(_payload()), encoding="utf-8")

    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    stub = SimpleNamespace(output_dir=(config_dir / "out" / "run").resolve(), validate=lambda: None)
    with patch("dce_cli.DcePipelineConfig.from_dict", return_value=stub) as from_dict_mock:
        with patch("dce_cli.run_dce_pipeline", return_value={"meta": {"status": "ok"}}):
            with patch("builtins.print"):
                dce_cli.main(
                    [
                        "--config",
                        str(config_path),
                        "--events",
                        "off",
                        "--set",
                        "dce_metadata_path=./typed_here.json",
                    ]
                )

    supplied = from_dict_mock.call_args.args[0]["stage_overrides"]["dce_metadata_path"]
    assert supplied == str((elsewhere / "typed_here.json").resolve())


@pytest.mark.unit
def test_the_gui_resolvers_agree_with_the_pipeline(tmp_path) -> None:
    """Both GUIs resolve a path before writing the run config they launch, so a third rule
    can hide there. The DCE GUI used to anchor to the repository root, which meant a config
    opened from anywhere else showed the CLI one set of files and ran another.

    Skips where PySide6 is absent -- it ships in requirements_gui.txt, not the base set.
    """
    pytest.importorskip("PySide6")

    import dce_gui
    import parametric_gui

    relative = "../data/sub-01/anat/t1.nii.gz"
    config_dir = tmp_path / "study" / "configs"
    config_dir.mkdir(parents=True)

    expected = str(parametric_pipeline._resolve_path(relative, config_dir))
    assert dce_gui._resolve_path(relative, config_dir) == expected
    assert parametric_gui._resolve_path(relative, config_dir) == expected
    assert dce_gui._resolve_paths([relative], config_dir) == [expected]
