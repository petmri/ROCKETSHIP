"""Parametric settings resolve from the shipped defaults file, and only from there.

The rule these pin is the same one `dce_config` follows: one user-editable file holds every
default, source carries none, and a value in neither the run config nor that file is an
error rather than a guess. The tests read the *shipped* file rather than a fixture, because
a test that resolves from somewhere else stops being evidence about what a user gets.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import parametric_config  # noqa: E402
from parametric_config import ParametricConfigError  # noqa: E402
from parametric_pipeline import ParametricT1Config  # noqa: E402


SHIPPED = REPO_ROOT / "python/parametric_defaults.json"
RUN_EXAMPLE = REPO_ROOT / "python/parametric_run_example.json"


@pytest.mark.unit
def test_the_shipped_defaults_file_is_what_gets_loaded() -> None:
    defaults = parametric_config.load_defaults()
    assert defaults.path == SHIPPED
    assert defaults.default_for("fit_type") == "t1_fa_fit"


@pytest.mark.unit
def test_source_carries_no_fallback_for_anything_the_file_defines() -> None:
    """Every defaulted key must come from the file, so editing the file changes a run."""
    defaults = parametric_config.load_defaults()
    resolved = ParametricT1Config.from_dict({"output_dir": "/tmp/x", "vfa_files": ["a.nii"]})
    for key, value in defaults.defaults.items():
        if not hasattr(resolved, key):
            continue
        assert getattr(resolved, key) == value, f"{key} did not come from {SHIPPED.name}"


@pytest.mark.unit
def test_a_run_config_beats_the_defaults_file() -> None:
    config = ParametricT1Config.from_dict(
        {"output_dir": "/tmp/x", "vfa_files": ["a.nii"], "rsquared_threshold": 0.95}
    )
    assert config.rsquared_threshold == 0.95
    value, source = parametric_config.resolve_with_source({"backend": "cpu"}, "backend")
    assert (value, source) == ("cpu", "run_config")
    assert parametric_config.resolve_with_source({}, "backend")[1] == "defaults_file"


@pytest.mark.unit
def test_a_required_setting_with_no_value_stops_the_run() -> None:
    with pytest.raises(ParametricConfigError) as excinfo:
        ParametricT1Config.from_dict({"vfa_files": ["a.nii"]})
    message = str(excinfo.value)
    assert "output_dir" in message
    # The message must say where to put the value, not just that it is absent.
    assert "run config" in message and str(SHIPPED) in message


@pytest.mark.unit
def test_a_misspelled_setting_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ParametricConfigError) as excinfo:
        ParametricT1Config.from_dict({"output_dir": "/tmp/x", "rsquared_threshhold": 0.5})
    assert "rsquared_threshhold" in str(excinfo.value)


@pytest.mark.unit
def test_input_paths_are_not_preferences_and_pass_the_key_guard() -> None:
    """A run names its inputs; those keys deliberately have no defaults-file entry."""
    parametric_config.validate_keys(
        {"output_dir": "/tmp/x", "vfa_files": [], "mask_file": None, "b1_map_file": None}
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "legacy,current,value,expected",
    [
        ("file_list", "vfa_files", ["a.nii"], [Path("a.nii").resolve()]),
        ("tr", "tr_ms", 8.0, 8.0),
        ("parameters", "flip_angles_deg", [2, 5], [2.0, 5.0]),
        ("xy_smooth_size", "xy_smooth_sigma", 1.5, 1.5),
    ],
)
def test_older_config_spellings_still_work(legacy: str, current: str, value, expected) -> None:
    """These appear in configs written against earlier releases; a run using one is fine."""
    parametric_config.validate_keys({"output_dir": "/tmp/x", legacy: value})
    config = ParametricT1Config.from_dict({"output_dir": "/tmp/x", legacy: value})
    assert getattr(config, current) == expected


@pytest.mark.unit
def test_a_null_in_a_config_means_absent_not_none() -> None:
    """`"tr_ms": null` is how someone hand-editing the file says "read it from the sidecar"."""
    config = ParametricT1Config.from_dict(
        {"output_dir": "/tmp/x", "vfa_files": ["a.nii"], "tr_ms": None}
    )
    assert config.tr_ms is None
    # The same must not swallow a defaulted key into None.
    assert (
        ParametricT1Config.from_dict(
            {"output_dir": "/tmp/x", "vfa_files": ["a.nii"], "backend": None}
        ).backend
        == parametric_config.load_defaults().default_for("backend")
    )


@pytest.mark.unit
def test_a_file_that_is_not_a_defaults_file_is_refused(tmp_path) -> None:
    bogus = tmp_path / "not_defaults.json"
    bogus.write_text(json.dumps({"defaults": {"fit_type": "nonsense"}}))
    with pytest.raises(ParametricConfigError, match="_schema"):
        parametric_config.load_defaults(bogus)


@pytest.mark.unit
def test_the_run_example_is_a_run_config_not_a_second_defaults_file() -> None:
    """The two roles were one file, which is what made the defaults uneditable in practice.

    The example may name its inputs and whatever it deliberately differs on; it must not
    restate the preference values, or editing the defaults file stops changing this run.
    """
    payload = json.loads(RUN_EXAMPLE.read_text())
    defaults = parametric_config.load_defaults()
    restated = [
        key
        for key, value in payload.items()
        if key in defaults.defaults and defaults.defaults[key] == value
    ]
    assert not restated, f"{RUN_EXAMPLE.name} repeats defaults-file values: {restated}"
    assert "output_dir" in payload and "vfa_files" in payload


@pytest.mark.unit
def test_relative_paths_re_anchor_against_the_config_that_holds_them(tmp_path) -> None:
    # Built from tmp_path rather than written as "/abs/b.nii": a leading slash is not an
    # absolute path on Windows, so that literal would be re-anchored there and the test
    # would be asserting the opposite of what it says.
    elsewhere = tmp_path.parent / "b.nii"
    resolved = parametric_config.resolve_override_paths(
        {"output_dir": "out", "vfa_files": ["a.nii", str(elsewhere)], "backend": "cpu"}, tmp_path
    )
    assert resolved["output_dir"] == str(tmp_path / "out")
    assert resolved["vfa_files"] == [str(tmp_path / "a.nii"), str(elsewhere)]
    assert resolved["backend"] == "cpu", "non-path keys must pass through untouched"
