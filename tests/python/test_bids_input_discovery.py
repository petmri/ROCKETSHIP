"""Filling a run config's file lists from the BIDS naming convention.

`run_dce_bids_batch.py` and the GUI's auto-find have always discovered inputs this way; the
pipeline did not, so the same BIDS config meant different things depending on which interface
read it -- the GUI silently replaced the file lists it was given, while the CLI insisted on
them. Discovery now runs in one place and all three agree.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import dce_file_discovery  # noqa: E402
from dce_pipeline import DcePipelineConfig  # noqa: E402


SESSION = REPO_ROOT / "tests/data/BIDS_test/derivatives/sub-02downsample/ses-01"
RAWDATA = REPO_ROOT / "tests/data/BIDS_test/rawdata/sub-02downsample/ses-01"
BIDS_EXAMPLE = REPO_ROOT / "python" / "dce_run_example_bids.json"


def _bids_payload(tmp_path: Path, **extra) -> dict:
    payload = {
        "output_dir": str(tmp_path / "out"),
        "subject_source_path": str(RAWDATA),
        "subject_tp_path": str(SESSION),
    }
    payload.update(extra)
    return payload


@pytest.mark.unit
def test_empty_file_lists_are_filled_from_the_session(tmp_path) -> None:
    config = DcePipelineConfig.from_dict(_bids_payload(tmp_path))

    assert config.dynamic_files and config.dynamic_files[0].name.endswith("desc-bfcz_DCE.nii")
    assert config.aif_files and "label-AIF_T1map" in config.aif_files[0].name
    assert config.roi_files and "label-brain_mask" in config.roi_files[0].name
    assert config.t1map_files and "space-DCEref_T1map" in config.t1map_files[0].name
    config.validate()


@pytest.mark.unit
def test_a_named_file_beats_the_convention(tmp_path) -> None:
    """Discovery fills gaps; it must never override something the config states."""
    decoy = SESSION / "dce" / "sub-02downsample_ses-01_label-AIF_T1map.nii"
    config = DcePipelineConfig.from_dict(_bids_payload(tmp_path, dynamic_files=[str(decoy)]))

    assert config.dynamic_files == [decoy.resolve()]
    # The lists it did not state are still discovered.
    assert config.roi_files


@pytest.mark.unit
def test_discovery_says_what_it_found(tmp_path, capsys) -> None:
    """An empty config runs on files the user never typed, so the run has to name them."""
    DcePipelineConfig.from_dict(_bids_payload(tmp_path))

    out = capsys.readouterr().out
    assert "Found by BIDS convention" in out
    assert "desc-bfcz_DCE.nii" in out


@pytest.mark.unit
def test_drift_files_are_never_discovered(tmp_path) -> None:
    """It has no naming convention, so there is nothing to look for and guessing would be
    worse than leaving it empty."""
    config = DcePipelineConfig.from_dict(_bids_payload(tmp_path))

    assert config.drift_files == []
    assert "drift_files" not in dict(dce_file_discovery.DISCOVERABLE_FILE_LISTS)


@pytest.mark.unit
def test_without_a_session_folder_nothing_is_discovered(tmp_path) -> None:
    """The non-BIDS case: no convention applies, so the config must name its files and the
    error has to say so rather than silently running on nothing."""
    config = DcePipelineConfig.from_dict({"output_dir": str(tmp_path / "out")})

    assert config.dynamic_files == []
    with pytest.raises(ValueError, match="dynamic_files is required"):
        config.validate()


@pytest.mark.unit
def test_a_missing_aif_points_at_both_ways_to_supply_one(tmp_path) -> None:
    """The old message said pipeline AIF discovery did not exist. It does now, so the error
    names both routes instead of ruling one out."""
    config = DcePipelineConfig.from_dict(
        {
            "output_dir": str(tmp_path / "out"),
            "dynamic_files": [str(tmp_path / "dyn.nii.gz")],
            "t1map_files": [str(tmp_path / "t1.nii.gz")],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        config.validate()

    message = str(excinfo.value)
    assert "subject_tp_path" in message
    assert dce_file_discovery.AIF_MASK_PATTERN in message


@pytest.mark.unit
def test_a_nonexistent_session_folder_is_not_an_error_here(tmp_path) -> None:
    """Discovery is best-effort: a wrong folder must fall through to validate()'s message
    about the missing inputs, not raise from inside config construction."""
    config = DcePipelineConfig.from_dict(
        {"output_dir": str(tmp_path / "out"), "subject_tp_path": str(tmp_path / "nope")}
    )

    assert config.dynamic_files == []


@pytest.mark.unit
def test_the_bids_example_names_no_files_at_all() -> None:
    """Guards what the example exists to show. Re-adding the file lists would make it a
    config that happens to be in BIDS rather than one that demonstrates the convention."""
    payload = json.loads(BIDS_EXAMPLE.read_text(encoding="utf-8"))

    for key, _kind in dce_file_discovery.DISCOVERABLE_FILE_LISTS:
        assert key not in payload, f"{key} is discoverable and should not be listed"

    config = DcePipelineConfig.from_dict(payload, base_dir=BIDS_EXAMPLE.parent)
    config.validate()
    for path in config.dynamic_files + config.aif_files + config.t1map_files:
        assert path.is_file()


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload, expected, why",
    [
        ({"subject_tp_path": "/x"}, True, "folders only: the config asks for the convention"),
        ({"subject_tp_path": "/x", "dynamic_files": ["a.nii"]}, False, "names a file: be explicit"),
        ({"dynamic_files": ["a.nii"]}, False, "non-BIDS: nothing to discover from"),
        ({}, False, "nothing to go on"),
        ({"subject_tp_path": "   "}, False, "blank field is not a folder"),
    ],
)
def test_the_gui_lets_the_config_decide_whether_to_auto_find(payload, expected, why) -> None:
    """Auto-find used to be on regardless, so it overwrote file lists a config had stated and
    left non-BIDS configs locked behind a warning about a folder they do not have.

    Skips where PySide6 is absent -- it ships in requirements_gui.txt, not the base set.
    """
    pytest.importorskip("PySide6")
    import dce_gui

    assert dce_gui.DceGuiWindow._config_wants_auto_find(payload) is expected, why
