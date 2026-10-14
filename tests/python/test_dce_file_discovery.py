"""Tests for dceprep BIDS input discovery."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import BidsSession  # noqa: E402
from dce_file_discovery import (  # noqa: E402
    discover_dce_input_paths,
    discover_dce_inputs,
    missing_required_inputs,
)


def _make_session(root: Path) -> BidsSession:
    return BidsSession(
        bids_root=root,
        subject="sub-01",
        session="ses-01",
        rawdata_path=root / "rawdata" / "sub-01" / "ses-01",
        derivatives_path=root / "derivatives" / "sub-01" / "ses-01",
    )


def _write_conventional_session(root: Path, *, with_noise: bool = True) -> BidsSession:
    session = _make_session(root)
    dce = session.derivatives_path / "dce"
    anat = session.derivatives_path / "anat"
    dce.mkdir(parents=True)
    anat.mkdir(parents=True)
    (dce / "sub-01_ses-01_desc-bfcz_DCE.nii.gz").touch()
    (dce / "sub-01_ses-01_desc-bfcz_DCE.json").touch()
    (dce / "sub-01_ses-01_label-AIF_T1map.nii.gz").touch()
    (anat / "sub-01_ses-01_space-DCEref_label-brain_mask.nii.gz").touch()
    (anat / "sub-01_ses-01_space-DCEref_T1map.nii.gz").touch()
    if with_noise:
        (anat / "sub-01_ses-01_label-noise_mask.nii.gz").touch()
    return session


def test_discovers_every_input_kind(tmp_path: Path) -> None:
    session = _write_conventional_session(tmp_path)

    found = discover_dce_input_paths(session)

    assert found["dynamic"].name == "sub-01_ses-01_desc-bfcz_DCE.nii.gz"
    assert found["aif_mask"].name == "sub-01_ses-01_label-AIF_T1map.nii.gz"
    assert found["roi_mask"].name == "sub-01_ses-01_space-DCEref_label-brain_mask.nii.gz"
    assert found["t1_map"].name == "sub-01_ses-01_space-DCEref_T1map.nii.gz"
    assert found["noise_mask"].name == "sub-01_ses-01_label-noise_mask.nii.gz"
    assert found["metadata_json"].name == "sub-01_ses-01_desc-bfcz_DCE.json"
    assert missing_required_inputs(found) == []


def test_optional_inputs_may_be_absent(tmp_path: Path) -> None:
    session = _write_conventional_session(tmp_path, with_noise=False)

    found = discover_dce_input_paths(session)

    assert found["noise_mask"] is None
    assert missing_required_inputs(found) == []
    assert discover_dce_inputs(session).noise_mask is None


def test_partial_discovery_reports_what_is_missing_instead_of_raising(tmp_path: Path) -> None:
    session = _write_conventional_session(tmp_path)
    (session.derivatives_path / "dce" / "sub-01_ses-01_desc-bfcz_DCE.nii.gz").unlink()
    (session.derivatives_path / "anat" / "sub-01_ses-01_space-DCEref_T1map.nii.gz").unlink()

    found = discover_dce_input_paths(session)

    assert found["dynamic"] is None
    assert found["t1_map"] is None
    # Everything else still resolves, which is what lets the GUI show a partial fill.
    assert found["aif_mask"] is not None
    assert found["roi_mask"] is not None
    assert missing_required_inputs(found) == ["dynamic", "t1_map"]


def test_bias_corrected_dynamic_wins_over_the_fallback_pattern(tmp_path: Path) -> None:
    session = _write_conventional_session(tmp_path)
    (session.derivatives_path / "dce" / "sub-01_ses-01_DCE.nii.gz").touch()

    found = discover_dce_input_paths(session)

    assert found["dynamic"].name == "sub-01_ses-01_desc-bfcz_DCE.nii.gz"


def test_missing_directories_yield_no_matches(tmp_path: Path) -> None:
    session = _make_session(tmp_path)

    found = discover_dce_input_paths(session)

    assert set(missing_required_inputs(found)) == {"dynamic", "aif_mask", "roi_mask", "t1_map"}
    assert all(value is None for value in found.values())


def test_strict_discovery_still_raises_and_names_the_missing_kinds(tmp_path: Path) -> None:
    session = _write_conventional_session(tmp_path)
    (session.derivatives_path / "dce" / "sub-01_ses-01_label-AIF_T1map.nii.gz").unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        discover_dce_inputs(session)

    message = str(excinfo.value)
    assert "sub-01_ses-01" in message
    assert "aif_mask" in message
    assert "*label-AIF_T1map.nii*" in message
