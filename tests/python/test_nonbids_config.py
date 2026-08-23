"""A run config does not have to describe a BIDS session.

`subject_source_path` and `subject_tp_path` used to be mandatory, and a config without them
died on a bare `KeyError` naming a key that means nothing for data outside BIDS. The pipeline
reads its images from the explicit file lists, so both are conveniences: the first enables
sidecar discovery by BIDS convention, the second is recorded in the run summary.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import dce_cli  # noqa: E402
import dce_config  # noqa: E402
from dce_pipeline import DcePipelineConfig  # noqa: E402


NONBIDS_EXAMPLE = REPO_ROOT / "python" / "dce_run_example_nonbids.json"
BIDS_EXAMPLE = REPO_ROOT / "python" / "dce_run_example_bids.json"


def _minimal_payload(tmp_path: Path) -> dict:
    """The least a non-BIDS run can say: where to write, and which files to read."""
    return {
        "output_dir": str(tmp_path / "out"),
        "dynamic_files": [str(tmp_path / "dyn.nii.gz")],
        "aif_files": [str(tmp_path / "aif.nii.gz")],
        "t1map_files": [str(tmp_path / "t1.nii.gz")],
    }


@pytest.mark.unit
def test_a_config_without_bids_session_paths_builds(tmp_path) -> None:
    config = DcePipelineConfig.from_dict(_minimal_payload(tmp_path))

    assert config.subject_source_path is None
    assert config.subject_tp_path is None
    assert config.output_dir == (tmp_path / "out").resolve()


@pytest.mark.unit
def test_empty_bids_session_paths_are_treated_as_absent(tmp_path) -> None:
    """The GUI writes "" for a blank field rather than dropping the key."""
    payload = _minimal_payload(tmp_path)
    payload["subject_source_path"] = ""
    payload["subject_tp_path"] = ""

    config = DcePipelineConfig.from_dict(payload)

    assert config.subject_source_path is None
    assert config.subject_tp_path is None


@pytest.mark.unit
def test_a_missing_output_dir_says_which_key_is_missing(tmp_path) -> None:
    """`output_dir` is the one genuinely required key, so its absence must read as guidance
    rather than as the KeyError traceback every missing key used to produce."""
    payload = _minimal_payload(tmp_path)
    del payload["output_dir"]

    with pytest.raises(dce_config.DceConfigError, match="output_dir"):
        DcePipelineConfig.from_dict(payload)


@pytest.mark.unit
def test_the_cli_reports_a_bad_config_without_a_traceback(tmp_path, capsys) -> None:
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps({"dynamic_files": []}), encoding="utf-8")

    rc = dce_cli.main(["--config", str(config_path), "--events", "off"])

    assert rc == 2
    assert "output_dir" in capsys.readouterr().err


@pytest.mark.unit
def test_both_shipped_examples_load_and_validate() -> None:
    """The two examples are the documented starting points, so a broken one is a broken
    first run. Loading them also proves the non-BIDS one needs no session paths."""
    for example in (BIDS_EXAMPLE, NONBIDS_EXAMPLE):
        payload = json.loads(example.read_text(encoding="utf-8"))
        config = DcePipelineConfig.from_dict(payload, base_dir=example.parent)
        config.validate()

        for path in config.dynamic_files + config.aif_files + config.t1map_files:
            assert path.is_file(), f"{example.name} points at a missing file: {path}"


@pytest.mark.unit
def test_the_nonbids_example_omits_the_bids_keys_and_states_its_metadata() -> None:
    """Guards the point of the example: it must keep demonstrating the non-BIDS route.

    Adding a subject path or a sidecar to it would quietly turn it into a second BIDS
    example, and the difference between the two files is what they exist to show.
    """
    payload = json.loads(NONBIDS_EXAMPLE.read_text(encoding="utf-8"))

    assert "subject_source_path" not in payload
    assert "subject_tp_path" not in payload

    overrides = {str(k).strip().lower() for k in payload["stage_overrides"]}
    assert {"tr_ms", "fa_deg", "time_resolution_sec", "relaxivity"} <= overrides
    assert "dce_metadata_path" not in overrides


@pytest.mark.unit
def test_the_bids_example_relies_on_sidecar_discovery() -> None:
    """The mirror-image guard: the BIDS example must keep demonstrating the convention, so
    it names a session folder and states none of the values the sidecar carries."""
    payload = json.loads(BIDS_EXAMPLE.read_text(encoding="utf-8"))

    session = Path(payload["subject_source_path"])
    sidecars = sorted((BIDS_EXAMPLE.parent / session / "dce").glob("*DCE.json"))
    assert sidecars, "the BIDS example's session folder has no discoverable DCE sidecar"

    overrides = {str(k).strip().lower() for k in payload["stage_overrides"]}
    assert not ({"tr_ms", "fa_deg", "time_resolution_sec", "relaxivity"} & overrides)
    assert "dce_metadata_path" not in overrides
