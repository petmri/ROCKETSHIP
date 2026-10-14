"""Unit tests for the shared `--set KEY=VALUE` parser.

Every entry point (both CLIs, both batch processors, the GUI override table) reads its
values through this module. They each used to carry their own copy and disagreed, which is
how `--set write_param_maps=false` came to mean the opposite of what it said.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import cli_overrides  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected",
    [
        # Booleans are the reason this module exists: the string "false" is truthy.
        ("false", False),
        ("true", True),
        ("False", False),
        ("TRUE", True),
        # Numbers keep their type so downstream int()/float() reads stay honest.
        ("123", 123),
        ("1600", 1600),
        ("5.7", 5.7),
        ("1e5", 100000.0),
        ("-1", -1),
        # Bare words are not JSON and must survive as typed.
        ("tv", "tv"),
        ("Dyn-1", "Dyn-1"),
        ("./data/sub-01_DCE.nii.gz", "./data/sub-01_DCE.nii.gz"),
        # Explicit emptiness and explicit nothing.
        ("", ""),
        ("none", None),
        ("null", None),
    ],
)
def test_values_keep_the_type_their_text_declares(text, expected) -> None:
    assert cli_overrides.coerce_override_value(text) == expected


@pytest.mark.unit
def test_booleans_are_real_bools_not_truthy_strings() -> None:
    """`==` would pass for 0/1, so check identity: the pipeline branches on these."""
    assert cli_overrides.coerce_override_value("false") is False
    assert cli_overrides.coerce_override_value("true") is True


@pytest.mark.unit
def test_surrounding_whitespace_is_ignored() -> None:
    assert cli_overrides.coerce_override_value("  false  ") is False
    assert cli_overrides.coerce_override_value("  5.7 ") == 5.7


@pytest.mark.unit
def test_parses_repeated_key_value_pairs() -> None:
    parsed = cli_overrides.parse_set_overrides(
        ["voxel_MaxFunEvals=123", "write_param_maps=false", "rootname=Dyn-1"]
    )
    assert parsed == {
        "voxel_MaxFunEvals": 123,
        "write_param_maps": False,
        "rootname": "Dyn-1",
    }


@pytest.mark.unit
def test_values_may_contain_equals_signs() -> None:
    assert cli_overrides.parse_set_overrides(["note=a=b"]) == {"note": "a=b"}


@pytest.mark.unit
def test_rejects_malformed_entries() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        cli_overrides.parse_set_overrides(["bad_entry"])
    with pytest.raises(ValueError, match="Empty KEY"):
        cli_overrides.parse_set_overrides(["=value"])


@pytest.mark.unit
def test_every_entry_point_shares_this_parser() -> None:
    """Guards the reason the module exists: four copies had drifted apart."""
    import dce_cli
    import parametric_cli
    import run_dce_bids_batch
    import run_parametric_bids_batch

    sys.path.insert(0, str(REPO_ROOT))
    for module in (dce_cli, parametric_cli, run_dce_bids_batch, run_parametric_bids_batch):
        assert module.parse_set_overrides is cli_overrides.parse_set_overrides
