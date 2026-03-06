"""Coverage checks for script_preferences parity audit metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PREFS = REPO_ROOT / "script_preferences.txt"
AUDIT_PATH = REPO_ROOT / "docs" / "project-management" / "projects" / "script-preferences-audit" / "script_preferences_option_audit.json"


def _parse_script_pref_keys(path: Path) -> list[str]:
    keys: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip().lower()
        if key:
            keys.append(key)
    return keys


@pytest.mark.integration
def test_script_preferences_audit_covers_all_keys() -> None:
    script_keys = _parse_script_pref_keys(SCRIPT_PREFS)
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    entries = payload.get("entries", [])
    assert isinstance(entries, list)
    audit_keys = [str(item.get("key", "")).strip().lower() for item in entries]

    assert len(audit_keys) == len(set(audit_keys)), "Duplicate keys in script-preferences audit"
    assert set(script_keys) == set(audit_keys), (
        "script_preferences and audit keys differ\n"
        f"only_in_script={sorted(set(script_keys) - set(audit_keys))}\n"
        f"only_in_audit={sorted(set(audit_keys) - set(script_keys))}"
    )


@pytest.mark.integration
def test_script_preferences_audit_status_values_and_new_alias_families() -> None:
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    allowed = {str(v) for v in payload.get("status_values", [])}
    assert allowed == {"supported", "intentionally_dropped", "pending"}

    entries = payload.get("entries", [])
    by_key = {str(item.get("key", "")).strip().lower(): item for item in entries}
    for key, item in by_key.items():
        status = str(item.get("status", "")).strip()
        assert status in allowed, f"Unexpected audit status for '{key}': {status!r}"
        assert str(item.get("python_target", "")).strip(), f"Missing python_target for '{key}'"

    # Newly-wired script-level alias families should remain explicitly marked supported.
    for key in (
        "start_t",
        "end_t",
        "tr",
        "fa",
        "time_resolution",
        "auto_find_injection",
        "blood_t1",
        "start_injection",
        "end_injection",
        "aif_type",
        "import_aif_path",
        "timevectyn",
    ):
        assert by_key[key]["status"] == "supported", f"Expected '{key}' to be marked supported"
