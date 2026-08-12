"""The defaults file is the single source of truth; these tests keep it that way.

See `docs/project-management/projects/defaults-single-source/PLAN.md`. The AST scans here
are the anti-rot mechanism: they fail when someone adds a preference read without adding the
key to `python/dce_defaults.json`, or reintroduces a hardcoded fallback in source.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
DEFAULTS_FILE = PYTHON_DIR / "dce_defaults.json"

# Modules that resolve DCE preferences.
PREFERENCE_MODULES = ("dce_pipeline.py", "dce_fit_backends.py", "dce_models.py")

# Reader functions whose second positional argument is a preference key.
KEY_READERS = ("_stage_override", "resolve", "resolve_optional", "resolve_scan_value")


def _defaults_payload() -> dict:
    return json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))


def _known_keys() -> set[str]:
    payload = _defaults_payload()
    keys: set[str] = set()
    for section in ("defaults", "required", "optional"):
        keys |= {str(k).lower() for k in payload.get(section, {})}
    return keys


def _read_keys(path: Path) -> dict[str, int]:
    """Preference keys read in a module -> first line number."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name not in KEY_READERS or len(node.args) < 2:
            continue
        key_node = node.args[1]
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            found.setdefault(key_node.value.lower(), node.lineno)
    return found


def test_defaults_file_is_wellformed() -> None:
    payload = _defaults_payload()
    assert str(payload.get("_schema", "")).startswith("rocketship-dce-defaults/")
    for section in ("defaults", "required", "optional", "_units"):
        assert isinstance(payload.get(section), dict), f"missing/invalid section: {section}"

    # A key must live in exactly one of the three behavioural sections.
    sections = {s: {k.lower() for k in payload[s]} for s in ("defaults", "required", "optional")}
    for a, b in (("defaults", "required"), ("defaults", "optional"), ("required", "optional")):
        overlap = sections[a] & sections[b]
        assert not overlap, f"keys in both '{a}' and '{b}': {sorted(overlap)}"


def test_relaxivity_has_no_default() -> None:
    """Relaxivity depends on the contrast agent, so shipping a value would be a wrong guess."""
    payload = _defaults_payload()
    assert "relaxivity" in {k.lower() for k in payload["required"]}
    assert "relaxivity" not in {k.lower() for k in payload["defaults"]}


def test_hematocrit_has_a_default() -> None:
    """Hematocrit is usually study-wide, so a default is appropriate (unlike relaxivity)."""
    payload = _defaults_payload()
    assert "hematocrit" in {k.lower() for k in payload["defaults"]}


@pytest.mark.parametrize("module", PREFERENCE_MODULES)
def test_every_preference_key_read_is_declared(module: str) -> None:
    """No code may read a preference key the defaults file does not declare."""
    path = PYTHON_DIR / module
    if not path.exists():
        pytest.skip(f"{module} not present")
    known = _known_keys()
    undeclared = {k: line for k, line in _read_keys(path).items() if k not in known}
    assert not undeclared, (
        f"{module} reads preference keys missing from {DEFAULTS_FILE.name}: "
        + ", ".join(f"{k} (line {line})" for k, line in sorted(undeclared.items()))
        + ". Add them to that file, or fix the spelling."
    )


def test_deleted_keys_are_gone() -> None:
    """Keys removed by the migration must not reappear in any shipped config."""
    removed = {"use_dce_preferences", "injection_duration", "dce_preferences_path"}
    for name in ("dce_defaults.json", "dce_default.json", "dceprep_default.json"):
        path = PYTHON_DIR / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        present = {k.lower() for k in payload.get("stage_overrides", {})}
        present |= {k.lower() for k in payload.get("defaults", {})}
        present |= {k.lower() for k in payload.get("optional", {})}
        clash = present & removed
        assert not clash, f"{name} still declares removed key(s): {sorted(clash)}"
