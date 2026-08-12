"""The defaults file is the single source of truth; these tests keep it that way.

See `docs/project-management/projects/defaults-single-source/PLAN.md`. The AST scans here
are the anti-rot mechanism: they fail when someone adds a preference read without adding the
key to `python/dce_defaults.json`, or reintroduces a hardcoded fallback in source.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
DEFAULTS_FILE = PYTHON_DIR / "dce_defaults.json"

sys.path.insert(0, str(PYTHON_DIR))

import dce_config  # noqa: E402
from dce_pipeline import DcePipelineConfig  # noqa: E402

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


class TestResolution:
    """The behaviours the single-source migration exists to produce."""

    def _config(self, tmp_path, **overrides):
        return DcePipelineConfig(
            subject_source_path=tmp_path,
            subject_tp_path=tmp_path,
            output_dir=tmp_path,
            stage_overrides=dict(overrides),
        )

    def test_missing_relaxivity_is_a_hard_stop(self, tmp_path) -> None:

        with pytest.raises(dce_config.DceConfigError) as excinfo:
            dce_config.resolve(self._config(tmp_path), "relaxivity")
        message = str(excinfo.value)
        assert "relaxivity" in message
        # The error has to say what to do, not just that something is missing.
        assert "sidecar" in message.lower()
        assert "dce_defaults.json" in message

    def test_unknown_key_is_rejected_rather_than_ignored(self, tmp_path) -> None:

        with pytest.raises(dce_config.DceConfigError, match="Unknown DCE preference"):
            dce_config.resolve(self._config(tmp_path), "voxel_uper_limit_ktrans")

    def test_typo_in_run_config_is_caught_by_validate(self, tmp_path) -> None:

        config = self._config(tmp_path, voxel_upper_limit_ktrnas=2.0)
        with pytest.raises(dce_config.DceConfigError, match="voxel_upper_limit_ktrnas"):
            dce_config.validate_override_keys(config.stage_overrides)

    def test_run_config_beats_the_defaults_file(self, tmp_path) -> None:

        config = self._config(tmp_path, voxel_upper_limit_ktrans=0.5)
        value, source = dce_config.resolve_with_source(config, "voxel_upper_limit_ktrans")
        assert value == 0.5
        assert source == "run_config"

        plain = dce_config.resolve_with_source(self._config(tmp_path), "voxel_upper_limit_ktrans")
        assert plain[1] == "defaults_file"

    def test_sidecar_beats_the_run_config_for_per_scan_values(self, tmp_path) -> None:
        """Inverted relative to every other key: the sidecar is the per-scan record."""

        config = self._config(tmp_path, relaxivity=3.6, hematocrit=0.40)
        sidecar = {"relaxivity": 4.5, "hematocrit": 0.38}
        assert dce_config.resolve_scan_value(config, "relaxivity", sidecar) == 4.5
        assert dce_config.resolve_scan_value(config, "hematocrit", sidecar) == 0.38
        # With no sidecar value the run config still wins over the defaults file.
        assert dce_config.resolve_scan_value(config, "relaxivity", {}) == 3.6
        assert dce_config.resolve_scan_value(config, "hematocrit", None) == 0.40

    def test_hematocrit_falls_through_to_the_defaults_file(self, tmp_path) -> None:

        assert dce_config.resolve_scan_value(self._config(tmp_path), "hematocrit", {}) == pytest.approx(0.45)
