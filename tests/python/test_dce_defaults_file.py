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
KEY_READERS = (
    "_stage_override",
    "resolve",
    "resolve_optional",
    "resolve_scan_value",
    "resolve_scan_value_with_source",
)

# Resolution calls -> index of the first argument acting as a fallback. A literal there is
# a default living in source, which is what this migration exists to remove.
FALLBACK_ARG = {
    "resolve": 2,
    "resolve_with_source": 2,
    "resolve_optional": 2,
    "resolve_scan_value": 3,  # (config, key, sidecar)
    "resolve_scan_value_with_source": 3,  # (config, key, sidecar)
    "_stage_override": 2,
    "_stage_override_optional": 2,
    "_scan_override": 3,
    "default_for": 1,
}


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


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _literal_fallback(node: ast.AST) -> str | None:
    """The literal a fallback expression bottoms out in, or None if it has no literal."""
    if isinstance(node, ast.Constant):
        # None is not a default -- it is the "unset" sentinel optional keys resolve to.
        return None if node.value is None else repr(node.value)
    if isinstance(node, ast.Call):
        name = _call_name(node)
        # A nested `d.get(key, literal)` hides a default just as effectively.
        if name == "get" and len(node.args) == 2:
            return _literal_fallback(node.args[1])
        if name in {"float", "int", "str", "bool"} and len(node.args) == 1:
            return _literal_fallback(node.args[0])
    return None


def _literal_defaults(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        index = FALLBACK_ARG.get(_call_name(node) or "")
        if index is None:
            continue
        candidates = list(node.args[index:])
        candidates += [kw.value for kw in node.keywords if kw.arg in {"default", "fallback"}]
        for candidate in candidates:
            literal = _literal_fallback(candidate)
            if literal is not None:
                hits.append(f"line {node.lineno}: {_call_name(node)}(..., {literal})")
    return hits


def _shipped_sources() -> list[Path]:
    return sorted(PYTHON_DIR.glob("*.py")) + sorted(REPO_ROOT.glob("run_*.py"))


@pytest.mark.parametrize("path", _shipped_sources(), ids=lambda p: p.name)
def test_no_literal_defaults_in_resolution_calls(path: Path) -> None:
    """A default passed at a call site is a default living in source. There are none.

    Without this the next feature adds `_stage_override(config, "new_key", 0.5)` and the
    single source of truth quietly stops being single.
    """
    hits = _literal_defaults(path)
    assert not hits, (
        f"{path.name} passes a literal default to a config-resolution call: "
        + "; ".join(hits)
        + f". Put the value in {DEFAULTS_FILE.name} instead."
    )


def test_the_literal_default_guard_actually_catches_one() -> None:
    """Guard the guard: an AST scan that silently matches nothing is worse than none."""
    source = (
        "_stage_override(config, 'a', 1.0)\n"
        "_stage_override_optional(config, 'b', stage_a.get('b', 2.0))\n"
        "_scan_override(config, 'c', sidecar, float('3.0'))\n"
        "resolve_optional(config, 'd', None)\n"  # None is the unset sentinel, not a default
        "_stage_override(config, 'e')\n"
        "_scan_override(config, 'f', sidecar)\n"
        "some_other_call(config, 'g', 4.0)\n"
    )
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in FALLBACK_ARG:
            index = FALLBACK_ARG[_call_name(node)]
            for candidate in node.args[index:]:
                literal = _literal_fallback(candidate)
                if literal is not None:
                    hits.append(literal)
    assert hits == ["1.0", "2.0", "'3.0'"]


def _shipped_configs() -> list[Path]:
    """Every committed run config, found by shape rather than by a hand-kept filename list."""
    candidates = sorted(PYTHON_DIR.glob("*.json")) + sorted((REPO_ROOT / "tests" / "python").glob("*.json"))
    out = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("stage_overrides"), dict):
            out.append(path)
    return out


@pytest.mark.parametrize("path", _shipped_configs(), ids=lambda p: p.name)
def test_shipped_configs_only_use_known_keys(path: Path) -> None:
    """A shipped config with a stale key is a documented command that errors on contact.

    Discovered by shape, not by a filename list: an earlier version of this test named three
    files and so missed `dce_cli_config.example.json`, which AGENTS.md tells users to run and
    which carried a `dce_preferences_path` the migration had already removed.
    """
    overrides = json.loads(path.read_text(encoding="utf-8"))["stage_overrides"]
    dce_config.validate_override_keys(overrides)


def test_deleted_keys_are_gone() -> None:
    """Keys removed by the migration must not reappear in the defaults file itself."""
    removed = {"use_dce_preferences", "injection_duration", "dce_preferences_path"}
    payload = _defaults_payload()
    declared = {k.lower() for section in ("defaults", "required", "optional") for k in payload.get(section, {})}
    clash = declared & removed
    assert not clash, f"{DEFAULTS_FILE.name} still declares removed key(s): {sorted(clash)}"


def test_retired_aliases_are_not_declared() -> None:
    """A retired spelling must not be re-declared, or it silently starts working again."""
    payload = _defaults_payload()
    declared = {k.lower() for section in ("defaults", "required", "optional") for k in payload.get(section, {})}
    clash = declared & set(dce_config.REMOVED_OVERRIDE_ALIASES)
    assert not clash, f"{DEFAULTS_FILE.name} re-declares retired alias(es): {sorted(clash)}"


@pytest.mark.parametrize("alias", sorted(dce_config.REMOVED_OVERRIDE_ALIASES))
def test_retired_alias_names_its_replacement(alias: str) -> None:
    """Rejecting the old spelling is not enough on its own.

    These are the MATLAB script names, so they arrive in configs translated by hand from
    `run_dce_cli.m`. A bare "unknown key" would read as "unsupported" rather than "renamed",
    so the message has to carry the surviving key.
    """
    replacement = dce_config.REMOVED_OVERRIDE_ALIASES[alias]
    with pytest.raises(dce_config.DceConfigError) as excinfo:
        dce_config.validate_override_keys({alias: 1})
    message = str(excinfo.value)
    assert "was removed" in message
    assert replacement in message


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

    def test_per_scan_resolution_reports_which_source_won(self, tmp_path) -> None:
        """A run-config value losing to the sidecar is normal, so Stage A logs which won.

        Without the reported source that outcome is invisible: the run config names a
        relaxivity, the sidecar quietly supplies a different one, and nothing says so.
        """

        config = self._config(tmp_path, relaxivity=3.6, hematocrit=0.40)

        assert dce_config.resolve_scan_value_with_source(
            config, "relaxivity", {"relaxivity": 4.5}
        ) == (4.5, "sidecar")
        assert dce_config.resolve_scan_value_with_source(config, "relaxivity", {}) == (
            3.6,
            "run_config",
        )

        bare = self._config(tmp_path)
        value, source = dce_config.resolve_scan_value_with_source(bare, "hematocrit", {})
        assert source == "defaults_file"
        assert value == pytest.approx(0.45)

    def test_hematocrit_falls_through_to_the_defaults_file(self, tmp_path) -> None:

        assert dce_config.resolve_scan_value(self._config(tmp_path), "hematocrit", {}) == pytest.approx(0.45)
