"""Resolution of parametric T1 preferences against `parametric_defaults.json`.

The parametric counterpart of `dce_config.py`, and deliberately the same shape: one
user-editable file holds every default, source carries none, and a key absent from both
the run config and the defaults file is an error rather than a guess.

Two things differ from the DCE side, both because the parametric config is flat:

* There is no `stage_overrides` block -- a run config's own top-level keys are the
  overrides, so `resolve` reads the payload directly.
* Input paths (`output_dir`, `vfa_files`, `mask_file`, `b1_map_file`) describe one study
  rather than how the software behaves, so they live in a run config and never in the
  defaults file. `PATH_VALUED_KEYS` names them for the callers that re-anchor relative
  paths.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

DEFAULTS_FILENAME = "parametric_defaults.json"
DEFAULTS_PATH = Path(__file__).resolve().parent / DEFAULTS_FILENAME

# Keys whose values are paths, and so move with the config file they are written in.
# Mirrors dce_config.PATH_VALUED_KEYS; kept here rather than imported because the two
# pipelines' key sets are unrelated and coupling them would be a false economy.
PATH_VALUED_KEYS = ("output_dir", "vfa_files", "mask_file", "b1_map_file")

# Older spellings still accepted in run configs, mapped to the current key. Kept because
# they appear in user configs written against earlier releases; a run using one is not
# doing anything wrong.
LEGACY_KEY_ALIASES = {
    "file_list": "vfa_files",
    "parameters": "flip_angles_deg",
    "tr": "tr_ms",
    "xy_smooth_size": "xy_smooth_sigma",
}

_UNSET = object()


class ParametricConfigError(ValueError):
    """A required preference is missing, or an unknown preference key was supplied."""


class ParametricDefaults:
    """The parsed `parametric_defaults.json`, with case-insensitive lookup."""

    __slots__ = ("path", "defaults", "required", "optional", "units", "_lc_defaults", "_known_lc")

    def __init__(self, payload: Mapping[str, Any], path: Path) -> None:
        self.path = path
        self.defaults: Dict[str, Any] = dict(payload.get("defaults", {}))
        self.required: Dict[str, str] = dict(payload.get("required", {}))
        self.optional: Dict[str, str] = dict(payload.get("optional", {}))
        self.units: Dict[str, str] = dict(payload.get("_units", {}))
        self._lc_defaults = {k.lower(): v for k, v in self.defaults.items()}
        self._known_lc = (
            set(self._lc_defaults)
            | {k.lower() for k in self.required}
            | {k.lower() for k in self.optional}
        )

    def knows(self, key: str) -> bool:
        """Whether `key` is a recognised preference in any section."""
        return str(key).lower() in self._known_lc

    def is_optional(self, key: str) -> bool:
        """Whether absence of `key` is meaningful rather than an error."""
        return str(key).lower() in {k.lower() for k in self.optional}

    def default_for(self, key: str, fallback: Any = _UNSET) -> Any:
        lc = str(key).lower()
        if lc in self._lc_defaults:
            return self._lc_defaults[lc]
        if fallback is not _UNSET:
            return fallback
        raise ParametricConfigError(self._missing_message(key))

    def _missing_message(self, key: str) -> str:
        lc = str(key).lower()
        if lc in {k.lower() for k in self.required}:
            explanation = next(v for k, v in self.required.items() if k.lower() == lc)
            return (
                f"Required parametric setting '{key}' was not supplied.\n{explanation}\n"
                f"Set it in your run config, or add it to the 'defaults' block of {self.path}."
            )
        if self.knows(key):
            return (
                f"Parametric setting '{key}' has no value. It is listed as optional in "
                f"{self.path}, so this is a bug in the caller -- optional keys must be read "
                f"with a caller-supplied fallback."
            )
        return (
            f"Unknown parametric setting '{key}'. It is not listed in {self.path}. "
            f"Check the spelling, or add it to that file if it is a new setting."
        )


@lru_cache(maxsize=4)
def _load(path_text: str, mtime_ns: int) -> ParametricDefaults:
    del mtime_ns  # cache key only: a rewritten file reloads
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = str(payload.get("_schema", ""))
    if not schema.startswith("rocketship-parametric-defaults/"):
        raise ParametricConfigError(
            f"{path} is not a ROCKETSHIP parametric defaults file (_schema={schema!r})"
        )
    return ParametricDefaults(payload, path)


def load_defaults(path: Optional[Path] = None) -> ParametricDefaults:
    """Load (and cache) the defaults file."""
    target = Path(path) if path is not None else DEFAULTS_PATH
    if not target.exists():
        raise ParametricConfigError(
            f"Parametric defaults file not found: {target}. This file ships with ROCKETSHIP "
            f"and is required -- the code carries no built-in fallback values."
        )
    return _load(str(target), target.stat().st_mtime_ns)


def canonical_key(key: str) -> str:
    """The current spelling of `key`, translating the legacy aliases."""
    return LEGACY_KEY_ALIASES.get(str(key).lower(), str(key))


def _from_payload(payload: Mapping[str, Any], key: str) -> Any:
    """Case-insensitive lookup in a run config, honouring legacy spellings.

    A `null` in the config reads as absent rather than as the value None: that is what
    writing `"tr_ms": null` means to someone editing the file by hand, and the pipeline's
    own examples use it that way.
    """
    wanted = {str(key).lower()}
    wanted |= {alias for alias, current in LEGACY_KEY_ALIASES.items() if current == key}
    for candidate, value in payload.items():
        if str(candidate).lower() in wanted and value is not None:
            return value
    return _UNSET


def resolve(
    payload: Mapping[str, Any], key: str, *, defaults: Optional[ParametricDefaults] = None
) -> Any:
    """Resolve one setting: run config, then the defaults file, then raise."""
    return resolve_with_source(payload, key, defaults=defaults)[0]


def resolve_with_source(
    payload: Mapping[str, Any], key: str, *, defaults: Optional[ParametricDefaults] = None
) -> tuple:
    """Resolve a setting and report where it came from, as `"run_config"`/`"defaults_file"`.

    Callers that record provenance need this: once every value has a defaults-file entry,
    "was this asked for or defaulted?" can no longer be inferred from the value alone.
    """
    table = defaults if defaults is not None else load_defaults()
    found = _from_payload(payload, key)
    if found is not _UNSET:
        return found, "run_config"
    return table.default_for(key), "defaults_file"


def resolve_optional(
    payload: Mapping[str, Any],
    key: str,
    fallback: Any = None,
    *,
    defaults: Optional[ParametricDefaults] = None,
) -> Any:
    """Resolve a setting whose absence is meaningful, without raising."""
    table = defaults if defaults is not None else load_defaults()
    found = _from_payload(payload, key)
    if found is not _UNSET:
        return found
    return table.default_for(key, fallback)


def validate_keys(
    payload: Mapping[str, Any], *, defaults: Optional[ParametricDefaults] = None
) -> None:
    """Reject unknown keys in a run config, so a typo stops the run instead of being ignored.

    Path keys are permitted without being preferences: they are how a run names its inputs,
    and deliberately have no entry in the defaults file. Keys starting with `_` are
    comments, matching the convention the shipped configs already use.
    """
    table = defaults if defaults is not None else load_defaults()
    unknown = [
        str(key)
        for key in payload
        if not str(key).startswith("_")
        and str(key).lower() not in {k.lower() for k in PATH_VALUED_KEYS}
        and str(key).lower() not in LEGACY_KEY_ALIASES
        and not table.knows(canonical_key(str(key)))
    ]
    if unknown:
        raise ParametricConfigError(
            f"Unknown parametric setting(s) in run config: {', '.join(sorted(unknown))}. "
            f"Known settings are listed in {table.path}."
        )


def resolve_override_paths(payload: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Re-anchor a config's path-valued keys against `base_dir`.

    Mirrors `dce_config.resolve_override_paths`, so a parametric config's relative paths
    move with the file the same way a DCE config's do.
    """
    resolved = dict(payload)
    for key in PATH_VALUED_KEYS:
        if key not in resolved or resolved[key] is None:
            continue
        value = resolved[key]
        if isinstance(value, (list, tuple)):
            resolved[key] = [str(_anchor(str(item), base_dir)) for item in value]
        else:
            resolved[key] = str(_anchor(str(value), base_dir))
    return resolved


def _anchor(text: str, base_dir: Path) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else (Path(base_dir) / path)
