"""Single-source resolution of DCE defaults, limits and preferences.

Every default the DCE code uses lives in `python/dce_defaults.json`. Nothing in `python/`
carries a fallback value: a key absent from both the run config and the defaults file is an
error, not a guess. See
`docs/project-management/projects/defaults-single-source/PLAN.md`.

Resolution order, highest first:

    1. the run config's `stage_overrides`
    2. `dce_defaults.json`'s `defaults` block
    3. raise `DceConfigError`

`relaxivity` and `hematocrit` may legitimately differ per scan, so they resolve with the
image's JSON sidecar ahead of everything else -- see `resolve_scan_value`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

DEFAULTS_FILENAME = "dce_defaults.json"
DEFAULTS_PATH = Path(__file__).resolve().parent / DEFAULTS_FILENAME

# Keys that may be set per scan in the image's JSON sidecar, which outranks the run config.
SCAN_LEVEL_KEYS = ("relaxivity", "hematocrit")

_UNSET = object()


class DceConfigError(ValueError):
    """A required preference is missing, or an unknown preference key was supplied."""


class DceDefaults:
    """The parsed `dce_defaults.json`, with case-insensitive lookup."""

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
        raise DceConfigError(self._missing_message(key))

    def _missing_message(self, key: str) -> str:
        lc = str(key).lower()
        if lc in {k.lower() for k in self.required}:
            explanation = next(v for k, v in self.required.items() if k.lower() == lc)
            return (
                f"Required DCE preference '{key}' was not supplied.\n{explanation}\n"
                f"Set it in the image JSON sidecar, in your run config's stage_overrides, "
                f"or add it to the 'defaults' block of {self.path}."
            )
        if self.knows(key):
            return (
                f"DCE preference '{key}' has no value. It is listed as optional in "
                f"{self.path}, so this is a bug in the caller -- optional keys must be read "
                f"with a caller-supplied fallback."
            )
        return (
            f"Unknown DCE preference '{key}'. It is not listed in {self.path}. "
            f"Check the spelling, or add it to that file if it is a new setting."
        )


@lru_cache(maxsize=4)
def _load(path_text: str, mtime_ns: int) -> DceDefaults:
    del mtime_ns  # cache key only: a rewritten file reloads
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = str(payload.get("_schema", ""))
    if not schema.startswith("rocketship-dce-defaults/"):
        raise DceConfigError(f"{path} is not a ROCKETSHIP DCE defaults file (_schema={schema!r})")
    return DceDefaults(payload, path)


def load_defaults(path: Optional[Path] = None) -> DceDefaults:
    """Load (and cache) the defaults file."""
    target = Path(path) if path is not None else DEFAULTS_PATH
    if not target.exists():
        raise DceConfigError(
            f"DCE defaults file not found: {target}. This file ships with ROCKETSHIP and is "
            f"required -- the code carries no built-in fallback values."
        )
    return _load(str(target), target.stat().st_mtime_ns)


def _override(stage_overrides: Mapping[str, Any], key: str) -> Any:
    """Case-insensitive lookup in a run config's stage_overrides; `_UNSET` when absent."""
    if key in stage_overrides:
        return stage_overrides[key]
    lc = str(key).lower()
    for candidate, value in stage_overrides.items():
        if str(candidate).lower() == lc:
            return value
    return _UNSET


def resolve(config: Any, key: str, *, defaults: Optional[DceDefaults] = None) -> Any:
    """Resolve one preference: run config, then the defaults file, then raise."""
    return resolve_with_source(config, key, defaults=defaults)[0]


def resolve_with_source(
    config: Any, key: str, *, defaults: Optional[DceDefaults] = None
) -> tuple[Any, str]:
    """Resolve a preference and report where it came from.

    The source is `"run_config"` or `"defaults_file"`. Callers that record provenance need
    this: once every value has a defaults-file entry, "was this asked for or defaulted?"
    can no longer be inferred from the value alone.
    """
    table = defaults if defaults is not None else load_defaults()
    found = _override(getattr(config, "stage_overrides", {}) or {}, key)
    if found is not _UNSET:
        return found, "run_config"
    return table.default_for(key), "defaults_file"


def resolve_optional(
    config: Any, key: str, fallback: Any = None, *, defaults: Optional[DceDefaults] = None
) -> Any:
    """Resolve a key whose absence is meaningful; returns `fallback` when unset everywhere."""
    table = defaults if defaults is not None else load_defaults()
    found = _override(getattr(config, "stage_overrides", {}) or {}, key)
    if found is not _UNSET:
        return found
    return table.default_for(key, fallback)


def resolve_scan_value(
    config: Any,
    key: str,
    sidecar: Optional[Mapping[str, Any]] = None,
    *,
    defaults: Optional[DceDefaults] = None,
) -> Any:
    """Resolve a per-scan value: image JSON sidecar wins, then run config, then defaults.

    The sidecar outranks the run config here — unlike every other key — because these
    values legitimately differ between scans and the sidecar is the per-scan record.
    `dce2bids` writes them there.
    """
    if sidecar:
        found = _override(sidecar, key)
        if found is not _UNSET and found is not None:
            return found
    return resolve(config, key, defaults=defaults)


def validate_override_keys(stage_overrides: Mapping[str, Any], *, defaults: Optional[DceDefaults] = None) -> None:
    """Raise if a run config sets a key the defaults file does not recognise (typo guard)."""
    table = defaults if defaults is not None else load_defaults()
    unknown = sorted(k for k in stage_overrides if not table.knows(k))
    if unknown:
        raise DceConfigError(
            f"Unknown DCE preference key(s) in stage_overrides: {', '.join(unknown)}. "
            f"Recognised keys are listed in {table.path}."
        )
