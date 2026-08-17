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

# Override spellings that were retired, mapped to the surviving key and its units. Each of
# these was a second name for a setting that already had one; carrying both meant the
# resolution order decided which won, and that order was not something a user could infer
# from the config. Kept here so the "unknown key" guard can name the replacement instead of
# just rejecting a spelling that used to work -- these are the MATLAB script names, so they
# turn up in configs translated by hand from `run_dce_cli.m`.
REMOVED_OVERRIDE_ALIASES: Dict[str, str] = {
    "start_time": "restrict_fit_start_min (minutes)",
    "end_time": "restrict_fit_end_min (minutes)",
    "start_time_min": "restrict_fit_start_min (minutes)",
    "end_time_min": "restrict_fit_end_min (minutes)",
    "end_injection": "end_injection_min (minutes)",
    "time_resolution": "time_resolution_sec (seconds)",
    "imported_aif_path": "import_aif_path",
    "aif_type": "aif_mode (top-level config field, not a stage override)",
    "aif_curve_mode": "aif_mode (top-level config field, not a stage override)",
    "tr": "tr_ms (milliseconds)",
    "fa": "fa_deg (degrees)",
    # Second units for a quantity that only needs one. Convert the number yourself; the
    # code no longer guesses which unit you meant.
    "tr_sec": "tr_ms (milliseconds)",
    "blood_t1": "blood_t1_ms (milliseconds)",
    "blood_t1_sec": "blood_t1_ms (milliseconds)",
    "time_resolution_min": "time_resolution_sec (seconds)",
}

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


# --- Stage-D fit settings -------------------------------------------------------------
#
# The fit backends take short, unprefixed setting names (`lower_limit_ktrans`); the
# user-facing file spells them `voxel_lower_limit_ktrans`. This table is the single
# definition of that mapping, so the pipeline, the backends and the tests cannot resolve
# different numbers -- which is exactly what they did before this file existed.

_STAGE_D_SHARED: tuple[tuple[str, str, str], ...] = (
    ("lower_limit_ktrans", "voxel_lower_limit_ktrans", "float"),
    ("upper_limit_ktrans", "voxel_upper_limit_ktrans", "float"),
    ("initial_value_ktrans", "voxel_initial_value_ktrans", "float"),
    ("lower_limit_ve", "voxel_lower_limit_ve", "float"),
    ("upper_limit_ve", "voxel_upper_limit_ve", "float"),
    ("initial_value_ve", "voxel_initial_value_ve", "float"),
    ("lower_limit_vp", "voxel_lower_limit_vp", "float"),
    ("upper_limit_vp", "voxel_upper_limit_vp", "float"),
    ("initial_value_vp", "voxel_initial_value_vp", "float"),
    ("lower_limit_fp", "voxel_lower_limit_fp", "float"),
    ("upper_limit_fp", "voxel_upper_limit_fp", "float"),
    ("initial_value_fp", "voxel_initial_value_fp", "float"),
    ("lower_limit_tp", "voxel_lower_limit_tp", "float"),
    ("upper_limit_tp", "voxel_upper_limit_tp", "float"),
    ("initial_value_tp", "voxel_initial_value_tp", "float"),
    ("lower_limit_tau", "voxel_lower_limit_tau", "float"),
    ("upper_limit_tau", "voxel_upper_limit_tau", "float"),
    ("initial_value_tau", "voxel_initial_value_tau", "float"),
    ("lower_limit_ktrans_rr", "voxel_lower_limit_ktrans_RR", "float"),
    ("upper_limit_ktrans_rr", "voxel_upper_limit_ktrans_RR", "float"),
    ("initial_value_ktrans_rr", "voxel_initial_value_ktrans_RR", "float"),
    ("value_ve_rr", "voxel_value_ve_RR", "float"),
    ("tol_fun", "voxel_TolFun", "float"),
    ("tol_x", "voxel_TolX", "float"),
    ("max_iter", "voxel_MaxIter", "int"),
    ("max_nfev", "voxel_MaxFunEvals", "int"),
    ("robust", "voxel_Robust", "str"),
    ("gpu_tolerance", "gpu_tolerance", "float"),
    ("gpu_max_n_iterations", "gpu_max_n_iterations", "int"),
    ("gpu_initial_value_ktrans", "gpu_initial_value_ktrans", "float"),
    ("gpu_initial_value_ve", "gpu_initial_value_ve", "float"),
    ("gpu_initial_value_vp", "gpu_initial_value_vp", "float"),
    ("gpu_initial_value_fp", "gpu_initial_value_fp", "float"),
    ("fxr_fw", "fxr_fw", "float"),
)

# Per-model overrides, emitted under a `<model>_` prefix that the pipeline promotes for the
# model being fitted. They exist so an unstable model can be tuned without moving the others.
_STAGE_D_PER_MODEL: Dict[str, tuple[tuple[str, str, str], ...]] = {
    "2cxm": (
        ("lower_limit_ktrans", "voxel_lower_limit_ktrans_2cxm", "raw"),
        ("upper_limit_ktrans", "voxel_upper_limit_ktrans_2cxm", "raw"),
        ("initial_value_ktrans", "voxel_initial_value_ktrans_2cxm", "raw"),
        ("lower_limit_ve", "voxel_lower_limit_ve_2cxm", "raw"),
        ("upper_limit_ve", "voxel_upper_limit_ve_2cxm", "raw"),
        ("initial_value_ve", "voxel_initial_value_ve_2cxm", "raw"),
        ("lower_limit_vp", "voxel_lower_limit_vp_2cxm", "raw"),
        ("upper_limit_vp", "voxel_upper_limit_vp_2cxm", "raw"),
        ("initial_value_vp", "voxel_initial_value_vp_2cxm", "raw"),
        ("lower_limit_fp", "voxel_lower_limit_fp_2cxm", "raw"),
        ("upper_limit_fp", "voxel_upper_limit_fp_2cxm", "raw"),
        ("initial_value_fp", "voxel_initial_value_fp_2cxm", "raw"),
        ("max_nfev", "voxel_MaxFunEvals_2cxm", "raw"),
        ("max_iter", "voxel_MaxIter_2cxm", "raw"),
        ("robust", "voxel_Robust_2cxm", "optional"),
    ),
    "tissue_uptake": (
        ("lower_limit_ktrans", "voxel_lower_limit_ktrans_tissue_uptake", "raw"),
        ("upper_limit_ktrans", "voxel_upper_limit_ktrans_tissue_uptake", "raw"),
        ("initial_value_ktrans", "voxel_initial_value_ktrans_tissue_uptake", "raw"),
        ("lower_limit_vp", "voxel_lower_limit_vp_tissue_uptake", "raw"),
        ("upper_limit_vp", "voxel_upper_limit_vp_tissue_uptake", "raw"),
        ("initial_value_vp", "voxel_initial_value_vp_tissue_uptake", "raw"),
        ("lower_limit_fp", "voxel_lower_limit_fp_tissue_uptake", "raw"),
        ("upper_limit_fp", "voxel_upper_limit_fp_tissue_uptake", "raw"),
        ("initial_value_fp", "voxel_initial_value_fp_tissue_uptake", "raw"),
        ("lower_limit_tp", "voxel_lower_limit_tp_tissue_uptake", "raw"),
        ("upper_limit_tp", "voxel_upper_limit_tp_tissue_uptake", "raw"),
        ("initial_value_tp", "voxel_initial_value_tp_tissue_uptake", "raw"),
        ("max_nfev", "voxel_MaxFunEvals_tissue_uptake", "raw"),
        ("max_iter", "voxel_MaxIter_tissue_uptake", "raw"),
        ("robust", "voxel_Robust_tissue_uptake", "optional"),
    ),
}


def _coerce(value: Any, kind: str, key: str) -> Any:
    if kind == "raw":
        return value
    if kind == "str":
        return str(value).strip()
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DceConfigError(
            f"DCE preference '{key}' must be numeric, got {value!r}. Fix it in the run "
            f"config or in {DEFAULTS_PATH}."
        ) from exc
    return int(number) if kind == "int" else number


def stage_d_prefs(config: Any = None, *, defaults: Optional[DceDefaults] = None) -> Dict[str, Any]:
    """The complete Stage-D fit settings dict.

    Pass a run config to layer its `stage_overrides` on top; pass none to get exactly what
    `dce_defaults.json` specifies, which is what the contract and reliability tests want --
    they should measure the shipped configuration, not a separate one.
    """
    table = defaults if defaults is not None else load_defaults()
    out: Dict[str, Any] = {}
    for setting_key, file_key, kind in _STAGE_D_SHARED:
        out[setting_key] = _coerce(resolve(config, file_key, defaults=table), kind, file_key)
    for model, entries in _STAGE_D_PER_MODEL.items():
        for setting_key, file_key, kind in entries:
            if kind == "optional":
                # Unset means "use the shared value"; `_apply_model_specific_prefs` skips
                # None entries, so it must stay None rather than being coerced.
                out[f"{model}_{setting_key}"] = resolve_optional(
                    config, file_key, None, defaults=table
                )
                continue
            value = resolve(config, file_key, defaults=table)
            out[f"{model}_{setting_key}"] = _coerce(value, kind, file_key)
    return out


def model_settings(
    model_name: str, config: Any = None, *, defaults: Optional[DceDefaults] = None
) -> Dict[str, Any]:
    """Stage-D settings for one model, with its `<model>_*` overrides already promoted.

    Values are in canonical per-minute units, matching `_units` in the defaults file.
    """
    prefs = stage_d_prefs(config, defaults=defaults)
    prefix = f"{model_name}_"
    resolved = {
        k: v
        for k, v in prefs.items()
        if not any(k.startswith(f"{m}_") for m in _STAGE_D_PER_MODEL)
    }
    for key, value in prefs.items():
        if not key.startswith(prefix) or value is None:
            continue
        base = key[len(prefix):]
        resolved[base] = int(float(value)) if base in {"max_iter", "max_nfev"} else value
    return resolved


def validate_override_keys(stage_overrides: Mapping[str, Any], *, defaults: Optional[DceDefaults] = None) -> None:
    """Raise if a run config sets a key the defaults file does not recognise (typo guard)."""
    table = defaults if defaults is not None else load_defaults()
    unknown = sorted(k for k in stage_overrides if not table.knows(k))
    if not unknown:
        return

    retired = [(k, REMOVED_OVERRIDE_ALIASES[k.lower()]) for k in unknown if k.lower() in REMOVED_OVERRIDE_ALIASES]
    if retired:
        detail = "; ".join(f"'{key}' was removed, use '{replacement}'" for key, replacement in retired)
        raise DceConfigError(
            f"Retired DCE preference key(s) in stage_overrides: {detail}. "
            f"Recognised keys are listed in {table.path}."
        )
    raise DceConfigError(
        f"Unknown DCE preference key(s) in stage_overrides: {', '.join(unknown)}. "
        f"Recognised keys are listed in {table.path}."
    )
