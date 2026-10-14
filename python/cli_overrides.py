"""One parser for `--set KEY=VALUE` arguments, shared by every entry point.

Each CLI had grown its own copy and they disagreed. The DCE ones kept every value as the
string it was typed as, so `--set write_param_maps=false` resolved to the truthy string
`"false"` and quietly did the opposite of what was asked, while the GUI parsed the same
setting through `json.loads` and got it right.

Values are read as JSON where that succeeds, so numbers, booleans, `null` and lists arrive
with their type intact. Anything JSON rejects stays the string it was typed as, which is
what keeps bare words (`tv`, `Dyn-1`) and file paths working.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable

# Spellings a shell user reaches for that JSON does not accept. `none` matters because the
# alternative is a string that reads as a value: `steady_state_auto_method=none` and
# `=None` must mean the same thing the config file's `null` does.
_BOOL_WORDS = {"true": True, "false": False}
_NULL_WORDS = {"none", "null"}


def coerce_override_value(raw: str) -> Any:
    """Read one `--set` value, keeping its type where the text declares one."""
    text = str(raw).strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except ValueError:
        pass
    lowered = text.lower()
    if lowered in _BOOL_WORDS:
        return _BOOL_WORDS[lowered]
    if lowered in _NULL_WORDS:
        return None
    return text


def parse_set_overrides(values: Iterable[str]) -> Dict[str, Any]:
    """Parse repeated `KEY=VALUE` arguments into a dict, coercing each value."""
    overrides: Dict[str, Any] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --set entry '{raw}'. Expected KEY=VALUE")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set entry '{raw}'. Empty KEY")
        overrides[key] = coerce_override_value(value)
    return overrides
