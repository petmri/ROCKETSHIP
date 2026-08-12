#!/usr/bin/env python3
"""Dump every default/limit/preference value the DCE code resolves, from every source.

This is the oracle for the single-source-defaults migration
(`docs/project-management/projects/defaults-single-source/PLAN.md`): run it before the
migration to freeze the current numbers, run it after to prove that the only values which
moved are the ones decided in that plan's D2.

It records, per key:
  * `code_default`   -- the literal fallback in `_stage_override(config, key, <literal>)`
  * `dce_default`    -- the value in python/dce_default.json
  * `dceprep_default`-- the value in python/dceprep_default.json
plus the fully-resolved Stage-D preference dicts (bare config and JSON-config, per model)
and the standalone `dce_fit_backends` settings tables.

Usage:
    .venv/bin/python tests/python/dump_resolved_defaults.py --output <path.json>
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

MODELS = ("patlak", "tofts", "ex_tofts", "tissue_uptake", "2cxm")


def _literal(node: ast.AST) -> Any:
    """Best-effort constant folding for a default expression; None when not a literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def scan_code_defaults(path: Path) -> Dict[str, Any]:
    """Every `_stage_override(config, "key", <literal>)` fallback in a module."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "_stage_override" or len(node.args) < 3:
            continue
        key_node = node.args[1]
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue
        key = key_node.value
        value = _literal(node.args[2])
        expr = ast.get_source_segment(src, node.args[2])
        entry = {"value": value, "expr": expr, "line": node.lineno}
        # Same key read at several sites: keep them all so a disagreement is visible.
        out.setdefault(key, []).append(entry)
    return out


def scan_inline_settings_defaults(path: Path) -> Dict[str, Any]:
    """Every `settings.get("key", <literal>)` / `prefs.get(...)` fallback in a module."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "get" or len(node.args) != 2:
            continue
        base = ast.get_source_segment(src, node.func.value) or ""
        if base not in ("settings", "prefs", "inputs.prefs"):
            continue
        key_node = node.args[0]
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue
        out.setdefault(key_node.value, []).append(
            {
                "base": base,
                "value": _literal(node.args[1]),
                "expr": ast.get_source_segment(src, node.args[1]),
                "line": node.lineno,
            }
        )
    return out


def scan_module_constants(path: Path) -> Dict[str, Any]:
    """Module-level UPPER_CASE scalar constants (the tunables not in any prefs file)."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                value = _literal(node.value)
                if isinstance(value, (int, float, str, bool)):
                    out[target.id] = {"value": value, "line": node.lineno}
    return out


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "defaults_snapshot_pre.json",
        help="Where to write the snapshot JSON.",
    )
    args = parser.parse_args()

    from dce_pipeline import (  # noqa: E402
        DcePipelineConfig,
        _apply_model_specific_prefs,
        _stage_d_fit_prefs,
    )
    import dce_fit_backends as fb  # noqa: E402

    dce_json = json.loads((REPO_ROOT / "python" / "dce_default.json").read_text())
    prep_json = json.loads((REPO_ROOT / "python" / "dceprep_default.json").read_text())
    dce_over = {k.lower(): v for k, v in dce_json.get("stage_overrides", {}).items()}
    prep_over = {k.lower(): v for k, v in prep_json.get("stage_overrides", {}).items()}

    code_defaults = scan_code_defaults(REPO_ROOT / "python" / "dce_pipeline.py")

    here = Path(".")
    bare = DcePipelineConfig(subject_source_path=here, subject_tp_path=here, output_dir=here)
    with_json = DcePipelineConfig(
        subject_source_path=here,
        subject_tp_path=here,
        output_dir=here,
        stage_overrides=dict(dce_json.get("stage_overrides", {})),
    )

    prefs_bare = _stage_d_fit_prefs(bare)
    prefs_json = _stage_d_fit_prefs(with_json)

    # Per-key three-way view: what the code falls back to vs what each shipped JSON says.
    per_key: Dict[str, Any] = {}
    for key, sites in sorted(code_defaults.items()):
        lc = key.lower()
        per_key[key] = {
            "code_default": sites[0]["value"],
            "code_default_expr": sites[0]["expr"],
            "code_sites": [s["line"] for s in sites],
            "code_sites_disagree": len({json.dumps(jsonable(s["value"])) for s in sites}) > 1,
            "in_dce_default_json": lc in dce_over,
            "dce_default": dce_over.get(lc),
            "in_dceprep_default_json": lc in prep_over,
            "dceprep_default": prep_over.get(lc),
        }
    for lc, value in sorted(dce_over.items()):
        if lc not in {k.lower() for k in code_defaults}:
            per_key.setdefault(
                lc,
                {
                    "code_default": None,
                    "code_default_expr": None,
                    "code_sites": [],
                    "code_sites_disagree": False,
                    "in_dce_default_json": True,
                    "dce_default": value,
                    "in_dceprep_default_json": lc in prep_over,
                    "dceprep_default": prep_over.get(lc),
                    "note": "present in dce_default.json; not read via _stage_override",
                },
            )

    snapshot = {
        "_generated_by": "tests/python/dump_resolved_defaults.py",
        "_purpose": (
            "Pre-migration freeze of every DCE default/limit/preference. See "
            "docs/project-management/projects/defaults-single-source/PLAN.md"
        ),
        "per_key": jsonable(per_key),
        "resolved_stage_d_prefs": {
            "bare_config": jsonable(prefs_bare),
            "dce_default_json_config": jsonable(prefs_json),
            "per_model_bare": {
                m: jsonable(_apply_model_specific_prefs(prefs_bare, m)) for m in MODELS
            },
            "per_model_dce_default_json": {
                m: jsonable(_apply_model_specific_prefs(prefs_json, m)) for m in MODELS
            },
        },
        "backend_settings_tables": {
            "patlak": jsonable(fb._patlak_settings(None)),
            "tofts": jsonable(fb._tofts_settings(None)),
            "ex_tofts": jsonable(fb._ex_tofts_settings(None)),
            "tissue_uptake": jsonable(fb._tissue_uptake_settings(None)),
            "2cxm": jsonable(fb._2cxm_settings(None)),
        },
        "inline_settings_defaults": {
            "dce_models.py": jsonable(
                scan_inline_settings_defaults(REPO_ROOT / "python" / "dce_models.py")
            ),
            "dce_fit_backends.py": jsonable(
                scan_inline_settings_defaults(REPO_ROOT / "python" / "dce_fit_backends.py")
            ),
        },
        "module_constants": {
            "dce_pipeline.py": jsonable(
                scan_module_constants(REPO_ROOT / "python" / "dce_pipeline.py")
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n_keys = len(per_key)
    n_invisible = sum(1 for v in per_key.values() if not v["in_dce_default_json"])
    print(f"Wrote {args.output}")
    print(f"  keys tracked                 : {n_keys}")
    print(f"  not in dce_default.json      : {n_invisible}")
    print(f"  resolved stage-D prefs (bare): {len(prefs_bare)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
