"""Path-first CLI entrypoint for running a single parametric T1 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List, Optional

from cli_overrides import parse_set_overrides
from parametric_cli import _build_event_logger, _stdout_event_sink
from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline


def _default_defaults_json_path() -> Path:
    return Path(__file__).resolve().parent / "parametric_defaults.json"


def _default_anat_output_dir(output_dir: Path) -> Path:
    return output_dir / "anat"


def _default_reports_output_dir(output_dir: Path) -> Path:
    return output_dir / "reports"


def _load_defaults(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Defaults JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_one(parent: Path, pattern: str) -> Optional[Path]:
    if not parent.is_dir():
        return None
    matches = sorted(parent.glob(pattern))
    if not matches:
        return None
    return matches[0].resolve()


def _canonical_dceref_vfa_files(deriv_anat: Path) -> List[Path]:
    pattern = re.compile(r"_flip-(\d+)_space-DCEref_VFA\.nii(?:\.gz)?$", re.IGNORECASE)
    flip_to_path: Dict[int, Path] = {}
    for candidate in sorted(deriv_anat.glob("*space-DCEref*_VFA.nii*")):
        match = pattern.search(candidate.name)
        if match is None:
            continue
        flip_to_path[int(match.group(1))] = candidate.resolve()
    return [flip_to_path[key] for key in sorted(flip_to_path)]


def _discover_parametric_inputs(subject_source: Path, subject_tp: Path) -> Dict[str, Any]:
    raw_anat = subject_source / "anat"
    deriv_anat = subject_tp / "anat"

    unified_vfa = _find_one(deriv_anat, "*space-DCEref_desc-bfczunified_VFA.nii*")
    if unified_vfa is not None:
        vfa_files = [unified_vfa]
    else:
        vfa_files = _canonical_dceref_vfa_files(deriv_anat)
    if not vfa_files:
        vfa_files = sorted(path.resolve() for path in raw_anat.glob("*flip-*_VFA.nii*"))
    if not vfa_files:
        vfa_files = sorted(path.resolve() for path in raw_anat.glob("*_VFA.nii*"))

    raw_sidecars = sorted(path.resolve() for path in raw_anat.glob("*flip-*_VFA.json"))
    if not raw_sidecars:
        raw_sidecars = sorted(path.resolve() for path in raw_anat.glob("*_VFA.json"))

    b1_map = _find_one(deriv_anat, "B1_scaled_FAreg.nii*")
    if b1_map is None:
        b1_map = _find_one(raw_anat, "B1_scaled_FAreg.nii*")

    mask_file = _find_one(deriv_anat, "*space-DCEref_desc-brain_mask.nii*")

    return {
        "vfa_files": vfa_files,
        "b1_map_file": b1_map,
        "mask_file": mask_file,
        "raw_sidecars": raw_sidecars,
    }


def _load_flip_angles_and_tr_from_sidecars(sidecar_paths: List[Path]) -> tuple[List[float], Optional[float]]:
    if not sidecar_paths:
        return [], None

    flip_angles: List[float] = []
    tr_values_ms: List[float] = []
    for path in sidecar_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "FlipAngle" in payload:
            flip_angles.append(float(payload["FlipAngle"]))
        if "RepetitionTime" in payload:
            tr_values_ms.append(float(payload["RepetitionTime"]) * 1000.0)

    flip_angles = sorted(flip_angles)
    if not tr_values_ms:
        return flip_angles, None

    tr_ms = tr_values_ms[0]
    for value in tr_values_ms[1:]:
        if abs(float(value) - float(tr_ms)) > 0.1:
            raise ValueError("Inconsistent RepetitionTime across raw VFA sidecars")
    return flip_angles, float(tr_ms)


def _find_bids_part(path: Path, prefix: str) -> Optional[str]:
    for part in path.parts:
        if part.startswith(prefix):
            return part
    return None


def _find_bids_entity_in_name(path: Optional[Path], entity: str) -> Optional[str]:
    if path is None:
        return None
    match = re.search(rf"(^|_){re.escape(entity)}-([^_]+)", path.name)
    if match is None:
        return None
    return f"{entity}-{match.group(2)}"


def _derive_bids_output_prefix(subject_source: Path, subject_tp: Path, candidate_paths: List[Path]) -> str:
    tokens: List[str] = []
    seen: set[str] = set()

    for prefix in ("sub-", "ses-"):
        value = _find_bids_part(subject_source, prefix) or _find_bids_part(subject_tp, prefix)
        if value is not None and value not in seen:
            tokens.append(value)
            seen.add(value)

    space_value = None
    for candidate in candidate_paths:
        space_value = _find_bids_entity_in_name(candidate, "space")
        if space_value is not None:
            break
    if space_value is not None and space_value not in seen:
        tokens.append(space_value)
        seen.add(space_value)

    if tokens:
        return "_".join(tokens)
    return "parametric"


def _derive_output_label(subject_source: Path, subject_tp: Path) -> Optional[str]:
    tokens = []
    for prefix in ("sub-", "ses-"):
        value = _find_bids_part(subject_source, prefix) or _find_bids_part(subject_tp, prefix)
        if value is not None:
            tokens.append(value)
    if tokens:
        return "_".join(tokens)
    return None


def _replace_paths(value: Any, path_map: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, dict):
        return {key: _replace_paths(item, path_map) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, path_map) for item in value]
    return value


def _rewrite_jsonl_paths(path: Path, path_map: Dict[str, str]) -> None:
    if not path.exists() or not path_map:
        return
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        lines.append(json.dumps(_replace_paths(payload, path_map), default=str))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _rename_case_artifacts(case_root: Path, result: Dict[str, Any], bids_prefix: str) -> Dict[str, Any]:
    anat_dir = _default_anat_output_dir(case_root)
    outputs = dict(result.get("outputs", {}))
    path_map: Dict[str, str] = {}

    rename_targets = {
        "t1_map_path": f"{bids_prefix}_T1map.nii.gz",
        "rsquared_map_path": f"{bids_prefix}_desc-rsquared_T1map.nii.gz",
        "rho_map_path": f"{bids_prefix}_desc-rho_T1map.nii.gz",
    }

    for output_key, file_name in rename_targets.items():
        old_path_text = str(outputs.get(output_key, "")).strip()
        if not old_path_text:
            continue
        old_path = Path(old_path_text)
        if not old_path.exists():
            continue
        new_path = anat_dir / file_name
        if new_path != old_path:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            path_map[str(old_path)] = str(new_path)
        outputs[output_key] = str(new_path)

    updated = dict(result)
    updated["outputs"] = outputs
    if path_map:
        updated = _replace_paths(updated, path_map)
    return updated


def _organize_case_reports(case_root: Path, result: Dict[str, Any], event_log_path: Path) -> Dict[str, Any]:
    reports_dir = _default_reports_output_dir(case_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    path_map: Dict[str, str] = {}
    summary_path_text = str(result.get("meta", {}).get("summary_path", "")).strip()
    if summary_path_text:
        summary_path = Path(summary_path_text)
        destination = reports_dir / summary_path.name
        if destination != summary_path:
            shutil.move(str(summary_path), str(destination))
            path_map[str(summary_path)] = str(destination)

    updated_result = _replace_paths(result, path_map)
    final_event_log = Path(path_map.get(str(event_log_path), str(event_log_path)))
    _rewrite_jsonl_paths(final_event_log, path_map)

    meta = dict(updated_result.get("meta", {}))
    meta["summary_path"] = path_map.get(summary_path_text, summary_path_text) if summary_path_text else None
    meta["event_log_path"] = str(final_event_log)
    updated_result["meta"] = meta

    final_summary_text = str(meta.get("summary_path", "")).strip()
    if final_summary_text:
        Path(final_summary_text).write_text(json.dumps(updated_result, indent=2), encoding="utf-8")

    return updated_result


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-source", type=Path, required=True, help="Raw subject/session path used for VFA and sidecar lookup.")
    parser.add_argument("--subject-tp", type=Path, required=True, help="Processed subject/session path containing derivative VFA/B1 inputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Case output root; parametric T1 artifacts are written under <output_dir>/anat.")
    parser.add_argument(
        "--defaults-json",
        type=Path,
        default=_default_defaults_json_path(),
        help="Path to JSON defaults template (default: python/parametric_default.json).",
    )
    parser.add_argument(
        "--fit-type",
        choices=["t1_fa_fit", "t1_fa_linear_fit", "t1_fa_two_point_fit"],
        default="t1_fa_fit",
        help="Parametric fit type (default: t1_fa_fit).",
    )
    parser.add_argument("--backend", choices=["auto", "cpu", "gpufit"], help="Optional backend override.")
    parser.add_argument("--tr-ms", type=float, help="Optional TR override in milliseconds.")
    parser.add_argument("--rsquared-threshold", type=float, help="Optional R-squared threshold override.")
    parser.add_argument("--vfa-file", dest="vfa_files", action="append", type=Path, default=[], help="Optional explicit VFA file override. Repeat for multi-file VFA inputs.")
    parser.add_argument("--b1-map-file", type=Path, help="Optional explicit B1 map override.")
    parser.add_argument("--mask-file", type=Path, help="Optional explicit mask override.")
    parser.add_argument("--events", choices=["on", "off"], default="on", help="Emit JSON progress events on stdout (default: on).")
    parser.add_argument("--event-log", type=Path, help="Optional JSONL path for event log (default: <output_dir>/reports/parametric_t1_events.jsonl).")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[], metavar="KEY=VALUE", help="Override top-level config key/value (repeatable).")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    subject_source = args.subject_source.expanduser().resolve()
    subject_tp = args.subject_tp.expanduser().resolve()
    case_root = args.output_dir.expanduser().resolve()
    output_dir = _default_anat_output_dir(case_root)
    defaults_json = args.defaults_json.expanduser().resolve()

    discovered = _discover_parametric_inputs(subject_source, subject_tp)
    vfa_files = [path.expanduser().resolve() for path in args.vfa_files] if args.vfa_files else discovered["vfa_files"]
    if not vfa_files:
        raise FileNotFoundError(
            "Unable to resolve VFA inputs for single-case parametric CLI. "
            "Pass --vfa-file explicitly if your layout is nonstandard."
        )

    defaults_payload = _load_defaults(defaults_json)
    payload = dict(defaults_payload.get("defaults", {}))
    payload["output_dir"] = str(output_dir)
    payload["fit_type"] = args.fit_type
    payload["vfa_files"] = [str(path) for path in vfa_files]
    payload["output_label"] = _derive_output_label(subject_source, subject_tp) or payload.get("output_label", "")

    if args.backend:
        payload["backend"] = args.backend
    if args.rsquared_threshold is not None:
        payload["rsquared_threshold"] = float(args.rsquared_threshold)

    flip_angles_deg, discovered_tr_ms = _load_flip_angles_and_tr_from_sidecars(discovered["raw_sidecars"])
    if flip_angles_deg:
        payload["flip_angles_deg"] = flip_angles_deg
    if args.tr_ms is not None:
        payload["tr_ms"] = float(args.tr_ms)
    elif discovered_tr_ms is not None:
        payload["tr_ms"] = float(discovered_tr_ms)

    b1_map_file = args.b1_map_file.expanduser().resolve() if args.b1_map_file else discovered["b1_map_file"]
    if b1_map_file is not None:
        payload["b1_map_file"] = str(b1_map_file)
    else:
        payload["b1_map_file"] = None

    mask_file = args.mask_file.expanduser().resolve() if args.mask_file else discovered["mask_file"]
    if mask_file is not None:
        payload["mask_file"] = str(mask_file)
    else:
        payload["mask_file"] = None

    bids_output_prefix = _derive_bids_output_prefix(
        subject_source,
        subject_tp,
        [*vfa_files, *( [b1_map_file] if b1_map_file is not None else [] ), *( [mask_file] if mask_file is not None else [] )],
    )

    payload.update(parse_set_overrides(args.set_overrides))

    config = ParametricT1Config.from_dict(payload)
    event_log_path = (
        args.event_log.expanduser().resolve() if args.event_log else (_default_reports_output_dir(case_root) / "parametric_t1_events.jsonl")
    )

    sinks = [_stdout_event_sink] if args.events == "on" else []
    event_log_handle, emit_event = _build_event_logger(event_log_path, sinks)
    try:
        emit_event(
            {
                "type": "cli_config",
                "mode": "single-case",
                "subject_source_path": str(subject_source),
                "subject_tp_path": str(subject_tp),
                "resolved_output_dir": str(config.output_dir),
                "defaults_json_path": str(defaults_json),
                "event_log_path": str(event_log_path),
                "config": payload,
                "resolved_inputs": {
                    "vfa_files": [str(path) for path in vfa_files],
                    "raw_sidecars": [str(path) for path in discovered["raw_sidecars"]],
                    "b1_map_file": str(b1_map_file) if b1_map_file else None,
                    "mask_file": str(mask_file) if mask_file else None,
                },
            }
        )
        result = run_parametric_t1_pipeline(config, event_callback=emit_event)
    finally:
        event_log_handle.close()

    result = _rename_case_artifacts(case_root, result, bids_output_prefix)
    result = _organize_case_reports(case_root, result, event_log_path)

    meta = dict(result["meta"])
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())