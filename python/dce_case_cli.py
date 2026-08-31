"""Path-first CLI entrypoint for running a single DCE case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, Iterable, Optional

from dce_cli import _build_event_logger, _stdout_event_sink, parse_set_overrides
from dce_pipeline import DcePipelineConfig, run_dce_pipeline


def _default_defaults_json_path() -> Path:
    return Path(__file__).resolve().parent / "dce_defaults.json"


def _default_reports_dir(output_dir: Path) -> Path:
    return output_dir / "reports"


def _find_first(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _glob_first(parent: Path, *patterns: str) -> Optional[Path]:
    if not parent.is_dir():
        return None
    for pattern in patterns:
        matches = sorted(parent.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def _dynamic_sidecar(dynamic_file: Optional[Path]) -> Optional[Path]:
    if dynamic_file is None:
        return None
    name = dynamic_file.name
    if name.endswith(".nii.gz"):
        candidate = dynamic_file.with_name(name[:-7] + ".json")
    elif name.endswith(".nii"):
        candidate = dynamic_file.with_suffix(".json")
    else:
        return None
    return candidate.resolve() if candidate.exists() else None


def _resolve_single_case_inputs(subject_source: Path, subject_tp: Path) -> Dict[str, Optional[Path]]:
    source = subject_source.expanduser().resolve()
    tp = subject_tp.expanduser().resolve()
    processed = tp / "processed"
    dce_dir = tp / "dce"
    anat_dir = tp / "anat"
    source_dce_dir = source / "dce"

    dynamic = _find_first(
        filter(
            None,
            [
                _glob_first(dce_dir, "*desc-bfcz_DCE.nii*", "*DCE.nii*"),
                _glob_first(source_dce_dir, "*DCE.nii*"),
                source / "Dynamic_t1w.nii",
                source / "Dynamic_t1w.nii.gz",
                tp / "Dynamic_t1w.nii",
                tp / "Dynamic_t1w.nii.gz",
            ],
        )
    )
    aif = _find_first(
        filter(
            None,
            [
                _glob_first(dce_dir, "*label-AIF_T1map.nii*"),
                tp / "T1_AIF_roi.nii",
                tp / "T1_AIF_roi.nii.gz",
                processed / "T1_AIF_roi.nii",
                processed / "T1_AIF_roi.nii.gz",
            ],
        )
    )
    roi = _find_first(
        filter(
            None,
            [
                _glob_first(anat_dir, "*space-DCEref_label-brain_mask.nii*"),
                tp / "T1_brain_roi.nii",
                tp / "T1_brain_roi.nii.gz",
                processed / "T1_brain_roi.nii",
                processed / "T1_brain_roi.nii.gz",
            ],
        )
    )
    t1map = _find_first(
        filter(
            None,
            [
                _glob_first(anat_dir, "*space-DCEref_T1map.nii*"),
                tp / "T1_map_t1_fa_fit_fa10.nii",
                tp / "T1_map_t1_fa_fit_fa10.nii.gz",
                processed / "T1_map_t1_fa_fit_fa10.nii",
                processed / "T1_map_t1_fa_fit_fa10.nii.gz",
            ],
        )
    )
    noise = _find_first(
        filter(
            None,
            [
                _glob_first(anat_dir, "*desc-noise_mask.nii*"),
                tp / "T1_noise_roi.nii",
                tp / "T1_noise_roi.nii.gz",
                processed / "T1_noise_roi.nii",
                processed / "T1_noise_roi.nii.gz",
            ],
        )
    )
    metadata = _find_first(
        filter(
            None,
            [
                _dynamic_sidecar(dynamic),
                _glob_first(source_dce_dir, "*DCE.json"),
                _glob_first(dce_dir, "*DCE.json"),
            ],
        )
    )

    return {
        "dynamic": dynamic,
        "aif": aif,
        "roi": roi,
        "t1map": t1map,
        "noise": noise,
        "metadata_json": metadata,
    }


def _parse_models(raw: str) -> Dict[str, int]:
    mapping = {
        "tofts": "tofts",
        "ex_tofts": "ex_tofts",
        "patlak": "patlak",
        "tissue_uptake": "tissue_uptake",
        "2cxm": "two_cxm",
        "two_cxm": "two_cxm",
        "fxr": "fxr",
        "auc": "auc",
        "nested": "nested",
        "fxl_rr": "FXL_rr",
        "FXL_rr": "FXL_rr",
    }
    flags = {
        "tofts": 0,
        "ex_tofts": 0,
        "patlak": 0,
        "tissue_uptake": 0,
        "two_cxm": 0,
        "fxr": 0,
        "auc": 0,
        "nested": 0,
        "FXL_rr": 0,
    }
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        key = mapping.get(token)
        if key is None:
            raise ValueError(f"Unsupported model '{token}'")
        flags[key] = 1
    if not any(flags.values()):
        flags["tofts"] = 1
    return flags


def _load_stage_defaults(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    return dict(payload.get("stage_overrides", payload.get("defaults", {})))


def _find_bids_part(path: Path, prefix: str) -> Optional[str]:
    for part in path.parts:
        if part.startswith(prefix):
            return part
    return None


def _derive_case_rootname(subject_source: Path, subject_tp: Path, dynamic_file: Optional[Path]) -> Optional[str]:
    entity_order = ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo")
    entities: Dict[str, str] = {}

    for prefix in ("sub-", "ses-"):
        value = _find_bids_part(subject_source, prefix) or _find_bids_part(subject_tp, prefix)
        if value is not None:
            entities[prefix[:-1]] = value

    if dynamic_file is not None:
        stem = dynamic_file.name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]
        for token in stem.split("_"):
            match = re.match(r"^(sub|ses|task|acq|ce|rec|dir|run|echo)-.+$", token)
            if match is not None and match.group(1) not in entities:
                entities[match.group(1)] = token

    root_tokens = [entities[key] for key in entity_order if key in entities]
    if root_tokens:
        return "_".join(root_tokens)
    return None


def _artifact_target_dir(output_dir: Path, artifact: Path) -> Optional[Path]:
    name = artifact.name
    lower_name = name.lower()
    lower_suffix = artifact.suffix.lower()
    if lower_name.endswith(".json") or lower_name.endswith(".jsonl"):
        return output_dir / "reports"
    if lower_suffix == ".png":
        return output_dir / "figures"
    if lower_name.endswith(".nii.gz") or lower_suffix in {".nii", ".npy", ".xls", ".npz"}:
        return output_dir / "dce"
    return None


def _replace_paths(value: Any, path_map: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, dict):
        return {key: _replace_paths(val, path_map) for key, val in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, path_map) for item in value]
    return value


def _rewrite_jsonl_paths(path: Path, path_map: Dict[str, str]) -> None:
    if not path.exists() or not path_map:
        return
    rewritten_lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        rewritten_lines.append(json.dumps(_replace_paths(payload, path_map), default=str))
    path.write_text("\n".join(rewritten_lines) + ("\n" if rewritten_lines else ""), encoding="utf-8")


def _organize_case_outputs(output_dir: Path, result: Dict[str, Any], event_log_path: Path) -> Dict[str, Any]:
    path_map: Dict[str, str] = {}

    for artifact in sorted(output_dir.iterdir()):
        if artifact.is_dir():
            continue
        target_dir = _artifact_target_dir(output_dir, artifact)
        if target_dir is None:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / artifact.name
        if destination == artifact:
            continue
        shutil.move(str(artifact), str(destination))
        path_map[str(artifact)] = str(destination)

    updated_result = _replace_paths(result, path_map)
    summary_path_text = str(updated_result.get("meta", {}).get("summary_path", "")).strip()
    if summary_path_text:
        summary_path = Path(summary_path_text)
        summary_path.write_text(json.dumps(updated_result, indent=2), encoding="utf-8")

    final_event_log = Path(path_map.get(str(event_log_path), str(event_log_path)))
    _rewrite_jsonl_paths(final_event_log, path_map)

    meta = dict(updated_result.get("meta", {}))
    meta["event_log_path"] = str(final_event_log)
    updated_result["meta"] = meta
    return updated_result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-source", type=Path, required=True, help="Raw subject/session path used for DCE metadata lookup.")
    parser.add_argument("--subject-tp", type=Path, required=True, help="Processed subject/session path containing DCE derivative inputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for pipeline artifacts.")
    parser.add_argument(
        "--defaults-json",
        type=Path,
        default=_default_defaults_json_path(),
        help="Path to JSON defaults template (default: python/dceprep_default.json).",
    )
    parser.add_argument("--checkpoint-dir", type=Path, help="Optional checkpoint directory override.")
    parser.add_argument("--backend", choices=["auto", "cpu", "gpufit"], default="auto", help="DCE fitting backend.")
    parser.add_argument("--models", default="patlak", help="Comma-separated model list (default: patlak).")
    parser.add_argument("--dynamic-file", type=Path, help="Optional explicit dynamic file override.")
    parser.add_argument("--aif-file", type=Path, help="Optional explicit AIF mask override.")
    parser.add_argument("--roi-file", type=Path, help="Optional explicit ROI mask override.")
    parser.add_argument("--t1map-file", type=Path, help="Optional explicit T1 map override.")
    parser.add_argument("--noise-file", type=Path, help="Optional explicit noise mask override.")
    parser.add_argument("--metadata-json", type=Path, help="Optional DCE acquisition metadata JSON.")
    parser.add_argument("--tr-ms", type=float, help="Repetition time in milliseconds when no metadata JSON is available.")
    parser.add_argument("--fa-deg", type=float, help="Flip angle in degrees when no metadata JSON is available.")
    parser.add_argument("--time-resolution-sec", type=float, help="DCE frame spacing in seconds when no metadata JSON is available.")
    parser.add_argument("--relaxivity", type=float, help="Contrast-agent relaxivity in /mM/s when no metadata JSON is available.")
    parser.add_argument("--events", choices=["on", "off"], default="on", help="Emit JSON progress events on stdout (default: on).")
    parser.add_argument("--event-log", type=Path, help="Optional JSONL path for event log (default: <output_dir>/reports/dce_pipeline_events.jsonl).")
    parser.add_argument("--no-checkpoints", action="store_true", help="Skip writing checkpoint files.")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[], metavar="KEY=VALUE", help="Override stage_overrides key/value (repeatable).")
    return parser.parse_args(argv)


def _missing_required_inputs(inputs: Dict[str, Optional[Path]]) -> list[str]:
    missing = []
    for key in ("dynamic", "aif", "roi", "t1map"):
        if inputs.get(key) is None:
            missing.append(key)
    return missing


def _missing_acquisition_settings(
    metadata_path: Optional[Path], stage_overrides: Dict[str, Any]
) -> list[str]:
    if metadata_path is not None:
        return []
    return [
        key
        for key in ("tr_ms", "fa_deg", "time_resolution_sec", "relaxivity")
        if stage_overrides.get(key) is None
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    subject_source = args.subject_source.expanduser().resolve()
    subject_tp = args.subject_tp.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    defaults_json = args.defaults_json.expanduser().resolve()
    reports_dir = _default_reports_dir(output_dir)
    checkpoint_dir = None if args.no_checkpoints else (args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else reports_dir / "checkpoints")

    inputs = _resolve_single_case_inputs(subject_source, subject_tp)
    if args.dynamic_file:
        inputs["dynamic"] = args.dynamic_file.expanduser().resolve()
    if args.aif_file:
        inputs["aif"] = args.aif_file.expanduser().resolve()
    if args.roi_file:
        inputs["roi"] = args.roi_file.expanduser().resolve()
    if args.t1map_file:
        inputs["t1map"] = args.t1map_file.expanduser().resolve()
    if args.noise_file:
        inputs["noise"] = args.noise_file.expanduser().resolve()
    if args.metadata_json:
        inputs["metadata_json"] = args.metadata_json.expanduser().resolve()

    missing = _missing_required_inputs(inputs)
    if missing:
        raise FileNotFoundError(
            "Unable to resolve required DCE inputs for single-case CLI. "
            f"Missing: {', '.join(missing)}. "
            "Pass explicit --dynamic-file/--aif-file/--roi-file/--t1map-file overrides if your layout is nonstandard."
        )

    stage_overrides: Dict[str, Any] = _load_stage_defaults(defaults_json)
    stage_overrides.update({
        "stage_a_mode": "real",
        "stage_b_mode": "real",
        "stage_d_mode": "real",
        "write_param_maps": True,
    })
    derived_rootname = _derive_case_rootname(subject_source, subject_tp, inputs["dynamic"])
    if derived_rootname is not None:
        stage_overrides["rootname"] = derived_rootname
    if inputs["metadata_json"] is not None:
        stage_overrides["dce_metadata_path"] = str(inputs["metadata_json"])
    stage_overrides.update(parse_set_overrides(args.set_overrides))
    for key in ("tr_ms", "fa_deg", "time_resolution_sec", "relaxivity"):
        value = getattr(args, key)
        if value is not None:
            stage_overrides[key] = value

    missing_settings = _missing_acquisition_settings(inputs["metadata_json"], stage_overrides)
    if missing_settings:
        raise ValueError(
            "DCE acquisition metadata was not found. Pass --metadata-json or provide all "
            f"of: {', '.join(missing_settings)}."
        )

    config = DcePipelineConfig(
        subject_source_path=subject_source,
        subject_tp_path=subject_tp,
        output_dir=output_dir,
        backend=args.backend,
        checkpoint_dir=checkpoint_dir,
        write_xls=True,
        aif_mode="fitted",
        dynamic_files=[inputs["dynamic"]],
        aif_files=[inputs["aif"]],
        roi_files=[inputs["roi"]],
        t1map_files=[inputs["t1map"]],
        noise_files=([inputs["noise"]] if inputs["noise"] is not None else []),
        drift_files=[],
        model_flags=_parse_models(args.models),
        stage_overrides=stage_overrides,
    )

    event_log_path = args.event_log.expanduser().resolve() if args.event_log else (reports_dir / "dce_pipeline_events.jsonl")
    sinks = [_stdout_event_sink] if args.events == "on" else []
    event_log_handle, emit_event = _build_event_logger(event_log_path, sinks)
    try:
        emit_event(
            {
                "type": "cli_config",
                "mode": "single-case",
                "subject_source_path": str(subject_source),
                "subject_tp_path": str(subject_tp),
                "resolved_output_dir": str(output_dir),
                "defaults_json_path": str(defaults_json),
                "event_log_path": str(event_log_path),
                "resolved_inputs": {key: (str(value) if value is not None else None) for key, value in inputs.items()},
                "models": args.models,
                "backend": args.backend,
                "stage_overrides": stage_overrides,
            }
        )
        result = run_dce_pipeline(config, event_callback=emit_event)
    finally:
        event_log_handle.close()

    result = _organize_case_outputs(output_dir, result, event_log_path)

    meta = dict(result["meta"])
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())