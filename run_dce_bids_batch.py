#!/usr/bin/env python3
"""Batch processor for DCE pipeline across BIDS datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import BidsSession, discover_bids_sessions  # noqa: E402
from dce_file_discovery import discover_dce_inputs  # noqa: E402
from dce_pipeline import DcePipelineConfig, run_dce_pipeline  # noqa: E402


def _load_config_template(path: Path) -> Dict[str, Any]:
    """Load JSON config template."""
    if not path.exists():
        raise FileNotFoundError(f"Config template not found: {path}")
    return json.loads(path.read_text())


def _parse_set_overrides(values: list[str]) -> Dict[str, Any]:
    """Parse KEY=VALUE overrides."""
    overrides: Dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --set entry '{raw}'. Expected KEY=VALUE")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set entry '{raw}'. Empty KEY")
        overrides[key] = value.strip()
    return overrides


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process DCE pipeline across BIDS dataset sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sessions with BIDS-compliant output structure
  %(prog)s --bids-root /data/my_study --pipeline-folder dceprep

  # Use config template with custom overrides
  %(prog)s --bids-root /data/my_study --pipeline-folder dceprep \\
    --config-template python/dce_default.json \\
    --set blood_t1_ms=1600 \\
    --set aif_curve_mode=fitted

  # Process with specific backend and models
  %(prog)s --bids-root /data/my_study --pipeline-folder dceprep \\
    --backend gpufit \\
    --dce-models tofts,ex_tofts,patlak

  # Read from one pipeline, write to another
  %(prog)s --bids-root /data/my_study \\
    --pipeline-folder dceprep \\
    --output-pipeline dce-postproc

  # Process only specific subjects/sessions
  %(prog)s --bids-root /data/my_study --pipeline-folder dceprep \\
    --subject sub-01 sub-02 \\
    --session ses-baseline

  # Use custom flat output directory instead of BIDS structure
  %(prog)s --bids-root /data/my_study \\
    --output-root /scratch/dce_results
        """,
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        required=True,
        help="BIDS root directory (must contain rawdata/ and derivatives/).",
    )
    parser.add_argument(
        "--pipeline-folder",
        type=str,
        default=None,
        help="Pipeline folder name within derivatives for inputs AND outputs (e.g., 'dceprep'). Creates bids_root/derivatives/{pipeline-folder}/ with BIDS subject/session structure.",
    )
    parser.add_argument(
        "--output-pipeline",
        type=str,
        default=None,
        help="Optional different pipeline folder for outputs only (e.g., read from 'dceprep' but write to 'dce-analysis'). If omitted, outputs go to same location as --pipeline-folder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Custom flat output directory (ignores BIDS structure). Mutually exclusive with --pipeline-folder.",
    )
    parser.add_argument(
        "--config-template",
        type=Path,
        default=None,
        help="Optional JSON config template for base settings (e.g., dce_default.json). Auto-discovered inputs and CLI args override template values.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "gpufit"],
        default="auto",
        help="DCE fitting backend.",
    )
    parser.add_argument(
        "--dce-models",
        type=str,
        default=None,
        help="Comma-separated DCE models to run (overrides config file). If omitted, uses config file settings.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        help="Process only specific subject(s) (e.g., sub-01, sub-02). Repeatable.",
    )
    parser.add_argument(
        "--session",
        action="append",
        dest="sessions",
        help="Process only specific session(s) (e.g., ses-baseline). Repeatable.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip input file validation checks.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Skip writing checkpoint files (stage A/B/D .npz files) for faster processing.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override stage_overrides key/value (repeatable). Example: --set blood_t1_ms=1600",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write batch summary JSON (default: derivatives/reports/[pipeline]/batch_summary_YYYYMMDD.json).",
    )
    return parser.parse_args(argv)


def _build_session_config(
    session: BidsSession,
    output_dir: Path,
    backend: str,
    models: List[str],
    config_template: Optional[Dict[str, Any]],
    set_overrides: Dict[str, Any],
    enable_checkpoints: bool = True,
) -> DcePipelineConfig:
    """Build DCE config for a single session.
    
    Precedence (lowest to highest):
    1. Hardcoded defaults
    2. Config template JSON (if provided) - EXPLICIT USER INTENT
    3. Auto-discovered BIDS inputs (only if template didn't specify files)
    4. CLI --set overrides
    5. CLI --backend and --dce-models
    
    IMPORTANT: Config template file paths take precedence over auto-discovery.
    """
    from dce_file_discovery import discover_dce_inputs

    inputs = discover_dce_inputs(session)
    
    # Start with hardcoded sensible defaults
    base_config = {
        "subject_source_path": str(session.rawdata_path),
        "subject_tp_path": str(session.derivatives_path),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoints") if enable_checkpoints else None,
        "backend": backend,
        "write_xls": True,
        "aif_mode": "auto",
        "dynamic_files": [],
        "aif_files": [],
        "roi_files": [],
        "t1map_files": [],
        "noise_files": [],
        "drift_files": [],
        "model_flags": {},
        "stage_overrides": {
            "stage_a_mode": "real",
            "stage_b_mode": "real",
            "stage_d_mode": "real",
            "aif_curve_mode": "raw",
            "write_param_maps": True,
        },
    }
    
    # Merge config template if provided
    if config_template:
        # Preserve base paths and output settings, merge the rest
        for key in ["aif_mode", "write_xls", "drift_files", "model_flags"]:
            if key in config_template:
                base_config[key] = config_template[key]
        
        # Merge stage_overrides from template
        if "stage_overrides" in config_template:
            base_config["stage_overrides"].update(config_template["stage_overrides"])

    # Injection timing should default to automatic Stage-A/B detection unless
    # explicitly provided via CLI --set. Strip template hard-codes here.
    injection_override_keys = {
        "start_injection",
        "end_injection",
        "start_injection_min",
        "end_injection_min",
    }
    explicit_injection_override = any(
        str(key).strip().lower() in injection_override_keys for key in set_overrides
    )
    if not explicit_injection_override:
        for key in list(base_config["stage_overrides"].keys()):
            if str(key).strip().lower() in injection_override_keys:
                base_config["stage_overrides"].pop(key, None)
    
    # Helper to resolve file lists from template patterns and optional auto-discovery fallback.
    def _resolve_file_list(
        key: str,
        auto_discovered_path: Optional[Path],
        *,
        required: bool,
    ) -> List[str]:
        if config_template and key in config_template and config_template[key]:
            # Template explicitly specifies this file list
            patterns = config_template[key]
            if not isinstance(patterns, list):
                patterns = [patterns]
            
            resolved_paths = []
            for pattern in patterns:
                pattern_str = str(pattern)
                
                # If pattern contains wildcards, resolve as glob relative to session derivatives
                if '*' in pattern_str or '?' in pattern_str:
                    matches = list(session.derivatives_path.glob(pattern_str))
                    if matches:
                        resolved_paths.extend([str(p) for p in sorted(matches)])
                    # If glob finds nothing, continue to auto-discovery below
                else:
                    # Literal path - resolve relative to session derivatives
                    full_path = session.derivatives_path / pattern_str
                    if full_path.exists():
                        resolved_paths.append(str(full_path))
            
            # If template patterns resolved to files, use them
            if resolved_paths:
                return resolved_paths

            # Template had patterns but they didn't match.
            if required:
                print(
                    f"Warning: Config template patterns for {key} didn't match any files. Falling back to auto-discovery.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Warning: Config template patterns for optional {key} didn't match any files. Leaving empty.",
                    file=sys.stderr,
                )
                return []

        # No template (or no match): use auto-discovered path when available.
        if auto_discovered_path is not None:
            return [str(auto_discovered_path)]

        # Optional paths can remain empty.
        if not required:
            return []

        # Required path with no fallback should fail loudly.
        raise FileNotFoundError(f"Missing required input for {key} and no auto-discovered fallback is available")
    
    # Resolve file lists with template taking precedence
    base_config["dynamic_files"] = _resolve_file_list("dynamic_files", inputs.dynamic, required=True)
    base_config["aif_files"] = _resolve_file_list("aif_files", inputs.aif_mask, required=True)
    base_config["roi_files"] = _resolve_file_list("roi_files", inputs.roi_mask, required=True)
    base_config["t1map_files"] = _resolve_file_list("t1map_files", inputs.t1_map, required=True)
    base_config["noise_files"] = _resolve_file_list("noise_files", inputs.noise_mask, required=False)
    
    # Set rootname from session ID for BIDS-compliant output naming (unless already set in template)
    rootname = f"{session.subject}"
    if session.session:
        rootname += f"_{session.session}"
    if not config_template or "rootname" not in config_template or config_template.get("rootname") == "Dyn-1":
        base_config["stage_overrides"]["rootname"] = rootname
    
    # Prefer per-session metadata JSON for timing by default.
    # Template timing values are often study-level defaults and can silently
    # mask sidecar-specific values; only keep explicit CLI --set timing overrides.
    tr_fa_override_keys = {
        "tr",
        "tr_ms",
        "tr_sec",
        "fa",
        "fa_deg",
    }
    metadata_candidates: List[Path] = []
    if inputs.metadata_json is not None:
        metadata_candidates.append(inputs.metadata_json)
    metadata_candidates.extend(sorted((session.rawdata_path / "dce").glob("*DCE.json")))

    explicit_metadata_override = any(str(key).lower() == "dce_metadata_path" for key in set_overrides)

    if metadata_candidates:
        for key in tr_fa_override_keys:
            if key in base_config["stage_overrides"] and key not in set_overrides:
                base_config["stage_overrides"].pop(key, None)
        # In batch mode, always bind metadata to the current session sidecar unless
        # user explicitly set dce_metadata_path via --set.
        if not explicit_metadata_override:
            base_config["stage_overrides"]["dce_metadata_path"] = str(metadata_candidates[0])
    elif not explicit_metadata_override:
        # Prevent template test-fixture metadata paths from leaking into real-data runs.
        base_config["stage_overrides"].pop("dce_metadata_path", None)

    # Apply --set overrides to stage_overrides
    if set_overrides:
        base_config["stage_overrides"].update(set_overrides)
    
    # Apply CLI --backend (always override)
    base_config["backend"] = backend
    
    # Apply CLI --dce-models only if explicitly provided (highest priority), otherwise keep config file's model_flags
    if models:  # Only set if --dce-models was explicitly provided
        model_flags = {
            "tofts": int("tofts" in models),
            "ex_tofts": int("ex_tofts" in models),
            "patlak": int("patlak" in models),
            "tissue_uptake": int("tissue_uptake" in models),
            "two_cxm": int("2cxm" in models or "two_cxm" in models),
            "fxr": int("fxr" in models),
            "auc": int("auc" in models),
            "nested": int("nested" in models),
            "FXL_rr": int("FXL_rr" in models),
        }
        base_config["model_flags"] = model_flags
    
    return DcePipelineConfig.from_dict(base_config)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    bids_root = args.bids_root.expanduser().resolve()
    if not bids_root.is_dir():
        print(f"Error: BIDS root not found: {bids_root}", file=sys.stderr)
        return 2

    # Validate mutually exclusive options
    if args.output_root and args.pipeline_folder:
        print("Error: --output-root and --pipeline-folder are mutually exclusive", file=sys.stderr)
        return 2
    
    if args.output_pipeline and not args.pipeline_folder:
        print("Error: --output-pipeline requires --pipeline-folder", file=sys.stderr)
        return 2

    # Determine input and output strategy
    if args.pipeline_folder:
        # BIDS-compliant workflow
        input_pipeline = args.pipeline_folder
        output_pipeline = args.output_pipeline or args.pipeline_folder
        output_root = bids_root / "derivatives" / output_pipeline
        use_bids_structure = True
    elif args.output_root:
        # User-specified flat output root
        input_pipeline = None  # Will look directly under derivatives/
        output_root = args.output_root.expanduser().resolve()
        use_bids_structure = False
    else:
        # Default: flat structure, no specific pipeline
        input_pipeline = None
        output_root = bids_root / "derivatives" / "dce_batch_output"
        use_bids_structure = False
    
    output_root.mkdir(parents=True, exist_ok=True)

    # Batch summary goes to derivatives/reports/[pipeline]/ with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    if args.summary_json:
        summary_json_path = args.summary_json.expanduser().resolve()
    else:
        reports_dir = bids_root / "derivatives" / "reports"
        if use_bids_structure and output_pipeline:
            reports_dir = reports_dir / output_pipeline
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_json_path = reports_dir / f"batch_summary_{timestamp}.json"

    # Load config template if provided
    config_template: Optional[Dict[str, Any]] = None
    if args.config_template:
        template_path = args.config_template.expanduser().resolve()
        try:
            config_template = _load_config_template(template_path)
            print(f"Loaded config template: {template_path}", flush=True)
        except Exception as exc:
            print(f"Error loading config template: {exc}", file=sys.stderr)
            return 2
    
    # Parse models and overrides (models is empty list if --dce-models not provided)
    models = [m.strip().lower() for m in args.dce_models.split(",") if m.strip()] if args.dce_models else []
    try:
        set_overrides = _parse_set_overrides(args.set_overrides)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # Discover sessions
    try:
        sessions = discover_bids_sessions(bids_root, pipeline_folder=input_pipeline)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # Filter by subject/session if specified
    if args.subjects:
        subjects_set = set(args.subjects)
        sessions = [s for s in sessions if s.subject in subjects_set]

    if args.sessions:
        sessions_set = set(args.sessions)
        sessions = [s for s in sessions if s.session in sessions_set]

    if not sessions:
        print("Error: No sessions matched filter criteria", file=sys.stderr)
        return 2

    print(f"Processing {len(sessions)} session(s) under {bids_root}", flush=True)

    # Run pipeline for each session
    session_results: List[Dict[str, Any]] = []
    failed_count = 0

    for idx, session in enumerate(sessions, 1):
        # Determine session output directory based on output strategy
        if use_bids_structure:
            # BIDS derivatives structure: derivatives/{pipeline}/sub-X/[ses-Y/]dce/
            if session.session:
                session_output_dir = output_root / session.subject / session.session / "dce"
                session_reports_dir = output_root / session.subject / session.session / "reports"
            else:
                session_output_dir = output_root / session.subject / "dce"
                session_reports_dir = output_root / session.subject / "reports"
        else:
            # Flat structure: output_root/sub-X_ses-Y/dce/
            session_output_dir = output_root / session.id / "dce"
            session_reports_dir = output_root / session.id / "reports"
        
        session_output_dir.mkdir(parents=True, exist_ok=True)
        session_reports_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(sessions)}] {session.id}...", flush=True)

        try:
            # Validate inputs before config
            if not args.skip_validation:
                from dce_file_discovery import discover_dce_inputs

                discover_dce_inputs(session)

            # Build and validate config
            config = _build_session_config(
                session,
                session_output_dir,
                args.backend,
                models,
                config_template,
                set_overrides,
                enable_checkpoints=not args.no_checkpoints,
            )
            config.validate()

            # Run pipeline
            result = run_dce_pipeline(config)
            
            # Move dce_pipeline_run.json to reports folder with BIDS naming
            pipeline_run_source = session_output_dir / "dce_pipeline_run.json"
            if pipeline_run_source.exists():
                pipeline_run_dest = session_reports_dir / f"{session.id}_desc-provenance.json"
                pipeline_run_source.rename(pipeline_run_dest)
                result["meta"]["summary_path"] = str(pipeline_run_dest)
            status = "success" if result.get("meta", {}).get("status") == "ok" else "partial"
            session_results.append(
                {
                    "session_id": session.id,
                    "status": status,
                    "output_dir": str(session_output_dir),
                    "notes": result.get("meta", {}).get("notes", ""),
                }
            )
            print(f"  ✓ {status}", flush=True)

        except Exception as exc:
            failed_count += 1
            session_results.append(
                {
                    "session_id": session.id,
                    "status": "failed",
                    "output_dir": str(session_output_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  ✗ failed: {exc}", flush=True)

    # Write summary
    summary = {
        "metadata": {
            "bids_root": str(bids_root),
            "input_pipeline": input_pipeline or "rawdata",
            "output_pipeline": output_pipeline if use_bids_structure else None,
            "output_root": str(output_root),
            "output_structure": "bids_derivatives" if use_bids_structure else "flat",
            "backend": args.backend,
            "dce_models": models,
        },
        "sessions_total": len(sessions),
        "sessions_success": sum(1 for r in session_results if r["status"] == "success"),
        "sessions_partial": sum(1 for r in session_results if r["status"] == "partial"),
        "sessions_failed": failed_count,
        "results": session_results,
    }

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to: {summary_json_path}", flush=True)
    print(
        f"  success={summary['sessions_success']}, "
        f"partial={summary['sessions_partial']}, "
        f"failed={summary['sessions_failed']}",
        flush=True,
    )

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
