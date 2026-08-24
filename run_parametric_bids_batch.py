#!/usr/bin/env python3
"""Batch processor for parametric T1 mapping across BIDS datasets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import BidsSession, discover_bids_sessions  # noqa: E402
import run_reporting  # noqa: E402
from run_reporting import Reporter, Verbosity  # noqa: E402
from cli_overrides import parse_set_overrides  # noqa: E402
from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline  # noqa: E402


def _load_config_template(path: Path) -> Dict[str, Any]:
    """Load JSON config template."""
    if not path.exists():
        raise FileNotFoundError(f"Config template not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process parametric T1 VFA fitting across BIDS dataset sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sessions using derivatives/{pipeline}/... BIDS layout
  %(prog)s --bids-root /data/my_study --pipeline-folder t1prep

  # Read from one derivatives pipeline and write to another
  %(prog)s --bids-root /data/my_study --pipeline-folder t1prep --output-pipeline t1maps

  # Use a config template and override specific keys
  %(prog)s --bids-root /data/my_study --pipeline-folder t1prep \\
    --config-template python/parametric_default.json \\
    --set tr_ms=5.0 --set rsquared_threshold=0.7

  # Run only selected subjects/sessions
  %(prog)s --bids-root /data/my_study --pipeline-folder t1prep \\
    --subject sub-01 --session ses-01
        """,
    )
    parser.add_argument("--bids-root", type=Path, required=True, help="BIDS root directory (must contain rawdata/ and derivatives/).")
    parser.add_argument("--pipeline-folder", type=str, default=None,
        help=(
            "Pipeline folder name within derivatives for session discovery and default outputs "
            "(e.g., 't1prep')."
        ),
    )
    parser.add_argument("--output-pipeline", type=str, default=None,
        help=(
            "Optional different pipeline folder for outputs only. "
            "Requires --pipeline-folder."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Custom flat output directory (ignores BIDS structure). Mutually exclusive with --pipeline-folder.")
    parser.add_argument("--config-template", type=Path, default=None, help="Optional JSON config template for base settings (e.g., python/parametric_default.json).")
    parser.add_argument("--fit-type", choices=["t1_fa_fit", "t1_fa_linear_fit", "t1_fa_two_point_fit"], default="t1_fa_fit", help="Parametric fit type (default: t1_fa_fit).")
    parser.add_argument("--subject", action="append", dest="subjects", help="Process only specific subject(s) (e.g., sub-01). Repeatable.")
    parser.add_argument("--session", action="append", dest="sessions", help="Process only specific session(s) (e.g., ses-01). Repeatable.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip input file validation checks before run.")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[], metavar="KEY=VALUE", help="Override top-level config key/value (repeatable). Example: --set tr_ms=5.0")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional path to write batch summary JSON (default: derivatives/reports/[pipeline]/batch_summary_YYYYMMDD.json).")
    run_reporting.add_verbosity_argument(parser)
    return parser.parse_args(argv)


def _find_one(parent: Path, pattern: str) -> Optional[Path]:
    if not parent.is_dir():
        return None
    matches = sorted(parent.glob(pattern))
    return matches[0] if matches else None


def _discover_parametric_inputs(session: BidsSession) -> Dict[str, Any]:
    """Discover VFA and optional B1/mask inputs for a session."""
    raw_anat = session.rawdata_path / "anat"
    deriv_anat = session.derivatives_path / "anat"

    def _preprocessed_unified_vfa() -> Optional[Path]:
        # Prefer preprocessed unified stack used by MATLAB parity workflows.
        return _find_one(deriv_anat, "*space-DCEref_desc-bfczunified_VFA.nii*")

    def _canonical_dceref_vfa_files() -> List[Path]:
        # Keep only canonical DCEref VFA files: ..._flip-XX_space-DCEref_VFA.nii[.gz]
        # Excludes derivative variants such as desc-bfc/desc-brain/RAS.
        pattern = re.compile(r"_flip-(\d+)_space-DCEref_VFA\.nii(?:\.gz)?$", re.IGNORECASE)
        flip_to_path: Dict[int, Path] = {}
        for candidate in sorted(deriv_anat.glob("*space-DCEref*_VFA.nii*")):
            match = pattern.search(candidate.name)
            if not match:
                continue
            flip_idx = int(match.group(1))
            flip_to_path[flip_idx] = candidate
        return [flip_to_path[key] for key in sorted(flip_to_path)]

    unified_vfa = _preprocessed_unified_vfa()
    if unified_vfa is not None:
        vfa_files = [unified_vfa]
    else:
        # Fall back to canonical registration outputs in DCE reference space.
        vfa_files = _canonical_dceref_vfa_files()
    if not vfa_files:
        vfa_files = sorted(raw_anat.glob("*flip-*_VFA.nii*"))
    if not vfa_files:
        vfa_files = sorted(raw_anat.glob("*_VFA.nii*"))

    if not vfa_files:
        raise FileNotFoundError(
            f"Missing VFA inputs for {session.id}: expected DCEref VFA under {deriv_anat} or raw VFA under {raw_anat}"
        )

    raw_sidecars = sorted(raw_anat.glob("*flip-*_VFA.json"))
    if not raw_sidecars:
        raw_sidecars = sorted(raw_anat.glob("*_VFA.json"))

    b1_map = _find_one(deriv_anat, "B1_scaled_FAreg.nii*")
    if b1_map is None:
        b1_map = _find_one(raw_anat, "B1_scaled_FAreg.nii*")

    mask_file = _find_one(deriv_anat, "*label-brain_mask.nii*")

    return {
        "vfa_files": [str(path) for path in vfa_files],
        "b1_map_file": str(b1_map) if b1_map else None,
        "mask_file": str(mask_file) if mask_file else None,
        "raw_sidecars": [str(path) for path in raw_sidecars],
    }


def _load_flip_angles_and_tr_from_sidecars(sidecar_paths: List[str]) -> tuple[List[float], Optional[float]]:
    if not sidecar_paths:
        return [], None

    flip_angles: List[float] = []
    tr_values_ms: List[float] = []
    for path_str in sidecar_paths:
        payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
        if "FlipAngle" in payload:
            flip_angles.append(float(payload["FlipAngle"]))
        if "RepetitionTime" in payload:
            tr_values_ms.append(float(payload["RepetitionTime"]) * 1000.0)

    flip_angles = sorted(flip_angles)
    if not tr_values_ms:
        return flip_angles, None

    tr_ms = tr_values_ms[0]
    for value in tr_values_ms[1:]:
        if abs(float(value) - float(tr_ms)) > 1e-9:
            raise ValueError("Inconsistent RepetitionTime across raw VFA sidecars")
    return flip_angles, float(tr_ms)


def _resolve_session_paths(session: BidsSession, raw_pattern: str) -> List[Path]:
    pattern = str(raw_pattern)
    if not pattern:
        return []

    path_value = Path(pattern)
    if path_value.is_absolute():
        if "*" in pattern or "?" in pattern:
            return sorted(path_value.parent.glob(path_value.name))
        return [path_value] if path_value.exists() else []

    resolved: List[Path] = []
    for root in (session.rawdata_path, session.derivatives_path, session.bids_root):
        if "*" in pattern or "?" in pattern:
            resolved.extend(sorted(root.glob(pattern)))
        else:
            candidate = root / pattern
            if candidate.exists():
                resolved.append(candidate)

    unique: List[Path] = []
    seen = set()
    for candidate in resolved:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _session_problem(session: BidsSession, skip_validation: bool) -> Optional[str]:
    """Describe why a session cannot run, or None when it is ready.

    Uses the same discovery the run loop uses, so the preflight cannot disagree with it.
    """
    if skip_validation:
        return None
    try:
        _discover_parametric_inputs(session)
    except Exception as exc:
        return str(exc).split(":", 1)[0] if ":" in str(exc) else str(exc)
    return None


def _build_session_config(
    session: BidsSession,
    output_dir: Path,
    fit_type: str,
    config_template: Optional[Dict[str, Any]],
    set_overrides: Dict[str, Any],
    template_dir: Optional[Path] = None,
) -> ParametricT1Config:
    """Build parametric config for a single session.

    Precedence (lowest to highest):
    1. Hardcoded defaults
    2. Config template JSON (if provided)
    3. Auto-discovered BIDS inputs (when template did not resolve explicit files)
    4. CLI --set overrides
    5. CLI --fit-type
    """
    discovered = _discover_parametric_inputs(session)

    payload: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "fit_type": "t1_fa_fit",
        "output_basename": "T1_map",
        "output_label": session.id,
        "rsquared_threshold": 0.6,
        "write_r_squared": True,
        "write_rho_map": False,
        "invalid_fill_value": -1.0,
        "odd_echoes": False,
        "xy_smooth_sigma": 0.0,
        "vfa_files": discovered["vfa_files"],
        "b1_map_file": discovered["b1_map_file"],
    }

    discovered_flip_angles, discovered_tr_ms = _load_flip_angles_and_tr_from_sidecars(discovered["raw_sidecars"])
    if discovered_flip_angles:
        payload["flip_angles_deg"] = discovered_flip_angles
    if discovered_tr_ms is not None:
        payload["tr_ms"] = discovered_tr_ms

    if config_template:
        for key, value in config_template.items():
            payload[key] = value

        if config_template.get("vfa_files"):
            raw_values = config_template["vfa_files"]
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            resolved: List[str] = []
            for raw in values:
                resolved.extend([str(path.resolve()) for path in _resolve_session_paths(session, str(raw))])
            if resolved:
                payload["vfa_files"] = resolved
            else:
                run_reporting.notice(
                    "config template vfa_files did not match this session; falling back to auto-discovered VFA files",
                    Verbosity.NORMAL,
                )
                payload["vfa_files"] = discovered["vfa_files"]

        if config_template.get("b1_map_file"):
            resolved_b1 = _resolve_session_paths(session, str(config_template["b1_map_file"]))
            if resolved_b1:
                payload["b1_map_file"] = str(resolved_b1[0].resolve())
            elif discovered["b1_map_file"]:
                run_reporting.notice(
                    "config template b1_map_file did not match this session; falling back to auto-discovered B1 map",
                    Verbosity.NORMAL,
                )
                payload["b1_map_file"] = discovered["b1_map_file"]
            else:
                payload["b1_map_file"] = None

    if payload.get("mask_file"):
        resolved_mask = _resolve_session_paths(session, str(payload["mask_file"]))
        if resolved_mask:
            payload["mask_file"] = str(resolved_mask[0].resolve())
        else:
            payload["mask_file"] = None

    if payload.get("b1_map_file") is None and discovered["b1_map_file"]:
        payload["b1_map_file"] = discovered["b1_map_file"]

    payload.update(set_overrides)
    payload["fit_type"] = fit_type

    # Session inputs are already absolute -- they come from discovery or from the per-session
    # glob resolution above. The anchor is for whatever the template supplied verbatim.
    return ParametricT1Config.from_dict(payload, base_dir=template_dir)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    verbosity = run_reporting.verbosity_from_args(args)
    reporter = Reporter(verbosity=verbosity)
    run_reporting.set_notice_sink(run_reporting.reporter_notice_sink(reporter))
    try:
        return _run_batch(args, reporter, verbosity)
    finally:
        run_reporting.set_notice_sink(None)


def _run_batch(args: argparse.Namespace, reporter: Reporter, verbosity: Verbosity) -> int:

    bids_root = args.bids_root.expanduser().resolve()
    if not bids_root.is_dir():
        print(f"Error: BIDS root not found: {bids_root}", file=sys.stderr)
        return 2

    if args.output_root and args.pipeline_folder:
        print("Error: --output-root and --pipeline-folder are mutually exclusive", file=sys.stderr)
        return 2

    if args.output_pipeline and not args.pipeline_folder:
        print("Error: --output-pipeline requires --pipeline-folder", file=sys.stderr)
        return 2

    if args.pipeline_folder:
        input_pipeline = args.pipeline_folder
        output_pipeline = args.output_pipeline or args.pipeline_folder
        output_root = bids_root / "derivatives" / output_pipeline
        use_bids_structure = True
    elif args.output_root:
        input_pipeline = None
        output_pipeline = None
        output_root = args.output_root.expanduser().resolve()
        use_bids_structure = False
    else:
        input_pipeline = None
        output_pipeline = None
        output_root = bids_root / "derivatives" / "parametric_batch_output"
        use_bids_structure = False

    output_root.mkdir(parents=True, exist_ok=True)

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

    config_template: Optional[Dict[str, Any]] = None
    template_dir: Optional[Path] = None
    if args.config_template:
        template_path = args.config_template.expanduser().resolve()
        template_dir = template_path.parent
        try:
            config_template = _load_config_template(template_path)
            reporter.note(f"config template  {run_reporting.shorten_path(template_path)}")
        except Exception as exc:
            print(f"Error loading config template: {exc}", file=sys.stderr)
            return 2

    try:
        set_overrides = parse_set_overrides(args.set_overrides)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        sessions = discover_bids_sessions(bids_root, pipeline_folder=input_pipeline)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.subjects:
        subjects_set = set(args.subjects)
        sessions = [s for s in sessions if s.subject in subjects_set]

    if args.sessions:
        sessions_set = set(args.sessions)
        sessions = [s for s in sessions if s.session in sessions_set]

    if not sessions:
        print("Error: No sessions matched filter criteria", file=sys.stderr)
        return 2

    reporter.header(
        "ROCKETSHIP parametric T1 batch",
        [
            ("Dataset", run_reporting.shorten_path(bids_root)),
            ("Input", input_pipeline or "rawdata"),
            ("Output", run_reporting.shorten_path(output_root)),
            ("Fit", args.fit_type),
        ],
    )
    run_reporting.report_queue(
        reporter,
        [(session.id, _session_problem(session, args.skip_validation)) for session in sessions],
        checked=not args.skip_validation,
    )

    session_results: List[Dict[str, Any]] = []
    failed_count = 0
    batch_started = time.perf_counter()

    for idx, session in enumerate(sessions, 1):
        if use_bids_structure:
            if session.session:
                session_output_dir = output_root / session.subject / session.session / "anat"
                session_reports_dir = output_root / session.subject / session.session / "reports"
            else:
                session_output_dir = output_root / session.subject / "anat"
                session_reports_dir = output_root / session.subject / "reports"
        else:
            session_output_dir = output_root / session.id / "anat"
            session_reports_dir = output_root / session.id / "reports"

        session_output_dir.mkdir(parents=True, exist_ok=True)
        session_reports_dir.mkdir(parents=True, exist_ok=True)

        reporter.start(f"[{idx}/{len(sessions)}] {session.id}")

        try:
            if not args.skip_validation:
                _discover_parametric_inputs(session)

            config = _build_session_config(
                session=session,
                output_dir=session_output_dir,
                fit_type=args.fit_type,
                config_template=config_template,
                set_overrides=set_overrides,
                template_dir=template_dir,
            )
            config.validate()

            session_reporter = (
                Reporter(verbosity=verbosity, nested=True)
                if verbosity >= Verbosity.DETAILED
                else None
            )
            if session_reporter is not None:
                run_reporting.set_notice_sink(
                    run_reporting.reporter_notice_sink(session_reporter)
                )
            try:
                # Only hand over a callback when something will render it, so a plain
                # batch run calls the pipeline exactly as it did before.
                result = (
                    run_parametric_t1_pipeline(config, event_callback=session_reporter.handle_event)
                    if session_reporter is not None
                    else run_parametric_t1_pipeline(config)
                )
            finally:
                run_reporting.set_notice_sink(run_reporting.reporter_notice_sink(reporter))

            pipeline_run_source = session_output_dir / "parametric_t1_run.json"
            if pipeline_run_source.exists():
                pipeline_run_dest = session_reports_dir / f"{session.id}_desc-parametricT1-provenance.json"
                pipeline_run_source.rename(pipeline_run_dest)
                result.setdefault("meta", {})["summary_path"] = str(pipeline_run_dest)

            status = "success" if result.get("meta", {}).get("status") == "ok" else "partial"
            session_results.append(
                {
                    "session_id": session.id,
                    "status": status,
                    "output_dir": str(session_output_dir),
                    "notes": result.get("meta", {}).get("notes", ""),
                }
            )
            reporter.settle(status=status, ok=(status == "success"))

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
            reporter.settle(status="failed", ok=False)
            reporter.warn(f"{type(exc).__name__}: {exc}")

    summary = {
        "metadata": {
            "bids_root": str(bids_root),
            "input_pipeline": input_pipeline or "rawdata",
            "output_pipeline": output_pipeline if use_bids_structure else None,
            "output_root": str(output_root),
            "output_structure": "bids_derivatives" if use_bids_structure else "flat",
            "fit_type": args.fit_type,
        },
        "sessions_total": len(sessions),
        "sessions_success": sum(1 for r in session_results if r["status"] == "success"),
        "sessions_partial": sum(1 for r in session_results if r["status"] == "partial"),
        "sessions_failed": failed_count,
        "results": session_results,
    }

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_reporting.report_batch_finish(
        reporter,
        time.perf_counter() - batch_started,
        [
            ("succeeded", summary["sessions_success"]),
            ("partial", summary["sessions_partial"]),
            ("failed", summary["sessions_failed"]),
        ],
        [("Summary", run_reporting.shorten_path(summary_json_path))],
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
