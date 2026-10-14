"""CLI entrypoint for the in-memory DCE A->B->D pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, IO, Optional

import dce_config
import run_reporting
from banner import print_banner
from cli_overrides import parse_set_overrides
from dce_pipeline import DcePipelineConfig, run_dce_pipeline
from run_reporting import Reporter, Verbosity
import version


EVENT_PREFIX = "ROCKETSHIP_EVENT "


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "dce_run_example_bids.json"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text())


EXAMPLES_EPILOG = """\
Example run configs, one per data layout -- copy one and edit it:

  python/dce_run_example_bids.json     data in BIDS layout (this is the default config)
  python/dce_run_example_nonbids.json  data in any other layout, e.g. a flat folder

BIDS is not required. The BIDS example names two session folders and no files: images and
masks are found by the dceprep naming convention, and TR, flip angle, temporal resolution
and relaxivity are read from the acquisition sidecar. The non-BIDS example names every
file outright and gives those values under stage_overrides. Naming a file always wins over
the convention. Full reference: docs/dce_options.md.
"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EXAMPLES_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to JSON pipeline config (default: python/dce_run_example_bids.json)")
    parser.add_argument("--output-dir", type=Path, help="Optional override for output_dir in config")
    parser.add_argument("--checkpoint-dir", type=Path, help="Optional override for checkpoint_dir in config")
    parser.add_argument("--backend", choices=["auto", "cpu", "gpufit"], help="Optional override for backend in config")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[], metavar="KEY=VALUE", help="Override stage_overrides key/value (repeatable)")
    parser.add_argument(
        "--events",
        choices=["on", "off"],
        default="off",
        help=(
            "Put the machine-readable JSON event stream on stdout instead of human-readable "
            "progress; the GUI uses this (default: off). The JSONL event log is written either way."
        ),
    )
    parser.add_argument("--event-log", type=Path, help="Optional JSONL path for event log (default: <output_dir>/dce_pipeline_events.jsonl)")
    run_reporting.add_verbosity_argument(parser)
    return parser.parse_args(argv)


def _build_event_logger(
    event_log_path: Path, sinks: list[Callable[[Dict[str, Any]], None]]
) -> tuple[IO[str], Any]:
    """Open the JSONL event log and return an emitter that also feeds `sinks`.

    The log records every event regardless of what the console shows, so a run is
    always reconstructable at full detail after the fact.
    """
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = event_log_path.open("w", encoding="utf-8")

    def _emit(event: Dict[str, Any]) -> None:
        handle.write(json.dumps(event, default=str) + "\n")
        handle.flush()
        for sink in sinks:
            sink(event)

    return handle, _emit


def _stdout_event_sink(event: Dict[str, Any]) -> None:
    print(EVENT_PREFIX + json.dumps(event, default=str), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    verbosity = run_reporting.verbosity_from_args(args)
    # stdout carries one audience at a time: either the machine event stream (what the
    # GUI reads and renders itself) or human-readable progress, never both interleaved.
    machine_events = args.events == "on"

    if verbosity >= Verbosity.NORMAL:
        print_banner()

    config_path = args.config.expanduser().resolve()
    payload = _load_config(config_path)

    if args.output_dir:
        payload["output_dir"] = str(args.output_dir.expanduser().resolve())
    if args.checkpoint_dir:
        payload["checkpoint_dir"] = str(args.checkpoint_dir.expanduser().resolve())
    if args.backend:
        payload["backend"] = args.backend

    stage_overrides = dict(payload.get("stage_overrides", {}))
    # A path typed on the command line is relative to where it was typed, so --set anchors
    # to the cwd while the config file's own paths anchor to the config (below).
    stage_overrides.update(
        dce_config.resolve_override_paths(parse_set_overrides(args.set_overrides), Path.cwd())
    )
    if stage_overrides:
        payload["stage_overrides"] = stage_overrides

    # Config resolution reports what it discovered, but the console it should report to
    # does not exist until output_dir is known -- which resolution is what determines.
    # Hold those notices and replay them once there is somewhere to put them.
    held: list[tuple[str, Verbosity]] = []
    run_reporting.set_notice_sink(lambda text, level: held.append((text, level)))
    reporter: Optional[Reporter] = None
    try:
        # Relative paths in a config are anchored to that config's own directory, so the same
        # file works from any working directory. `parametric_cli` resolves the same way.
        try:
            config = DcePipelineConfig.from_dict(payload, base_dir=config_path.parent)
            config.validate()
        except (ValueError, KeyError) as exc:
            # A malformed run config is a user error, not a crash: say what is wrong and stop.
            print(f"Error in {config_path}: {exc}", file=sys.stderr)
            return 2

        resolved_output_dir = config.output_dir
        event_log_path = (
            args.event_log.expanduser().resolve()
            if args.event_log
            else (resolved_output_dir / "dce_pipeline_events.jsonl")
        )

        sinks: list[Callable[[Dict[str, Any]], None]] = []
        if machine_events:
            sinks.append(_stdout_event_sink)
        else:
            reporter = Reporter(verbosity=verbosity)
            sinks.append(reporter.handle_event)
        event_log_handle, emit_event = _build_event_logger(event_log_path, sinks)

        run_reporting.set_notice_sink(
            run_reporting.event_notice_sink(emit_event)
            if machine_events
            else run_reporting.reporter_notice_sink(reporter)
        )

        try:
            emit_event(
                {
                    "type": "cli_config",
                    **version.build_identity(),
                    "config_path": str(config_path),
                    "resolved_output_dir": str(resolved_output_dir),
                    "event_log_path": str(event_log_path),
                    "config": payload,
                }
            )
            for text, level in held:
                run_reporting.notice(text, level)
            held.clear()
            result = run_dce_pipeline(config, event_callback=emit_event)
        except Exception as exc:
            # A failure inside a stage already emitted run_error; one outside them (writing
            # the summary, say) did not, so emit here too -- the reporter shows whichever
            # arrives first. A traceback on top helps only someone debugging the pipeline.
            emit_event(
                {"type": "run_error", "error_type": type(exc).__name__, "error": str(exc)}
            )
            if verbosity >= Verbosity.DEBUG:
                raise
            return 1
        finally:
            event_log_handle.close()
    finally:
        run_reporting.set_notice_sink(None)

    meta = dict(result["meta"])
    meta["event_log_path"] = str(event_log_path)
    if machine_events or verbosity >= Verbosity.DEBUG:
        print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
