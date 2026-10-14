"""CLI entrypoint for parametric T1 mapping workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, IO, Optional

import run_reporting
from banner import print_banner
from cli_overrides import parse_set_overrides
from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline
from run_reporting import Reporter, Verbosity
import version


EVENT_PREFIX = "ROCKETSHIP_EVENT "


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "parametric_run_example.json"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to JSON pipeline config (default: python/parametric_run_example.json)",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional override for output_dir in config")
    parser.add_argument("--tr-ms", type=float, help="Optional override for tr_ms in config")
    parser.add_argument(
        "--rsquared-threshold",
        type=float,
        help="Optional override for rsquared_threshold in config",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override top-level config key/value (repeatable)",
    )
    parser.add_argument(
        "--events",
        choices=["on", "off"],
        default="off",
        help=(
            "Put the machine-readable JSON event stream on stdout instead of human-readable "
            "progress; the GUI uses this (default: off). The JSONL event log is written either way."
        ),
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        help="Optional JSONL path for event log (default: <output_dir>/parametric_t1_events.jsonl)",
    )
    run_reporting.add_verbosity_argument(parser)
    return parser.parse_args(argv)


def _build_event_logger(
    event_log_path: Path, sinks: list[Callable[[Dict[str, Any]], None]]
) -> tuple[IO[str], Any]:
    """Open the JSONL event log and return an emitter that also feeds `sinks`."""
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
    # As in dce_cli: stdout carries the machine event stream or human progress, not both.
    machine_events = args.events == "on"
    if verbosity >= Verbosity.NORMAL:
        print_banner()
    config_path = args.config.expanduser().resolve()
    payload = _load_config(config_path)

    if args.output_dir:
        payload["output_dir"] = str(args.output_dir.expanduser().resolve())
    if args.tr_ms is not None:
        payload["tr_ms"] = float(args.tr_ms)
    if args.rsquared_threshold is not None:
        payload["rsquared_threshold"] = float(args.rsquared_threshold)

    payload.update(parse_set_overrides(args.set_overrides))

    config = ParametricT1Config.from_dict(payload, base_dir=config_path.parent)
    resolved_output_dir = config.output_dir
    event_log_path = (
        args.event_log.expanduser().resolve() if args.event_log else (resolved_output_dir / "parametric_t1_events.jsonl")
    )

    reporter: Optional[Reporter] = None
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
        try:
            result = run_parametric_t1_pipeline(config, event_callback=emit_event)
        except Exception as exc:
            # The parametric pipeline has no run_error event of its own, so the failure is
            # reported here -- through the event stream, so the GUI and the log see it too.
            emit_event(
                {"type": "run_error", "error_type": type(exc).__name__, "error": str(exc)}
            )
            if verbosity >= Verbosity.DEBUG:
                raise
            return 1
    finally:
        event_log_handle.close()
        run_reporting.set_notice_sink(None)

    meta = dict(result["meta"])
    meta["event_log_path"] = str(event_log_path)
    if machine_events or verbosity >= Verbosity.DEBUG:
        print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
