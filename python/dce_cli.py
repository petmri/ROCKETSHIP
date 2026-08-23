"""CLI entrypoint for the in-memory DCE A->B->D pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, IO, Optional

import dce_config
from banner import print_banner
from cli_overrides import parse_set_overrides
from dce_pipeline import DcePipelineConfig, run_dce_pipeline


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
    parser.add_argument("--events", choices=["on", "off"], default="on", help="Emit JSON progress events on stdout (default: on)")
    parser.add_argument("--event-log", type=Path, help="Optional JSONL path for event log (default: <output_dir>/dce_pipeline_events.jsonl)")
    return parser.parse_args(argv)


def _build_event_logger(event_log_path: Path, emit_stdout: bool) -> tuple[IO[str], Any]:
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = event_log_path.open("w", encoding="utf-8")

    def _emit(event: Dict[str, Any]) -> None:
        line = json.dumps(event, default=str)
        handle.write(line + "\n")
        handle.flush()
        if emit_stdout:
            print(EVENT_PREFIX + line, flush=True)

    return handle, _emit


def main(argv: list[str] | None = None) -> int:
    print_banner()
    args = parse_args(argv if argv is not None else sys.argv[1:])
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
    event_log_path = args.event_log.expanduser().resolve() if args.event_log else (resolved_output_dir / "dce_pipeline_events.jsonl")
    event_log_handle, emit_event = _build_event_logger(event_log_path, emit_stdout=(args.events == "on"))
    try:
        emit_event(
            {
                "type": "cli_config",
                "config_path": str(config_path),
                "resolved_output_dir": str(resolved_output_dir),
                "event_log_path": str(event_log_path),
                "config": payload,
            }
        )
        result = run_dce_pipeline(config, event_callback=emit_event)
    finally:
        event_log_handle.close()

    meta = dict(result["meta"])
    meta["event_log_path"] = str(event_log_path)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
