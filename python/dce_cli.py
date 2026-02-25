"""CLI entrypoint for the in-memory DCE A->B->D pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, IO, Optional

from dce_pipeline import DcePipelineConfig, run_dce_pipeline


EVENT_PREFIX = "ROCKETSHIP_EVENT "


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "dce_default.json"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text())


def _parse_set_overrides(values: list[str]) -> Dict[str, Any]:
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to JSON pipeline config (default: python/dce_default.json)",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional override for output_dir in config")
    parser.add_argument("--checkpoint-dir", type=Path, help="Optional override for checkpoint_dir in config")
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "gpufit"],
        help="Optional override for backend in config",
    )
    parser.add_argument(
        "--dce-preferences",
        type=Path,
        help="Optional path to MATLAB-style dce_preferences.txt (applied as Python defaults)",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override stage_overrides key/value (repeatable)",
    )
    parser.add_argument(
        "--events",
        choices=["on", "off"],
        default="on",
        help="Emit JSON progress events on stdout (default: on)",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        help="Optional JSONL path for event log (default: <output_dir>/dce_pipeline_events.jsonl)",
    )
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
    if args.dce_preferences:
        stage_overrides["dce_preferences_path"] = str(args.dce_preferences.expanduser().resolve())
    stage_overrides.update(_parse_set_overrides(args.set_overrides))
    if stage_overrides:
        payload["stage_overrides"] = stage_overrides

    config = DcePipelineConfig.from_dict(payload)
    resolved_output_dir = Path(str(payload["output_dir"])).expanduser().resolve()
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
