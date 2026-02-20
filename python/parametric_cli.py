"""CLI entrypoint for parametric T1 mapping workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, IO, Optional

from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline


EVENT_PREFIX = "ROCKETSHIP_EVENT "


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "parametric_default.json"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_value(raw: str) -> Any:
    text = raw.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." in text or "e" in lower:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_set_overrides(values: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --set entry '{raw}'. Expected KEY=VALUE")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set entry '{raw}'. Empty KEY")
        overrides[key] = _coerce_value(value)
    return overrides


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to JSON pipeline config (default: python/parametric_default.json)",
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
        default="on",
        help="Emit JSON progress events on stdout (default: on)",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        help="Optional JSONL path for event log (default: <output_dir>/parametric_t1_events.jsonl)",
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
    if args.tr_ms is not None:
        payload["tr_ms"] = float(args.tr_ms)
    if args.rsquared_threshold is not None:
        payload["rsquared_threshold"] = float(args.rsquared_threshold)

    payload.update(_parse_set_overrides(args.set_overrides))

    config = ParametricT1Config.from_dict(payload, base_dir=config_path.parent)
    resolved_output_dir = config.output_dir
    event_log_path = (
        args.event_log.expanduser().resolve() if args.event_log else (resolved_output_dir / "parametric_t1_events.jsonl")
    )

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
        result = run_parametric_t1_pipeline(config, event_callback=emit_event)
    finally:
        event_log_handle.close()

    meta = dict(result["meta"])
    meta["event_log_path"] = str(event_log_path)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
