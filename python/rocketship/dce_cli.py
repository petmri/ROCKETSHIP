"""CLI entrypoint for the in-memory DCE A->B->D pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from .dce_pipeline import DcePipelineConfig, run_dce_pipeline


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON pipeline config")
    parser.add_argument("--output-dir", type=Path, help="Optional override for output_dir in config")
    parser.add_argument("--checkpoint-dir", type=Path, help="Optional override for checkpoint_dir in config")
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "gpufit"],
        help="Optional override for backend in config",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    payload = _load_config(args.config)

    if args.output_dir:
        payload["output_dir"] = str(args.output_dir.expanduser().resolve())
    if args.checkpoint_dir:
        payload["checkpoint_dir"] = str(args.checkpoint_dir.expanduser().resolve())
    if args.backend:
        payload["backend"] = args.backend

    config = DcePipelineConfig.from_dict(payload)
    result = run_dce_pipeline(config)

    print(json.dumps(result["meta"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
