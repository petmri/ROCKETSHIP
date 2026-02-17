"""Cross-platform runner for DCE parity tests.

This helper avoids manual environment-variable setup and keeps parity output readable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import warnings


SUITES = {
    "multi-model": "tests.python.test_dce_pipeline_parity_metrics."
    "TestDcePipelineParityMetrics.test_downsample_bbb_p19_models_cpu_and_auto",
    "tofts-downsample": "tests.python.test_dce_pipeline_parity_metrics."
    "TestDcePipelineParityMetrics.test_downsample_bbb_p19_tofts_ktrans",
    "tofts-full": "tests.python.test_dce_pipeline_parity_metrics."
    "TestDcePipelineParityMetrics.test_full_bbb_p19_tofts_ktrans",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROCKETSHIP DCE parity tests with discoverable options."
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        default="multi-model",
        help="Parity test suite to run (default: multi-model).",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Override ROCKETSHIP_BBB_DOWNSAMPLED_ROOT.",
    )
    parser.add_argument(
        "--full-root",
        default="",
        help="Override ROCKETSHIP_BBB_FULL_ROOT for tofts-full suite.",
    )
    parser.add_argument(
        "--roi-stride",
        type=int,
        default=12,
        help="ROI stride for multi-model suite (default: 12).",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show deprecation warnings (suppressed by default).",
    )
    return parser.parse_args()


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ["ROCKETSHIP_RUN_PIPELINE_PARITY"] = "1"
    os.environ["ROCKETSHIP_PARITY_MULTI_MODEL_ROI_STRIDE"] = str(max(1, int(args.roi_stride)))

    if args.dataset_root:
        os.environ["ROCKETSHIP_BBB_DOWNSAMPLED_ROOT"] = args.dataset_root
    if args.full_root:
        os.environ["ROCKETSHIP_BBB_FULL_ROOT"] = args.full_root

    os.environ["ROCKETSHIP_RUN_MULTI_MODEL_BACKEND_PARITY"] = "1" if args.suite == "multi-model" else "0"
    if args.suite == "tofts-full":
        os.environ["ROCKETSHIP_RUN_FULL_VOLUME_PARITY"] = "1"

    if not args.show_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _repo_root()
    args = _parse_args()
    os.chdir(root)
    _configure_environment(args)

    test_name = SUITES[args.suite]
    print(f"[PARITY-RUNNER] repo={root}")
    print(f"[PARITY-RUNNER] suite={args.suite} test={test_name}")
    print(f"[PARITY-RUNNER] python={sys.executable}")
    if args.dataset_root:
        print(f"[PARITY-RUNNER] datasetRoot={args.dataset_root}")
    if args.full_root:
        print(f"[PARITY-RUNNER] fullRoot={args.full_root}")
    print(f"[PARITY-RUNNER] roiStride={max(1, int(args.roi_stride))}")
    if not args.show_warnings:
        print("[PARITY-RUNNER] deprecation warnings suppressed (use --show-warnings to enable)")

    env = os.environ.copy()
    if not args.show_warnings:
        env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::PendingDeprecationWarning"

    cmd = [sys.executable, "-m", "unittest", test_name, "-v"]
    completed = subprocess.run(cmd, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
