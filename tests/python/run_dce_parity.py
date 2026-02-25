"""Cross-platform runner for DCE parity tests.

This helper avoids manual environment-variable setup and keeps parity output readable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import warnings


SUITES = {
    "multi-model": "tests/python/test_dce_pipeline_parity_metrics.py::test_downsample_bbb_p19_models_cpu_and_auto",
    "model-map-roi-cpu": "tests/python/test_dce_pipeline_parity_metrics.py::test_downsample_bbb_p19_model_maps_and_roi_xls_cpu",
    "tofts-downsample": "tests/python/test_dce_pipeline_parity_metrics.py::test_downsample_bbb_p19_tofts_ktrans",
    "tofts-full": "tests/python/test_dce_pipeline_parity_metrics.py::test_full_bbb_p19_tofts_ktrans",
}
SUITE_SUMMARY_FILES = {
    "multi-model": "parity_multi_model_summary.json",
    "model-map-roi-cpu": "parity_model_map_roi_cpu_summary.json",
    "tofts-downsample": "parity_tofts_downsample_summary.json",
    "tofts-full": "parity_tofts_full_summary.json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROCKETSHIP DCE parity tests with discoverable options."
    )
    parser.add_argument(
        "-s",
        "--suite",
        choices=sorted(SUITES.keys()),
        default="multi-model",
        help="Parity test suite to run (default: multi-model).",
    )
    parser.add_argument(
        "-d",
        "--dataset-root",
        default="",
        help="Override the downsample parity dataset root (pytest alias: --ds-root).",
    )
    parser.add_argument(
        "-f",
        "--full-root",
        default="",
        help="Override full-volume dataset root for tofts-full suite (pytest alias: --fr-root).",
    )
    parser.add_argument(
        "-r",
        "--roi-stride",
        type=int,
        default=12,
        help="ROI stride for multi-model suite (default: 12; pytest alias: --stride).",
    )
    parser.add_argument(
        "-w",
        "--show-warnings",
        action="store_true",
        help="Show deprecation warnings (suppressed by default).",
    )
    parser.add_argument(
        "--summary-dir",
        default=".pytest_cache/parity_summaries",
        help="Directory where parity summary JSON artifacts are written.",
    )
    return parser.parse_args()


def _configure_environment(args: argparse.Namespace) -> None:
    if not args.show_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _render_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    widths = {h: len(h) for h in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def _line(values: dict[str, str]) -> str:
        return "  ".join(str(values.get(h, "")).ljust(widths[h]) for h in headers)

    lines = [_line({h: h for h in headers}), _line({h: "-" * widths[h] for h in headers})]
    lines.extend(_line(row) for row in rows)
    return "\n".join(lines)


def _parse_metric_values_from_error(error_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not error_text:
        return out
    metric_keys = ("n", "corr", "mse", "mae", "p95_abs_err", "rows", "max_abs_err")
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    for key in metric_keys:
        match = re.search(rf"{re.escape(key)}=({number})", error_text)
        if not match:
            continue
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError:
            continue
        out[key] = int(value) if key in {"n", "rows"} and value.is_integer() else value
    return out


def _load_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        print(f"[PARITY-SUMMARY] no summary file found at {summary_path}")
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[PARITY-SUMMARY] failed to parse {summary_path}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"[PARITY-SUMMARY] invalid payload in {summary_path}: expected object")
        return None
    return payload


def _print_tofts_summary(payload: dict[str, Any]) -> None:
    rows: list[dict[str, str]] = []
    for key, label in (("ktrans", "Ktrans"), ("ve", "ve")):
        metrics = payload.get(key)
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "Metric": label,
                "n": str(metrics.get("n", "-")),
                "corr": _format_float(metrics.get("corr")),
                "mse": _format_float(metrics.get("mse")),
                "mae": _format_float(metrics.get("mae")),
                "p95_abs_err": _format_float(metrics.get("p95_abs_err")),
            }
        )
    if not rows:
        print("[PARITY-SUMMARY] no Ktrans/ve metrics found")
        return
    print(_render_table(rows, ["Metric", "n", "corr", "mse", "mae", "p95_abs_err"]))


def _print_multi_model_summary(payload: dict[str, Any]) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        print("[PARITY-SUMMARY] no multi-model checks found")
        return

    rows: list[dict[str, str]] = []
    pass_count = 0
    fail_count = 0
    skip_count = 0
    for check in checks:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status", "unknown"))
        if status == "pass":
            pass_count += 1
        elif status == "failed":
            fail_count += 1
        elif status == "skipped":
            skip_count += 1
        metrics = check.get("metrics") if isinstance(check.get("metrics"), dict) else {}
        error_metrics = _parse_metric_values_from_error(str(check.get("error", "")))
        merged_metrics = {**error_metrics, **(metrics or {})}
        rows.append(
            {
                "check": str(check.get("label", "")),
                "status": status,
                "required": str(bool(check.get("required", False))).lower(),
                "n": str(merged_metrics.get("n", check.get("valid_voxels", "-"))),
                "corr": _format_float(merged_metrics.get("corr")),
                "mse": _format_float(merged_metrics.get("mse")),
                "mae": _format_float(merged_metrics.get("mae")),
            }
        )

    print(
        "[PARITY-SUMMARY] checks="
        f"{len(rows)} pass={pass_count} fail={fail_count} skipped={skip_count} "
        f"required_failures={len(payload.get('required_failures', []) or [])} "
        f"diagnostic_failures={len(payload.get('diagnostic_failures', []) or [])}"
    )
    if rows:
        print(_render_table(rows, ["check", "status", "required", "n", "corr", "mse", "mae"]))


def _print_model_map_roi_summary(payload: dict[str, Any]) -> None:
    map_checks = payload.get("map_checks")
    if isinstance(map_checks, list) and map_checks:
        rows: list[dict[str, str]] = []
        for check in map_checks:
            if not isinstance(check, dict):
                continue
            metrics = check.get("metrics") if isinstance(check.get("metrics"), dict) else {}
            error_metrics = _parse_metric_values_from_error(str(check.get("error", "")))
            merged_metrics = {**error_metrics, **(metrics or {})}
            rows.append(
                {
                    "map_check": str(check.get("label", "")),
                    "status": str(check.get("status", "unknown")),
                    "required": str(bool(check.get("required", False))).lower(),
                    "n": str(merged_metrics.get("n", check.get("valid_voxels", "-"))),
                    "corr": _format_float(merged_metrics.get("corr")),
                    "mse": _format_float(merged_metrics.get("mse")),
                    "mae": _format_float(merged_metrics.get("mae")),
                }
            )
        print("[PARITY-SUMMARY] map checks")
        print(_render_table(rows, ["map_check", "status", "required", "n", "corr", "mse", "mae"]))

    roi_checks = payload.get("roi_checks")
    if isinstance(roi_checks, list) and roi_checks:
        rows = []
        for check in roi_checks:
            if not isinstance(check, dict):
                continue
            metrics = check.get("metrics") if isinstance(check.get("metrics"), dict) else {}
            error_metrics = _parse_metric_values_from_error(str(check.get("error", "")))
            merged_metrics = {**error_metrics, **(metrics or {})}
            rows.append(
                {
                    "roi_check": str(check.get("label", "")),
                    "status": str(check.get("status", "unknown")),
                    "required": str(bool(check.get("required", False))).lower(),
                    "rows": str(merged_metrics.get("rows", "-")),
                    "mae": _format_float(merged_metrics.get("mae")),
                    "max_abs_err": _format_float(merged_metrics.get("max_abs_err")),
                }
            )
        print("[PARITY-SUMMARY] roi checks")
        print(_render_table(rows, ["roi_check", "status", "required", "rows", "mae", "max_abs_err"]))


def _print_summary_for_suite(suite: str, summary_path: Path) -> None:
    print(f"\n[PARITY-SUMMARY] source={summary_path}")
    payload = _load_summary(summary_path)
    if payload is None:
        return

    dataset_root = payload.get("dataset_root")
    if dataset_root:
        print(f"[PARITY-SUMMARY] dataset_root={dataset_root}")

    if suite in {"tofts-downsample", "tofts-full"}:
        _print_tofts_summary(payload)
    elif suite == "multi-model":
        _print_multi_model_summary(payload)
    elif suite == "model-map-roi-cpu":
        _print_model_map_roi_summary(payload)
    else:
        print(f"[PARITY-SUMMARY] no suite formatter for {suite}")


def main() -> int:
    root = _repo_root()
    args = _parse_args()
    os.chdir(root)
    _configure_environment(args)

    test_name = SUITES[args.suite]
    summary_dir = Path(args.summary_dir).expanduser()
    if not summary_dir.is_absolute():
        summary_dir = (root / summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / SUITE_SUMMARY_FILES[args.suite]
    if summary_file.exists():
        summary_file.unlink()

    print(f"[PARITY-RUNNER] repo={root}", flush=True)
    print(f"[PARITY-RUNNER] suite={args.suite} test={test_name}", flush=True)
    print(f"[PARITY-RUNNER] python={sys.executable}", flush=True)
    print(f"[PARITY-RUNNER] summaryDir={summary_dir}", flush=True)
    if args.dataset_root:
        print(f"[PARITY-RUNNER] datasetRoot={args.dataset_root}", flush=True)
    if args.full_root:
        print(f"[PARITY-RUNNER] fullRoot={args.full_root}", flush=True)
    print(f"[PARITY-RUNNER] roiStride={max(1, int(args.roi_stride))}", flush=True)
    if not args.show_warnings:
        print(
            "[PARITY-RUNNER] deprecation warnings suppressed (use --show-warnings to enable)",
            flush=True,
        )

    env = os.environ.copy()
    if not args.show_warnings:
        env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::PendingDeprecationWarning"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_name,
        "-v",
        "--parity",
        "--parity-summary-dir",
        str(summary_dir),
        "--stride",
        str(max(1, int(args.roi_stride))),
    ]
    if args.dataset_root:
        cmd.extend(["--ds-root", args.dataset_root])
    if args.full_root:
        cmd.extend(["--fr-root", args.full_root])
    if args.suite == "multi-model":
        cmd.append("--mm-parity")
    if args.suite == "tofts-full":
        cmd.append("--full-parity")

    completed = subprocess.run(cmd, env=env, check=False)
    _print_summary_for_suite(args.suite, summary_file)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
