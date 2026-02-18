"""Benchmark full DCE pipeline runtime across MATLAB/Python backend configurations.

This script targets a fast BIDS-style fixture by default and runs as many
backend configurations as are available on the current machine:

- matlab_cpu
- matlab_gpufit
- python_cpu
- python_cpufit
- python_gpufit

Unavailable configurations are reported as SKIP.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "test_data" / "ci_fixtures" / "dce" / "downsample_x2_bids"
ALL_MODELS = ["tofts", "ex_tofts", "patlak", "tissue_uptake", "two_cxm", "fxr", "auc", "nested", "FXL_rr"]
ALL_CONFIGS = ["matlab_cpu", "matlab_gpufit", "python_cpu", "python_cpufit", "python_gpufit"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full DCE pipeline runtime across available backends.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"BIDS-style dataset root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--subject", default="sub-01", help="Subject ID within dataset root (default: sub-01)")
    parser.add_argument("--session", default="ses-01", help="Session ID within dataset root (default: ses-01)")
    parser.add_argument(
        "--models",
        default="patlak",
        help="Comma-separated model flags to run (default: patlak). Use 'all' for every model flag.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions per configuration (default: 1)")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per-run timeout in seconds (default: 900)")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable for Python runs")
    parser.add_argument("--matlab-cmd", default="matlab", help="MATLAB command (default: matlab)")
    parser.add_argument(
        "--configs",
        default=",".join(ALL_CONFIGS),
        help="Comma-separated subset of configs to attempt",
    )
    parser.add_argument("--keep-workdir", action="store_true", help="Keep temporary benchmark work directory")
    parser.add_argument("--output-json", type=Path, help="Optional path to write detailed JSON results")
    return parser.parse_args()


def _escape_matlab_string(path_or_text: str) -> str:
    return path_or_text.replace("'", "''")


def _parse_csv_tokens(text: str) -> List[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def _parse_models(raw: str) -> Dict[str, int]:
    model_flags = {key: 0 for key in ALL_MODELS}
    tokens = [token.lower() for token in _parse_csv_tokens(raw)]
    if not tokens:
        tokens = ["patlak"]
    if "all" in tokens:
        for key in ALL_MODELS:
            model_flags[key] = 1
        return model_flags

    alias = {
        "2cxm": "two_cxm",
        "two_cxm": "two_cxm",
        "tissue": "tissue_uptake",
        "tofts": "tofts",
        "ex_tofts": "ex_tofts",
        "patlak": "patlak",
        "fxr": "fxr",
        "auc": "auc",
        "nested": "nested",
        "fxl_rr": "FXL_rr",
    }
    unknown: List[str] = []
    for token in tokens:
        key = alias.get(token)
        if key is None:
            unknown.append(token)
            continue
        model_flags[key] = 1
    if unknown:
        raise ValueError(f"Unknown model tokens: {unknown}")
    return model_flags


def _parse_configs(raw: str) -> List[str]:
    requested = _parse_csv_tokens(raw)
    if not requested:
        requested = list(ALL_CONFIGS)
    bad = [name for name in requested if name not in ALL_CONFIGS]
    if bad:
        raise ValueError(f"Unsupported configs requested: {bad}. Allowed={ALL_CONFIGS}")
    return requested


def _subject_paths(dataset_root: Path, subject: str, session: str) -> Tuple[Path, Path]:
    source = dataset_root / "rawdata" / subject / session
    tp = dataset_root / "derivatives" / subject / session
    return source, tp


def _find_single_file(parent: Path, pattern: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {parent / pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one match for {parent / pattern}, found {len(matches)}")
    return matches[0]


def _discover_python_inputs(tp_root: Path) -> Dict[str, Path]:
    dce_dir = tp_root / "dce"
    anat_dir = tp_root / "anat"
    return {
        "dynamic": _find_single_file(dce_dir, "*desc-bfcz_DCE.nii*"),
        "aif": _find_single_file(dce_dir, "*desc-AIF_T1map.nii*"),
        "roi": _find_single_file(anat_dir, "*desc-brain_mask.nii*"),
        "t1map": _find_single_file(anat_dir, "*space-DCEref_T1map.nii*"),
    }


def _replace_pref(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{value}", text)
    return text + f"\n{key} = {value}\n"


def _prepare_matlab_pref_dir(dest: Path, model_flags: Dict[str, int], force_cpu: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    script_text = (REPO_ROOT / "script_preferences.txt").read_text(encoding="utf-8")
    dce_pref_text = (REPO_ROOT / "dce" / "dce_preferences.txt").read_text(encoding="utf-8")

    script_keys = {
        "tofts": "tofts",
        "ex_tofts": "ex_tofts",
        "patlak": "patlak",
        "tissue_uptake": "tissue_uptake",
        "two_cxm": "two_cxm",
        "fxr": "fxr",
        "auc": "auc",
        "nested": "nested",
        "FXL_rr": "FXL_rr",
    }
    for model_key, pref_key in script_keys.items():
        script_text = _replace_pref(script_text, pref_key, str(int(model_flags[model_key])))

    dce_pref_text = _replace_pref(dce_pref_text, "force_cpu", str(int(force_cpu)))

    (dest / "script_preferences.txt").write_text(script_text, encoding="utf-8")
    (dest / "dce_preferences.txt").write_text(dce_pref_text, encoding="utf-8")


def _run_subprocess(
    cmd: List[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]],
    timeout_sec: int,
) -> Tuple[bool, float, str, str]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return False, elapsed, "", f"timeout after {timeout_sec}s: {exc}"
    elapsed = time.perf_counter() - start
    ok = completed.returncode == 0
    stderr = completed.stderr or ""
    if not ok and not stderr:
        stderr = f"command failed with exit code {completed.returncode}"
    return ok, elapsed, completed.stdout or "", stderr


def _probe_python_backend(python_exe: str, env_extra: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    snippet = (
        "import json,sys; from pathlib import Path; "
        "repo=Path(sys.argv[1]); sys.path.insert(0,str(repo/'python')); "
        "import dce_pipeline as d; d.probe_acceleration_backend.cache_clear(); "
        "print(json.dumps(d.probe_acceleration_backend()))"
    )
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    completed = subprocess.run(
        [python_exe, "-c", snippet, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return {"backend": "none", "error": completed.stderr.strip() or "probe failed"}
    try:
        return json.loads(completed.stdout.strip())
    except Exception:
        return {"backend": "none", "error": "failed to parse python backend probe"}


def _probe_matlab_gpufit(matlab_cmd: str) -> Dict[str, Any]:
    if shutil.which(matlab_cmd) is None:
        return {"available": False, "reason": f"'{matlab_cmd}' not found in PATH"}

    batch = (
        f"cd('{_escape_matlab_string(str(REPO_ROOT))}'); "
        "addpath(fullfile(pwd,'dce')); addpath(fullfile(pwd,'external_programs')); "
        "avail=0; if exist('GpufitCudaAvailableMex','file'), "
        "try, avail=GpufitCudaAvailableMex; catch, avail=0; end; end; "
        "fprintf('ROCKETSHIP_MATLAB_GPUFIT_AVAILABLE=%d\\n', double(avail>0));"
    )
    completed = subprocess.run(
        [matlab_cmd, "-noFigureWindows", "-batch", batch],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    marker = "ROCKETSHIP_MATLAB_GPUFIT_AVAILABLE="
    for line in output.splitlines():
        if marker in line:
            try:
                value = int(line.split(marker, 1)[1].strip())
                return {"available": value == 1, "reason": "probe marker"}
            except Exception:
                break
    return {"available": False, "reason": "probe marker not found"}


def _python_config_payload(
    source_root: Path,
    tp_root: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    model_flags: Dict[str, int],
    backend: str,
) -> Dict[str, Any]:
    files = _discover_python_inputs(tp_root)
    return {
        "subject_source_path": str(source_root),
        "subject_tp_path": str(tp_root),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "backend": backend,
        "write_xls": True,
        "aif_mode": "auto",
        "dynamic_files": [str(files["dynamic"])],
        "aif_files": [str(files["aif"])],
        "roi_files": [str(files["roi"])],
        "t1map_files": [str(files["t1map"])],
        "noise_files": [str(files["aif"])],
        "drift_files": [],
        "model_flags": {
            "tofts": int(model_flags["tofts"]),
            "ex_tofts": int(model_flags["ex_tofts"]),
            "patlak": int(model_flags["patlak"]),
            "tissue_uptake": int(model_flags["tissue_uptake"]),
            "two_cxm": int(model_flags["two_cxm"]),
            "fxr": int(model_flags["fxr"]),
            "auc": int(model_flags["auc"]),
            "nested": int(model_flags["nested"]),
            "FXL_rr": int(model_flags["FXL_rr"]),
        },
        "stage_overrides": {
            "stage_a_mode": "real",
            "stage_b_mode": "real",
            "stage_d_mode": "real",
            "aif_curve_mode": "fitted",
            "time_smoothing": "none",
            "time_smoothing_window": 0,
            "steady_state_start": 1,
            "steady_state_end": 2,
            "snr_filter": 0.0,
        },
    }


def _read_python_backend_used(summary_path: Path) -> str:
    if not summary_path.exists():
        return "unknown"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    stage_d = payload.get("stages", {}).get("D", {}) if isinstance(payload, dict) else {}
    if not isinstance(stage_d, dict):
        return "unknown"
    backend_used = stage_d.get("backend_used")
    return str(backend_used) if backend_used is not None else "unknown"


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.3f}"


def _summarize_times(times: List[float]) -> Dict[str, Optional[float]]:
    if not times:
        return {"mean": None, "std": None, "min": None, "max": None}
    mean = sum(times) / len(times)
    if len(times) <= 1:
        std = 0.0
    else:
        var = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
        std = var ** 0.5
    return {"mean": mean, "std": std, "min": min(times), "max": max(times)}


def _render_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    widths: Dict[str, int] = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row.get(h, "")))

    def _line(values: Dict[str, str]) -> str:
        return "  ".join(values.get(h, "").ljust(widths[h]) for h in headers)

    sep = "  ".join("-" * widths[h] for h in headers)
    lines = [_line({h: h for h in headers}), sep]
    lines.extend(_line(row) for row in rows)
    return "\n".join(lines)


def _build_row(result: Dict[str, Any]) -> Dict[str, str]:
    stats = _summarize_times(result.get("times_sec", []))
    return {
        "Configuration": str(result.get("name", "")),
        "Status": str(result.get("status", "")),
        "Runs": str(len(result.get("times_sec", []))),
        "Mean(s)": _fmt_float(stats["mean"]),
        "Std(s)": _fmt_float(stats["std"]),
        "Min(s)": _fmt_float(stats["min"]),
        "Max(s)": _fmt_float(stats["max"]),
        "BackendUsed": str(result.get("backend_used", "-")),
        "Notes": str(result.get("notes", "")),
    }


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    source_root, tp_root = _subject_paths(dataset_root, str(args.subject), str(args.session))
    model_flags = _parse_models(str(args.models))
    requested_configs = _parse_configs(str(args.configs))
    repeats = max(1, int(args.repeats))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not source_root.exists():
        raise FileNotFoundError(f"subject source path not found: {source_root}")
    if not tp_root.exists():
        raise FileNotFoundError(f"subject tp path not found: {tp_root}")

    print("[BENCH] ROCKETSHIP full-pipeline DCE benchmark")
    print(f"[BENCH] repo_root      = {REPO_ROOT}")
    print(f"[BENCH] dataset_root   = {dataset_root}")
    print(f"[BENCH] subject_source = {source_root}")
    print(f"[BENCH] subject_tp     = {tp_root}")
    print(f"[BENCH] models         = {[k for k, v in model_flags.items() if v]}")
    print(f"[BENCH] repeats        = {repeats}")
    print(f"[BENCH] timeout_sec    = {args.timeout_sec}")

    python_probe_default = _probe_python_backend(str(args.python_exe))
    python_probe_no_cuda = _probe_python_backend(str(args.python_exe), env_extra={"CUDA_VISIBLE_DEVICES": ""})
    matlab_probe = _probe_matlab_gpufit(str(args.matlab_cmd))
    matlab_available = shutil.which(str(args.matlab_cmd)) is not None

    print(f"[BENCH] python probe(default) = {python_probe_default}")
    print(f"[BENCH] python probe(no_cuda) = {python_probe_no_cuda}")
    print(f"[BENCH] matlab gpufit probe   = {matlab_probe}")

    run_root = Path(tempfile.mkdtemp(prefix="rocketship_bench_"))
    results: List[Dict[str, Any]] = []

    try:
        for config_name in requested_configs:
            result: Dict[str, Any] = {
                "name": config_name,
                "status": "SKIP",
                "times_sec": [],
                "backend_used": "-",
                "notes": "",
            }

            if config_name.startswith("matlab") and not matlab_available:
                result["notes"] = f"MATLAB command '{args.matlab_cmd}' not found"
                results.append(result)
                continue

            if config_name == "matlab_gpufit" and not bool(matlab_probe.get("available", False)):
                result["notes"] = "GpufitCudaAvailableMex unavailable"
                results.append(result)
                continue

            if config_name == "python_gpufit" and not bool(python_probe_default.get("pygpufit_imported", False)):
                result["notes"] = "pygpufit import unavailable"
                results.append(result)
                continue

            if config_name == "python_cpufit" and str(python_probe_no_cuda.get("backend", "")) != "cpufit_cpu":
                result["notes"] = f"cpufit unavailable with CUDA hidden (probe={python_probe_no_cuda.get('backend')})"
                results.append(result)
                continue

            result["status"] = "OK"
            for rep in range(repeats):
                run_dir = run_root / config_name / f"rep_{rep + 1:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                if config_name.startswith("matlab"):
                    pref_dir = run_dir / "prefs"
                    force_cpu = 1 if config_name == "matlab_cpu" else 0
                    _prepare_matlab_pref_dir(pref_dir, model_flags=model_flags, force_cpu=force_cpu)

                    batch_expr = (
                        f"cd('{_escape_matlab_string(str(pref_dir))}'); "
                        f"addpath('{_escape_matlab_string(str(REPO_ROOT))}'); "
                        f"run_dce_cli('{_escape_matlab_string(str(source_root))}','{_escape_matlab_string(str(tp_root))}');"
                    )
                    cmd = [str(args.matlab_cmd), "-noFigureWindows", "-batch", batch_expr]
                    ok, elapsed, _stdout, stderr = _run_subprocess(
                        cmd,
                        cwd=REPO_ROOT,
                        env=os.environ.copy(),
                        timeout_sec=int(args.timeout_sec),
                    )
                    if not ok:
                        result["status"] = "FAIL"
                        result["notes"] = stderr.strip().splitlines()[-1] if stderr.strip() else "MATLAB run failed"
                        break
                    result["times_sec"].append(elapsed)
                    result["backend_used"] = "cpu_forced" if force_cpu else "gpufit_cuda"
                else:
                    output_dir = run_dir / "out"
                    checkpoint_dir = output_dir / "checkpoints"
                    backend = {
                        "python_cpu": "cpu",
                        "python_cpufit": "auto",
                        "python_gpufit": "gpufit",
                    }[config_name]

                    cfg_payload = _python_config_payload(
                        source_root=source_root,
                        tp_root=tp_root,
                        output_dir=output_dir,
                        checkpoint_dir=checkpoint_dir,
                        model_flags=model_flags,
                        backend=backend,
                    )
                    cfg_path = run_dir / "config.json"
                    cfg_path.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

                    env = os.environ.copy()
                    if config_name == "python_cpufit":
                        env["CUDA_VISIBLE_DEVICES"] = ""

                    cmd = [
                        str(args.python_exe),
                        str(REPO_ROOT / "run_dce_python_cli.py"),
                        "--config",
                        str(cfg_path),
                        "--events",
                        "off",
                    ]
                    ok, elapsed, _stdout, stderr = _run_subprocess(
                        cmd,
                        cwd=REPO_ROOT,
                        env=env,
                        timeout_sec=int(args.timeout_sec),
                    )
                    if not ok:
                        result["status"] = "FAIL"
                        result["notes"] = stderr.strip().splitlines()[-1] if stderr.strip() else "Python run failed"
                        break

                    summary_backend = _read_python_backend_used(output_dir / "dce_pipeline_run.json")
                    result["backend_used"] = summary_backend

                    if config_name == "python_cpufit" and summary_backend != "cpufit_cpu":
                        result["status"] = "SKIP"
                        result["times_sec"] = []
                        result["notes"] = f"requested cpufit but backend_used={summary_backend}"
                        break

                    if config_name == "python_gpufit" and summary_backend not in {"gpufit_cuda", "gpufit_cpu_fallback"}:
                        result["status"] = "SKIP"
                        result["times_sec"] = []
                        result["notes"] = f"requested gpufit but backend_used={summary_backend}"
                        break

                    result["times_sec"].append(elapsed)

            results.append(result)

        rows = [_build_row(item) for item in results]
        table = _render_table(
            rows,
            headers=[
                "Configuration",
                "Status",
                "Runs",
                "Mean(s)",
                "Std(s)",
                "Min(s)",
                "Max(s)",
                "BackendUsed",
                "Notes",
            ],
        )

        print("\n[BENCH] Results\n")
        print(table)

        payload = {
            "repo_root": str(REPO_ROOT),
            "dataset_root": str(dataset_root),
            "subject_source_path": str(source_root),
            "subject_tp_path": str(tp_root),
            "models": model_flags,
            "repeats": repeats,
            "timeout_sec": int(args.timeout_sec),
            "probes": {
                "python_default": python_probe_default,
                "python_no_cuda": python_probe_no_cuda,
                "matlab": matlab_probe,
            },
            "results": results,
            "workdir": str(run_root),
        }
        if args.output_json:
            output_json = args.output_json.expanduser().resolve()
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\n[BENCH] wrote JSON report: {output_json}")

        if any(item.get("status") == "FAIL" for item in results):
            return 1
        return 0
    finally:
        if args.keep_workdir:
            print(f"\n[BENCH] retained workdir: {run_root}")
        else:
            shutil.rmtree(run_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
