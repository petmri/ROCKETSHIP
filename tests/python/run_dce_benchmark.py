"""Benchmark full DCE pipeline runtime across MATLAB/Python backend configurations.

The reported number is deliberately whole-process wall clock -- that is what a user
actually waits for. To keep that number interpretable the table also breaks it into
Stage A / Stage B / Stage D and an ``Other`` remainder (interpreter startup, I/O, and
on the MATLAB side the figure export), and reports the measured interpreter startup
floor for each language.

Configurations:

- matlab_cpu
- matlab_gpufit
- python_cpu
- python_cpufit
- python_gpufit

Unavailable configurations are reported as SKIP.

Dataset layout (see ``--raw-subdir`` / ``--derivatives-subdir``)::

    <dataset-root>/sourcedata/raw/<subject>/<session>/{dce,anat}
    <dataset-root>/derivatives/<deriv-subdir>/<subject>/<session>/{dce,anat}

Both pipelines are pointed at byte-identical inputs through a per-run scratch
timepoint directory of symlinks, so MATLAB never writes into the source dataset and
repeats stay independent.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "tests/data" / "BIDS_test"
DEFAULT_SUBJECT = "sub-203103"
DEFAULT_SESSION = "ses-01"
DEFAULT_RAW_SUBDIR = "sourcedata/raw"
DEFAULT_DERIVATIVES_SUBDIR = "dceprep-multihance_fix"

# Upper bound on how many derivatives subdirectories the fallback scan will stat.
# The scan only runs when --derivatives-subdir misses, and these trees usually live
# on a network mount, so it must not turn into a full-tree walk.
MAX_DERIVATIVES_PROBES = 64

# Relative spread in fitted-voxel count above which configurations are declared
# incomparable rather than merely inconsistent.
VOXEL_COUNT_COMPARABLE_TOLERANCE = 0.01

ALL_MODELS = ["tofts", "ex_tofts", "patlak", "tissue_uptake", "two_cxm", "fxr", "auc", "nested", "FXL_rr"]
ALL_CONFIGS = ["matlab_cpu", "matlab_gpufit", "python_cpu", "python_cpufit", "python_gpufit"]

# Subdirectory + glob for each pipeline input, relative to the subject timepoint dir.
INPUT_PATTERNS: Dict[str, Tuple[str, str]] = {
    "dynamic": ("dce", "*desc-bfcz_DCE.nii*"),
    "aif": ("dce", "*desc-AIF_T1map.nii*"),
    "roi": ("anat", "*space-DCEref_desc-brain_mask.nii*"),
    "t1map": ("anat", "*space-DCEref_T1map.nii*"),
}

# `rootname` in script_preferences.txt; drives the MATLAB log filenames we parse.
MATLAB_ROOTNAME = "dce"
# D_fit_voxels_func names its per-model log with these slugs, not the pref keys.
MATLAB_MODEL_SLUG = {"two_cxm": "2cxm"}

# run_dce_cli.m swallows Stage-A/B/D failures and still exits 0 (see its `return`
# statements and the Stage-D catch). These markers are how a swallowed failure looks
# on stdout; without them a run that did nothing scores as a fast OK.
MATLAB_FAILURE_MARKERS = (
    "RUND failed",
    "RUNB failed and could not recover",
    "Not enough baseline acquisitions",
    "File does not exist",
)
# Recoverable retries: not a failure, but they re-run Stage A and inflate the time.
MATLAB_WARNING_MARKERS = (
    "Trying again by retaining",
    "RUNB failed. Repeating in case of bad read",
)

_MATLAB_ELAPSED_RE = re.compile(r"Elapsed time is ([0-9.eE+-]+) seconds")
_MATLAB_VOXELS_RE = re.compile(r"Starting fitting for (\d+) voxels")

# Preferences read from script_preferences.txt and mirrored into the Python config so
# both pipelines run the same settings. Value is the Python stage_overrides key.
SHARED_PREF_KEYS = {
    "snr_filter": "snr_filter",
    "noise_pixsize": "noise_pixsize",
    "start_t": "start_t",
    "end_t": "end_t",
    "time_smoothing": "time_smoothing",
    "time_smoothing_window": "time_smoothing_window",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full DCE pipeline runtime across available backends.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"BIDS-style dataset root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--subject", default=DEFAULT_SUBJECT, help=f"Subject ID (default: {DEFAULT_SUBJECT})")
    parser.add_argument("--session", default=DEFAULT_SESSION, help=f"Session ID (default: {DEFAULT_SESSION})")
    parser.add_argument(
        "--raw-subdir",
        default=DEFAULT_RAW_SUBDIR,
        help=f"Raw data path relative to dataset root (default: {DEFAULT_RAW_SUBDIR})",
    )
    parser.add_argument(
        "--derivatives-subdir",
        default=DEFAULT_DERIVATIVES_SUBDIR,
        help=(
            "Preferred subdirectory under <dataset-root>/derivatives "
            f"(default: {DEFAULT_DERIVATIVES_SUBDIR}). If it does not hold the subject, "
            "'dceprep*' siblings are probed as a fallback."
        ),
    )
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
    parser.add_argument(
        "--checkpoints",
        choices=["off", "on"],
        default="off",
        help=(
            "Write Python stage checkpoints (default: off). Checkpointing np.savez_compressed's "
            "every stage array -- ~950 MB and tens of seconds on a full-resolution subject, inside "
            "the timed window -- and a normal run does not do it."
        ),
    )
    parser.add_argument(
        "--no-startup-probe",
        action="store_true",
        help="Skip measuring the interpreter startup floors (saves ~11s of MATLAB startup)",
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
        "tissue_uptake": "tissue_uptake",
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


def _resolve_derivatives_tp(
    dataset_root: Path,
    subject: str,
    session: str,
    preferred_subdir: str,
) -> Tuple[Path, str]:
    """Locate <root>/derivatives/<something>/<subject>/<session>.

    The intermediate directory name varies per dataset, so try the caller's preference
    first (the common case, and the only one that touches no extra directories), then
    fall back to a bounded scan that favours 'dceprep*' siblings.
    """
    deriv_base = dataset_root / "derivatives"
    if not deriv_base.is_dir():
        raise FileNotFoundError(f"derivatives directory not found: {deriv_base}")

    candidates: List[Tuple[Path, str]] = []
    if preferred_subdir:
        candidates.append((deriv_base / preferred_subdir, f"--derivatives-subdir={preferred_subdir}"))
    # Some datasets have no intermediate directory at all.
    candidates.append((deriv_base, "derivatives root (no intermediate directory)"))

    for cand, note in candidates:
        if (cand / subject / session).is_dir():
            return cand / subject / session, note

    try:
        entries = sorted(p for p in deriv_base.iterdir() if p.is_dir())
    except OSError as exc:
        raise FileNotFoundError(f"could not list {deriv_base}: {exc}") from exc

    dceprep = [p for p in entries if "dceprep" in p.name.lower()]
    others = [p for p in entries if "dceprep" not in p.name.lower()]
    scanned = (dceprep + others)[:MAX_DERIVATIVES_PROBES]

    for cand in scanned:
        if (cand / subject / session).is_dir():
            return cand / subject / session, f"fallback scan matched {cand.name}"

    raise FileNotFoundError(
        f"no derivatives subdirectory under {deriv_base} contains {subject}/{session} "
        f"(tried '{preferred_subdir}', the derivatives root, and {len(scanned)} of "
        f"{len(entries)} subdirectories, 'dceprep*' first). "
        "Pass --derivatives-subdir explicitly."
    )


def _resolve_subject_paths(
    dataset_root: Path,
    subject: str,
    session: str,
    raw_subdir: str,
    derivatives_subdir: str,
) -> Tuple[Path, Path, str]:
    source = dataset_root / raw_subdir / subject / session
    if not source.is_dir():
        raise FileNotFoundError(f"subject source path not found: {source} (adjust --raw-subdir)")
    tp, note = _resolve_derivatives_tp(dataset_root, subject, session, derivatives_subdir)
    return source, tp, note


def _find_single_file(parent: Path, pattern: str) -> Path:
    # *_RAS.nii.gz are reoriented duplicates that sit next to the real input and would
    # otherwise make every glob ambiguous.
    matches = sorted(p for p in parent.glob(pattern) if "_RAS." not in p.name)
    if not matches:
        raise FileNotFoundError(f"No files matched {parent / pattern}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise RuntimeError(f"Expected one match for {parent / pattern}, found {len(matches)}: {names}")
    return matches[0]


def _discover_inputs(tp_root: Path) -> Dict[str, Path]:
    return {
        key: _find_single_file(tp_root / subdir, pattern)
        for key, (subdir, pattern) in INPUT_PATTERNS.items()
    }


def _materialize_scratch_tp(run_dir: Path, files: Dict[str, Path]) -> Tuple[Path, Dict[str, str]]:
    """Symlink the resolved inputs into a private timepoint tree.

    MATLAB writes its stage outputs next to the dynamic file, so without this every run
    would write into the real dataset and repeats would not be independent.
    """
    tp_root = run_dir / "tp"
    relative: Dict[str, str] = {}
    for key, src in files.items():
        subdir = INPUT_PATTERNS[key][0]
        dest_dir = tp_root / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        link = dest_dir / src.name
        if not link.exists():
            link.symlink_to(src)
        relative[key] = f"/{subdir}/{src.name}"
    return tp_root, relative


def _replace_pref(text: str, key: str, value: str) -> str:
    # [ \t] rather than \s: \s spans newlines, so a key with an empty value would
    # reach across the blank line and rewrite whatever follows it.
    pattern = re.compile(rf"(?m)^([ \t]*{re.escape(key)}[ \t]*=[ \t]*).*$")
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{value}", text)
    return text + f"\n{key} = {value}\n"


def _read_pref(text: str, key: str) -> Optional[str]:
    match = re.search(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*(.*)$", text)
    if not match:
        return None
    return match.group(1).strip()


def _shared_pref_overrides(script_text: str) -> Dict[str, Any]:
    """Mirror MATLAB's preferences into Python stage overrides.

    Anything hardcoded on only one side (this used to be snr_filter, the noise mask and
    the start_t frame window) makes the two pipelines fit different data, which shows up
    as a runtime difference that has nothing to do with either implementation.
    """
    overrides: Dict[str, Any] = {}
    for pref_key, override_key in SHARED_PREF_KEYS.items():
        raw = _read_pref(script_text, pref_key)
        if raw is None or raw == "":
            continue
        try:
            value: Any = float(raw)
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = raw
        overrides[override_key] = value
    return overrides


def _prepare_matlab_pref_dir(
    dest: Path,
    model_flags: Dict[str, int],
    force_cpu: int,
    relative_inputs: Dict[str, str],
) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    script_text = (REPO_ROOT / "script_preferences.txt").read_text(encoding="utf-8")
    dce_pref_text = (REPO_ROOT / "dce" / "dce_preferences.txt").read_text(encoding="utf-8")

    for model_key in ALL_MODELS:
        script_text = _replace_pref(script_text, model_key, str(int(model_flags[model_key])))

    # Pin the exact files rather than leaving a glob: the scratch tree holds one file per
    # role, and this guarantees MATLAB and Python read the same bytes.
    script_text = _replace_pref(script_text, "dynamic_files", relative_inputs["dynamic"])
    script_text = _replace_pref(script_text, "aif_files", relative_inputs["aif"])
    script_text = _replace_pref(script_text, "roi_files", relative_inputs["roi"])
    script_text = _replace_pref(script_text, "t1map_files", relative_inputs["t1map"])

    dce_pref_text = _replace_pref(dce_pref_text, "force_cpu", str(int(force_cpu)))

    (dest / "script_preferences.txt").write_text(script_text, encoding="utf-8")
    (dest / "dce_preferences.txt").write_text(dce_pref_text, encoding="utf-8")
    return script_text


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


def _measure_matlab_startup(matlab_cmd: str) -> Optional[float]:
    """Wall clock for a MATLAB batch process that does nothing."""
    start = time.perf_counter()
    completed = subprocess.run(
        [matlab_cmd, "-noFigureWindows", "-batch", "1;"],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - start
    return elapsed if completed.returncode == 0 else None


def _measure_python_startup(python_exe: str) -> Optional[float]:
    """Wall clock for interpreter startup plus importing the pipeline module."""
    snippet = (
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, str(Path(sys.argv[1])/'python')); "
        "import dce_pipeline"
    )
    start = time.perf_counter()
    completed = subprocess.run(
        [python_exe, "-c", snippet, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - start
    return elapsed if completed.returncode == 0 else None


def _python_config_payload(
    source_root: Path,
    tp_root: Path,
    output_dir: Path,
    checkpoint_dir: Optional[Path],
    model_flags: Dict[str, int],
    backend: str,
    shared_overrides: Dict[str, Any],
    files: Dict[str, Path],
) -> Dict[str, Any]:
    stage_overrides: Dict[str, Any] = {
        "stage_a_mode": "real",
        "stage_b_mode": "real",
        "stage_d_mode": "real",
    }
    stage_overrides.update(shared_overrides)
    return {
        "subject_source_path": str(source_root),
        "subject_tp_path": str(tp_root),
        "output_dir": str(output_dir),
        "aif_mode": "fitted",
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "backend": backend,
        # False to match MATLAB: script_preferences.txt leaves roi_list empty, so
        # D_fit_voxels_func writes no ROI table. Leaving this on makes Python do an extra
        # whole-brain average-then-fit that the MATLAB side never runs.
        "write_xls": False,
        "aif_mode": "fitted",
        "dynamic_files": [str(files["dynamic"])],
        "aif_files": [str(files["aif"])],
        "roi_files": [str(files["roi"])],
        "t1map_files": [str(files["t1map"])],
        # Empty, to match script_preferences.txt's `noise_files =`: both pipelines then
        # derive noise from the same noise_pixsize corner square.
        "noise_files": [],
        "drift_files": [],
        "model_flags": {key: int(model_flags[key]) for key in ALL_MODELS},
        "stage_overrides": stage_overrides,
    }


def _python_run_metrics(output_dir: Path, model_keys: Sequence[str]) -> Dict[str, Any]:
    """Per-stage seconds and fitted voxel count for one Python run."""
    metrics: Dict[str, Any] = {"backend_used": "unknown", "stage_sec": {}, "voxels": None}

    summary_path = output_dir / "dce_pipeline_run.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        stage_d = payload.get("stages", {}).get("D", {}) if isinstance(payload, dict) else {}
        if isinstance(stage_d, dict):
            backend_used = stage_d.get("backend_used")
            if backend_used is not None:
                metrics["backend_used"] = str(backend_used)
            outputs = stage_d.get("model_outputs", {})
            if isinstance(outputs, dict):
                for key in model_keys:
                    entry = outputs.get(key)
                    if isinstance(entry, dict):
                        shape = entry.get("voxel_result_shape")
                        if isinstance(shape, list) and shape:
                            metrics["voxels"] = int(shape[0])
                            break

    events_path = output_dir / "dce_pipeline_events.jsonl"
    if events_path.exists():
        starts: Dict[str, datetime] = {}
        try:
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                stamp = event.get("timestamp_utc")
                stage = event.get("stage")
                if not stamp or not stage:
                    continue
                when = datetime.fromisoformat(stamp)
                if event.get("type") == "stage_start":
                    starts[stage] = when
                elif event.get("type") == "stage_done" and stage in starts:
                    metrics["stage_sec"][stage] = (when - starts[stage]).total_seconds()
        except Exception:
            pass
    return metrics


def _matlab_run_metrics(dce_dir: Path, model_keys: Sequence[str], run_started_at: float) -> Dict[str, Any]:
    """Per-stage seconds and fitted voxel count from MATLAB's own diary logs.

    Also the failure detector: run_dce_cli.m exits 0 after a swallowed Stage-A/B/D
    failure, and the one thing it cannot fake is a fresh Stage-D log.
    """
    metrics: Dict[str, Any] = {"stage_sec": {}, "voxels": None, "missing": []}

    def _fresh_text(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        # 2s of slack for filesystem timestamp granularity on network mounts.
        if path.stat().st_mtime < run_started_at - 2.0:
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    stage_logs = {
        "A": dce_dir / f"A_{MATLAB_ROOTNAME}R1info.log",
        "B": dce_dir / f"B_{MATLAB_ROOTNAME}fitted_R1info.log",
    }
    for stage, path in stage_logs.items():
        text = _fresh_text(path)
        if text is None:
            metrics["missing"].append(f"stage {stage} log ({path.name})")
            continue
        elapsed = _MATLAB_ELAPSED_RE.findall(text)
        if elapsed:
            metrics["stage_sec"][stage] = float(elapsed[-1])

    total_d = 0.0
    saw_d = False
    voxels: Optional[int] = None
    for model_key in model_keys:
        slug = MATLAB_MODEL_SLUG.get(model_key, model_key)
        path = dce_dir / f"{MATLAB_ROOTNAME}_{slug}_fit.log"
        text = _fresh_text(path)
        if text is None:
            metrics["missing"].append(f"stage D log for {model_key} ({path.name})")
            continue
        elapsed = _MATLAB_ELAPSED_RE.findall(text)
        if elapsed:
            total_d += float(elapsed[-1])
            saw_d = True
        found = _MATLAB_VOXELS_RE.search(text)
        if found and voxels is None:
            voxels = int(found.group(1))
    if saw_d:
        metrics["stage_sec"]["D"] = total_d
    metrics["voxels"] = voxels
    return metrics


def _scan_markers(text: str, markers: Sequence[str]) -> Optional[str]:
    for line in text.splitlines():
        for marker in markers:
            if marker in line:
                return line.strip()
    return None


def _fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _stddev(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _stage_mean(result: Dict[str, Any], stage: str) -> Optional[float]:
    values = [entry[stage] for entry in result.get("stage_sec", []) if stage in entry]
    return _mean(values)


def _other_mean(result: Dict[str, Any]) -> Optional[float]:
    """Wall clock not accounted for by stages A/B/D: startup, I/O, figure export."""
    leftovers: List[float] = []
    for total, stages in zip(result.get("times_sec", []), result.get("stage_sec", [])):
        if {"A", "B", "D"} <= set(stages):
            leftovers.append(total - (stages["A"] + stages["B"] + stages["D"]))
    return _mean(leftovers)


def _voxel_summary(result: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    counts = [c for c in result.get("voxels", []) if c is not None]
    if not counts:
        return "-", None
    unique = sorted(set(counts))
    if len(unique) > 1:
        return "mixed", None
    return f"{unique[0]:,}", unique[0]


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
    times = result.get("times_sec", [])
    voxel_text, _ = _voxel_summary(result)
    return {
        "Configuration": str(result.get("name", "")),
        "Status": str(result.get("status", "")),
        "Runs": str(len(times)),
        "Total(s)": _fmt_float(_mean(times)),
        "Std(s)": _fmt_float(_stddev(times)),
        "A(s)": _fmt_float(_stage_mean(result, "A")),
        "B(s)": _fmt_float(_stage_mean(result, "B")),
        "D(s)": _fmt_float(_stage_mean(result, "D")),
        "Other(s)": _fmt_float(_other_mean(result)),
        "Voxels": voxel_text,
        "BackendUsed": str(result.get("backend_used", "-")),
        "Notes": str(result.get("notes", "")),
    }


TABLE_HEADERS = [
    "Configuration",
    "Status",
    "Runs",
    "Total(s)",
    "Std(s)",
    "A(s)",
    "B(s)",
    "D(s)",
    "Other(s)",
    "Voxels",
    "BackendUsed",
    "Notes",
]


def _run_matlab_rep(
    *,
    config_name: str,
    run_dir: Path,
    source_root: Path,
    scratch_tp: Path,
    relative_inputs: Dict[str, str],
    model_flags: Dict[str, int],
    model_keys: Sequence[str],
    matlab_cmd: str,
    timeout_sec: int,
    matlab_startup_sec: Optional[float],
) -> Dict[str, Any]:
    pref_dir = run_dir / "prefs"
    force_cpu = 1 if config_name == "matlab_cpu" else 0
    _prepare_matlab_pref_dir(pref_dir, model_flags, force_cpu, relative_inputs)

    batch_expr = (
        f"cd('{_escape_matlab_string(str(pref_dir))}'); "
        f"addpath('{_escape_matlab_string(str(REPO_ROOT))}'); "
        f"run_dce_cli('{_escape_matlab_string(str(source_root))}','{_escape_matlab_string(str(scratch_tp))}');"
    )
    cmd = [matlab_cmd, "-noFigureWindows", "-batch", batch_expr]

    started_at = time.time()
    ok, elapsed, stdout, stderr = _run_subprocess(
        cmd, cwd=REPO_ROOT, env=os.environ.copy(), timeout_sec=timeout_sec
    )
    (run_dir / "matlab_stdout.txt").write_text(stdout, encoding="utf-8")
    if stderr:
        (run_dir / "matlab_stderr.txt").write_text(stderr, encoding="utf-8")

    if not ok:
        note = stderr.strip().splitlines()[-1] if stderr.strip() else "MATLAB run failed"
        return {"ok": False, "note": note, "elapsed": elapsed}

    # run_dce_cli.m returns 0 after a swallowed stage failure, so exit code alone is not
    # evidence the pipeline ran.
    failure = _scan_markers(stdout, MATLAB_FAILURE_MARKERS)
    if failure:
        return {"ok": False, "note": f"MATLAB reported: {failure}", "elapsed": elapsed}

    metrics = _matlab_run_metrics(scratch_tp / "dce", model_keys, started_at)
    if metrics["missing"]:
        return {
            "ok": False,
            "note": f"exited 0 but produced no fresh {metrics['missing'][0]}",
            "elapsed": elapsed,
        }
    if metrics["voxels"] is None:
        return {"ok": False, "note": "exited 0 but Stage D logged no voxel count", "elapsed": elapsed}

    notes: List[str] = []
    warning = _scan_markers(stdout, MATLAB_WARNING_MARKERS)
    if warning:
        notes.append(f"retried: {warning}")
    if matlab_startup_sec is not None and elapsed < matlab_startup_sec:
        notes.append(f"total below MATLAB startup floor ({matlab_startup_sec:.1f}s)")

    return {
        "ok": True,
        "elapsed": elapsed,
        "stage_sec": metrics["stage_sec"],
        "voxels": metrics["voxels"],
        "backend_used": "cpu_forced" if force_cpu else "gpufit_cuda",
        "notes": notes,
    }


def _run_python_rep(
    *,
    config_name: str,
    run_dir: Path,
    source_root: Path,
    scratch_tp: Path,
    files: Dict[str, Path],
    model_flags: Dict[str, int],
    model_keys: Sequence[str],
    shared_overrides: Dict[str, Any],
    python_exe: str,
    timeout_sec: int,
    checkpoints: bool,
) -> Dict[str, Any]:
    output_dir = run_dir / "out"
    checkpoint_dir = (output_dir / "checkpoints") if checkpoints else None
    backend = {"python_cpu": "cpu", "python_cpufit": "auto", "python_gpufit": "gpufit"}[config_name]

    cfg_payload = _python_config_payload(
        source_root=source_root,
        tp_root=scratch_tp,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        model_flags=model_flags,
        backend=backend,
        shared_overrides=shared_overrides,
        files=files,
    )
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    env = os.environ.copy()
    if config_name == "python_cpufit":
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        python_exe,
        str(REPO_ROOT / "run_dce_python_cli.py"),
        "--config",
        str(cfg_path),
        "--events",
        "off",
    ]
    ok, elapsed, _stdout, stderr = _run_subprocess(cmd, cwd=REPO_ROOT, env=env, timeout_sec=timeout_sec)
    if not ok:
        note = stderr.strip().splitlines()[-1] if stderr.strip() else "Python run failed"
        return {"ok": False, "note": note, "elapsed": elapsed}

    metrics = _python_run_metrics(output_dir, model_keys)
    backend_used = metrics["backend_used"]

    if config_name == "python_cpufit" and backend_used != "cpufit_cpu":
        return {"ok": False, "skip": True, "note": f"requested cpufit but backend_used={backend_used}"}
    if config_name == "python_gpufit" and backend_used not in {"gpufit_cuda", "gpufit_cpu_fallback"}:
        return {"ok": False, "skip": True, "note": f"requested gpufit but backend_used={backend_used}"}

    return {
        "ok": True,
        "elapsed": elapsed,
        "stage_sec": metrics["stage_sec"],
        "voxels": metrics["voxels"],
        "backend_used": backend_used,
        "notes": [],
    }


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    source_root, tp_root, deriv_note = _resolve_subject_paths(
        dataset_root,
        str(args.subject),
        str(args.session),
        str(args.raw_subdir),
        str(args.derivatives_subdir),
    )
    model_flags = _parse_models(str(args.models))
    model_keys = [key for key, value in model_flags.items() if value]
    requested_configs = _parse_configs(str(args.configs))
    repeats = max(1, int(args.repeats))

    files = _discover_inputs(tp_root)
    script_text = (REPO_ROOT / "script_preferences.txt").read_text(encoding="utf-8")
    shared_overrides = _shared_pref_overrides(script_text)

    print("[BENCH] ROCKETSHIP full-pipeline DCE benchmark")
    print(f"[BENCH] repo_root       = {REPO_ROOT}")
    print(f"[BENCH] dataset_root    = {dataset_root}")
    print(f"[BENCH] subject_source  = {source_root}")
    print(f"[BENCH] subject_tp      = {tp_root}")
    print(f"[BENCH] derivatives via = {deriv_note}")
    for key in sorted(files):
        print(f"[BENCH]   input {key:8s} = {files[key].name}")
    print(f"[BENCH] models          = {model_keys}")
    print(f"[BENCH] shared prefs    = {shared_overrides}")
    print(f"[BENCH] repeats         = {repeats}")
    print(f"[BENCH] checkpoints     = {args.checkpoints}")
    print(f"[BENCH] timeout_sec     = {args.timeout_sec}")

    python_probe_default = _probe_python_backend(str(args.python_exe))
    python_probe_no_cuda = _probe_python_backend(str(args.python_exe), env_extra={"CUDA_VISIBLE_DEVICES": ""})
    matlab_probe = _probe_matlab_gpufit(str(args.matlab_cmd))
    matlab_available = shutil.which(str(args.matlab_cmd)) is not None

    print(f"[BENCH] python probe(default) = {python_probe_default}")
    print(f"[BENCH] python probe(no_cuda) = {python_probe_no_cuda}")
    print(f"[BENCH] matlab gpufit probe   = {matlab_probe}")

    wants_matlab = any(name.startswith("matlab") for name in requested_configs)
    wants_python = any(name.startswith("python") for name in requested_configs)
    matlab_startup_sec: Optional[float] = None
    python_startup_sec: Optional[float] = None
    if not args.no_startup_probe:
        if wants_matlab and matlab_available:
            matlab_startup_sec = _measure_matlab_startup(str(args.matlab_cmd))
            print(f"[BENCH] matlab startup floor = {_fmt_float(matlab_startup_sec, 2)} s (matlab -batch \"1;\")")
        if wants_python:
            python_startup_sec = _measure_python_startup(str(args.python_exe))
            print(
                f"[BENCH] python startup floor = {_fmt_float(python_startup_sec, 2)} s "
                "(interpreter + dce_pipeline import)"
            )

    run_root = Path(tempfile.mkdtemp(prefix="rocketship_bench_"))
    results: List[Dict[str, Any]] = []

    try:
        for config_name in requested_configs:
            result: Dict[str, Any] = {
                "name": config_name,
                "status": "SKIP",
                "times_sec": [],
                "stage_sec": [],
                "voxels": [],
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
            notes: List[str] = []
            for rep in range(repeats):
                run_dir = run_root / config_name / f"rep_{rep + 1:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                scratch_tp, relative_inputs = _materialize_scratch_tp(run_dir, files)

                if config_name.startswith("matlab"):
                    outcome = _run_matlab_rep(
                        config_name=config_name,
                        run_dir=run_dir,
                        source_root=source_root,
                        scratch_tp=scratch_tp,
                        relative_inputs=relative_inputs,
                        model_flags=model_flags,
                        model_keys=model_keys,
                        matlab_cmd=str(args.matlab_cmd),
                        timeout_sec=int(args.timeout_sec),
                        matlab_startup_sec=matlab_startup_sec,
                    )
                else:
                    outcome = _run_python_rep(
                        config_name=config_name,
                        run_dir=run_dir,
                        source_root=source_root,
                        scratch_tp=scratch_tp,
                        files=files,
                        model_flags=model_flags,
                        model_keys=model_keys,
                        shared_overrides=shared_overrides,
                        python_exe=str(args.python_exe),
                        timeout_sec=int(args.timeout_sec),
                        checkpoints=(args.checkpoints == "on"),
                    )

                if not outcome.get("ok"):
                    result["status"] = "SKIP" if outcome.get("skip") else "FAIL"
                    result["times_sec"] = []
                    result["stage_sec"] = []
                    result["voxels"] = []
                    result["notes"] = str(outcome.get("note", "run failed"))
                    break

                result["times_sec"].append(outcome["elapsed"])
                result["stage_sec"].append(outcome["stage_sec"])
                result["voxels"].append(outcome["voxels"])
                result["backend_used"] = outcome["backend_used"]
                for note in outcome.get("notes", []):
                    if note not in notes:
                        notes.append(note)
            else:
                result["notes"] = "; ".join(notes)

            results.append(result)

        rows = [_build_row(item) for item in results]
        print("\n[BENCH] Results\n")
        print(_render_table(rows, headers=TABLE_HEADERS))

        print(
            "\n[BENCH] Total(s) is whole-process wall clock -- what a user waits for. "
            "A/B/D are the pipelines' own stage timers,"
        )
        print("[BENCH] and Other(s) = Total - (A+B+D).")
        print(
            "[BENCH] The stage columns do NOT cover the same work on both sides: MATLAB's timers "
            "bracket computation only,"
        )
        print(
            "[BENCH] so reading the dynamic (before A) and writing maps/figures (after D) land in "
            "its Other(s), while Python's"
        )
        print("[BENCH] stage timers include that I/O. Only Total(s) is directly comparable.")
        if matlab_startup_sec is not None:
            print(f"[BENCH] Of Other(s), ~{matlab_startup_sec:.1f}s is unavoidable MATLAB process startup.")
        if python_startup_sec is not None:
            print(f"[BENCH] Of Other(s), ~{python_startup_sec:.1f}s is unavoidable Python startup + imports.")

        # A runtime comparison only means something if both sides fit the same voxels.
        compared = {}
        for item in results:
            if item.get("status") != "OK":
                continue
            _, count = _voxel_summary(item)
            compared[item["name"]] = count
        known = [count for count in compared.values() if count is not None]
        unknown = any(count is None for count in compared.values())
        spread = (max(known) - min(known)) / max(known) if len(known) > 1 and max(known) > 0 else 0.0
        # A handful of voxels is a parity discrepancy worth reporting, but it does not
        # make the runtimes measure different workloads; a material difference does.
        not_comparable = unknown or spread > VOXEL_COUNT_COMPARABLE_TOLERANCE
        voxel_mismatch = unknown or (len(set(known)) > 1)
        if compared and voxel_mismatch:
            level = "WARN" if not_comparable else "NOTE"
            print(f"\n[BENCH][{level}] Configurations did not fit the same number of voxels:")
            for name, count in compared.items():
                print(f"[BENCH][{level}]   {name}: {'unknown' if count is None else f'{count:,}'}")
            if not_comparable:
                print(
                    f"[BENCH][WARN] Spread is {spread:.1%} -- these rows are not comparable, "
                    "the runtimes measure different workloads."
                )
            else:
                print(
                    f"[BENCH][NOTE] Spread is {spread:.2%}, small enough that the runtimes remain "
                    "comparable, but the counts should match exactly -- this is a parity discrepancy."
                )

        payload = {
            "repo_root": str(REPO_ROOT),
            "dataset_root": str(dataset_root),
            "subject_source_path": str(source_root),
            "subject_tp_path": str(tp_root),
            "derivatives_resolution": deriv_note,
            "inputs": {key: str(path) for key, path in files.items()},
            "shared_stage_overrides": shared_overrides,
            "models": model_flags,
            "repeats": repeats,
            "timeout_sec": int(args.timeout_sec),
            "startup_floor_sec": {"matlab": matlab_startup_sec, "python": python_startup_sec},
            "probes": {
                "python_default": python_probe_default,
                "python_no_cuda": python_probe_no_cuda,
                "matlab": matlab_probe,
            },
            "results": results,
            "voxel_counts": compared,
            "voxel_count_mismatch": bool(compared and voxel_mismatch),
            "voxel_count_spread": spread,
            "configs_comparable": not (compared and not_comparable),
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
