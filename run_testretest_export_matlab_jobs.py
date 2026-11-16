#!/usr/bin/env python3
"""Export what the Python fits used, as a job list for the MATLAB arm on another machine.

The MATLAB arm has to run on the same inputs with the same settings as the Python arm.
Rather than restate those settings by hand, this reads every fitted session's own record
(run_config.json and dce_pipeline_run.json) and writes one job per session: the four input
files and the values Stage A resolved -- TR, flip angle, frame spacing, relaxivity,
hematocrit, blood T1, the frame window and the last baseline frame. Paths are stored
relative to the DCE BIDS root, so the job list works wherever that share is mounted.

    run_testretest_export_matlab_jobs.py --rater AC \\
        --fit-root  <DCE_BIDS>/derivatives/rocketship-testretest/python_AC \\
        --bids-root <DCE_BIDS> \\
        --out       <DCE_BIDS>/derivatives/rocketship-testretest/matlab_jobs_AC.json

Run the jobs with project-management/projects/test-retest-validation/matlab/
run_testretest_matlab.m (see matlab_arm.md in that project folder).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

AIF_TYPE = {"fitted": 1, "raw": 2}  # script_preferences.txt: raw = 2, fitted = 1
INPUTS = {"dynamic": "dynamic_files", "aif": "aif_files", "roi": "roi_files", "t1map": "t1map_files"}


def session_job(fit_dir: Path, bids_root: Path) -> Dict[str, Any]:
    """One MATLAB job from one Python fit folder. Raises ValueError if it cannot be matched."""
    config = json.loads((fit_dir / "run_config.json").read_text(encoding="utf-8"))
    run = json.loads((fit_dir / "dce_pipeline_run.json").read_text(encoding="utf-8"))
    stage_a = run["stages"]["A"]
    if config["aif_mode"] not in AIF_TYPE:
        raise ValueError(f"aif_mode '{config['aif_mode']}' has no MATLAB aif_type")
    if stage_a.get("blood_t1_override_sec") is None:
        raise ValueError("the Python fit took blood T1 from the AIF T1 map; the MATLAB driver expects a fixed blood T1")

    job: Dict[str, Any] = {"subject": fit_dir.parent.name, "session": fit_dir.name}
    job["rootname"] = config["stage_overrides"].get("rootname", f"{job['subject']}_{job['session']}")
    for name, key in INPUTS.items():
        files = config[key]
        if len(files) != 1:
            raise ValueError(f"{key} lists {len(files)} files")
        # not resolved: a derived rater run links its preprocessed files from the base run,
        # and the job should name the file the fit was given
        path = Path(files[0])
        try:
            job[name] = path.relative_to(bids_root).as_posix()
        except ValueError:
            raise ValueError(f"{path} is not under --bids-root {bids_root}") from None

    window = stage_a.get("timepoint_window", {})
    full_range = window.get("source") == "default_full_range"
    baseline = stage_a.get("steady_state_auto", {}) or {}
    job.update(
        tr_ms=float(stage_a["tr_ms"]),
        fa_deg=float(stage_a["fa_deg"]),
        time_resolution_sec=float(stage_a["time_resolution_min"]) * 60.0,
        relaxivity=float(stage_a["relaxivity"]),
        hematocrit=float(stage_a["hematocrit"]),
        blood_t1_sec=float(stage_a["blood_t1_override_sec"]),
        # null = the whole series, which is what MATLAB gets from an empty start_t / end_t
        start_t=None if full_range else int(window["start_1b"]),
        end_t=None if full_range else int(window["end_1b"]),
        n_frames=int(window.get("n_timepoints_input", 0)) or None,
        # last baseline frame, 1-based, as the Python fit used it
        steady_state_end=int(stage_a["steady_state_time"][1]),
        # 'aif_sidecar' = marked by the rater; anything else was auto-detected by Python
        steady_state_source=str(baseline.get("method_used", "unknown")),
        aif_type=AIF_TYPE[config["aif_mode"]],
        models={k: int(v) for k, v in config["model_flags"].items() if int(v)},
        python_git_revision=run.get("provenance", {}).get("git_revision"),
    )
    return job


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-root", type=Path, required=True, help="Python fits from run_testretest_fits.py.")
    parser.add_argument("--bids-root", type=Path, required=True, help="DCE BIDS root; job paths are relative to it.")
    parser.add_argument("--rater", required=True, help="Rater label recorded in the job list.")
    parser.add_argument("--out", type=Path, required=True, help="Job list JSON to write.")
    args = parser.parse_args(argv)

    bids_root = args.bids_root.absolute()
    ok = None
    manifest = args.fit_root / "fit_manifest.tsv"
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as handle:
            ok = {(r["subject"], r["session"]) for r in csv.DictReader(handle, delimiter="\t") if r["status"] == "ok"}
    jobs: List[Dict[str, Any]] = []
    left_out: List[str] = []
    for fit_dir in sorted(p.parent for p in args.fit_root.glob("sub-*/ses-*/dce_pipeline_run.json")):
        key = (fit_dir.parent.name, fit_dir.name)
        if ok is not None and key not in ok:
            left_out.append(f"{key[0]} {key[1]}: Python fit not ok")
            continue
        try:
            jobs.append(session_job(fit_dir.absolute(), bids_root))
        except (ValueError, KeyError) as error:
            left_out.append(f"{key[0]} {key[1]}: {error}")
    payload = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rater": args.rater,
        "fit_root": args.fit_root.absolute().relative_to(bids_root).as_posix(),
        "paths_relative_to": "the DCE BIDS root (pass it to run_testretest_matlab as bids_root)",
        "jobs": jobs,
        "left_out": left_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{len(jobs)} jobs, {len(left_out)} left out -> {args.out}")
    for line in left_out:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
