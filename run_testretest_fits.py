#!/usr/bin/env python3
"""Fit every session of a DCEPrep rater run with the Python pipeline, one config for all.

The Python arm of the test-retest validation. Inputs are read from a DCEPrep run
(derivatives/dceprep-testretest_<RATER>) and never written to; outputs go to a separate
root, one folder per session, each with the literal config it ran and its log.

    run_testretest_fits.py \\
        --dceprep-run  <DCE_BIDS>/derivatives/dceprep-testretest_AC \\
        --staged-root  <DCE_BIDS>/sourcedata/raw_dceprep_testretest \\
        --output-root  <DCE_BIDS>/derivatives/rocketship-testretest/python_AC

Plan: project-management/projects/test-retest-validation/test_retest_validation.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import version  # noqa: E402

DEFAULT_CONFIG = (
    REPO_ROOT / "project-management/projects/test-retest-validation/testretest_run_config.json"
)
INPUT_LISTS = ("dynamic_files", "aif_files", "roi_files", "t1map_files")
MANIFEST_COLUMNS = ("subject", "session", "status", "detail", "duration_sec", "git_revision")


def build_session_config(
    template: Dict[str, Any], session_dir: Path, staged_session: Path, output_dir: Path,
    regions_roots: Sequence[Path] = (),
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Resolve the template for one session. Returns (config, "") or (None, reason).

    `roi_files` may list several patterns: each must match exactly one file, in the session
    or else under one of `regions_roots` (<root>/sub-*/ses-*), and they are passed in order.
    The first is the fit ROI; with `fit_voxels` false every ROI's averaged curve is fitted
    (the `_rois.xls` outputs). The other inputs take one file in all.
    """
    subject, session = session_dir.parent.name, session_dir.name
    config = {k: v for k, v in template.items() if k not in ("_doc", "cohort_overrides")}
    for key in INPUT_LISTS:
        if key == "roi_files" and len(template[key]) > 1:
            resolved: List[str] = []
            for pattern in template[key]:
                for folder in (session_dir, *(Path(r) / subject / session for r in regions_roots)):
                    found = sorted(folder.glob(pattern))
                    if found:
                        break
                if len(found) != 1:
                    return None, f"{key}: {len(found)} files match {pattern}, need exactly 1"
                resolved.append(str(found[0]))
            config[key] = resolved
            continue
        matches: List[Path] = []
        for pattern in template[key]:
            matches.extend(sorted(session_dir.glob(pattern)))
        if len(matches) != 1:
            return None, f"{key}: {len(matches)} files match {template[key]}, need exactly 1"
        config[key] = [str(matches[0])]
    sidecars = sorted((staged_session / "dce").glob("*_DCE.json"))
    if len(sidecars) != 1:
        return None, f"staged DCE sidecar: {len(sidecars)} found under {staged_session / 'dce'}"

    overrides = dict(template.get("stage_overrides", {}))
    prefixes = [p for p in template.get("cohort_overrides", {}) if subject.startswith(p)]
    if template.get("cohort_overrides") and not prefixes:
        return None, f"no cohort_overrides entry matches {subject}"
    if prefixes:
        overrides.update(template["cohort_overrides"][max(prefixes, key=len)])
    overrides["rootname"] = f"{subject}_{session}"
    overrides["dce_metadata_path"] = str(sidecars[0])

    config.update(
        subject_source_path=str(staged_session),
        subject_tp_path=str(session_dir),
        output_dir=str(output_dir),
        stage_overrides=overrides,
    )
    return config, ""


def fit_session(config: Dict[str, Any], output_dir: Path) -> Tuple[str, str]:
    """Run the DCE CLI on one resolved config. Returns (status, detail)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(REPO_ROOT / "run_dce_python_cli.py"),
        "--config",
        str(config_path),
        "--verbosity",
        "detailed",
    ]
    with open(output_dir / "fit.log", "w", encoding="utf-8") as log:
        returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if returncode != 0:
        lines = (output_dir / "fit.log").read_text(encoding="utf-8", errors="replace").splitlines()
        return "failed", (lines[-1] if lines else f"exit {returncode}")[:300]
    return "ok", ""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dceprep-run", type=Path, required=True, help="DCEPrep run folder holding sub-*/ses-*.")
    parser.add_argument("--staged-root", type=Path, required=True, help="Staged raw data root (DCE sidecars).")
    parser.add_argument("--output-root", type=Path, required=True, help="Where fits are written; must differ from --dceprep-run.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Run config template.")
    parser.add_argument("--subject", action="append", default=[], help="Only this subject (repeatable).")
    parser.add_argument("--session", action="append", default=[], help="Only this session (repeatable).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sessions the manifest already lists as ok.")
    parser.add_argument("--allow-dirty", action="store_true", help="Run from a checkout with uncommitted changes.")
    parser.add_argument(
        "--regions-root", action="append", default=[], type=Path,
        help="Another run to look for roi_files in (e.g. regions_SG for the tumor ROI); repeatable.",
    )
    args = parser.parse_args(argv)

    dceprep_run = args.dceprep_run.resolve()
    output_root = args.output_root.resolve()
    if output_root == dceprep_run or dceprep_run in output_root.parents:
        parser.error("--output-root must be outside --dceprep-run: the rater runs are inputs")
    identity = version.build_identity()
    revision = str(identity.get("git_revision"))
    if revision.endswith("-dirty") and not args.allow_dirty:
        parser.error(f"checkout is {revision}; commit first so the fits record a real commit")

    template = json.loads(args.config.read_text(encoding="utf-8"))
    template_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "fit_manifest.tsv"
    done: Dict[Tuple[str, str], Dict[str, str]] = {}
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            done = {(r["subject"], r["session"]): r for r in csv.DictReader(handle, delimiter="\t")}

    sessions = sorted(p for p in dceprep_run.glob("sub-*/ses-*") if p.is_dir())
    sessions = [
        p for p in sessions
        if (not args.subject or p.parent.name in args.subject) and (not args.session or p.name in args.session)
    ]
    print(f"ROCKETSHIP {identity.get('version')} {revision}; {len(sessions)} sessions from {dceprep_run}")
    for i, session_dir in enumerate(sessions, 1):
        key = (session_dir.parent.name, session_dir.name)
        if args.skip_existing and done.get(key, {}).get("status") == "ok":
            continue
        output_dir = output_root / key[0] / key[1]
        started = time.monotonic()
        config, reason = build_session_config(
            template, session_dir, args.staged_root.resolve() / key[0] / key[1], output_dir,
            [r.resolve() for r in args.regions_root],
        )
        status, detail = ("skipped", reason) if config is None else fit_session(config, output_dir)
        done[key] = {
            "subject": key[0],
            "session": key[1],
            "status": status,
            "detail": detail,
            "duration_sec": f"{time.monotonic() - started:.1f}",
            "git_revision": revision,
        }
        print(f"[{i}/{len(sessions)}] {key[0]} {key[1]}: {status} {detail}", flush=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, MANIFEST_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(done[k] for k in sorted(done))

    provenance = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rocketship": identity,
        "dceprep_run": str(dceprep_run),
        "dceprep_run_provenance": "PROVENANCE.md in that folder records the DCEPrep commit",
        "staged_root": str(args.staged_root.resolve()),
        "config_template": str(args.config.resolve()),
        "config_template_sha256": template_sha,
        "command": [Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
        "status_counts": {
            s: sum(1 for r in done.values() if r["status"] == s) for s in ("ok", "failed", "skipped")
        },
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["status_counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
