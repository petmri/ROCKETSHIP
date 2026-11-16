#!/usr/bin/env python3
"""Test-retest validation: collect ROI medians, audit metadata, run the blinded analysis.

    # 1. one tidy table from every pipeline arm (repeat --arm; pipeline:rater:fits:dceprep-run)
    run_testretest_analysis.py collect --out tables \\
        --arm python:AC:<fits>/python_AC:<derivatives>/dceprep-testretest_AC \\
        --arm matlab:AC:<fits>/matlab_AC:<derivatives>/dceprep-testretest_AC

    # 2. acquisition metadata, session against session
    run_testretest_analysis.py audit --fit-root <fits>/python_AC --staged-root <staged> --out tables

    # 3. the analysis. 'shuffled' scrambles pairing and pipeline labels; 'unblinded' is
    #    refused until prereg.json is complete and signed off.
    run_testretest_analysis.py analyse --mode shuffled --table tables/roi_medians.csv \\
        --session-qc tables/session_qc.csv --out reports

    # code check on generated data with known answers
    run_testretest_analysis.py selfcheck

Plan: project-management/projects/test-retest-validation/test_retest_validation.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import repeatability as rp  # noqa: E402
import testretest_study as study  # noqa: E402
import version  # noqa: E402

PROJECT_DIR = REPO_ROOT / "project-management/projects/test-retest-validation"
DEFAULT_PREREG = PROJECT_DIR / "prereg.json"


def cmd_collect(args: argparse.Namespace) -> int:
    rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for arm in args.arm:
        try:
            pipeline, rater, fit_root, dceprep_run = arm.split(":", 3)
        except ValueError:
            raise SystemExit(f"--arm must be pipeline:rater:fit_root:dceprep_run, got '{arm}'")
        arm_rows, arm_qc, arm_problems = study.collect_arm(
            Path(fit_root), Path(dceprep_run), pipeline=pipeline, rater=rater
        )
        print(f"{pipeline} / {rater}: {len(arm_qc)} sessions, {len(arm_rows)} ROI medians, {len(arm_problems)} problems")
        rows += arm_rows
        qc_rows += arm_qc
        problems += arm_problems
    out = Path(args.out)
    study.write_table(out / "roi_medians.csv", rows, study.TABLE_COLUMNS)
    study.write_table(out / "session_qc.csv", qc_rows, study.SESSION_QC_COLUMNS)
    (out / "collect_problems.txt").write_text("\n".join(problems) + ("\n" if problems else ""), encoding="utf-8")
    print(f"wrote {out / 'roi_medians.csv'} (values not shown: the table is blinded input)")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    rows = study.metadata_audit(Path(args.fit_root), Path(args.staged_root) if args.staged_root else None)
    out = Path(args.out) / "metadata_audit.csv"
    study.write_table(out, rows, ("subject", "cohort", "sessions", "flags"))
    flagged = [row for row in rows if row["flags"]]
    print(f"{len(rows)} subjects, {len(flagged)} with a difference between sessions -> {out}")
    for row in flagged:
        print(f"  {row['subject']}: {row['flags']}")
    return 0


def _subset(rows: List[Dict[str, Any]], prereg: Dict[str, Any], which: str) -> List[Dict[str, Any]]:
    if which == "all":
        return rows
    by_cohort: Dict[str, set] = {}
    for row in rows:
        by_cohort.setdefault(str(row["cohort"]), set()).add(str(row["subject"]))
    split = rp.stratified_split(
        by_cohort, float(prereg["split"]["development_fraction"]), int(prereg["split"]["seed"])
    )
    return [row for row in rows if split[str(row["subject"])] == which]


def cmd_analyse(args: argparse.Namespace) -> int:
    prereg = study.load_prereg(Path(args.prereg))
    mode = args.mode
    identity = version.build_identity()
    notes = [f"ROCKETSHIP {identity.get('version')} at {identity.get('git_revision')}", f"pre-registration {args.prereg} sha256 {study.file_sha256(Path(args.prereg))[:16]}"]
    if mode == "unblinded":
        try:
            study.require_unblinded(prereg)
        except study.BlindingError as error:
            print(f"REFUSED: {error}", file=sys.stderr)
            return 2
    else:
        missing = study.missing_decisions(prereg)
        if missing:
            notes.append("pre-registration still undecided: " + ", ".join(missing))

    exclusions: List[Dict[str, Any]] = []
    if mode == "synthetic":
        rows = study.synthetic_table({"A": 24, "B": 19}, {"python": 0.30, "matlab": 0.30}, seed=args.seed)
        notes.append("synthetic table: 24 + 19 subjects, true wSD 0.30 in both pipelines (ratio 1)")
    else:
        if not args.table:
            raise SystemExit("--table is required for the shuffled and unblinded modes")
        rows = study.read_roi_table(Path(args.table))
        notes.append(f"table {args.table} sha256 {study.file_sha256(Path(args.table))[:16]}, {len(rows)} rows")
        session_qc = study.read_table(Path(args.session_qc), numeric=("max_displacement_mm", "aif_fit_r2_adj")) if args.session_qc else []
        manual = study.read_table(Path(args.manual_exclusions)) if args.manual_exclusions else []
        rows, exclusions = study.apply_exclusions(rows, session_qc, prereg.get("exclusions", {}), manual)
        if mode == "unblinded":
            rows = _subset(rows, prereg, args.subset)
            notes.append(f"subset: {args.subset}")
        else:
            rows = study.shuffle_labels(rows, seed=args.seed)
            notes.append(f"labels shuffled with seed {args.seed}: second sessions exchanged between subjects, pipeline labels swapped at random")

    placeholders: List[str] = []
    settings = {
        key: study.decision(prereg, f"noninferiority.{key}", mode, placeholders)
        for key in ("margin", "confidence", "interval")
    }
    if placeholders:
        notes.append("placeholders for undecided settings (not decisions): " + "; ".join(placeholders))
    bootstrap = prereg.get("bootstrap", {})
    results = study.analyse(
        rows,
        margin=float(settings["margin"]),
        confidence=float(settings["confidence"]),
        interval=str(settings["interval"]),
        report_wcv_for=prereg.get("report_wcv_for", []),
        primary_endpoint=prereg.get("primary_endpoint"),
        n_boot=int(args.n_boot or bootstrap.get("n_boot", 5000)),
        seed=int(bootstrap.get("seed", 0)),
    )
    out_dir = Path(args.out) / mode
    report = study.write_report(out_dir, results, mode=mode, exclusions=exclusions, notes=notes, plots=not args.no_plots)
    print(study.BANNERS[mode])
    print(f"{len(results['metrics'])} groups, {len(results['noninferiority'])} pipeline comparisons, "
          f"{len(exclusions)} exclusions, {len(results['dropped'])} incomplete -> {report}")
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """End-to-end check on generated data with known answers. Exit 1 on any failure."""
    failures: List[str] = []

    def check(label: str, ok: bool, detail: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")
        if not ok:
            failures.append(label)

    def run(table, **kwargs):
        return study.analyse(table, margin=1.1, confidence=0.95, interval="pitman", n_boot=400, seed=1, **kwargs)

    def wsd(results, pipeline, cohort="A"):
        return next(m["wSD"] for m in results["metrics"] if m["pipeline"] == pipeline and m["cohort"] == cohort)

    equal = run(study.synthetic_table({"A": 600, "B": 600}, {"python": 0.30, "matlab": 0.30}, seed=1))
    check("wSD recovered (python)", abs(wsd(equal, "python") - 0.30) < 0.02, f"{wsd(equal, 'python'):.3f} vs 0.300")
    check("wSD recovered (matlab)", abs(wsd(equal, "matlab") - 0.30) < 0.02, f"{wsd(equal, 'matlab'):.3f} vs 0.300")
    ni = next(r for r in equal["noninferiority"] if r["cohort"] == "A")
    check("equal pipelines are non-inferior", ni["non_inferior"] and abs(ni["wSD_ratio"] - 1) < 0.05, f"ratio {ni['wSD_ratio']:.3f}, upper {ni['pitman_ci_high']:.3f}")
    pooled = next(r for r in equal["noninferiority"] if r["cohort"] == study.POOLED)
    check("pooled cohort analysed", pooled["n"] == 1200, f"n = {pooled['n']}")

    worse_table = study.synthetic_table({"A": 600, "B": 600}, {"python": 0.45, "matlab": 0.30}, seed=2)
    worse = run(worse_table)
    ni = next(r for r in worse["noninferiority"] if r["cohort"] == "A")
    check("a 1.5x worse pipeline is not non-inferior", (not ni["non_inferior"]) and abs(ni["wSD_ratio"] - 1.5) < 0.08, f"ratio {ni['wSD_ratio']:.3f}, upper {ni['pitman_ci_high']:.3f}")

    scrambled = run(study.shuffle_labels(worse_table, seed=3))
    ni = next(r for r in scrambled["noninferiority"] if r["cohort"] == "A")
    check("shuffling hides the pipeline difference", abs(ni["wSD_ratio"] - 1) < 0.1, f"ratio {ni['wSD_ratio']:.3f} (true 1.5)")
    check("shuffling destroys the pairing", wsd(scrambled, "matlab") > 2 * 0.30, f"wSD {wsd(scrambled, 'matlab'):.3f} (true 0.30, between-subject SD 1.0)")

    try:
        study.require_unblinded(study.load_prereg(DEFAULT_PREREG))
        state = "complete: the unblinded mode is open"
    except study.BlindingError:
        state = "incomplete: the unblinded mode is locked"
    print(f"INFO  committed pre-registration is {state}")
    print("selfcheck " + ("FAILED: " + ", ".join(failures) if failures else "passed"))
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="ROI medians of every arm into one tidy table")
    collect.add_argument("--arm", action="append", required=True, metavar="PIPELINE:RATER:FIT_ROOT:DCEPREP_RUN")
    collect.add_argument("--out", required=True, help="Folder for roi_medians.csv and session_qc.csv")
    collect.set_defaults(run=cmd_collect)

    audit = commands.add_parser("audit", help="Compare acquisition metadata between paired sessions")
    audit.add_argument("--fit-root", required=True)
    audit.add_argument("--staged-root")
    audit.add_argument("--out", required=True)
    audit.set_defaults(run=cmd_audit)

    analyse = commands.add_parser("analyse", help="Repeatability and non-inferiority, blinded unless signed off")
    analyse.add_argument("--mode", choices=study.MODES, required=True)
    analyse.add_argument("--table", help="roi_medians.csv from 'collect'")
    analyse.add_argument("--session-qc", help="session_qc.csv from 'collect'")
    analyse.add_argument("--manual-exclusions", help="TSV/CSV of subject, session, rater, reason")
    analyse.add_argument("--prereg", default=str(DEFAULT_PREREG))
    analyse.add_argument("--subset", choices=("all", "development", "held_out"), default="all", help="Unblinded mode only")
    analyse.add_argument("--seed", type=int, default=0, help="Seed of the synthetic table or of the shuffle")
    analyse.add_argument("--n-boot", type=int, help="Override the pre-registered bootstrap size (blinded modes)")
    analyse.add_argument("--no-plots", action="store_true")
    analyse.add_argument("--out", required=True, help="Reports folder; the mode becomes a subfolder")
    analyse.set_defaults(run=cmd_analyse)

    selfcheck = commands.add_parser("selfcheck", help="Check the analysis on generated data with known answers")
    selfcheck.set_defaults(run=cmd_selfcheck)

    args = parser.parse_args(argv)
    if args.command == "analyse" and args.mode == "unblinded" and args.n_boot:
        parser.error("--n-boot cannot override the pre-registration in the unblinded mode")
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
