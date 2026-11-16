#!/usr/bin/env python3
"""Processing-variance runner: refit sessions while changing one processing choice at a time.

Comparison 2 of the test-retest plan. Each session starts from the literal config its fit
ran with (run_config.json, written by run_testretest_fits.py). A factors file lists the
factors and their levels; every level is one refit that differs from the base in that
factor only. The spread of the ROI medians across a factor's levels, pooled over sessions,
is that factor's processing SD.

    run_testretest_processing_variance.py \\
        --fit-root <fits>/python_AC --dceprep-run <derivatives>/dceprep-testretest_AC \\
        --rater-run WN=<derivatives>/dceprep-testretest_WN \\
        --subject sub-X --out <fits>/processing_variance_AC

Relating the processing SD to the scan-rescan wSD needs a real repeatability result, so
--rescan-metrics is refused until the pre-registration is signed off.

Plan: project-management/projects/test-retest-validation/test_retest_validation.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import repeatability as rp  # noqa: E402
import testretest_study as study  # noqa: E402

PROJECT_DIR = REPO_ROOT / "project-management/projects/test-retest-validation"
DEFAULT_FACTORS = PROJECT_DIR / "processing_factors.json"
LEVEL_COLUMNS = ("subject", "session", "factor", "level", "status", "model", "region", "parameter", "value")
SD_COLUMNS = ("factor", "model", "region", "parameter", "n_sessions", "n_levels", "processing_sd", "rescan_wSD", "sd_ratio", "variance_ratio")


def _strip_nii(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def _sidecar(image: Path) -> Path:
    return image.with_name(_strip_nii(image.name) + ".json")


def _morph(mask: np.ndarray, operation: str) -> np.ndarray:
    """Grow or shrink a mask by one voxel in-plane (the AIF is drawn slice by slice)."""
    from scipy import ndimage

    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[:, 1, 1] = structure[1, :, 1] = True
    if operation == "dilate":
        return ndimage.binary_dilation(mask, structure)
    if operation == "erode":
        return ndimage.binary_erosion(mask, structure)
    raise ValueError(f"unknown aif_mask operation '{operation}'")


def stage_aif(
    level: Mapping[str, Any], base_aif: Path, variant_dir: Path, rater_runs: Mapping[str, Path],
    subject: str, session: str, blood_t1_ms: Optional[float], dynamic: Optional[Path] = None,
) -> Tuple[Optional[Path], str]:
    """Write the AIF file a level asks for into its folder. Returns (path, "") or (None, reason).

    The pipeline reads the baseline end from a JSON sidecar next to the AIF file, so which
    sidecar travels with the file is part of the level: 'baseline' is 'as_drawn' (keep the
    source's sidecar), 'auto' (no sidecar: auto-detect) or {'from': RATER}.
    """
    import nibabel as nib

    source = base_aif
    if level.get("aif_from"):
        run = rater_runs.get(str(level["aif_from"]))
        if run is None:
            return None, f"no --rater-run given for {level['aif_from']}"
        matches = sorted((run / subject / session / "dce").glob("*_label-AIF_T1map.nii*"))
        if len(matches) != 1:
            return None, f"{len(matches)} AIF files for rater {level['aif_from']}"
        source = matches[0]
    image = nib.load(str(source))
    if tuple(image.shape) != tuple(nib.load(str(base_aif)).shape):
        return None, "AIF grid differs from the base session's"
    data = np.asarray(image.get_fdata())
    if dynamic is not None and (level.get("aif_from") or level.get("aif_mask") == "dilate"):
        # DCEPrep masks the series by brain-or-AIF of the rater it ran for, so a voxel
        # another rater drew, or one added by dilation, may hold no signal in this run
        voxels = np.isfinite(data) & (data > 0)
        if level.get("aif_mask"):
            voxels = _morph(voxels, str(level["aif_mask"]))
        series = np.asarray(nib.load(str(dynamic)).dataobj)[voxels]
        dark = int((~(np.abs(series) > 0).any(axis=1)).sum())
        if dark:
            return None, f"{dark} of {int(voxels.sum())} AIF voxels have no signal in this run's preprocessed series"
    if level.get("aif_mask"):
        selected = np.isfinite(data) & (data > 0)
        changed = _morph(selected, str(level["aif_mask"]))
        if not changed.any():
            return None, f"{level['aif_mask']} leaves no AIF voxel"
        # voxels added by dilation need a T1: the blood T1 the run uses, else the mask's mean
        fill = blood_t1_ms if blood_t1_ms else float(data[selected].mean())
        data = np.where(changed, np.where(selected, data, fill), 0.0)
    target = variant_dir / "aif" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), image.affine, image.header), str(target))

    baseline = level.get("baseline", "as_drawn")
    sidecar_source: Optional[Path] = None
    if baseline == "as_drawn":
        sidecar_source = _sidecar(source)
    elif isinstance(baseline, Mapping):
        run = rater_runs.get(str(baseline["from"]))
        if run is None:
            return None, f"no --rater-run given for {baseline['from']}"
        others = sorted((run / subject / session / "dce").glob("*_label-AIF_T1map.nii*"))
        sidecar_source = _sidecar(others[0]) if len(others) == 1 else None
    elif baseline != "auto":
        return None, f"unknown baseline '{baseline}'"
    if sidecar_source is not None and sidecar_source.exists():
        shutil.copyfile(sidecar_source, _sidecar(target))
    return target, ""


def run_level(base: Mapping[str, Any], level: Mapping[str, Any], variant_dir: Path, aif: Path) -> Tuple[str, str]:
    """Refit one level. Returns (status, detail)."""
    config = json.loads(json.dumps(base))
    config.update(level.get("top", {}))
    config["stage_overrides"] = {**config.get("stage_overrides", {}), **level.get("set", {})}
    config["output_dir"] = str(variant_dir)
    config["aif_files"] = [str(aif)]
    config.pop("checkpoint_dir", None)
    config_path = variant_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, str(REPO_ROOT / "run_dce_python_cli.py"), "--config", str(config_path)]
    with open(variant_dir / "fit.log", "w", encoding="utf-8") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0:
        lines = (variant_dir / "fit.log").read_text(encoding="utf-8", errors="replace").splitlines()
        return "failed", (lines[-1] if lines else f"exit {code}")[:300]
    return "ok", ""


def processing_sd_table(level_rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Pooled within-session SD across levels, per factor x model x region x parameter.

    A session counts for a factor only if it has a finite value at two or more levels.
    """
    grouped: Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], List[float]]] = {}
    for row in level_rows:
        if row["status"] != "ok" or not np.isfinite(float(row["value"])):
            continue
        key = (row["factor"], row["model"], row["region"], row["parameter"])
        grouped.setdefault(key, {}).setdefault((row["subject"], row["session"]), []).append(float(row["value"]))
    table: List[Dict[str, Any]] = []
    for key, sessions in sorted(grouped.items()):
        usable = [values for values in sessions.values() if len(values) >= 2]
        if not usable:
            continue
        table.append({
            "factor": key[0], "model": key[1], "region": key[2], "parameter": key[3],
            "n_sessions": len(usable), "n_levels": max(len(v) for v in usable),
            "processing_sd": rp.processing_sd(usable),
        })
    return table


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-root", type=Path, required=True, help="Fits from run_testretest_fits.py (base configs).")
    parser.add_argument("--dceprep-run", type=Path, required=True, help="DCEPrep run the fits used (region masks).")
    parser.add_argument("--rater-run", action="append", default=[], metavar="RATER=PATH", help="Another rater's DCEPrep run, for levels with aif_from / baseline from.")
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTORS)
    parser.add_argument("--factor", action="append", default=[], help="Only this factor (repeatable).")
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--regions-json", type=Path, help='Region masks as {"regions": {name: glob}, "fit_roi": glob}; default WM/GM.')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rescan-metrics", type=Path, help="metrics.csv of an unblinded report; adds the ratio to the scan-rescan wSD.")
    parser.add_argument("--rescan-cohort", default=study.POOLED, help="Cohort row of --rescan-metrics to compare with.")
    parser.add_argument("--rescan-rater", help="Rater row of --rescan-metrics to compare with.")
    parser.add_argument("--prereg", type=Path, default=PROJECT_DIR / "prereg.json")
    args = parser.parse_args(argv)

    if args.rescan_metrics:
        if not args.rescan_rater:
            parser.error("--rescan-metrics needs --rescan-rater")
        try:
            study.require_unblinded(study.load_prereg(args.prereg))
        except study.BlindingError as error:
            print(f"REFUSED: {error}", file=sys.stderr)
            return 2
    factors = json.loads(args.factors.read_text(encoding="utf-8"))["factors"]
    if args.factor:
        unknown = set(args.factor) - set(factors)
        if unknown:
            parser.error(f"unknown factor(s): {sorted(unknown)}")
        factors = {name: factors[name] for name in args.factor}
    rater_runs = {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in args.rater_run}
    collect_kwargs: Dict[str, Any] = {}
    if args.regions_json:
        spec = json.loads(args.regions_json.read_text(encoding="utf-8"))
        collect_kwargs = {"regions": spec["regions"], "fit_roi": spec["fit_roi"]}
        if "model_parameters" in spec:
            collect_kwargs["model_parameters"] = spec["model_parameters"]

    level_rows: List[Dict[str, Any]] = []
    sessions = sorted(p.parent for p in args.fit_root.glob("sub-*/ses-*/run_config.json"))
    sessions = [
        p for p in sessions
        if (not args.subject or p.parent.name in args.subject) and (not args.session or p.name in args.session)
    ]
    for fit_dir in sessions:
        subject, session = fit_dir.parent.name, fit_dir.name
        base = json.loads((fit_dir / "run_config.json").read_text(encoding="utf-8"))
        base_aif = Path(base["aif_files"][0])
        blood_t1 = base.get("stage_overrides", {}).get("blood_t1_ms")
        dceprep_session = args.dceprep_run / subject / session
        for factor, levels in factors.items():
            for level in levels:
                variant_dir = args.out / subject / session / factor / str(level["name"])
                variant_dir.mkdir(parents=True, exist_ok=True)
                aif, reason = stage_aif(level, base_aif, variant_dir, rater_runs, subject, session, blood_t1, Path(base["dynamic_files"][0]))
                status, detail = ("skipped", reason) if aif is None else run_level(base, level, variant_dir, aif)
                print(f"{subject} {session} {factor}={level['name']}: {status} {detail}", flush=True)
                rows: List[Dict[str, Any]] = []
                if status == "ok":
                    rows, _, problems = study.collect_session(variant_dir, dceprep_session, pipeline="python", rater="", **collect_kwargs)
                    for problem in problems:
                        print(f"  {problem}")
                common = {"subject": subject, "session": session, "factor": factor, "level": level["name"], "status": status}
                level_rows += [{**common, **{k: r[k] for k in ("model", "region", "parameter", "value")}} for r in rows] or [
                    {**common, "model": "", "region": "", "parameter": "", "value": float("nan")}
                ]
    study.write_table(args.out / "processing_levels.csv", level_rows, LEVEL_COLUMNS)

    table = processing_sd_table(level_rows)
    if args.rescan_metrics:
        rescan = {
            (m["model"], m["region"], m["parameter"]): float(m["wSD"])
            for m in study.read_table(args.rescan_metrics)
            if m["pipeline"] == "python" and m["cohort"] == args.rescan_cohort and m["rater"] == args.rescan_rater
        }
        for row in table:
            wsd = rescan.get((row["model"], row["region"], row["parameter"]))
            if wsd:
                budget = rp.variance_budget({row["factor"]: row["processing_sd"]}, wsd)[0]
                row.update(rescan_wSD=wsd, sd_ratio=budget["sd_ratio"], variance_ratio=budget["variance_ratio"])
    study.write_table(args.out / "processing_sd.csv", table, SD_COLUMNS)
    print(f"{len(sessions)} session(s), {len(factors)} factor(s) -> {args.out / 'processing_sd.csv'} (values not shown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
