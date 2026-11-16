#!/usr/bin/env python3
"""Build fit inputs for another AIF rater from an existing DCEPrep run, without DCEPrep.

DCEPrep's preprocessing does not depend on the rater except in one place: the dynamic
series is masked by brain-or-AIF, so an AIF voxel outside the brain mask survives only in
the run of the rater who drew it. Everything else (registration, bias correction,
z-normalization, T1 map) is the same for every rater. This script therefore takes a base
run and, for each session the new rater drew, writes a session folder that links the base
run's preprocessed files and adds the new rater's AIF:

    dce/<prefix>_label-AIF_mask.nii.gz     the rater's mask on the base run's grid
    dce/<prefix>_label-AIF_T1map.nii.gz    the T1 map inside it (what the fit reads)
    dce/<prefix>_label-AIF_T1map.json      the rater's baseline end, if marked
    anat/<prefix>_space-DCEref_label-tumor_mask.nii.gz   the rater's tumour ROI, if drawn

A session is derived only if every AIF voxel has signal in the base run's preprocessed
series. Otherwise it is listed as needing a real DCEPrep run and nothing is written.

    run_testretest_derive_rater.py --rater SG \\
        --aifartist <DCE_BIDS>/derivatives/AIFArtist \\
        --base-run  <DCE_BIDS>/derivatives/dceprep-testretest_AC \\
        --out       <DCE_BIDS>/derivatives/rocketship-testretest/dceprep-derived_SG

Plan: project-management/projects/test-retest-validation/test_retest_validation.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "project-management/projects/test-retest-validation/testretest_run_config.json"
# base-run files a fit or the ROI collection reads; AIF files are written, never linked
LINKED = (
    "anat/*_space-DCEref_label-brain_mask.nii*",
    "anat/*_space-DCEref_label-WM_mask.nii*",
    "anat/*_space-DCEref_label-GM_mask.nii*",
    "anat/*_space-DCEref_label-CSF_mask.nii*",
    "anat/*_space-DCEref_T1map.nii*",
    "dce/*_desc-bfcz_DCE.nii*",
    "dce/*_desc-hmc_maxdisp.txt",
)
MANIFEST_COLUMNS = (
    "subject", "session", "status", "detail", "aif_voxels", "aif_voxels_t1_filled",
    "baseline_end", "tumor_voxels",
)


def to_grid(mask_image: Any, reference_image: Any) -> Tuple[Optional[np.ndarray], str]:
    """Move a mask onto a grid that differs from its own by a whole-voxel shift.

    DCEPrep trims empty edge slices with fslroi, which keeps the header right, so a mask
    drawn on the untrimmed series maps onto the trimmed grid by an integer offset read from
    the two affines. Returns (mask, "") or (None, reason); voxels shifted off the grid are
    an error, not a silent loss.
    """
    transform = np.linalg.inv(reference_image.affine) @ mask_image.affine
    offset = transform[:3, 3]
    if not (np.allclose(transform[:3, :3], np.eye(3), atol=1e-3) and np.allclose(offset, np.round(offset), atol=1e-2)):
        return None, "mask grid is not a whole-voxel shift of the base run's grid"
    source = np.asarray(mask_image.dataobj) > 0
    if source.ndim != 3:
        return None, f"mask has {source.ndim} dimensions"
    index = np.argwhere(source) + np.round(offset).astype(int)
    shape = np.array(reference_image.shape[:3])
    inside = ((index >= 0) & (index < shape)).all(axis=1)
    if not inside.all():
        return None, f"{int((~inside).sum())} of {len(index)} mask voxels fall in slices the base run trimmed"
    out = np.zeros(tuple(shape), dtype=bool)
    out[tuple(index.T)] = True
    return out, ""


def _one(folder: Path, pattern: str) -> Optional[Path]:
    matches = sorted(folder.glob(pattern))
    return matches[0] if len(matches) == 1 else None


def _blood_t1_ms(config: Dict[str, Any], subject: str) -> Optional[float]:
    prefixes = [p for p in config.get("cohort_overrides", {}) if subject.startswith(p)]
    if not prefixes:
        return None
    return config["cohort_overrides"][max(prefixes, key=len)].get("blood_t1_ms")


def derive_session(
    rater: str, rater_dir: Path, base_session: Path, out_session: Path, blood_t1_ms: Optional[float]
) -> Dict[str, Any]:
    """Derive one session. Returns its manifest row; writes nothing unless status is 'ok'."""
    import nibabel as nib

    subject, session = base_session.parent.name, base_session.name
    prefix = f"{subject}_{session}"
    row: Dict[str, Any] = {"subject": subject, "session": session, "status": "skipped", "detail": ""}

    def stop(detail: str, status: str = "skipped") -> Dict[str, Any]:
        row.update(status=status, detail=detail)
        return row

    mask_path = _one(rater_dir, f"*desc-rater{rater}_label-AIF_mask.nii*")
    if mask_path is None:
        return stop(f"no single rater {rater} AIF mask")
    dynamic_path = _one(base_session, "dce/*_desc-bfcz_DCE.nii*")
    t1_path = _one(base_session, "anat/*_space-DCEref_T1map.nii*")
    if dynamic_path is None or t1_path is None:
        return stop("base run has no preprocessed series or T1 map for this session")
    dynamic = nib.load(str(dynamic_path))
    mask, reason = to_grid(nib.load(str(mask_path)), dynamic)
    if mask is None:
        return stop(reason, "needs_dceprep")
    if not mask.any():
        return stop("AIF mask is empty")
    signal = np.asarray(dynamic.dataobj)[mask]
    dark = int((~(np.abs(signal) > 0).any(axis=1)).sum())
    if dark:
        return stop(f"{dark} of {int(mask.sum())} AIF voxels lie outside the base run's brain-or-AIF mask", "needs_dceprep")

    t1_image = nib.load(str(t1_path))
    t1 = np.asarray(t1_image.get_fdata())
    if t1.shape != mask.shape:
        return stop("T1 map grid differs from the preprocessed series")
    valid = np.isfinite(t1) & (t1 > 0)
    fill = blood_t1_ms if blood_t1_ms else 1.0  # the fit replaces AIF T1 with blood_t1_ms; this only selects
    aif_t1 = np.where(mask, np.where(valid, t1, fill), 0.0).astype(np.float32)

    for pattern in LINKED:
        for source in sorted(base_session.glob(pattern)):
            target = out_session / source.relative_to(base_session)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                target.unlink()
            os.symlink(source.resolve(), target)
    (out_session / "dce").mkdir(parents=True, exist_ok=True)
    header = t1_image.header
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), dynamic.affine), str(out_session / f"dce/{prefix}_label-AIF_mask.nii.gz"))
    nib.save(nib.Nifti1Image(aif_t1, t1_image.affine, header), str(out_session / f"dce/{prefix}_label-AIF_T1map.nii.gz"))

    baseline_end = ""
    sidecar = mask_path.with_name(mask_path.name.split(".nii")[0] + ".json")
    if sidecar.exists():
        end = json.loads(sidecar.read_text(encoding="utf-8")).get("SteadyStateEndTimeIndex")
        if end is not None:
            baseline_end = int(end)
            # copied unchanged, as the staging of the other rater runs does
            (out_session / f"dce/{prefix}_label-AIF_T1map.json").write_text(
                json.dumps({"SteadyStateEndTimeIndex": baseline_end, "Source": sidecar.name}, indent=2) + "\n", encoding="utf-8"
            )

    tumor_voxels, reason = write_tumor(rater, rater_dir, dynamic, out_session, prefix)
    if reason:
        row["detail"] = f"tumour mask not written: {reason}"
    row.update(
        status="ok", aif_voxels=int(mask.sum()), aif_voxels_t1_filled=int((mask & ~valid).sum()),
        baseline_end=baseline_end, tumor_voxels=tumor_voxels,
    )
    return row


def write_tumor(rater: str, rater_dir: Path, reference: Any, out_session: Path, prefix: str) -> Tuple[Any, str]:
    """Write the rater's tumour ROI on the reference grid. Returns (voxel count or "", reason)."""
    import nibabel as nib

    tumor_path = _one(rater_dir, f"*desc-rater{rater}_label-Tumor_mask.nii*")
    if tumor_path is None:
        return "", ""
    tumor, reason = to_grid(nib.load(str(tumor_path)), reference)
    if tumor is None:
        return "", reason
    target = out_session / f"anat/{prefix}_space-DCEref_label-tumor_mask.nii.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(tumor.astype(np.uint8), reference.affine), str(target))
    return int(tumor.sum()), ""


def write_regions(rater: str, aifartist: Path, base_run: Path, regions_out: Path) -> List[Dict[str, Any]]:
    """Tumour ROIs of every session the rater drew one for, whatever became of its AIF.

    The ROI does not depend on the AIF, so sessions whose AIF needs a real DCEPrep run still
    get their tumour region. Pass the folder to 'collect --regions-root'.
    """
    import nibabel as nib

    rows: List[Dict[str, Any]] = []
    for rater_dir in sorted(aifartist.glob("sub-*/ses-*/dce")):
        if not list(rater_dir.glob(f"*desc-rater{rater}_label-Tumor_mask.nii*")):
            continue
        subject, session = rater_dir.parts[-3], rater_dir.parts[-2]
        reference_path = _one(base_run / subject / session, "dce/*_desc-bfcz_DCE.nii*")
        if reference_path is None:
            rows.append({"subject": subject, "session": session, "status": "skipped", "detail": "no preprocessed series in the base run"})
            continue
        voxels, reason = write_tumor(rater, rater_dir, nib.load(str(reference_path)), regions_out / subject / session, f"{subject}_{session}")
        rows.append({"subject": subject, "session": session, "status": "skipped" if reason else "ok", "detail": reason, "tumor_voxels": voxels})
    regions_out.mkdir(parents=True, exist_ok=True)
    with open(regions_out / "regions_manifest.tsv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ("subject", "session", "status", "detail", "tumor_voxels"), delimiter="\t", restval="")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rater", required=True, help="AIFArtist rater code, e.g. SG.")
    parser.add_argument("--aifartist", type=Path, required=True, help="derivatives/AIFArtist")
    parser.add_argument("--base-run", type=Path, required=True, help="DCEPrep run to take the preprocessing from.")
    parser.add_argument("--out", type=Path, help="Derived run folder; must be outside --base-run.")
    parser.add_argument("--regions-out", type=Path, help="Also write the rater's tumour ROIs for every session that has one, into this folder.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Run config (cohort blood T1).")
    args = parser.parse_args(argv)
    base_run = args.base_run.resolve()
    if args.out is None and args.regions_out is None:
        parser.error("give --out, --regions-out, or both")
    for target in (args.out, args.regions_out):
        if target is not None and (target.resolve() == base_run or base_run in target.resolve().parents):
            parser.error("outputs must be outside --base-run")
    if args.regions_out is not None:
        regions = write_regions(args.rater, args.aifartist, base_run, args.regions_out.resolve())
        print(f"tumour ROIs: {sum(r['status'] == 'ok' for r in regions)} written, {sum(r['status'] != 'ok' for r in regions)} skipped -> {args.regions_out}")
    if args.out is None:
        return 0
    out = args.out.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    for rater_dir in sorted(args.aifartist.glob("sub-*/ses-*/dce")):
        if not list(rater_dir.glob(f"*desc-rater{args.rater}_label-AIF_mask.nii*")):
            continue
        subject, session = rater_dir.parts[-3], rater_dir.parts[-2]
        base_session = base_run / subject / session
        if not base_session.is_dir():
            rows.append({"subject": subject, "session": session, "status": "skipped", "detail": "session is not in the base run"})
        else:
            rows.append(derive_session(args.rater, rater_dir, base_session, out / subject / session, _blood_t1_ms(config, subject)))
        print(f"{subject} {session}: {rows[-1]['status']} {rows[-1]['detail']}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "derive_manifest.tsv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, MANIFEST_COLUMNS, delimiter="\t", restval="")
        writer.writeheader()
        writer.writerows(rows)
    (out / "DERIVED.md").write_text(
        f"# Derived rater run: {args.rater}\n\n"
        f"Not a DCEPrep run. Preprocessing is linked from `{base_run}`; only the AIF mask, the AIF T1 map,\n"
        f"its baseline sidecar and the tumour ROI come from rater {args.rater} (`{args.aifartist.resolve()}`).\n"
        "Written by ROCKETSHIP run_testretest_derive_rater.py; per-session status in derive_manifest.tsv.\n",
        encoding="utf-8",
    )
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("ok", "needs_dceprep", "skipped")}
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
