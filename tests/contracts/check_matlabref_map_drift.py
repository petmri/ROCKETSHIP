#!/usr/bin/env python3
"""Fail if a freshly regenerated MATLAB region-parity baseline drifts from committed maps.

`check_baseline_drift.py` guards the synthetic-curve contract baseline
(`matlab_reference_v1.json`) but never touches the imaging pipeline (no
`A_make_R1maps_func`/`find_end_ss`). The `sub-10bbbdownsample` NIfTI maps under
`derivatives/matlabref/` (used by `test_bbb_p19_region_parity` /
`test_bbb_p19_roi_xls_parity`) had no analogous guard: CI regenerated them fresh every
run but never diffed the fresh output against the committed copy, so the committed maps
could silently go stale while the *comparison test* quietly started grading Python
against a moving target.

This script closes that gap: regenerate the same maps fresh (candidate) and compare them,
map-by-map, against the committed copies (reference). Any real drift in MATLAB's own
pipeline output (e.g. a `find_end_ss`/`A_make_R1maps_func` change) fails here, with a
message pointing at the fixture, instead of showing up later as a confusing
Python-vs-MATLAB numeric mismatch.

Usage:
    python tests/contracts/check_matlabref_map_drift.py --candidate-root /tmp/matlab_candidate
    python tests/contracts/check_matlabref_map_drift.py --candidate-root a/ --reference-root b/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ROOT = (
    REPO_ROOT / "tests/data/BIDS_test/derivatives/matlabref/sub-10bbbdownsample/ses-01/dce"
)

# MATLAB-vs-itself on identical inputs is close but *not* bit-identical: nonlinear
# voxel fits (especially confidence-interval estimates, which invert a Jacobian that can
# be near-singular in a handful of poorly-conditioned voxels) are known to be sensitive
# to MATLAB release / OS / parfor worker count, occasionally blowing up to huge values in
# a voxel or two even though the fit itself is unchanged (see dce_preferences.txt's tight
# voxel_MaxIter=voxel_MaxFunEvals=50 budget, and existing 2CXM/ve non-identifiability
# notes). Gate on rank (Spearman) correlation, not Pearson: a handful of such outliers
# barely move ranks over several thousand voxels, but Pearson is a sum-of-products
# statistic, so a single voxel a few orders of magnitude off the rest dominates it and
# can collapse it even though every other voxel agrees almost exactly (observed: one
# near-singular tofts ve CI voxel swung Pearson corr from ~1.0 to 0.21 on a platform
# where MATLAB's Jacobian inversion landed on the unstable side for that voxel). A
# genuine algorithm change (like the steady-state window bug this guard exists to catch)
# still collapses Spearman corr to ~0/negative, far below this margin. Max-abs-diff is
# still reported for debugging, just not gated.
CORR_MIN = 0.9

# Stage-B AIF contract payload (Dyn-1_stage_b_aif.json). Unlike the maps this is a
# deterministic AIF fit on a 64-sample ROI mean, not thousands of near-singular voxel
# fits, so it gets a strict absolute tolerance rather than a rank correlation. The
# committed copy is what tests/python/test_stage_b_aif_parity.py grades Python against,
# so it going stale would silently move that gate's target.
STAGE_B_CONTRACT_NAME = "Dyn-1_stage_b_aif.json"
STAGE_B_REL_TOL = 1e-6
STAGE_B_TIME_ATOL = 1e-6


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib  # type: ignore

    return np.asarray(np.squeeze(nib.load(str(path)).get_fdata()), dtype=np.float64)


def _compare_map(reference_path: Path, candidate_path: Path) -> Tuple[str, float, float, int]:
    ref = _load_nifti(reference_path)
    cand = _load_nifti(candidate_path)
    if ref.shape != cand.shape:
        return ("shape mismatch", float("nan"), float("nan"), 0)
    mask = np.isfinite(ref) & np.isfinite(cand)
    n = int(np.count_nonzero(mask))
    if n < 2:
        return ("collapsed", float("nan"), float("nan"), n)
    x, y = ref[mask], cand[mask]
    corr = float(spearmanr(x, y).correlation) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    max_abs_diff = float(np.max(np.abs(x - y)))
    return ("ok", corr, max_abs_diff, n)


def _compare_stage_b_contract(reference_path: Path, candidate_path: Path) -> List[str]:
    """Diff the Stage-B AIF contract payload field by field. Returns drift messages."""
    import json

    ref = json.loads(reference_path.read_text())
    cand = json.loads(candidate_path.read_text())
    drifts: List[str] = []

    ref_version = str(ref.get("meta", {}).get("version", ""))
    cand_version = str(cand.get("meta", {}).get("version", ""))
    if ref_version != cand_version:
        return [f"{STAGE_B_CONTRACT_NAME}: version {cand_version!r} != committed {ref_version!r}"]

    for name in sorted(set(ref.get("curves", {})) | set(cand.get("curves", {}))):
        r = np.asarray(ref["curves"].get(name, []), dtype=np.float64).ravel()
        c = np.asarray(cand["curves"].get(name, []), dtype=np.float64).ravel()
        if r.size != c.size:
            drifts.append(f"{STAGE_B_CONTRACT_NAME}/curves.{name}: length {c.size} != {r.size}")
            continue
        scale = float(np.max(np.abs(r))) if r.size else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        rel = float(np.max(np.abs(r - c))) / scale
        print(f"  {STAGE_B_CONTRACT_NAME}/curves.{name}: rel_max_abs={rel:.3e} (n={r.size})")
        if rel > STAGE_B_REL_TOL:
            drifts.append(f"{STAGE_B_CONTRACT_NAME}/curves.{name}: rel_max_abs={rel:.3e} > {STAGE_B_REL_TOL:.1e}")

    # Window terms are minutes or 1-based indices; both must reproduce exactly.
    for name in sorted(set(ref.get("window", {})) | set(cand.get("window", {}))):
        r = np.asarray(ref["window"].get(name), dtype=np.float64).ravel()
        c = np.asarray(cand["window"].get(name), dtype=np.float64).ravel()
        if r.size != c.size:
            drifts.append(f"{STAGE_B_CONTRACT_NAME}/window.{name}: length {c.size} != {r.size}")
            continue
        diff = float(np.max(np.abs(r - c))) if r.size else 0.0
        if diff > STAGE_B_TIME_ATOL:
            drifts.append(
                f"{STAGE_B_CONTRACT_NAME}/window.{name}: {c.tolist()} != committed {r.tolist()}"
            )

    for name in ("params_cp", "params_stlv"):
        r = np.asarray(ref.get("fit", {}).get(name) or [], dtype=np.float64).ravel()
        c = np.asarray(cand.get("fit", {}).get(name) or [], dtype=np.float64).ravel()
        if r.size != c.size:
            drifts.append(f"{STAGE_B_CONTRACT_NAME}/fit.{name}: length {c.size} != {r.size}")
            continue
        if r.size:
            rel = float(np.max(np.abs(r - c) / np.maximum(np.abs(r), 1e-12)))
            print(f"  {STAGE_B_CONTRACT_NAME}/fit.{name}: rel_max={rel:.3e}")
            if rel > STAGE_B_REL_TOL:
                drifts.append(f"{STAGE_B_CONTRACT_NAME}/fit.{name}: rel_max={rel:.3e} > {STAGE_B_REL_TOL:.1e}")

    return drifts


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-root", type=Path, required=True, help="Freshly regenerated maps directory.")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT, help="Committed maps directory.")
    parser.add_argument("--corr-min", type=float, default=CORR_MIN)
    args = parser.parse_args(argv)

    # Scope the comparison to whatever the candidate step actually regenerated (e.g. just
    # tofts+patlak): a committed reference map for a model the candidate step didn't
    # (re)generate this run is out of scope, not a failure.
    candidate_maps = sorted(args.candidate_root.glob("*.nii")) + sorted(args.candidate_root.glob("*.nii.gz"))
    # A stageBOnly regeneration produces the contract payload and no maps; that is a valid
    # candidate, not an empty one. Only a candidate with neither is a broken generation step.
    if not candidate_maps and not (args.candidate_root / STAGE_B_CONTRACT_NAME).exists():
        print(f"No candidate maps or Stage-B contract found under {args.candidate_root}")
        return 1

    print(f"matlabref map drift check: reference={args.reference_root} candidate={args.candidate_root} corr_min={args.corr_min:g}")

    drifts: List[str] = []
    missing: List[str] = []
    for candidate_path in candidate_maps:
        reference_path = args.reference_root / candidate_path.name
        if not reference_path.exists():
            missing.append(candidate_path.name)
            continue
        status, corr, max_abs_diff, n = _compare_map(reference_path, candidate_path)
        if status != "ok":
            drifts.append(f"{candidate_path.name}: {status} (n={n})")
            continue
        print(f"  {candidate_path.name}: corr={corr:.6f} max_abs_diff={max_abs_diff:.3e} (n={n})")
        if not (corr >= args.corr_min):
            drifts.append(f"{candidate_path.name}: corr={corr:.6f} (n={n}, corr_min={args.corr_min:g})")

    if missing:
        drifts.append(f"reference missing {len(missing)} map(s) present in candidate: {', '.join(sorted(missing))}")

    # Stage-B AIF contract, when the candidate step produced one. Absent on older
    # candidates (or a stageBOnly=false generator predating it), which is not drift.
    stage_b_checked = False
    candidate_contract = args.candidate_root / STAGE_B_CONTRACT_NAME
    reference_contract = args.reference_root / STAGE_B_CONTRACT_NAME
    if candidate_contract.exists():
        if not reference_contract.exists():
            drifts.append(f"{STAGE_B_CONTRACT_NAME}: present in candidate but missing from the committed reference")
        else:
            stage_b_checked = True
            drifts.extend(_compare_stage_b_contract(reference_contract, candidate_contract))

    if not drifts:
        extra = " + Stage-B AIF contract" if stage_b_checked else ""
        print(f"OK: {len(candidate_maps)} freshly regenerated map(s){extra} match the committed baseline.")
        return 0

    print(f"DRIFT DETECTED: {len(drifts)} item(s) out of tolerance:")
    for line in drifts:
        print(f"  {line}")
    print(
        "\nThe committed MATLAB region-parity baseline (derivatives/matlabref/sub-10bbbdownsample/...) "
        "no longer matches current MATLAB output. Regenerate and commit it, e.g.:\n"
        "  matlab -batch \"addpath('.'); addpath('tests/matlab'); "
        "generate_dce_tofts_parity_map("
        "'outputRoot', 'tests/data/BIDS_test/derivatives/matlabref/sub-10bbbdownsample/ses-01/dce', "
        "'dynamicPath', 'tests/data/BIDS_test/rawdata/sub-10bbbdownsample/ses-01/dce/sub-10bbbdownsample_ses-01_DCE.nii', "
        "'aifRoiPath', 'tests/data/BIDS_test/derivatives/sub-10bbbdownsample/ses-01/dce/sub-10bbbdownsample_ses-01_label-AIF_mask.nii', "
        "'brainRoiPath', 'tests/data/BIDS_test/derivatives/sub-10bbbdownsample/ses-01/anat/sub-10bbbdownsample_ses-01_label-brain_mask.nii', "
        "'t1MapPath', 'tests/data/BIDS_test/derivatives/sub-10bbbdownsample/ses-01/anat/sub-10bbbdownsample_ses-01_space-DCEref_T1map.nii', "
        "'noiseRoiPath', 'tests/data/BIDS_test/derivatives/sub-10bbbdownsample/ses-01/anat/sub-10bbbdownsample_ses-01_label-noise_mask.nii', "
        "'models', {'tofts', 'patlak'});\"\n"
        "Then review the diff and commit the refreshed tests/data/BIDS_test/derivatives/matlabref/... files."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
