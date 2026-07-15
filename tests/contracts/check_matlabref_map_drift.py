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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ROOT = (
    REPO_ROOT / "tests/data/BIDS_test/derivatives/matlabref/sub-10bbbdownsample/ses-01/dce"
)

# MATLAB-vs-itself on identical inputs should be near bit-identical; the only expected
# noise is optimizer/solver nondeterminism, several orders of magnitude below a genuine
# algorithm change (e.g. a different steady-state window shifts corr from ~1.0 to ~0.0).
CORR_MIN = 0.999
MAX_ABS_DIFF_MAX = 1e-3


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
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    max_abs_diff = float(np.max(np.abs(x - y)))
    return ("ok", corr, max_abs_diff, n)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-root", type=Path, required=True, help="Freshly regenerated maps directory.")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT, help="Committed maps directory.")
    parser.add_argument("--corr-min", type=float, default=CORR_MIN)
    parser.add_argument("--max-abs-diff-max", type=float, default=MAX_ABS_DIFF_MAX)
    args = parser.parse_args(argv)

    # Scope the comparison to whatever the candidate step actually regenerated (e.g. just
    # tofts+patlak): a committed reference map for a model the candidate step didn't
    # (re)generate this run is out of scope, not a failure.
    candidate_maps = sorted(args.candidate_root.glob("*.nii")) + sorted(args.candidate_root.glob("*.nii.gz"))
    if not candidate_maps:
        print(f"No candidate maps found under {args.candidate_root}")
        return 1

    print(
        f"matlabref map drift check: reference={args.reference_root} candidate={args.candidate_root} "
        f"corr_min={args.corr_min:g} max_abs_diff_max={args.max_abs_diff_max:g}"
    )

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
        if not (corr >= args.corr_min and max_abs_diff <= args.max_abs_diff_max):
            drifts.append(
                f"{candidate_path.name}: corr={corr:.6f} max_abs_diff={max_abs_diff:.3e} "
                f"(n={n}, corr_min={args.corr_min:g}, max_abs_diff_max={args.max_abs_diff_max:g})"
            )

    if missing:
        drifts.append(f"reference missing {len(missing)} map(s) present in candidate: {', '.join(sorted(missing))}")

    if not drifts:
        print(f"OK: {len(candidate_maps)} freshly regenerated map(s) match the committed baseline.")
        return 0

    print(f"DRIFT DETECTED: {len(drifts)} map(s) out of tolerance:")
    for line in drifts:
        print(f"  {line}")
    print(
        "\nThe committed MATLAB region-parity baseline (derivatives/matlabref/sub-10bbbdownsample/...) "
        "no longer matches current MATLAB output. Regenerate and commit it, e.g.:\n"
        "  matlab -batch \"addpath('.'); addpath('tests/matlab'); "
        "generate_dce_tofts_parity_map("
        "'outputRoot', 'tests/data/BIDS_test/derivatives/matlabref/sub-10bbbdownsample/ses-01/dce', "
        "'dynamicPath', 'tests/data/BIDS_test/rawdata/sub-10bbbdownsample/ses-01/dce/sub-10bbbdownsample_ses-01_DCE.nii', "
        "'aifRoiPath', 'tests/data/BIDS_test/derivatives/sub-10bbbdownsample/ses-01/dce/sub-10bbbdownsample_ses-01_desc-AIFroi_mask.nii', "
        "'brainRoiPath', 'tests/data/BIDS_test/derivatives/sub-10bbbdownsample/ses-01/anat/sub-10bbbdownsample_ses-01_desc-brain_mask.nii', "
        "'t1MapPath', 'tests/data/BIDS_test/derivatives/sub-10bbbdownsample/ses-01/anat/sub-10bbbdownsample_ses-01_space-DCEref_T1map.nii', "
        "'noiseRoiPath', 'tests/data/BIDS_test/derivatives/sub-10bbbdownsample/ses-01/anat/sub-10bbbdownsample_ses-01_desc-noise_mask.nii', "
        "'models', {'tofts', 'patlak'});\"\n"
        "Then review the diff and commit the refreshed tests/data/BIDS_test/derivatives/matlabref/... files."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
