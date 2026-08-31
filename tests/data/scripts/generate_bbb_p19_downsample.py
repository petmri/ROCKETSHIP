"""Generate a nearest-neighbor downsampled BBB p19 fixture for fast parity tests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "tests" / "data" / "BBB data p19"
# The downsampled BBB fixture is the DCE fit-parity subject sub-10bbbdownsample in BIDS_test.
# --output-root points at the BIDS dataset root; files land under sub-10bbbdownsample/ses-01.
# This script produces the pipeline INPUTS (DCE + preprocessed derivative maps/masks); the MATLAB
# fit baselines under derivatives/matlabref are produced separately by the MATLAB parity generator.
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "data" / "BIDS_test"
SUBJECT = "sub-10bbbdownsample"
SESSION = "ses-01"


def _scale_affine_xy(affine: np.ndarray, factor_x: int, factor_y: int) -> np.ndarray:
    out = np.array(affine, dtype=np.float64, copy=True)
    out[:3, 0] *= float(factor_x)
    out[:3, 1] *= float(factor_y)
    return out


def _downsample_xy(data: np.ndarray, factor_x: int, factor_y: int) -> np.ndarray:
    if data.ndim == 4:
        return data[::factor_x, ::factor_y, :, :]
    if data.ndim == 3:
        return data[::factor_x, ::factor_y, :]
    if data.ndim == 2:
        return data[::factor_x, ::factor_y]
    raise ValueError(f"Unsupported NIfTI dimensionality: {data.ndim}")


def _downsample_nifti_xy(src: Path, dst: Path, factor_x: int, factor_y: int) -> None:
    image = nib.load(str(src))
    data = np.asanyarray(image.dataobj)
    down = _downsample_xy(data, factor_x, factor_y)

    affine = _scale_affine_xy(image.affine, factor_x, factor_y)
    header = image.header.copy()
    out_img = nib.Nifti1Image(down, affine, header)
    out_img.set_data_dtype(data.dtype)
    nib.save(out_img, str(dst))


def _copy_or_downsample(src: Path, dst: Path, factor_x: int, factor_y: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lower = src.name.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        _downsample_nifti_xy(src, dst, factor_x, factor_y)
    else:
        shutil.copy2(src, dst)


def _build_file_map(source_root: Path) -> List[tuple[Path, Path]]:
    """Map source BBB p19 files to their BIDS destinations under sub-10bbbdownsample/ses-01."""
    stem = f"{SUBJECT}_{SESSION}"
    raw_dce = Path("rawdata") / SUBJECT / SESSION / "dce"
    der_anat = Path("derivatives") / SUBJECT / SESSION / "anat"
    der_dce = Path("derivatives") / SUBJECT / SESSION / "dce"
    wanted = [
        (source_root / "Dynamic_t1w.nii", raw_dce / f"{stem}_DCE.nii"),
        (source_root / "processed" / "T1_map_t1_fa_fit_fa10.nii", der_anat / f"{stem}_space-DCEref_T1map.nii"),
        (source_root / "processed" / "T1_brain_roi.nii", der_anat / f"{stem}_label-brain_mask.nii"),
        (source_root / "processed" / "T1_gm_roi.nii", der_anat / f"{stem}_label-GM_mask.nii"),
        (source_root / "processed" / "T1_wm_roi.nii", der_anat / f"{stem}_label-WM_mask.nii"),
        (source_root / "processed" / "T1_noise_roi.nii", der_anat / f"{stem}_label-noise_mask.nii"),
        (source_root / "processed" / "T1_AIF_roi.nii", der_dce / f"{stem}_label-AIF_mask.nii"),
    ]
    return [(src, dst) for src, dst in wanted if src.exists()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--factor-x", type=int, default=3)
    parser.add_argument("--factor-y", type=int, default=3)
    parser.add_argument("--clean", action="store_true", help="Delete output root before generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    factor_x = int(args.factor_x)
    factor_y = int(args.factor_y)

    if factor_x < 1 or factor_y < 1:
        raise ValueError("factor-x and factor-y must be >= 1")
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    file_map = _build_file_map(source_root)

    if args.clean:
        # Remove only the exact files this script owns. The GM/WM ROI masks (added for GM/WM
        # parity and not derivable from the base BBB source) and the MATLAB baselines under
        # derivatives/matlabref are committed assets that must survive a regeneration.
        stem = f"{SUBJECT}_{SESSION}"
        owned_files = [output_root / rel for _, rel in file_map]
        owned_files.append(output_root / "rawdata" / SUBJECT / SESSION / "dce" / f"{stem}_DCE.json")
        for f in owned_files:
            if f.exists():
                f.unlink()
    generated: List[str] = []
    for src, rel_dst in file_map:
        dst = output_root / rel_dst
        _copy_or_downsample(src, dst, factor_x, factor_y)
        generated.append(str(rel_dst))

    stem = f"{SUBJECT}_{SESSION}"
    dce_json = output_root / "rawdata" / SUBJECT / SESSION / "dce" / f"{stem}_DCE.json"
    dce_json.write_text(
        json.dumps(
            {
                "RepetitionTime": 0.00829,
                "TemporalResolution": 15.84,
                "FlipAngle": 15,
                "AcquisitionDateTime": "2000-01-01T00:00:00.000000",
            },
            indent=2,
        )
        + "\n"
    )

    subject_root = output_root / "rawdata" / SUBJECT / SESSION
    print(str(subject_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
