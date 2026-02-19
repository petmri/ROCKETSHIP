"""Generate a nearest-neighbor downsampled BBB p19 fixture for fast parity tests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "tests" / "data" / "BBB data p19"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "data" / "synthetic" / "generated" / "bbb_p19_downsample_x3y3"


def _iter_result_maps(results_dir: Path) -> Iterable[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("*.nii")) + sorted(results_dir.glob("*.nii.gz"))


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


def _build_file_list(source_root: Path) -> List[Path]:
    wanted = [
        source_root / "Dynamic_t1w.nii",
        source_root / "fa2.nii",
        source_root / "fa5.nii",
        source_root / "fa10.nii",
        source_root / "processed" / "T1_AIF_roi.nii",
        source_root / "processed" / "T1_brain_roi.nii",
        source_root / "processed" / "T1_map_t1_fa_fit_fa10.nii",
        source_root / "processed" / "T1_noise_roi.nii",
        source_root / "processed" / "Rsquared_t1_fa_fit_fa10.nii",
        source_root / "processed" / "User Inputs Log.txt",
    ]
    wanted.extend(_iter_result_maps(source_root / "processed" / "results"))
    return [p for p in wanted if p.exists()]


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

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    file_list = _build_file_list(source_root)
    generated: List[str] = []
    for src in file_list:
        rel = src.relative_to(source_root)
        dst = output_root / rel
        _copy_or_downsample(src, dst, factor_x, factor_y)
        generated.append(str(rel))

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "factor_x": factor_x,
        "factor_y": factor_y,
        "generated_files": generated,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(str(output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
