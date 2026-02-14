"""Generate a tiny DCE fixture for fast settings/feature tests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "test_data" / "ci_fixtures" / "dce" / "tiny_settings_case"


def _to_signal_from_conc(
    conc: np.ndarray,
    *,
    t1_sec: float,
    tr_sec: float,
    fa_deg: float,
    sss: float,
    relaxivity: float,
    plasma_scale: float = 1.0,
) -> np.ndarray:
    """Synthesize GRE signal using the same equation family inverted by Stage A."""
    fa_rad = np.deg2rad(float(fa_deg))
    e1 = np.exp(-tr_sec / t1_sec)
    sstar = (1.0 - e1) / (1.0 - np.cos(fa_rad) * e1)

    r1_t = (1.0 / t1_sec) + (relaxivity * plasma_scale * conc)
    ab = np.exp(tr_sec * r1_t)
    denom = ab - np.cos(fa_rad)
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
    x = (ab - 1.0) / denom
    signal = sss * x / sstar
    return np.asarray(signal, dtype=np.float64)


def _make_masks(shape_xyz: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape_xyz
    aif = np.zeros(shape_xyz, dtype=np.uint8)
    roi = np.zeros(shape_xyz, dtype=np.uint8)
    noise = np.zeros(shape_xyz, dtype=np.uint8)

    aif[1:3, 1:3, :] = 1
    roi[4:8, 3:7, :] = 1
    noise[nx - 2 : nx, 0:2, :] = 1
    return aif, roi, noise


def _generate_fixture(output_root: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    nx, ny, nz, nt = 10, 10, 1, 18
    tr_ms = 8.29
    tr_sec = tr_ms / 1000.0
    fa_deg = 15.0
    time_resolution_sec = 15.84
    timer_min = np.arange(nt, dtype=np.float64) * (time_resolution_sec / 60.0)
    relaxivity = 3.6
    hematocrit = 0.42

    aif_mask, roi_mask, noise_mask = _make_masks((nx, ny, nz))

    t1_map = np.full((nx, ny, nz), 1.30, dtype=np.float64)  # sec
    t1_map[aif_mask > 0] = 0.95
    t1_map[roi_mask > 0] = 1.20

    # Plasma AIF concentration (mM), smooth rise + bi-exponential decay.
    cp = np.zeros(nt, dtype=np.float64)
    inj = 4
    post = np.arange(nt - inj, dtype=np.float64)
    cp[inj:] = 1.8 * np.exp(-0.35 * post) + 0.5 * np.exp(-0.06 * post)
    cp[inj : inj + 3] += np.linspace(0.2, 0.45, 3)

    dynamic = np.zeros((nx, ny, nz, nt), dtype=np.float64)
    base = 100.0 + rng.normal(0.0, 1.2, size=(nx, ny, nz))

    # Background voxels: mild drift/noise only.
    for t in range(nt):
        dynamic[:, :, :, t] = base + 0.15 * t + rng.normal(0.0, 0.5, size=(nx, ny, nz))

    # AIF voxels: convert Cp to signal with blood scaling (1-hematocrit).
    aif_indices = np.argwhere(aif_mask > 0)
    for idx_i, (x, y, z) in enumerate(aif_indices):
        scale = 1.0 + 0.03 * np.sin(idx_i)
        cp_vox = cp * scale
        s_curve = _to_signal_from_conc(
            cp_vox,
            t1_sec=float(t1_map[x, y, z]),
            tr_sec=tr_sec,
            fa_deg=fa_deg,
            sss=float(base[x, y, z]),
            relaxivity=relaxivity,
            plasma_scale=(1.0 - hematocrit),
        )
        dynamic[x, y, z, :] = s_curve + rng.normal(0.0, 0.35, size=nt)

    # ROI voxels: small spread of Tofts-like concentration curves.
    roi_indices = np.argwhere(roi_mask > 0)
    for idx_i, (x, y, z) in enumerate(roi_indices):
        ktrans = 0.08 + 0.06 * ((idx_i % 5) / 4.0)
        ve = 0.22 + 0.18 * (((idx_i // 5) % 4) / 3.0)
        ct = np.zeros(nt, dtype=np.float64)
        for t in range(nt):
            if t == 0:
                continue
            tau = timer_min[: t + 1]
            cp_tau = cp[: t + 1]
            decay = np.exp((-ktrans / ve) * (tau[-1] - tau))
            ct[t] = ktrans * np.trapezoid(cp_tau * decay, tau)
        s_curve = _to_signal_from_conc(
            ct,
            t1_sec=float(t1_map[x, y, z]),
            tr_sec=tr_sec,
            fa_deg=fa_deg,
            sss=float(base[x, y, z]),
            relaxivity=relaxivity,
            plasma_scale=1.0,
        )
        dynamic[x, y, z, :] = s_curve + rng.normal(0.0, 0.45, size=nt)

    # Noise ROI gets stronger random noise.
    noise_coords = np.argwhere(noise_mask > 0)
    for (x, y, z) in noise_coords:
        dynamic[x, y, z, :] = base[x, y, z] + rng.normal(0.0, 4.0, size=nt)

    affine = np.eye(4, dtype=np.float64)
    processed = output_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(dynamic.astype(np.float32), affine), str(output_root / "Dynamic_t1w.nii"))
    nib.save(nib.Nifti1Image(aif_mask.astype(np.uint8), affine), str(processed / "T1_AIF_roi.nii"))
    nib.save(nib.Nifti1Image(roi_mask.astype(np.uint8), affine), str(processed / "T1_brain_roi.nii"))
    nib.save(nib.Nifti1Image((t1_map * 1000.0).astype(np.float32), affine), str(processed / "T1_map_t1_fa_fit_fa10.nii"))
    nib.save(nib.Nifti1Image(noise_mask.astype(np.uint8), affine), str(processed / "T1_noise_roi.nii"))

    meta = {
        "seed": int(seed),
        "shape": [nx, ny, nz, nt],
        "tr_ms": tr_ms,
        "fa_deg": fa_deg,
        "time_resolution_sec": time_resolution_sec,
        "relaxivity": relaxivity,
        "hematocrit": hematocrit,
        "start_injection_min": float(timer_min[inj]),
        "end_injection_min": float(timer_min[min(nt - 1, inj + 2)]),
    }
    (processed / "tiny_fixture_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260214)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_root.expanduser().resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = _generate_fixture(out, int(args.seed))
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
