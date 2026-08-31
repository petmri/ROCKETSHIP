#!/usr/bin/env python3
"""Freeze Stage-B arrays from a real RUNNER_DATA session into a committed test fixture.

On-demand generator, not part of the automated suite. Runs Stage-A + Stage-B on
``RUNNER_DATA/sub-1101743/ses-01`` and writes a voxel-subsampled ``Ct``/``Cp_use``/``timer``
bundle for ``tests/python/test_backend_equivalence.py``.

Freezing matters for two reasons: the network drive is not available to CI or to most
contributors, and a backend-equivalence gate must compare backends on *identical* inputs --
re-running Stage-A/B per test would let upstream changes move both sides together and mask a
divergence. Real data rather than the synthetic downsample fixture because the measured
cpufit/gpufit-vs-MATLAB divergence lives in noisy, weakly-identifiable voxels that a clean
fixture does not contain.

Usage:
    python tests/python/freeze_stage_b_backend_fixture.py
    python tests/python/freeze_stage_b_backend_fixture.py --runner-data /path/to/RUNNER_DATA
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _run_stage_a_real,
    _run_stage_b_real,
)

DEFAULT_RUNNER_DATA = Path("/media/network_mriphysics/USC-PPG/RUNNER_DATA")
SUBJECT = "sub-1101743"
SESSION = "ses-01"
DEFAULT_OUTPUT = REPO_ROOT / "tests/data/stage_b_frozen/sub-1101743_ses-01_stage_b_backend.npz"

# Acquisition parameters as the MATLAB reference run used them, read off that run's
# A_dceR1info.log rather than guessed. TriggerDelayTime/n_reps/1000 = 980729/64/1000.
ACQUISITION = {
    "tr_ms": 8.012,
    "fa_deg": 15.0,
    "time_resolution_sec": 980729.0 / 64.0 / 1000.0,
    "hematocrit": 0.45,
    "relaxivity": 5.7,
    "snr_filter": 1.0,
    "start_t": 3,
}


def _paths(runner_data: Path) -> dict:
    dceprep = runner_data / "derivatives" / "dceprep" / SUBJECT / SESSION
    stem = f"{SUBJECT}_{SESSION}"
    return {
        "source": runner_data / "sourcedata" / SUBJECT / SESSION,
        "processed": dceprep,
        "dynamic": dceprep / "dce" / f"{stem}_desc-bfcz_DCE.nii.gz",
        "aif": dceprep / "dce" / f"{stem}_desc-AIF_mask.nii.gz",
        "roi": dceprep / "anat" / f"{stem}_space-DCEref_desc-brain_mask.nii.gz",
        "t1map": dceprep / "anat" / f"{stem}_space-DCEref_T1map.nii",
    }


def _build_config(paths: dict, out_dir: Path) -> DcePipelineConfig:
    return DcePipelineConfig(
        subject_source_path=paths["source"],
        subject_tp_path=paths["processed"],
        output_dir=out_dir,
        backend="cpu",
        aif_mode="fitted",
        write_xls=False,
        dynamic_files=[paths["dynamic"]],
        aif_files=[paths["aif"]],
        roi_files=[paths["roi"]],
        t1map_files=[paths["t1map"]],
        noise_files=[],  # dceprep produces no noise mask; optional upstream.
        model_flags={"patlak": 1},
        stage_overrides={
            "rootname": "dce",
            "stage_a_mode": "real",
            "stage_b_mode": "real",
            "stage_d_mode": "real",
            "steady_state_auto_method": "tv",
            "auto_find_injection": 1,
            "time_smoothing": "none",
            "time_smoothing_window": 0,
            "save_aif_figure": False,
            **ACQUISITION,
        },
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner-data", type=Path, default=DEFAULT_RUNNER_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-voxels", type=int, default=2000, help="Voxels to keep (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    paths = _paths(args.runner_data)
    missing = [str(p) for k, p in paths.items() if k not in {"source", "processed"} and not p.exists()]
    if missing:
        print(f"Missing RUNNER_DATA input(s):\n  " + "\n  ".join(missing))
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        config = _build_config(paths, Path(tmp) / "out")
        print(f"Running Stage-A on {paths['dynamic'].name} ...")
        stage_a = _run_stage_a_real(config)
        print("Running Stage-B ...")
        stage_b = _run_stage_b_real(config, stage_a)

    arrays = stage_b["arrays"]
    ct = np.asarray(arrays["Ct"], dtype=np.float64)
    cp_use = np.asarray(arrays["Cp_use"], dtype=np.float64).ravel()
    timer = np.asarray(arrays["timer"], dtype=np.float64).ravel()

    n_time, n_vox = ct.shape
    print(f"Stage-B arrays: Ct={ct.shape} Cp_use={cp_use.shape} timer={timer.shape}")

    # Keep only voxels every backend can actually fit; a non-finite curve would make the
    # comparison a NaN-vs-NaN tautology rather than a backend check.
    finite = np.all(np.isfinite(ct), axis=0)
    usable = np.flatnonzero(finite)
    print(f"Usable (all-finite) voxels: {usable.size} of {n_vox}")
    if usable.size == 0:
        print("No finite voxel curves in Stage-B output.")
        return 1

    keep = usable
    if args.n_voxels and args.n_voxels < usable.size:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(usable, size=int(args.n_voxels), replace=False))

    ct_keep = ct[:, keep]
    meta = {
        "source": f"RUNNER_DATA/{SUBJECT}/{SESSION}",
        "dynamic": paths["dynamic"].name,
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_time": int(n_time),
        "n_voxels_stage_b": int(n_vox),
        "n_voxels_frozen": int(keep.size),
        "sample_seed": int(args.seed),
        "acquisition": ACQUISITION,
        "aif_mode": str(stage_b.get("aif_mode", "")),
        "start_injection_min": float(stage_b.get("start_injection_min", float("nan"))),
        "end_injection_min": float(stage_b.get("end_injection_min", float("nan"))),
        "time_resolution_min": float(stage_b.get("time_resolution_min", float("nan"))),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        # float32 matches what the accelerated backends consume anyway, and halves the
        # committed fixture; the CPU reference reads the same values either way.
        Ct=ct_keep.astype(np.float32),
        Cp_use=cp_use.astype(np.float64),
        timer=timer.astype(np.float64),
        voxel_index=keep.astype(np.int64),
        meta=np.array(json.dumps(meta)),
    )
    size_kb = args.output.stat().st_size / 1024.0
    print(f"\nWrote {args.output} ({size_kb:.0f} KB)")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
