"""MATLAB-vs-Python numeric parity for the `tv` steady-state-end detector.

`find_end_ss_tv.m` is a from-scratch MATLAB port of `_tv_baseline_end`
(python/dce_pipeline.py) -- there was no prior MATLAB implementation to port from, so
unlike most MATLAB<->Python contracts here, Python isn't reproducing MATLAB; MATLAB is
reproducing Python. This test is the only thing that would catch an index-base
(0-based Python vs 1-based MATLAB) translation mistake in that port. Skips if MATLAB
isn't on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import _tv_baseline_end  # noqa: E402


def _require_matlab(matlab_cmd: str = "matlab") -> str:
    exe = shutil.which(matlab_cmd) or (matlab_cmd if Path(matlab_cmd).exists() else None)
    if not exe:
        pytest.skip(f"MATLAB command not found: {matlab_cmd!r}")
    return exe


def _run_find_end_ss_tv(matlab_exe: str, stlv: np.ndarray) -> int:
    """Run dce/find_end_ss_tv.m on `stlv` via a temp .mat round-trip and return end_ss."""
    from scipy.io import savemat

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mat_path = tmp_path / "curve.mat"
        savemat(mat_path, {"DYNAMLV": np.asarray(stlv, dtype=np.float64)})
        batch_code = (
            f"addpath('{REPO_ROOT / 'dce'}'); "
            f"s = load('{mat_path}'); "
            "[end_ss, ~] = find_end_ss_tv(s.DYNAMLV); "
            "fprintf('END_SS=%d\\n', end_ss);"
        )
        proc = subprocess.run(
            [matlab_exe, "-batch", batch_code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(f"MATLAB run failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        for line in proc.stdout.splitlines():
            if line.startswith("END_SS="):
                return int(line.split("=", 1)[1])
        raise AssertionError(f"Could not find END_SS in MATLAB output:\n{proc.stdout}\n{proc.stderr}")


def _test_cases() -> List[Tuple[str, np.ndarray]]:
    cases: List[Tuple[str, np.ndarray]] = []

    n_time = 48
    mean_curve = np.full(n_time, 95.0, dtype=np.float64)
    mean_curve[:10] += np.array(
        [0.0, -0.1, 0.15, -0.05, 0.08, -0.06, 0.1, -0.02, 0.04, 0.0], dtype=np.float64
    )
    mean_curve[10:] += 18.0 + 10.0 * np.exp(-np.arange(n_time - 10, dtype=np.float64) / 8.0)
    cases.append(("fixture_step_change", np.tile(mean_curve[:, np.newaxis], (1, 5))))

    cases.append(("fallback_short", np.full((2, 3), 100.0, dtype=np.float64)))

    cases.append(("flat_constant", np.full((30, 4), 50.0, dtype=np.float64)))

    rng = np.random.default_rng(42)
    n2 = 20
    c = np.full(n2, 200.0) + rng.normal(0, 0.5, n2)
    c[-1] += 40.0
    cases.append(("late_jump", np.tile(c[:, np.newaxis], (1, 3))))

    rng = np.random.default_rng(42)
    n3 = 64
    base = np.full(n3, 300.0)
    base[:6] += rng.normal(0, 1.0, 6)
    base[6:] += 60.0 * (1 - np.exp(-(np.arange(n3 - 6)) / 5.0))
    voxels = np.tile(base[:, np.newaxis], (1, 12)) + rng.normal(0, 2.0, (n3, 12))
    cases.append(("realistic_multivoxel", voxels))

    return cases


@pytest.mark.integration
@pytest.mark.parametrize("name,stlv", _test_cases(), ids=lambda v: v if isinstance(v, str) else "curve")
def test_find_end_ss_tv_matches_matlab_on_synthetic_curves(name: str, stlv: np.ndarray) -> None:
    matlab_exe = _require_matlab()
    python_end_ss = int(_tv_baseline_end(stlv)["end_ss_1b"])
    matlab_end_ss = _run_find_end_ss_tv(matlab_exe, stlv)
    assert matlab_end_ss == python_end_ss, f"{name}: matlab={matlab_end_ss} python={python_end_ss}"


@pytest.mark.integration
def test_find_end_ss_tv_matches_matlab_on_real_bbb_downsample_curve() -> None:
    matlab_exe = _require_matlab()
    import nibabel as nib

    S = REPO_ROOT / "tests/data/BIDS_test"
    sub, ses = "sub-10bbbdownsample", "ses-01"
    dynamic = np.asarray(
        nib.load(str(S / f"rawdata/{sub}/{ses}/dce/{sub}_{ses}_DCE.nii")).get_fdata(), dtype=np.float64
    )
    mask = np.asarray(
        nib.load(str(S / f"derivatives/{sub}/{ses}/dce/{sub}_{ses}_label-AIF_mask.nii")).get_fdata(),
        dtype=np.float64,
    )
    if mask.ndim == 4:
        mask = mask[..., 0]
    stlv = dynamic[mask > 0].T

    python_end_ss = int(_tv_baseline_end(stlv)["end_ss_1b"])
    matlab_end_ss = _run_find_end_ss_tv(matlab_exe, stlv)
    assert matlab_end_ss == python_end_ss, f"real curve: matlab={matlab_end_ss} python={python_end_ss}"
