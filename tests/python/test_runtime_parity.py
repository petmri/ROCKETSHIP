"""Python-vs-MATLAB runtime parity: run both stacks live and compare.

This wires up the ``--run-runtime-parity`` option surface (matlab command, DCE/T1
roots, max Python/MATLAB wall-clock ratio) to real tests. It runs the actual
MATLAB and Python workflows on a shared fixture and, above all, asserts the
NUMERICAL results agree (parity is about "same results"; runtime is secondary).

Wall-clock ratio is reported always but only asserted when the MATLAB run is long
enough that its ~10 s interpreter startup does not dominate the measurement;
otherwise the ratio is logged as informational. Point ``--runtime-parity-t1-root``
/ ``--runtime-parity-dce-root`` at a larger dataset for a meaningful ratio gate.

All tests here are gated behind ``--run-runtime-parity`` and require MATLAB.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

# Reuse the committed DCE parity plumbing rather than reimplementing it.
from test_dce_pipeline_parity_metrics import (  # noqa: E402
    _dataset_paths,
    _default_downsample_root,
    _load_nifti,
    _make_tofts_post_8ef4988_config,
)

from dce_pipeline import run_dce_pipeline  # noqa: E402
from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline  # noqa: E402

T1_FIXTURE = REPO_ROOT / "tests/data/ci_fixtures/t1/vfa_small"
T1_FLIP_FILES = ["flip-02deg_VFA.nii.gz", "flip-05deg_VFA.nii.gz", "flip-10deg_VFA.nii.gz"]
T1_FLIP_ANGLES = [2.0, 5.0, 10.0]
T1_TR_MS = 8.012

# Below this MATLAB wall-clock (seconds), interpreter startup dominates and the
# runtime ratio is not a meaningful performance measurement, so it is not gated.
MEANINGFUL_MATLAB_SECONDS = 15.0

# T1 plausibility band shared with test_t1_map_parity (non-identifiable voxels excluded).
T1_MIN_MS, T1_MAX_MS = 100.0, 4000.0


def _require_matlab(matlab_cmd: str) -> str:
    exe = shutil.which(matlab_cmd) or (matlab_cmd if Path(matlab_cmd).exists() else None)
    if not exe:
        pytest.skip(f"MATLAB command not found: {matlab_cmd!r}")
    return exe


def _time_matlab(matlab_cmd: str, batch_code: str) -> float:
    """Run a MATLAB -batch command from the repo root and return wall-clock seconds."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [matlab_cmd, "-batch", batch_code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise AssertionError(f"MATLAB run failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return dt


def _masked_corr_mae(py_map: np.ndarray, ml_map: np.ndarray, lo: float, hi: float) -> tuple[int, float, float]:
    m = (
        np.isfinite(py_map) & np.isfinite(ml_map)
        & (py_map > lo) & (py_map < hi) & (ml_map > lo) & (ml_map < hi)
    )
    n = int(np.count_nonzero(m))
    if n < 2:
        return n, float("nan"), float("nan")
    x, y = py_map[m], ml_map[m]
    return n, float(np.corrcoef(x, y)[0, 1]), float(np.mean(np.abs(x - y)))


def _report_ratio(py_seconds: float, matlab_seconds: float, max_ratio: float) -> None:
    ratio = py_seconds / matlab_seconds if matlab_seconds > 0 else float("inf")
    print(
        f"[RUNTIME-PARITY] python={py_seconds:.2f}s matlab={matlab_seconds:.2f}s "
        f"ratio={ratio:.3f} (max={max_ratio})",
        flush=True,
    )
    if matlab_seconds >= MEANINGFUL_MATLAB_SECONDS:
        assert ratio <= max_ratio, (
            f"Python/MATLAB runtime ratio {ratio:.3f} exceeds max {max_ratio} "
            f"(python={py_seconds:.2f}s matlab={matlab_seconds:.2f}s)"
        )
    else:
        print(
            f"[RUNTIME-PARITY] matlab={matlab_seconds:.2f}s < {MEANINGFUL_MATLAB_SECONDS}s "
            "(startup-dominated); ratio is informational, not gated.",
            flush=True,
        )


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.slow
def test_runtime_parity_t1(
    run_runtime_parity: bool,
    runtime_parity_matlab_cmd: str,
    runtime_parity_max_python_over_matlab_ratio: float,
    runtime_parity_t1_root: str,
    tmp_path: Path,
) -> None:
    if not run_runtime_parity:
        pytest.skip("Use --run-runtime-parity to run Python-vs-MATLAB runtime parity tests.")
    matlab_cmd = _require_matlab(runtime_parity_matlab_cmd)

    root = Path(runtime_parity_t1_root) if runtime_parity_t1_root else T1_FIXTURE
    vfa = [root / f for f in T1_FLIP_FILES]
    for f in vfa:
        assert f.exists(), f"Missing VFA file for runtime parity: {f}"

    # MATLAB T1 map (timed).
    ml_map_path = tmp_path / "matlab_T1.nii"
    vfa_cells = ",".join(f"'{p}'" for p in vfa)
    batch = (
        "addpath('tests/matlab'); addpath('tests/matlab/helpers'); "
        f"generate_t1_parity_map('vfaFiles', {{{vfa_cells}}}, "
        f"'flipAngles', {T1_FLIP_ANGLES}, 'trMs', {T1_TR_MS}, "
        f"'fitType', 't1_fa_fit', 'outputPath', '{ml_map_path}', 'rsquaredThreshold', 0);"
    )
    matlab_seconds = _time_matlab(matlab_cmd, batch)

    # Python T1 map (timed).
    t0 = time.perf_counter()
    cfg = ParametricT1Config(
        output_dir=tmp_path / "py_out",
        vfa_files=list(vfa),
        fit_type="t1_fa_fit",
        flip_angles_deg=list(T1_FLIP_ANGLES),
        tr_ms=T1_TR_MS,
        backend="cpu",
        rsquared_threshold=0.0,
        invalid_fill_value=float("nan"),
    )
    result = run_parametric_t1_pipeline(cfg)
    py_seconds = time.perf_counter() - t0
    assert result["meta"]["status"] == "ok", result["meta"]

    py_candidates = sorted((tmp_path / "py_out").glob("T1_map_*t1_fa_fit*.nii*"))
    assert py_candidates, "Python T1 map not written"
    py_map = np.asarray(np.squeeze(nib.load(str(py_candidates[0])).get_fdata()), dtype=np.float64)
    ml_map = np.asarray(np.squeeze(nib.load(str(ml_map_path)).get_fdata()), dtype=np.float64)

    n, corr, mae = _masked_corr_mae(py_map, ml_map, T1_MIN_MS, T1_MAX_MS)
    print(f"[RUNTIME-PARITY] t1 numerical: n={n} corr={corr:.8f} mae_ms={mae:.4g}", flush=True)
    assert n >= 300, f"only {n} plausible-T1 voxels; comparison collapsed"
    assert corr >= 0.999, f"T1 runtime parity corr {corr:.6f} < 0.999"
    assert mae <= 2.0, f"T1 runtime parity MAE {mae:.4g} ms > 2.0"

    _report_ratio(py_seconds, matlab_seconds, runtime_parity_max_python_over_matlab_ratio)


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.slow
def test_runtime_parity_dce_tofts(
    run_runtime_parity: bool,
    runtime_parity_matlab_cmd: str,
    runtime_parity_max_python_over_matlab_ratio: float,
    runtime_parity_dce_root: str,
    tmp_path: Path,
) -> None:
    if not run_runtime_parity:
        pytest.skip("Use --run-runtime-parity to run Python-vs-MATLAB runtime parity tests.")
    matlab_cmd = _require_matlab(runtime_parity_matlab_cmd)

    root = Path(runtime_parity_dce_root) if runtime_parity_dce_root else _default_downsample_root()
    paths = _dataset_paths(root)
    assert paths["dynamic"].exists(), f"Missing DCE dynamic input: {paths['dynamic']}"

    # MATLAB tofts Ktrans map (timed) into a private results dir.
    ml_out = tmp_path / "results_matlab"
    batch = (
        "addpath('.'); addpath('tests/matlab'); "
        f"generate_dce_tofts_parity_map('subjectRoot', '{paths['root']}', "
        f"'outputRoot', '{ml_out}', 'models', {{'tofts'}});"
    )
    matlab_seconds = _time_matlab(matlab_cmd, batch)
    ml_ktrans_path = ml_out / "Dyn-1_tofts_fit_Ktrans.nii"
    assert ml_ktrans_path.exists(), f"MATLAB did not write {ml_ktrans_path}"

    # Python tofts Ktrans map (timed).
    t0 = time.perf_counter()
    result = run_dce_pipeline(_make_tofts_post_8ef4988_config(paths, tmp_path / "py_out", backend="cpu"))
    py_seconds = time.perf_counter() - t0
    assert result["meta"]["status"] == "ok"

    py_ktrans = _load_nifti(tmp_path / "py_out" / "Dyn-1_tofts_fit_Ktrans.nii.gz")
    ml_ktrans = _load_nifti(ml_ktrans_path)
    roi_mask = _load_nifti(paths["roi"])
    m = np.isfinite(py_ktrans) & np.isfinite(ml_ktrans) & (roi_mask > 0)
    n = int(np.count_nonzero(m))
    assert n >= 100, f"only {n} ROI voxels; comparison collapsed"
    x, y = py_ktrans[m], ml_ktrans[m]
    corr = float(np.corrcoef(x, y)[0, 1])
    mse = float(np.mean((x - y) ** 2))
    print(f"[RUNTIME-PARITY] dce tofts numerical: n={n} corr={corr:.6f} mse={mse:.6g}", flush=True)
    assert corr >= 0.99, f"DCE runtime parity corr {corr:.6f} < 0.99"
    assert mse <= 0.001, f"DCE runtime parity mse {mse:.6g} > 0.001"

    _report_ratio(py_seconds, matlab_seconds, runtime_parity_max_python_over_matlab_ratio)
