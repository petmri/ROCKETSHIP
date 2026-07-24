"""Unit tests for the DCE quality-of-fit map builder (`dce_qof`).

Exercises the post-fit-NPZ contract (SSE column, wash-in window from `b_out.json`, χ²_ν) and the
Fortran-order voxel→volume reconstruction, using a small synthetic NPZ so no heavy fixture is
needed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import dce_qof  # noqa: E402
from dce_sigma import sigma_successive_difference  # noqa: E402


def _write_synthetic_npz(tmp: Path, *, model="tofts", n_frames=40, dims=(4, 3, 1)) -> Path:
    rng = np.random.default_rng(0)
    n_vox = dims[0] * dims[1] * dims[2] - 2  # leave 2 background voxels outside the ROI
    ramp = 1.0 / (1.0 + np.exp(-8.0 * (np.linspace(0, 1, n_frames) - 0.3)))
    ct = ramp[:, None] + rng.normal(0.0, 0.05, size=(n_frames, n_vox))  # (T, V)

    # voxel_results: tofts sse_col is 3 (1-based) -> put a known SSE in column index 2.
    sse = np.linspace(0.05, 0.5, n_vox)
    voxel_results = np.zeros((n_vox, 4), dtype=np.float64)
    voxel_results[:, 0] = 0.1  # ktrans
    voxel_results[:, 1] = 0.2  # ve
    voxel_results[:, 2] = sse  # SSE (column 3, 1-based)

    tumind = np.arange(n_vox, dtype=np.int64)  # first n_vox voxels (F-order) are the ROI

    npz = tmp / "Dyn-1_tofts_fit_postfit_arrays.npz"
    np.savez_compressed(
        npz,
        model_name=np.asarray(model),
        ct_voxel_mM=ct,
        timer_min=np.linspace(0.0, 2.0, n_frames),
        voxel_results=voxel_results,
        tumind_0based=tumind,
        dimensions_xyz=np.asarray(dims, dtype=np.int64),
    )
    return npz


def _write_checkpoint(npz: Path) -> None:
    ckpt = npz.parent / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "b_out.json").write_text(
        json.dumps(
            {
                "time_resolution_min": 0.05,
                "start_injection_min": 0.5,  # -> onset frame 10
                "end_injection_min": 0.6,  # -> +2 frames duration
                "start_time_index": 1,
            }
        )
    )


@pytest.mark.unit
def test_compute_qof_matches_manual_reduced_chi_square(tmp_path: Path) -> None:
    npz = _write_synthetic_npz(tmp_path)
    _write_checkpoint(npz)
    res = dce_qof.compute_qof(npz)

    assert res["model_name"] == "tofts"
    assert res["p"] == 2
    assert res["window"] == (10, 14)  # onset 10, duration 2, margin 2

    with np.load(npz) as d:
        ct = np.asarray(d["ct_voxel_mM"], float)
        sse = np.asarray(d["voxel_results"], float)[:, 2]
    n_obs = ct.shape[0]
    sigma = sigma_successive_difference(ct, exclude=res["window"])
    expected = sse / (sigma**2 * (n_obs - res["p"]))
    np.testing.assert_allclose(res["chi2v"], expected, rtol=1e-12)
    np.testing.assert_allclose(res["sigma"], sigma, rtol=1e-12)


@pytest.mark.unit
def test_compute_qof_without_checkpoint_uses_no_window(tmp_path: Path) -> None:
    npz = _write_synthetic_npz(tmp_path)  # no b_out.json written
    res = dce_qof.compute_qof(npz)
    assert res["window"] is None
    assert np.all(np.isfinite(res["chi2v"]))


@pytest.mark.unit
def test_reliable_mask_threshold(tmp_path: Path) -> None:
    npz = _write_synthetic_npz(tmp_path)
    res = dce_qof.compute_qof(npz, chi2_reliable_max=2.0)
    assert res["reliable"].dtype == bool
    np.testing.assert_array_equal(
        res["reliable"], np.isfinite(res["chi2v"]) & (res["chi2v"] <= 2.0)
    )


@pytest.mark.unit
def test_write_qof_maps_npy_reconstruction_is_fortran_order(tmp_path: Path) -> None:
    npz = _write_synthetic_npz(tmp_path)
    res = dce_qof.compute_qof(npz)
    out = dce_qof.write_qof_maps(npz, tmp_path / "maps")  # no reference -> .npy volumes

    vol = np.load(out["qof_chi2nu"])
    dims = res["dims"]
    assert vol.shape == dims
    # ROI voxels are the first len(tumind) in Fortran order; background is NaN.
    coords = np.unravel_index(res["tumind"], dims, order="F")
    np.testing.assert_allclose(vol[coords], res["chi2v"].astype(np.float32), rtol=1e-6)
    filled = np.zeros(dims, dtype=bool)
    filled[coords] = True
    assert np.isnan(vol[~filled]).all()


@pytest.mark.unit
def test_reliable_mask_percentile_excludes_worst_tail() -> None:
    chi2 = np.arange(1, 101, dtype=np.float64)  # 1..100
    mask, tau = dce_qof.reliable_mask(chi2, percentile=95.0)
    assert tau == pytest.approx(np.percentile(chi2, 95.0))
    # keep the best 95% (<= p95), exclude the worst ~5%
    assert mask.sum() == 95
    assert not mask[-1]  # the largest chi2 is excluded


@pytest.mark.unit
def test_reliable_mask_explicit_cutoff_and_nan() -> None:
    chi2 = np.array([0.5, 2.0, np.nan, 10.0])
    mask, tau = dce_qof.reliable_mask(chi2, chi2_max=3.0)
    assert tau == 3.0
    np.testing.assert_array_equal(mask, np.array([True, True, False, False]))


@pytest.mark.unit
def test_reliable_mask_requires_a_threshold() -> None:
    with pytest.raises(ValueError):
        dce_qof.reliable_mask(np.array([1.0, 2.0]))


@pytest.mark.unit
def test_qof_volumes_reconstructs_chi2_and_sigma(tmp_path: Path) -> None:
    npz = _write_synthetic_npz(tmp_path)
    vols = dce_qof.qof_volumes(npz)
    res = vols["res"]
    dims = vols["dims"]
    assert vols["chi2nu"].shape == dims
    coords = np.unravel_index(res["tumind"], dims, order="F")
    np.testing.assert_allclose(vols["chi2nu"][coords], res["chi2v"].astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(vols["sigma"][coords], res["sigma"].astype(np.float32), rtol=1e-6)


@pytest.mark.unit
def test_compute_qof_shrink_sigma_clamps_outlier_voxel(tmp_path: Path) -> None:
    # Build an NPZ whose last voxel has a grossly inflated concentration-curve noise (motion-like);
    # eBayes shrinkage should clamp its sigma to the prior mean, restoring a large chi2_nu there.
    rng = np.random.default_rng(0)
    n, dims = 48, (10, 10, 1)
    n_vox = 60
    ramp = 1.0 / (1.0 + np.exp(-8.0 * (np.linspace(0, 1, n) - 0.3)))
    ct = ramp[:, None] + rng.normal(0.0, 0.05, size=(n, n_vox))
    ct[:, -1] = ramp + rng.normal(0.0, 1.0, size=n)  # artifact voxel: 20x noise
    voxel_results = np.zeros((n_vox, 4))
    voxel_results[:, 2] = 0.1  # small SSE for everyone; the artifact's inflated sigma would hide it
    npz = tmp_path / "Dyn-1_tofts_fit_postfit_arrays.npz"
    np.savez_compressed(
        npz, model_name=np.asarray("tofts"), ct_voxel_mM=ct,
        timer_min=np.linspace(0, 2, n), voxel_results=voxel_results,
        tumind_0based=np.arange(n_vox, dtype=np.int64), dimensions_xyz=np.asarray(dims, np.int64),
    )

    plain = dce_qof.compute_qof(npz)
    shrunk = dce_qof.compute_qof(npz, shrink_sigma=True, clamp_quantile=0.999)

    assert "sigma_raw" in shrunk and "eb" in shrunk
    assert shrunk["eb"]["dof"] > 0
    # the artifact voxel had the largest raw sigma...
    assert np.argmax(shrunk["sigma_raw"]) == n_vox - 1
    # ...and shrinkage pulled it below its raw value (clamped toward the prior)
    assert shrunk["sigma"][-1] < shrunk["sigma_raw"][-1]
    assert shrunk["eb"]["n_clamped"] >= 1


@pytest.mark.unit
def test_unknown_model_raises(tmp_path: Path) -> None:
    npz = _write_synthetic_npz(tmp_path, model="not_a_model")
    with pytest.raises(ValueError):
        dce_qof.compute_qof(npz)
