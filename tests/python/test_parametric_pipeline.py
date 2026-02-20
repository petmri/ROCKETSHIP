"""Integration tests for the parametric T1 mapping pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline  # noqa: E402


def _write_tiny_vfa_fixture(path: Path, *, shape: tuple[int, int] = (4, 4), tr_ms: float = 8.0) -> tuple[list[float], float]:
    flip_angles = [2.0, 10.0, 15.0]
    true_t1 = 1300.0
    rho = 1000.0
    stack = []
    for fa in flip_angles:
        theta = np.deg2rad(float(fa))
        e1 = np.exp(-tr_ms / true_t1)
        signal = rho * (((1.0 - e1) * np.sin(theta)) / (1.0 - e1 * np.cos(theta)))
        stack.append(np.full(shape, float(signal), dtype=np.float32))
    volume = np.stack(stack, axis=-1)
    nii = nib.Nifti1Image(volume, np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(path))
    return flip_angles, tr_ms


def _write_tiny_b1_fixture(path: Path, *, shape: tuple[int, int] = (4, 4), scale: float = 1.0) -> None:
    b1 = np.full(shape, float(scale), dtype=np.float32)
    nii = nib.Nifti1Image(b1, np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(path))


@pytest.mark.integration
def test_parametric_t1_pipeline_with_multifile_vfa_sidecars(tmp_path: Path) -> None:
    payload = {
        "output_dir": str(tmp_path / "out_multifile"),
        "vfa_files": [
            "tests/data/BIDS_test/rawdata/sub-01/ses-01/anat/sub-01_ses-01_flip-01_VFA.nii.gz",
            "tests/data/BIDS_test/rawdata/sub-01/ses-01/anat/sub-01_ses-01_flip-02_VFA.nii.gz",
            "tests/data/BIDS_test/rawdata/sub-01/ses-01/anat/sub-01_ses-01_flip-03_VFA.nii.gz",
        ],
        "output_basename": "T1_map",
        "output_label": "multifile",
        "fit_type": "t1_fa_linear_fit",
        "rsquared_threshold": 0.6,
        "write_r_squared": True,
        "write_rho_map": False,
    }

    config = ParametricT1Config.from_dict(payload, base_dir=REPO_ROOT)
    summary = run_parametric_t1_pipeline(config)

    t1_map_path = Path(summary["outputs"]["t1_map_path"])
    rsq_path = Path(summary["outputs"]["rsquared_map_path"])

    assert t1_map_path.exists()
    assert rsq_path.exists()
    assert Path(summary["meta"]["summary_path"]).exists()

    t1_map = nib.load(str(t1_map_path)).get_fdata()
    rsq_map = nib.load(str(rsq_path)).get_fdata()

    assert t1_map.shape == (256, 256)
    assert rsq_map.shape == (256, 256)

    valid = t1_map > 0
    assert int(np.count_nonzero(valid)) > 0
    assert float(np.nanmean(t1_map[valid])) > 0.0
    assert summary["metrics"]["valid_fits"] > 0
    assert summary["resolved_inputs"]["n_flips"] == 3


@pytest.mark.integration
def test_parametric_t1_pipeline_with_single_stacked_vfa_file(tmp_path: Path) -> None:
    payload = {
        "output_dir": str(tmp_path / "out_stacked"),
        "vfa_files": [
            "tests/data/BIDS_test/derivatives/sub-01/ses-01/anat/sub-01_ses-01_space-DCEref_desc-bfczunified_VFA.nii"
        ],
        "flip_angles_deg": [2.0, 5.0, 10.0],
        "tr_ms": 8.012,
        "output_basename": "T1_map",
        "output_label": "stacked",
        "fit_type": "t1_fa_linear_fit",
        "rsquared_threshold": 0.6,
        "write_r_squared": True,
        "write_rho_map": True,
    }

    config = ParametricT1Config.from_dict(payload, base_dir=REPO_ROOT)
    summary = run_parametric_t1_pipeline(config)

    t1_map_path = Path(summary["outputs"]["t1_map_path"])
    rsq_path = Path(summary["outputs"]["rsquared_map_path"])
    rho_path = Path(summary["outputs"]["rho_map_path"])

    assert t1_map_path.exists()
    assert rsq_path.exists()
    assert rho_path.exists()

    t1_map = nib.load(str(t1_map_path)).get_fdata()
    rsq_map = nib.load(str(rsq_path)).get_fdata()
    rho_map = nib.load(str(rho_path)).get_fdata()

    assert t1_map.shape == (256, 256, 1)
    assert rsq_map.shape == (256, 256, 1)
    assert rho_map.shape == (256, 256, 1)

    assert np.isfinite(t1_map).any()
    assert np.isfinite(rsq_map).any()
    assert np.isfinite(rho_map).any()


@pytest.mark.integration
def test_parametric_t1_pipeline_realdata_default_output_naming_multifile(tmp_path: Path) -> None:
    payload = {
        "output_dir": str(tmp_path / "out_default_multifile"),
        "vfa_files": [
            "tests/data/BIDS_test/rawdata/sub-01/ses-01/anat/sub-01_ses-01_flip-01_VFA.nii.gz",
            "tests/data/BIDS_test/rawdata/sub-01/ses-01/anat/sub-01_ses-01_flip-02_VFA.nii.gz",
            "tests/data/BIDS_test/rawdata/sub-01/ses-01/anat/sub-01_ses-01_flip-03_VFA.nii.gz",
        ],
        "output_basename": "T1_map",
        "fit_type": "t1_fa_linear_fit",
        "rsquared_threshold": 0.6,
        "write_r_squared": True,
        "write_rho_map": False,
    }

    config = ParametricT1Config.from_dict(payload, base_dir=REPO_ROOT)
    summary = run_parametric_t1_pipeline(config)
    expected_t1 = tmp_path / "out_default_multifile" / "T1_map_t1_fa_linear_fit_sub-01_ses-01_flip-01_VFA.nii.gz"
    expected_rsq = tmp_path / "out_default_multifile" / "Rsquared_t1_fa_linear_fit_sub-01_ses-01_flip-01_VFA.nii.gz"

    assert Path(summary["outputs"]["t1_map_path"]) == expected_t1
    assert Path(summary["outputs"]["rsquared_map_path"]) == expected_rsq
    assert expected_t1.exists()
    assert expected_rsq.exists()


@pytest.mark.integration
def test_parametric_t1_pipeline_realdata_default_output_naming_stacked(tmp_path: Path) -> None:
    payload = {
        "output_dir": str(tmp_path / "out_default_stacked"),
        "vfa_files": [
            "tests/data/BIDS_test/derivatives/sub-01/ses-01/anat/sub-01_ses-01_space-DCEref_desc-bfczunified_VFA.nii"
        ],
        "flip_angles_deg": [2.0, 5.0, 10.0],
        "tr_ms": 8.012,
        "output_basename": "T1_map",
        "fit_type": "t1_fa_linear_fit",
        "rsquared_threshold": 0.6,
        "write_r_squared": True,
        "write_rho_map": True,
    }

    config = ParametricT1Config.from_dict(payload, base_dir=REPO_ROOT)
    summary = run_parametric_t1_pipeline(config)
    expected_t1 = (
        tmp_path
        / "out_default_stacked"
        / "T1_map_t1_fa_linear_fit_sub-01_ses-01_space-DCEref_desc-bfczunified_VFA.nii.gz"
    )
    expected_rsq = (
        tmp_path
        / "out_default_stacked"
        / "Rsquared_t1_fa_linear_fit_sub-01_ses-01_space-DCEref_desc-bfczunified_VFA.nii.gz"
    )
    expected_rho = (
        tmp_path
        / "out_default_stacked"
        / "T1_map_rho_t1_fa_linear_fit_sub-01_ses-01_space-DCEref_desc-bfczunified_VFA.nii.gz"
    )

    assert Path(summary["outputs"]["t1_map_path"]) == expected_t1
    assert Path(summary["outputs"]["rsquared_map_path"]) == expected_rsq
    assert Path(summary["outputs"]["rho_map_path"]) == expected_rho
    assert expected_t1.exists()
    assert expected_rsq.exists()
    assert expected_rho.exists()


@pytest.mark.integration
def test_parametric_t1_pipeline_supports_nonlinear_fit_type_tiny_fixture(tmp_path: Path) -> None:
    vfa_path = tmp_path / "tiny_nonlinear_vfa.nii.gz"
    flip_angles, tr_ms = _write_tiny_vfa_fixture(vfa_path)
    payload = {
        "output_dir": str(tmp_path / "out_nonlinear"),
        "vfa_files": [str(vfa_path)],
        "flip_angles_deg": flip_angles,
        "tr_ms": tr_ms,
        "fit_type": "t1_fa_fit",
        "output_basename": "T1_map",
        "output_label": "tiny_nonlinear",
        "rsquared_threshold": 0.2,
        "write_r_squared": True,
        "write_rho_map": True,
    }

    config = ParametricT1Config.from_dict(payload)
    summary = run_parametric_t1_pipeline(config)

    assert Path(summary["outputs"]["t1_map_path"]).exists()
    assert Path(summary["outputs"]["rsquared_map_path"]).exists()
    assert Path(summary["outputs"]["rho_map_path"]).exists()
    assert summary["metrics"]["valid_fits"] > 0


@pytest.mark.integration
def test_parametric_t1_pipeline_supports_two_point_fit_type_tiny_fixture(tmp_path: Path) -> None:
    vfa_path = tmp_path / "tiny_two_point_vfa.nii.gz"
    flip_angles, tr_ms = _write_tiny_vfa_fixture(vfa_path)
    payload = {
        "output_dir": str(tmp_path / "out_two_point"),
        "vfa_files": [str(vfa_path)],
        "flip_angles_deg": flip_angles,
        "tr_ms": tr_ms,
        "fit_type": "t1_fa_two_point_fit",
        "output_basename": "T1_map",
        "output_label": "tiny_two_point",
        "rsquared_threshold": 0.2,
        "write_r_squared": True,
        "write_rho_map": False,
    }

    config = ParametricT1Config.from_dict(payload)
    summary = run_parametric_t1_pipeline(config)

    assert Path(summary["outputs"]["t1_map_path"]).exists()
    assert Path(summary["outputs"]["rsquared_map_path"]).exists()
    assert summary["outputs"]["rho_map_path"] is None
    assert summary["metrics"]["valid_fits"] > 0


@pytest.mark.integration
def test_parametric_t1_pipeline_b1_map_file_is_applied_to_flip_angles(tmp_path: Path) -> None:
    vfa_path = tmp_path / "tiny_b1_vfa.nii.gz"
    b1_path = tmp_path / "tiny_b1_map.nii.gz"
    flip_angles, tr_ms = _write_tiny_vfa_fixture(vfa_path)
    _write_tiny_b1_fixture(b1_path, scale=0.8)

    payload_without_b1 = {
        "output_dir": str(tmp_path / "out_no_b1"),
        "vfa_files": [str(vfa_path)],
        "flip_angles_deg": flip_angles,
        "tr_ms": tr_ms,
        "fit_type": "t1_fa_linear_fit",
        "output_basename": "T1_map",
        "output_label": "tiny_no_b1",
        "rsquared_threshold": 0.2,
        "write_r_squared": True,
        "write_rho_map": False,
    }
    payload_with_b1 = {
        **payload_without_b1,
        "output_dir": str(tmp_path / "out_with_b1"),
        "output_label": "tiny_with_b1",
        "b1_map_file": str(b1_path),
    }

    summary_without_b1 = run_parametric_t1_pipeline(ParametricT1Config.from_dict(payload_without_b1))
    summary_with_b1 = run_parametric_t1_pipeline(ParametricT1Config.from_dict(payload_with_b1))

    mean_without_b1 = float(summary_without_b1["metrics"]["t1_mean_ms"])
    mean_with_b1 = float(summary_with_b1["metrics"]["t1_mean_ms"])
    assert not np.isclose(mean_without_b1, mean_with_b1)
    assert summary_with_b1["resolved_inputs"]["b1_mode"] == "explicit"
    assert Path(summary_with_b1["resolved_inputs"]["b1_map_path"]) == b1_path.resolve()


@pytest.mark.integration
def test_parametric_t1_pipeline_auto_detects_default_b1_map_name(tmp_path: Path) -> None:
    vfa_path = tmp_path / "tiny_auto_b1_vfa.nii.gz"
    b1_path = tmp_path / "B1_scaled_FAreg.nii.gz"
    flip_angles, tr_ms = _write_tiny_vfa_fixture(vfa_path)
    _write_tiny_b1_fixture(b1_path, scale=0.9)

    payload = {
        "output_dir": str(tmp_path / "out_auto_b1"),
        "vfa_files": [str(vfa_path)],
        "flip_angles_deg": flip_angles,
        "tr_ms": tr_ms,
        "fit_type": "t1_fa_linear_fit",
        "output_basename": "T1_map",
        "output_label": "tiny_auto_b1",
        "rsquared_threshold": 0.2,
        "write_r_squared": False,
        "write_rho_map": False,
    }

    summary = run_parametric_t1_pipeline(ParametricT1Config.from_dict(payload))
    assert summary["resolved_inputs"]["b1_mode"] == "auto"
    assert Path(summary["resolved_inputs"]["b1_map_path"]) == b1_path.resolve()
