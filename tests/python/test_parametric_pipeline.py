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
