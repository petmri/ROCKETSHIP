"""Fast tiny-fixture settings coverage for DCE Stage A/B/D behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import DcePipelineConfig, _run_stage_a_real, _run_stage_b_real, _run_stage_d_real  # noqa: E402


def _tiny_root() -> Path:
    return Path(
        os.environ.get(
            "ROCKETSHIP_TINY_SETTINGS_ROOT",
            str(REPO_ROOT / "test_data" / "ci_fixtures" / "dce" / "tiny_settings_case"),
        )
    ).expanduser().resolve()


def _load_meta(root: Path) -> dict:
    return json.loads((root / "processed" / "tiny_fixture_meta.json").read_text())


def _make_config(root: Path, output_dir: Path, extra_overrides: dict | None = None) -> DcePipelineConfig:
    meta = _load_meta(root)
    overrides = {
        "rootname": "Dyn-1",
        "stage_a_mode": "real",
        "stage_b_mode": "real",
        "stage_d_mode": "real",
        "aif_curve_mode": "fitted",
        "tr_ms": float(meta["tr_ms"]),
        "fa_deg": float(meta["fa_deg"]),
        "time_resolution_sec": float(meta["time_resolution_sec"]),
        "start_injection_min": float(meta["start_injection_min"]),
        "end_injection_min": float(meta["end_injection_min"]),
        "steady_state_start": 1,
        "steady_state_end": 3,
        "relaxivity": float(meta["relaxivity"]),
        "hematocrit": float(meta["hematocrit"]),
        "snr_filter": 0.0,
        "time_smoothing": "none",
        "time_smoothing_window": 0,
        "voxel_MaxFunEvals": 200,
    }
    if extra_overrides:
        overrides.update(extra_overrides)

    return DcePipelineConfig(
        subject_source_path=root,
        subject_tp_path=root / "processed",
        output_dir=output_dir,
        backend="cpu",
        checkpoint_dir=output_dir / "checkpoints",
        write_xls=False,
        dynamic_files=[root / "Dynamic_t1w.nii"],
        aif_files=[root / "processed" / "T1_AIF_roi.nii"],
        roi_files=[root / "processed" / "T1_brain_roi.nii"],
        t1map_files=[root / "processed" / "T1_map_t1_fa_fit_fa10.nii"],
        noise_files=[root / "processed" / "T1_noise_roi.nii"],
        model_flags={
            "tofts": 1,
            "ex_tofts": 0,
            "patlak": 0,
            "tissue_uptake": 0,
            "two_cxm": 0,
            "fxr": 0,
            "auc": 0,
            "nested": 0,
            "FXL_rr": 0,
        },
        stage_overrides=overrides,
    )


def _run_abd(config: DcePipelineConfig) -> tuple[dict, dict, dict]:
    stage_a = _run_stage_a_real(config)
    stage_b = _run_stage_b_real(config, stage_a)
    stage_d = _run_stage_d_real(config, stage_a, stage_b)
    return stage_a, stage_b, stage_d
@pytest.fixture(scope="module")
def tiny_root() -> Path:
    root = _tiny_root()
    if not root.exists():
        pytest.skip(f"Missing tiny settings fixture: {root}", allow_module_level=True)
    return root


@pytest.mark.integration
def test_tiny_fixture_end_to_end_tofts_smoke(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(tiny_root, Path(tmp) / "out")
        _, _, stage_d = _run_abd(config)
        vox = np.asarray(stage_d["arrays"]["tofts_voxel_results"], dtype=np.float64)
        assert vox.shape[0] > 0
        finite_rows = np.isfinite(vox[:, 0]) & np.isfinite(vox[:, 1])
        assert int(np.sum(finite_rows)) > 0


@pytest.mark.integration
def test_tofts_constraints_enforce_bounds(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(
            tiny_root,
            Path(tmp) / "out",
            {
                "voxel_lower_limit_ktrans": 0.12,
                "voxel_upper_limit_ktrans": 0.45,
                "voxel_initial_value_ktrans": 0.2,
                "voxel_lower_limit_ve": 0.30,
                "voxel_upper_limit_ve": 0.80,
                "voxel_initial_value_ve": 0.45,
            },
        )
        _, _, stage_d = _run_abd(config)
        vox = np.asarray(stage_d["arrays"]["tofts_voxel_results"], dtype=np.float64)
        finite = np.isfinite(vox[:, 0]) & np.isfinite(vox[:, 1])
        assert int(np.sum(finite)) > 0
        k = vox[finite, 0]
        ve = vox[finite, 1]
        assert float(np.min(k)) >= 0.12 - 1e-6
        assert float(np.max(k)) <= 0.45 + 1e-6
        assert float(np.min(ve)) >= 0.30 - 1e-6
        assert float(np.max(ve)) <= 0.80 + 1e-6


@pytest.mark.integration
def test_initial_guess_variants_are_stable(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config_low = _make_config(
            tiny_root,
            Path(tmp) / "low",
            {
                "voxel_initial_value_ktrans": 1e-6,
                "voxel_initial_value_ve": 0.95,
                "voxel_MaxFunEvals": 300,
            },
        )
        config_high = _make_config(
            tiny_root,
            Path(tmp) / "high",
            {
                "voxel_initial_value_ktrans": 1.5,
                "voxel_initial_value_ve": 0.03,
                "voxel_MaxFunEvals": 300,
            },
        )

        _, _, d_low = _run_abd(config_low)
        _, _, d_high = _run_abd(config_high)
        a = np.asarray(d_low["arrays"]["tofts_voxel_results"], dtype=np.float64)
        b = np.asarray(d_high["arrays"]["tofts_voxel_results"], dtype=np.float64)
        finite = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1]) & np.isfinite(b[:, 0]) & np.isfinite(b[:, 1])
        assert int(np.sum(finite)) > 0

        k_mae = float(np.mean(np.abs(a[finite, 0] - b[finite, 0])))
        ve_mae = float(np.mean(np.abs(a[finite, 1] - b[finite, 1])))
        assert k_mae <= 0.05, f"ktrans MAE too high: {k_mae:.6f}"
        assert ve_mae <= 0.05, f"ve MAE too high: {ve_mae:.6f}"


@pytest.mark.integration
def test_blood_t1_override_changes_aif_path(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config_default = _make_config(tiny_root, Path(tmp) / "default")
        stage_a_default = _run_stage_a_real(config_default)

        config_override = _make_config(tiny_root, Path(tmp) / "override", {"blood_t1_ms": 1600.0})
        stage_a_override = _run_stage_a_real(config_override)

        assert float(stage_a_override["blood_t1_override_sec"]) == pytest.approx(1.6)
        assert np.allclose(stage_a_override["arrays"]["T1LV"], 1.6)

        cp_default = np.mean(np.asarray(stage_a_default["arrays"]["Cp"], dtype=np.float64), axis=1)
        cp_override = np.mean(np.asarray(stage_a_override["arrays"]["Cp"], dtype=np.float64), axis=1)
        delta = float(np.mean(np.abs(cp_default - cp_override)))
        assert delta > 1e-3, f"Expected Cp to change with blood_t1 override, got mean abs diff {delta:.6e}"


@pytest.mark.integration
def test_blood_t1_sec_override_alias(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config_override = _make_config(tiny_root, Path(tmp) / "override_sec", {"blood_t1_sec": 1.55})
        stage_a_override = _run_stage_a_real(config_override)
        assert float(stage_a_override["blood_t1_override_sec"]) == pytest.approx(1.55)
        assert np.allclose(stage_a_override["arrays"]["T1LV"], 1.55)


@pytest.mark.integration
def test_blood_t1_override_rejects_nonpositive(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config_bad = _make_config(tiny_root, Path(tmp) / "bad", {"blood_t1_ms": 0.0})
        with pytest.raises(ValueError, match="blood_t1 override must be positive"):
            _run_stage_a_real(config_bad)
