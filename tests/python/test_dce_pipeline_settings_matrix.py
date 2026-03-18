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
            str(REPO_ROOT / "tests/data" / "ci_fixtures" / "dce" / "tiny_settings_case"),
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


def _drop_stage_overrides(config: DcePipelineConfig, *keys: str) -> None:
    for key in keys:
        config.stage_overrides.pop(key, None)


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
        assert stage_a_default["blood_t1_source"] == "aif_t1_map"
        assert stage_a_default["blood_t1_override_sec"] is None
        assert int(stage_a_default["blood_t1_voxel_count"]) == int(np.asarray(stage_a_default["arrays"]["T1LV"]).size)
        assert float(stage_a_default["blood_t1_mean_sec"]) == pytest.approx(
            float(np.mean(np.asarray(stage_a_default["arrays"]["T1LV"], dtype=np.float64)))
        )
        assert float(stage_a_default["blood_t1_median_sec"]) == pytest.approx(
            float(np.median(np.asarray(stage_a_default["arrays"]["T1LV"], dtype=np.float64)))
        )
        assert float(stage_a_default["blood_t1_mean_ms"]) == pytest.approx(float(stage_a_default["blood_t1_mean_sec"]) * 1000.0)
        assert float(stage_a_default["blood_t1_median_ms"]) == pytest.approx(float(stage_a_default["blood_t1_median_sec"]) * 1000.0)

        config_override = _make_config(tiny_root, Path(tmp) / "override", {"blood_t1_ms": 1600.0})
        stage_a_override = _run_stage_a_real(config_override)

        assert float(stage_a_override["blood_t1_override_sec"]) == pytest.approx(1.6)
        assert stage_a_override["blood_t1_source"] == "override"
        assert int(stage_a_override["blood_t1_voxel_count"]) == int(np.asarray(stage_a_override["arrays"]["T1LV"]).size)
        assert float(stage_a_override["blood_t1_mean_sec"]) == pytest.approx(1.6)
        assert float(stage_a_override["blood_t1_median_sec"]) == pytest.approx(1.6)
        assert float(stage_a_override["blood_t1_mean_ms"]) == pytest.approx(1600.0)
        assert float(stage_a_override["blood_t1_median_ms"]) == pytest.approx(1600.0)
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
        assert stage_a_override["blood_t1_source"] == "override"
        assert float(stage_a_override["blood_t1_mean_ms"]) == pytest.approx(1550.0)
        assert np.allclose(stage_a_override["arrays"]["T1LV"], 1.55)


@pytest.mark.integration
def test_blood_t1_override_rejects_nonpositive(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config_bad = _make_config(tiny_root, Path(tmp) / "bad", {"blood_t1_ms": 0.0})
        with pytest.raises(ValueError, match="blood_t1 override must be positive"):
            _run_stage_a_real(config_bad)


@pytest.mark.integration
def test_script_level_tr_fa_time_resolution_aliases(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(
            tiny_root,
            Path(tmp) / "alias_meta",
            {
                "tr": 9.7,
                "fa": 18.5,
                "time_resolution": 6.0,
            },
        )
        _drop_stage_overrides(config, "tr_ms", "fa_deg", "time_resolution_sec")
        stage_a = _run_stage_a_real(config)

        assert float(stage_a["tr_ms"]) == pytest.approx(9.7)
        assert float(stage_a["fa_deg"]) == pytest.approx(18.5)
        assert float(stage_a["time_resolution_min"]) == pytest.approx(6.0 / 60.0)


@pytest.mark.integration
def test_script_level_blood_t1_alias(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(tiny_root, Path(tmp) / "alias_blood_t1", {"blood_t1": 1.62})
        stage_a = _run_stage_a_real(config)
        assert float(stage_a["blood_t1_override_sec"]) == pytest.approx(1.62)
        assert stage_a["blood_t1_source"] == "override"
        assert float(stage_a["blood_t1_mean_ms"]) == pytest.approx(1620.0)
        assert np.allclose(stage_a["arrays"]["T1LV"], 1.62)


@pytest.mark.integration
def test_script_level_start_t_end_t_aliases_clip_stage_a_timepoints(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(
            tiny_root,
            Path(tmp) / "alias_start_end_t",
            {
                "start_t": 3,
                "end_t": 10,
            },
        )
        stage_a = _run_stage_a_real(config)

        time_window = stage_a["timepoint_window"]
        assert int(time_window["start_1b"]) == 3
        assert int(time_window["end_1b"]) == 10
        assert int(time_window["n_timepoints_input"]) == 18
        assert int(time_window["n_timepoints_output"]) == 8
        assert np.asarray(stage_a["arrays"]["Cp"], dtype=np.float64).shape[0] == 8
        assert np.asarray(stage_a["arrays"]["timer"], dtype=np.float64).shape[0] == 8


@pytest.mark.integration
def test_script_level_aif_type_and_injection_aliases(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(
            tiny_root,
            Path(tmp) / "alias_aif",
            {
                "aif_type": 2,
                "start_injection": 0.75,
                "end_injection": 1.05,
            },
        )
        _drop_stage_overrides(config, "aif_curve_mode", "start_injection_min", "end_injection_min")

        stage_a = _run_stage_a_real(config)
        stage_b = _run_stage_b_real(config, stage_a)

        assert stage_b["aif_mode"] == "raw"
        assert stage_b["aif_name"] == "raw"
        assert float(stage_b["start_injection_min"]) == pytest.approx(0.75)
        assert float(stage_b["end_injection_min"]) == pytest.approx(1.05)
        assert np.allclose(stage_b["arrays"]["Cp_use"], stage_b["arrays"]["CpROI"])


@pytest.mark.integration
def test_script_level_timevectyn_controls_timevectpath(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        timer_file = tmp_path / "timer.csv"
        custom_timer = np.linspace(0.0, 0.85, num=18, dtype=np.float64)
        np.savetxt(timer_file, custom_timer, delimiter=",")

        config_enabled = _make_config(
            tiny_root,
            tmp_path / "enabled",
            {
                "timevectpath": str(timer_file),
                "timevectyn": 1,
                "aif_type": 2,
            },
        )
        _drop_stage_overrides(config_enabled, "aif_curve_mode")
        stage_a_enabled = _run_stage_a_real(config_enabled)
        stage_b_enabled = _run_stage_b_real(config_enabled, stage_a_enabled)
        assert np.allclose(stage_b_enabled["arrays"]["timer"], custom_timer)

        config_disabled = _make_config(
            tiny_root,
            tmp_path / "disabled",
            {
                "timevectpath": str(timer_file),
                "timevectyn": 0,
                "aif_type": 2,
            },
        )
        _drop_stage_overrides(config_disabled, "aif_curve_mode")
        stage_a_disabled = _run_stage_a_real(config_disabled)
        stage_b_disabled = _run_stage_b_real(config_disabled, stage_a_disabled)
        assert not np.allclose(stage_b_disabled["arrays"]["timer"], custom_timer)
