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


TINY_SUBJECT = "sub-11tiny"
TINY_SESSION = "ses-01"


def _tiny_root() -> Path:
    return Path(
        os.environ.get(
            "ROCKETSHIP_TINY_SETTINGS_ROOT",
            str(REPO_ROOT / "tests/data" / "BIDS_test"),
        )
    ).expanduser().resolve()


def _tiny_paths(root: Path) -> dict:
    """Resolve the tiny DCE fixture inputs within the BIDS_test sub-11tiny subject."""
    raw = root / "rawdata" / TINY_SUBJECT / TINY_SESSION
    der = root / "derivatives" / TINY_SUBJECT / TINY_SESSION
    stem = f"{TINY_SUBJECT}_{TINY_SESSION}"
    return {
        "source": raw / "dce",
        "tp": der,
        "dynamic": raw / "dce" / f"{stem}_DCE.nii",
        "aif": der / "dce" / f"{stem}_label-AIF_mask.nii",
        "roi": der / "anat" / f"{stem}_label-brain_mask.nii",
        "t1map": der / "dce" / f"{stem}_space-DCEref_T1map.nii",
        "noise": der / "anat" / f"{stem}_label-noise_mask.nii",
        "meta": der / "dce" / f"{stem}_desc-tinymeta.json",
    }


def _load_meta(root: Path) -> dict:
    return json.loads(_tiny_paths(root)["meta"].read_text())


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
        # The injection start is always the resolved baseline end, so it is pinned via
        # steady_state_end rather than overridden directly.
        "steady_state_end": int(round(float(meta["start_injection_min"])
                                      / (float(meta["time_resolution_sec"]) / 60.0))) + 1,
        "end_injection_min": float(meta["end_injection_min"]),
        "relaxivity": float(meta["relaxivity"]),
        "hematocrit": float(meta["hematocrit"]),
        "snr_filter": 0.0,
        "time_smoothing": "none",
        "time_smoothing_window": 0,
        "voxel_MaxFunEvals": 200,
    }
    if extra_overrides:
        overrides.update(extra_overrides)

    paths = _tiny_paths(root)
    return DcePipelineConfig(
        subject_source_path=paths["source"],
        subject_tp_path=paths["tp"],
        output_dir=output_dir,
        backend="cpu",
        checkpoint_dir=output_dir / "checkpoints",
        write_xls=False,
        dynamic_files=[paths["dynamic"]],
        aif_files=[paths["aif"]],
        roi_files=[paths["roi"]],
        t1map_files=[paths["t1map"]],
        noise_files=[paths["noise"]],
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
    if not _tiny_paths(root)["dynamic"].exists():
        pytest.skip(f"Missing tiny settings fixture under: {root}", allow_module_level=True)
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
def test_manual_metadata_overrides_replace_the_sidecar(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(
            tiny_root,
            Path(tmp) / "manual_meta",
            {
                "tr_ms": 9.7,
                "fa_deg": 18.5,
                "time_resolution_sec": 6.0,
            },
        )
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
def test_raw_aif_mode_with_an_explicit_injection_end(tiny_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(
            tiny_root,
            Path(tmp) / "raw_aif",
            {
                "aif_curve_mode": "raw",
                "steady_state_end": 4,
                "end_injection_min": 1.05,
            },
        )

        stage_a = _run_stage_a_real(config)
        stage_b = _run_stage_b_real(config, stage_a)

        assert stage_b["aif_mode"] == "raw"
        assert stage_b["aif_name"] == "raw"
        # start_injection is derived: steady_state_end = 4 (1-based last baseline frame) sits at
        # timer[3], not overridden independently.
        dt = float(stage_a["time_resolution_min"])
        assert float(stage_b["start_injection_min"]) == pytest.approx(3.0 * dt)
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
                "aif_curve_mode": "raw",
            },
        )
        stage_a_enabled = _run_stage_a_real(config_enabled)
        stage_b_enabled = _run_stage_b_real(config_enabled, stage_a_enabled)
        assert np.allclose(stage_b_enabled["arrays"]["timer"], custom_timer)

        config_disabled = _make_config(
            tiny_root,
            tmp_path / "disabled",
            {
                "timevectpath": str(timer_file),
                "timevectyn": 0,
                "aif_curve_mode": "raw",
            },
        )
        stage_a_disabled = _run_stage_a_real(config_disabled)
        stage_b_disabled = _run_stage_b_real(config_disabled, stage_a_disabled)
        assert not np.allclose(stage_b_disabled["arrays"]["timer"], custom_timer)
