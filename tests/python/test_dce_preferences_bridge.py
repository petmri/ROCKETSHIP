"""Tests for MATLAB dce_preferences.txt bridging into Python DCE pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _fit_aif_biexp,
    _lsq_result_usable,
    _stage_d_fit_prefs,
    run_dce_pipeline,
)


def _write_prefs(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_config(tmp: Path, prefs_path: Path, extra_overrides: dict | None = None, *, backend: str = "auto") -> DcePipelineConfig:
    subject_source = tmp / "subject_source"
    subject_tp = tmp / "subject_tp"
    output_dir = tmp / "output"

    dynamic = subject_tp / "dynamic.nii.gz"
    aif = subject_tp / "aif_mask.nii.gz"
    t1map = subject_tp / "t1map.nii.gz"
    dynamic.parent.mkdir(parents=True, exist_ok=True)
    dynamic.write_text("")
    aif.write_text("")
    t1map.write_text("")

    overrides = {
        "stage_a_mode": "scaffold",
        "stage_b_mode": "scaffold",
        "stage_d_mode": "scaffold",
        "dce_preferences_path": str(prefs_path),
    }
    if extra_overrides:
        overrides.update(extra_overrides)

    return DcePipelineConfig(
        subject_source_path=subject_source,
        subject_tp_path=subject_tp,
        output_dir=output_dir,
        backend=backend,
        checkpoint_dir=tmp / "checkpoints",
        write_xls=False,
        dynamic_files=[dynamic],
        aif_files=[aif],
        roi_files=[],
        t1map_files=[t1map],
        noise_files=[],
        drift_files=[],
        model_flags={"tofts": 1},
        stage_overrides=overrides,
    )


@pytest.mark.integration
def test_stage_d_preferences_loaded_with_expression_parsing() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prefs = _write_prefs(
            tmp / "dce_preferences.txt",
            [
                "voxel_lower_limit_ktrans = 10^-5",
                "voxel_upper_limit_ve = 0.73",
                "voxel_MaxFunEvals = 77",
                "fxr_fw = 0.66",
            ],
        )
        config = _make_config(tmp, prefs)
        fit_prefs = _stage_d_fit_prefs(config)

        assert float(fit_prefs["lower_limit_ktrans"]) == pytest.approx(1e-5)
        assert float(fit_prefs["upper_limit_ve"]) == pytest.approx(0.73)
        assert int(fit_prefs["max_nfev"]) == 77
        assert float(fit_prefs["fxr_fw"]) == pytest.approx(0.66)


@pytest.mark.integration
def test_stage_overrides_take_precedence_over_preferences_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prefs = _write_prefs(tmp / "dce_preferences.txt", ["voxel_MaxFunEvals = 77"])
        config = _make_config(tmp, prefs, {"voxel_MaxFunEvals": 123})
        fit_prefs = _stage_d_fit_prefs(config)
        assert int(fit_prefs["max_nfev"]) == 123


@pytest.mark.integration
@patch(
    "dce_pipeline.probe_acceleration_backend",
    return_value={
        "backend": "gpufit_cuda",
        "reason": "test_gpu_available",
        "cuda_available": True,
        "pygpufit_imported": True,
        "pycpufit_imported": True,
        "pygpufit_error": None,
        "pycpufit_error": None,
    },
)
def test_force_cpu_preference_overrides_backend_auto(_probe_mock: object) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prefs = _write_prefs(tmp / "dce_preferences.txt", ["force_cpu = 1"])
        config = _make_config(tmp, prefs, backend="auto")
        result = run_dce_pipeline(config)
        assert result["stages"]["D"]["selected_backend"] == "cpu"


@pytest.mark.integration
@patch("scipy.optimize.least_squares")
def test_aif_advanced_preferences_flow_into_fit_kwargs(least_squares_mock: object) -> None:
    class _DummyResult:
        x = np.array([1.0, 1.0, 1.0, 0.1, 0.3], dtype=np.float64)
        success = True
        jac = np.zeros((8, 5), dtype=np.float64)

    least_squares_mock.return_value = _DummyResult()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prefs = _write_prefs(
            tmp / "dce_preferences.txt",
            [
                "aif_MaxFunEvals = 123",
                "aif_TolFun = 10^-8",
                "aif_TolX = 10^-9",
                "aif_Robust = Bisquare",
            ],
        )
        config = _make_config(tmp, prefs)
        timer = np.linspace(0.0, 1.0, num=8, dtype=np.float64)
        curve = np.array([0.0, 0.0, 0.4, 1.2, 0.8, 0.6, 0.4, 0.3], dtype=np.float64)

        _fit_aif_biexp(
            config=config,
            timer=timer,
            curve=curve,
            start_injection_min=timer[2],
            end_injection_min=timer[4],
            fitting_au=False,
        )

        kwargs = least_squares_mock.call_args.kwargs
        assert kwargs["method"] == "trf"
        assert int(kwargs["max_nfev"]) == 123
        assert float(kwargs["ftol"]) == pytest.approx(1e-8)
        assert float(kwargs["xtol"]) == pytest.approx(1e-9)
        # aif_Robust=Bisquare drives the hand-written Tukey IRLS loop, which solves an
        # ordinary weighted least-squares problem each iteration -- so no scipy `loss=` is
        # passed. scipy's loss functions are not the same estimator as MATLAB's Bisquare.
        assert "loss" not in kwargs
        # A, B, c, d, delta -- t_base_end is an input, not a fitted parameter.
        assert kwargs["x0"].shape == (5,)
        assert kwargs["bounds"][0].shape == (5,)
        assert kwargs["bounds"][1].shape == (5,)


@pytest.mark.integration
def test_aif_fit_holds_t_base_end_and_fits_delta() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prefs = _write_prefs(tmp / "dce_preferences.txt", [])
        config = _make_config(tmp, prefs)
        timer = np.linspace(0.0, 1.0, num=8, dtype=np.float64)
        curve = np.array([0.0, 0.0, 0.2, 1.1, 0.9, 0.6, 0.4, 0.25], dtype=np.float64)

        result = _fit_aif_biexp(
            config=config,
            timer=timer,
            curve=curve,
            start_injection_min=timer[2],
            end_injection_min=timer[4],
            fitting_au=False,
        )

        assert result["fit_t_base_end"] is False
        # t_base_end is pinned to the resolved baseline end; only delta is free.
        assert float(result["t_base_end"]) == pytest.approx(float(timer[2]))
        assert float(result["t0_exp"]) > float(result["t_base_end"])
        # delta bottoms out at one frame, never at zero.
        dt = float(timer[1] - timer[0])
        assert float(result["delta"]) >= dt - 1e-12
        # Coefficients are always reported as MATLAB's [A B c d t_base_end t0_exp].
        assert result["params"].shape == (6,)
        assert float(result["params"][4]) == pytest.approx(float(result["t_base_end"]))
        assert float(result["params"][5]) == pytest.approx(float(result["t0_exp"]))


def test_lsq_result_usable_accepts_exhausted_evaluation_budget() -> None:
    """status=0 is "ran out of evaluations", not "failed to fit".

    aif_TolFun/aif_TolX are inherited from MATLAB below machine epsilon, so no tolerance test
    can ever fire and trf always terminates this way -- returning a perfectly good best point.
    Gating on `result.success` (status > 0) discarded those fits and silently fell back to the
    `tv` detector.
    """

    class _Exhausted:
        status = 0  # "maximum number of function evaluations is exceeded"
        success = False
        x = np.array([1.0, 1.0, 1.0, 0.1, 0.3], dtype=np.float64)

    class _ImproperInput:
        status = -1
        success = False
        x = np.array([1.0, 1.0, 1.0, 0.1, 0.3], dtype=np.float64)

    class _NonFinite:
        status = 2
        success = True
        x = np.array([1.0, np.nan, 1.0, 0.1, 0.3], dtype=np.float64)

    assert _lsq_result_usable(_Exhausted()) is True
    assert _lsq_result_usable(_ImproperInput()) is False
    assert _lsq_result_usable(_NonFinite()) is False
    assert _lsq_result_usable(None) is False


@pytest.mark.integration
@patch("scipy.optimize.least_squares")
def test_aif_fit_succeeds_when_scipy_exhausts_max_nfev(least_squares_mock: object) -> None:
    """The whole-fit success flag must follow the same rule, not scipy's `success`."""

    class _Exhausted:
        status = 0
        success = False
        x = np.array([1.0, 1.0, 1.0, 0.1, 0.3], dtype=np.float64)
        jac = np.zeros((8, 5), dtype=np.float64)

    least_squares_mock.return_value = _Exhausted()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prefs = _write_prefs(tmp / "dce_preferences.txt", ["aif_Robust = off"])
        config = _make_config(tmp, prefs)
        timer = np.linspace(0.0, 1.0, num=8, dtype=np.float64)
        curve = np.array([0.0, 0.0, 0.2, 1.1, 0.9, 0.6, 0.4, 0.25], dtype=np.float64)

        result = _fit_aif_biexp(
            config=config,
            timer=timer,
            curve=curve,
            start_injection_min=timer[2],
            end_injection_min=timer[4],
            fitting_au=False,
        )

        assert result["fit_success"] is True


@pytest.mark.integration
def test_aif_timing_pass_fits_t_base_end() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prefs = _write_prefs(tmp / "dce_preferences.txt", [])
        config = _make_config(tmp, prefs)
        timer = np.linspace(0.0, 1.0, num=8, dtype=np.float64)
        curve = np.array([0.0, 0.0, 0.2, 1.1, 0.9, 0.6, 0.4, 0.25], dtype=np.float64)

        result = _fit_aif_biexp(
            config=config,
            timer=timer,
            curve=curve,
            start_injection_min=timer[2],
            end_injection_min=timer[4],
            fitting_au=False,
            fit_pass="timing",
        )

        assert result["fit_t_base_end"] is True
        assert result["params"].shape == (6,)
        assert float(result["t0_exp"]) > float(result["t_base_end"])


@pytest.mark.integration
def test_aif_biexp_timing_method_override_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prefs = _write_prefs(tmp / "dce_preferences.txt", [])
        config = _make_config(tmp, prefs, {"aif_biexp_timing_method": "fit_transition_times"})
        with pytest.raises(ValueError, match="aif_biexp_timing_method was removed"):
            config.validate()
