"""Stage-B AIF contract gate: Python Stage-B vs a frozen MATLAB Stage-B payload.

The rest of the parity suite compares only final parameter maps, which is how a
structurally different Stage-B AIF fit went undetected for months (issue #2 /
``docs/project-management/projects/archived/batch-parity/aif_fitting_parity.md``):
Ktrans maps still correlated well enough to pass while ``Cp_use`` itself was wrong.
This gates the Stage-B outputs directly, so that failure mode fails here first.

Reference payload: ``derivatives/matlabref/.../Dyn-1_stage_b_aif.json``, written by
``tests/matlab/helpers/write_stage_b_aif_contract.m`` during MATLAB baseline
generation. It is committed, so this test needs no MATLAB. MATLAB-side drift (a
stale committed payload) is caught separately by
``tests/contracts/check_matlabref_map_drift.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import _run_stage_a_real, _run_stage_b_real  # noqa: E402

# Reuse the parity suite's own fixture resolution and Stage-A/B settings rather than
# restating them: a private copy would drift, and identical settings are the premise
# of the comparison.
from test_dce_pipeline_parity_metrics import (  # noqa: E402
    _dataset_paths,
    _default_downsample_root,
    _make_config,
)

CONTRACT_FILENAME = "Dyn-1_stage_b_aif.json"
CONTRACT_VERSION = "stage_b_aif_v1"

# Measured agreement on sub-10bbbdownsample is ~1e-8 relative on every curve, so these
# sit 3-4 orders of magnitude above the noise floor: tight enough that a changed AIF
# model, a shifted injection window or an off-by-one baseline fails, loose enough to
# absorb MATLAB-release/BLAS variation in the shared nonlinear fit.
CURVE_REL_MAE_MAX = 1e-5
CURVE_REL_MAX_ABS_MAX = 1e-4
CURVE_CORR_MIN = 0.99999

# Timing terms are minutes on a 0.264-min grid; agreement is exact today. 1e-6 min is
# 60 microseconds -- far below a frame, so a one-frame shift (the actual failure mode)
# cannot pass, while float formatting through JSON cannot fail it.
TIME_ATOL = 1e-6
# AIF fit coefficients are nonlinear-fit outputs; matches tolerance_profiles.json's
# fit_recovery_strict rtol.
FIT_PARAM_RTOL = 1e-4
FIT_PARAM_ATOL = 1e-8

GATED_CURVES = ("timer", "CpROI", "Cp_use", "Stlv_use")


def _log(message: str) -> None:
    print(f"[STAGE-B-CONTRACT] {message}", flush=True)


def _curve_metrics(ref: np.ndarray, got: np.ndarray) -> dict:
    """Scale-relative metrics so concentration and signal curves share one threshold."""
    scale = float(np.max(np.abs(ref)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    diff = np.abs(ref - got)
    corr = float("nan")
    if ref.size > 2 and float(np.std(ref)) > 0.0 and float(np.std(got)) > 0.0:
        corr = float(np.corrcoef(ref, got)[0, 1])
    return {
        "n": int(ref.size),
        "scale": scale,
        "rel_mae": float(np.mean(diff)) / scale,
        "rel_max_abs": float(np.max(diff)) / scale,
        "corr": corr,
        "argmax_ref": int(np.argmax(ref)) + 1,
        "argmax_got": int(np.argmax(got)) + 1,
    }


def _run_python_stage_b(paths: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(paths, Path(tmp) / "stage_b_out", backend="cpu", models=["tofts"])
        # The QC figure is an artifact of the run, not part of the contract.
        config.stage_overrides = {**config.stage_overrides, "save_aif_figure": False}
        stage_a = _run_stage_a_real(config)
        return _run_stage_b_real(config, stage_a)


@pytest.mark.parity
@pytest.mark.integration
def test_stage_b_aif_contract_matches_matlab(parity_dataset_root: str) -> None:
    root = Path(parity_dataset_root) if parity_dataset_root else _default_downsample_root()
    paths = _dataset_paths(root)
    contract_path = Path(paths["matlabref"]) / CONTRACT_FILENAME

    required = [contract_path, paths["dynamic"], paths["aif"], paths["roi"], paths["t1map"], paths["noise"]]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        pytest.skip(f"Stage-B contract assets missing ({len(missing)}); first: {missing[0]}")

    reference = json.loads(contract_path.read_text())
    version = str(reference.get("meta", {}).get("version", ""))
    assert version == CONTRACT_VERSION, (
        f"Stage-B contract payload is version {version!r}, expected {CONTRACT_VERSION!r}. "
        "Regenerate with generate_dce_tofts_parity_map(..., 'stageBOnly', true)."
    )
    assert str(reference["meta"]["aif_name"]) == "fitted", (
        "Reference payload was not produced by the fitted-AIF branch; the contract only "
        f"describes fitted-mode Stage-B (got aif_name={reference['meta']['aif_name']!r})."
    )

    stage_b = _run_python_stage_b(paths)
    arrays = stage_b["arrays"]
    assert stage_b["aif_mode"] == "fitted", (
        f"Python Stage-B ran in {stage_b['aif_mode']!r} mode; the reference is fitted-mode."
    )

    failures: list[str] = []

    # --- Curves -----------------------------------------------------------------
    ref_curves = reference["curves"]
    for name in GATED_CURVES:
        ref = np.asarray(ref_curves[name], dtype=np.float64).ravel()
        got = np.asarray(arrays[name], dtype=np.float64).ravel()
        if ref.size != got.size:
            failures.append(f"{name}: length {got.size} != reference {ref.size}")
            _log(f"{name}: FAILED (length {got.size} vs {ref.size})")
            continue
        if not np.all(np.isfinite(got)):
            failures.append(f"{name}: python produced {int(np.sum(~np.isfinite(got)))} non-finite sample(s)")
            continue
        m = _curve_metrics(ref, got)
        _log(
            f"{name}: n={m['n']} rel_mae={m['rel_mae']:.3e} rel_max={m['rel_max_abs']:.3e} "
            f"corr={m['corr']:.8f}"
        )
        if m["rel_mae"] > CURVE_REL_MAE_MAX:
            failures.append(f"{name}: rel_mae={m['rel_mae']:.3e} > {CURVE_REL_MAE_MAX:.1e}")
        if m["rel_max_abs"] > CURVE_REL_MAX_ABS_MAX:
            failures.append(f"{name}: rel_max_abs={m['rel_max_abs']:.3e} > {CURVE_REL_MAX_ABS_MAX:.1e}")
        if not (m["corr"] >= CURVE_CORR_MIN):
            failures.append(f"{name}: corr={m['corr']:.8f} < {CURVE_CORR_MIN}")

    # --- Injection window / indices ---------------------------------------------
    ref_window = reference["window"]
    step_ref = np.asarray(ref_window["step"], dtype=np.float64).ravel()
    step_got = np.asarray(arrays["step"], dtype=np.float64).ravel()
    if step_got.size != step_ref.size:
        failures.append(f"step: length {step_got.size} != reference {step_ref.size}")
    else:
        step_diff = float(np.max(np.abs(step_ref - step_got)))
        _log(f"step: matlab={step_ref.tolist()} python={step_got.tolist()} max_abs={step_diff:.3e}")
        if step_diff > TIME_ATOL:
            failures.append(f"step: max_abs={step_diff:.3e} > {TIME_ATOL:.1e} (injection window moved)")

    scalar_checks = [
        ("start_injection", float(ref_window["start_injection"]), float(stage_b["start_injection_min"]), TIME_ATOL),
        ("end_injection", float(ref_window["end_injection"]), float(stage_b["end_injection_min"]), TIME_ATOL),
        ("time_resolution", float(ref_window["time_resolution"]), float(stage_b["time_resolution_min"]), TIME_ATOL),
    ]
    for name, ref_val, got_val, atol in scalar_checks:
        _log(f"{name}: matlab={ref_val:.10g} python={got_val:.10g}")
        if not (abs(ref_val - got_val) <= atol):
            failures.append(f"{name}: python={got_val:.10g} vs matlab={ref_val:.10g} (atol={atol:.1e})")

    # Indices are 1-based and must match exactly; an off-by-one here is the bug class
    # this contract exists to catch, so there is no tolerance to give.
    py_max_index = int(np.argmax(np.asarray(arrays["Cp_use"], dtype=np.float64).ravel())) + 1
    index_checks = [
        ("max_index", int(ref_window["max_index"]), py_max_index),
        ("start_time", int(ref_window["start_time"]), int(stage_b["start_time_index"])),
        ("end_time", int(ref_window["end_time"]), int(stage_b["end_time_index"])),
        ("numvoxels", int(ref_window["numvoxels"]), int(stage_b["numvoxels"])),
    ]
    for name, ref_i, got_i in index_checks:
        _log(f"{name}: matlab={ref_i} python={got_i}")
        if ref_i != got_i:
            failures.append(f"{name}: python={got_i} != matlab={ref_i}")

    # --- AIF fit coefficients ----------------------------------------------------
    # [A B c d t_base_end t0_exp]. t_base_end/t0_exp are the baseline and upslope
    # transition times -- the terms the Stage-B parity gap was actually about.
    ref_params = np.asarray(reference["fit"]["params_cp"], dtype=np.float64).ravel()
    got_params = np.asarray(stage_b["fit_params_cp"], dtype=np.float64).ravel()
    param_names = ["A", "B", "c", "d", "t_base_end", "t0_exp"]
    if got_params.size != ref_params.size:
        failures.append(f"fit_params_cp: length {got_params.size} != reference {ref_params.size}")
    else:
        for name, ref_v, got_v in zip(param_names, ref_params, got_params):
            tol = FIT_PARAM_ATOL + FIT_PARAM_RTOL * abs(ref_v)
            _log(f"fit.{name}: matlab={ref_v:.10g} python={got_v:.10g}")
            if not (abs(ref_v - got_v) <= tol):
                failures.append(f"fit.{name}: python={got_v:.10g} vs matlab={ref_v:.10g} (tol={tol:.3e})")

    # Reported, not gated: MATLAB's gof.adjrsquare and the Python port use different
    # degrees-of-freedom conventions, so a gate here would measure that convention
    # rather than the port. The curve checks above already constrain the fit.
    _log(
        f"rsquare_cp (reported): matlab={float(reference['fit']['rsquare_cp']):.8g} "
        f"python={float(stage_b['fit_rsquared_cp_adj']):.8g}"
    )

    assert not failures, "Stage-B AIF contract mismatch:\n  " + "\n  ".join(failures)
