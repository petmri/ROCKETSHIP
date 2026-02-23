"""Ground-truth reliability checks for synthetic phantom sessions in BIDS_test."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import probe_acceleration_backend  # noqa: E402

from phantom_gt_helpers import (  # noqa: E402
    PHANTOM_GT_TOLERANCES_PATH,
    discover_phantom_gt_sessions,
    load_phantom_gt_tolerances,
    run_phantom_gt_bids_summary,
)


def _resolve_dce_backend_tolerance_key(tolerances: Dict[str, Any], backend_used: str) -> str:
    dce_tols = dict(tolerances["tolerances"]["dce"])
    if backend_used in dce_tols:
        return backend_used
    aliases = dict(tolerances.get("backend_aliases", {}))
    alias_target = str(aliases.get(backend_used, ""))
    if alias_target and alias_target in dce_tols:
        return alias_target
    raise KeyError(f"No phantom GT DCE tolerance profile for backend '{backend_used}'")


def _require_gate_ready_profile_or_xfail(tolerances: Dict[str, Any]) -> None:
    if bool(tolerances.get("gate_ready", True)):
        return
    pytest.xfail(
        "Phantom GT tolerance profile is provisional (performance triage in progress); "
        "use tests/python/run_phantom_gt_reliability.py for exploratory review."
    )


def _assert_session_within_tolerances(session: Dict[str, Any], tolerances: Dict[str, Any]) -> None:
    session_id = str(session.get("session_id", "<unknown>"))
    t1_tols = dict(tolerances["tolerances"]["t1_ms"])
    for region_name, stats in dict(session.get("t1", {})).items():
        if region_name not in t1_tols:
            raise AssertionError(f"{session_id}: missing T1 tolerance for region '{region_name}'")
        mae = float(stats["mae"])
        tol = float(t1_tols[region_name])
        assert mae <= tol, (
            f"{session_id} T1 region={region_name} MAE {mae:.6g} exceeded tolerance {tol:.6g} ms"
        )

    backend_used = str(session.get("backend_used", ""))
    dce_key = _resolve_dce_backend_tolerance_key(tolerances, backend_used)
    dce_tols = dict(tolerances["tolerances"]["dce"][dce_key])
    for model_name, model_metrics in dict(session.get("dce", {})).items():
        model_tol = dce_tols.get(model_name)
        if not isinstance(model_tol, dict):
            raise AssertionError(f"{session_id}: missing DCE tolerance block for model '{model_name}' ({dce_key})")
        for param_name, region_metrics in dict(model_metrics).items():
            param_tol = model_tol.get(param_name)
            if not isinstance(param_tol, dict):
                raise AssertionError(
                    f"{session_id}: missing DCE tolerance for {model_name}:{param_name} ({dce_key})"
                )
            for region_name, stats in dict(region_metrics).items():
                if region_name not in param_tol:
                    raise AssertionError(
                        f"{session_id}: missing DCE tolerance for {model_name}:{param_name}:{region_name} ({dce_key})"
                    )
                mae = float(stats["mae"])
                tol = float(param_tol[region_name])
                assert mae <= tol, (
                    f"{session_id} {model_name}:{param_name} region={region_name} "
                    f"MAE {mae:.6g} exceeded tolerance {tol:.6g} ({dce_key})"
                )


def _run_phantom_summary_or_skip(
    *,
    run_qualification: bool,
    qualification_root: str,
    tmp_path: Path,
    backend: str,
    require_cpufit_cpu: bool,
) -> Dict[str, Any]:
    if not run_qualification:
        pytest.skip("Use --run-qualification to run phantom ground-truth reliability checks.")

    bids_root = Path(qualification_root).expanduser().resolve() if qualification_root else (
        REPO_ROOT / "tests" / "data" / "BIDS_test"
    )
    if not bids_root.exists():
        pytest.skip(f"Qualification root not found: {bids_root}")

    sessions = discover_phantom_gt_sessions(bids_root)
    if not sessions:
        pytest.skip(f"No phantom gt sessions found under {bids_root}")

    if require_cpufit_cpu:
        probe_acceleration_backend.cache_clear()
        probe = probe_acceleration_backend()
        if not bool(probe.get("pycpufit_imported", False)):
            pytest.skip(f"pycpufit unavailable on this platform: {probe.get('pycpufit_error')}")

    summary = run_phantom_gt_bids_summary(
        bids_root=bids_root,
        output_root=tmp_path / f"phantom_gt_{backend}",
        backend=backend,
    )
    out_json = tmp_path / f"phantom_gt_{backend}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if require_cpufit_cpu:
        unexpected = [s["session_id"] for s in summary["sessions"] if str(s.get("backend_used")) != "cpufit_cpu"]
        if unexpected:
            pytest.skip(
                "phantom auto-backend reliability expected cpufit_cpu on this machine, "
                f"but got {unexpected}"
            )

    return summary


@pytest.mark.integration
@pytest.mark.qualification
@pytest.mark.slow
def test_phantom_ground_truth_reliability_cpu_against_region_tolerances(
    run_qualification: bool, qualification_root: str, tmp_path: Path
) -> None:
    tolerances = load_phantom_gt_tolerances(PHANTOM_GT_TOLERANCES_PATH)
    _require_gate_ready_profile_or_xfail(tolerances)
    summary = _run_phantom_summary_or_skip(
        run_qualification=run_qualification,
        qualification_root=qualification_root,
        tmp_path=tmp_path,
        backend="cpu",
        require_cpufit_cpu=False,
    )
    for session in summary["sessions"]:
        _assert_session_within_tolerances(session, tolerances)


@pytest.mark.integration
@pytest.mark.qualification
def test_phantom_ground_truth_reliability_auto_cpufit_against_region_tolerances(
    run_qualification: bool, qualification_root: str, tmp_path: Path
) -> None:
    tolerances = load_phantom_gt_tolerances(PHANTOM_GT_TOLERANCES_PATH)
    _require_gate_ready_profile_or_xfail(tolerances)
    summary = _run_phantom_summary_or_skip(
        run_qualification=run_qualification,
        qualification_root=qualification_root,
        tmp_path=tmp_path,
        backend="auto",
        require_cpufit_cpu=True,
    )
    for session in summary["sessions"]:
        _assert_session_within_tolerances(session, tolerances)
