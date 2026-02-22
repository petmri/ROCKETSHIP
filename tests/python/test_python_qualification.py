"""Integration tests for dataset-level Python qualification runner."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import BidsSession  # noqa: E402
from qualification import QualificationRunConfig, run_bids_qualification  # noqa: E402
from qualification import _primary_model_map_blockers  # noqa: E402
from qualification import _session_dce_inputs  # noqa: E402


@pytest.mark.unit
def test_primary_model_map_blockers_flags_nonfinite_ex_tofts() -> None:
    map_stats = {
        "tofts:Ktrans": {"nonzero_fraction": 0.30, "finite_nonzero_fraction": 0.29},
        "tofts:ve": {"nonzero_fraction": 0.30, "finite_nonzero_fraction": 0.29},
        "ex_tofts:Ktrans": {"nonzero_fraction": 0.30, "finite_nonzero_fraction": 0.0},
        "ex_tofts:ve": {"nonzero_fraction": 0.30, "finite_nonzero_fraction": 0.0},
        "ex_tofts:vp": {"nonzero_fraction": 0.30, "finite_nonzero_fraction": 0.0},
        "patlak:Ktrans": {"nonzero_fraction": 0.30, "finite_nonzero_fraction": 0.30},
        "patlak:vp": {"nonzero_fraction": 0.30, "finite_nonzero_fraction": 0.30},
    }
    blockers = _primary_model_map_blockers(map_stats, ("tofts", "ex_tofts", "patlak"))
    assert any("ex_tofts:Ktrans" in msg for msg in blockers)
    assert any("ex_tofts:ve" in msg for msg in blockers)
    assert any("ex_tofts:vp" in msg for msg in blockers)
    assert not any(" for tofts:Ktrans:" in msg for msg in blockers)
    assert not any(" for patlak:Ktrans:" in msg for msg in blockers)


@pytest.mark.unit
def test_primary_model_map_blockers_ignores_all_zero_maps() -> None:
    map_stats = {
        "tofts:Ktrans": {"nonzero_fraction": 0.0, "finite_nonzero_fraction": 0.0},
        "tofts:ve": {"nonzero_fraction": 0.0, "finite_nonzero_fraction": 0.0},
    }
    blockers = _primary_model_map_blockers(map_stats, ("tofts",))
    assert blockers == []


@pytest.mark.unit
def test_session_dce_inputs_requires_explicit_aif_roi(tmp_path: Path) -> None:
    raw_session = tmp_path / "rawdata" / "sub-01" / "ses-01"
    deriv_session = tmp_path / "derivatives" / "sub-01" / "ses-01"
    (raw_session / "dce").mkdir(parents=True, exist_ok=True)
    (deriv_session / "dce").mkdir(parents=True, exist_ok=True)
    (deriv_session / "anat").mkdir(parents=True, exist_ok=True)

    (deriv_session / "dce" / "sub-01_ses-01_desc-bfcz_DCE.nii.gz").write_text("")
    (deriv_session / "anat" / "sub-01_ses-01_space-DCEref_desc-brain_mask.nii.gz").write_text("")
    (deriv_session / "anat" / "sub-01_ses-01_space-DCEref_T1map.nii").write_text("")

    session = BidsSession(
        bids_root=tmp_path,
        subject="sub-01",
        session="ses-01",
        rawdata_path=raw_session,
        derivatives_path=deriv_session,
    )

    with pytest.raises(FileNotFoundError, match=r"aif \(\*desc-AIF_T1map\.nii\*\)"):
        _session_dce_inputs(session)


@pytest.mark.integration
@pytest.mark.qualification
def test_bids_root_qualification_processes_all_sessions(
    run_qualification: bool, qualification_root: str, tmp_path: Path
) -> None:
    if not run_qualification:
        pytest.skip("Use --run-qualification to run dataset-level BIDS qualification.")

    bids_root = Path(qualification_root).expanduser().resolve() if qualification_root else (
        REPO_ROOT / "tests" / "data" / "BIDS_test"
    )
    if not bids_root.exists():
        pytest.skip(f"Qualification root not found: {bids_root}")

    summary = run_bids_qualification(
        QualificationRunConfig(
            bids_root=bids_root,
            output_root=tmp_path / "qualification",
            backend="auto",
            dce_models=("tofts", "ex_tofts", "patlak"),
            run_t1=True,
            run_dce=True,
            write_postfit_arrays=False,
        )
    )

    assert summary["meta"]["sessions_discovered"] > 0
    assert Path(summary["meta"]["summary_json_path"]).exists()
    assert Path(summary["meta"]["summary_markdown_path"]).exists()

    if int(summary["meta"]["sessions_failed"]) > 0:
        failed_details = []
        for session in summary["sessions"]:
            if session.get("status") == "failed":
                failed_details.append(
                    f"{session['session_id']}: blockers={session.get('blockers', [])}"
                )
        pytest.fail("Qualification failures:\n" + "\n".join(failed_details))
