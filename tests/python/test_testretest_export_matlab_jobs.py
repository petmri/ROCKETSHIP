"""run_testretest_export_matlab_jobs.py: MATLAB jobs carry what the Python fit used."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import run_testretest_export_matlab_jobs as export  # noqa: E402

pytestmark = pytest.mark.unit


def write_fit(bids: Path, subject: str, session: str, *, method="aif_sidecar", end=28, window=None, aif_mode="raw", blood=1.664):
    fit = bids / "derivatives/rocketship-testretest/python_AC" / subject / session
    fit.mkdir(parents=True)
    run_dir = bids / "derivatives/dceprep-testretest_AC" / subject / session
    prefix = f"{subject}_{session}"
    config = {
        "aif_mode": aif_mode,
        "dynamic_files": [str(run_dir / f"dce/{prefix}_desc-bfcz_DCE.nii.gz")],
        "aif_files": [str(run_dir / f"dce/{prefix}_label-AIF_T1map.nii.gz")],
        "roi_files": [str(run_dir / f"anat/{prefix}_space-DCEref_label-brain_mask.nii.gz")],
        "t1map_files": [str(run_dir / f"anat/{prefix}_space-DCEref_T1map.nii.gz")],
        "model_flags": {"tofts": 0, "ex_tofts": 1, "patlak": 1},
        "stage_overrides": {"rootname": prefix, "blood_t1_ms": 1664},
    }
    stage_a = {
        "tr_ms": 6.8, "fa_deg": 10.0, "time_resolution_min": 0.1, "relaxivity": 3.8, "hematocrit": 0.45,
        "blood_t1_override_sec": blood, "steady_state_time": [1, end],
        "steady_state_auto": {"method_used": method},
        "timepoint_window": window or {"source": "default_full_range", "start_1b": 1, "end_1b": 60, "n_timepoints_input": 60},
    }
    (fit / "run_config.json").write_text(json.dumps(config))
    (fit / "dce_pipeline_run.json").write_text(json.dumps({"stages": {"A": stage_a}, "provenance": {"git_revision": "abc1234"}}))
    return fit


def test_job_carries_the_resolved_settings_with_paths_relative_to_the_share(tmp_path):
    fit = write_fit(tmp_path, "sub-A1", "ses-01")

    job = export.session_job(fit, tmp_path)

    assert job["dynamic"] == "derivatives/dceprep-testretest_AC/sub-A1/ses-01/dce/sub-A1_ses-01_desc-bfcz_DCE.nii.gz"
    assert job["rootname"] == "sub-A1_ses-01"
    assert (job["tr_ms"], job["fa_deg"], job["relaxivity"], job["hematocrit"]) == (6.8, 10.0, 3.8, 0.45)
    assert job["time_resolution_sec"] == pytest.approx(6.0)
    assert job["blood_t1_sec"] == 1.664
    assert job["aif_type"] == 2  # raw
    assert job["models"] == {"ex_tofts": 1, "patlak": 1}
    assert (job["steady_state_end"], job["steady_state_source"]) == (28, "aif_sidecar")
    assert job["start_t"] is None and job["end_t"] is None  # the whole series
    assert job["python_git_revision"] == "abc1234"


def test_a_cropped_window_and_an_auto_baseline_are_passed_on(tmp_path):
    fit = write_fit(tmp_path, "sub-A1", "ses-01", method="tv", end=12,
                    window={"source": "stage_overrides", "start_1b": 3, "end_1b": 55, "n_timepoints_input": 60})
    job = export.session_job(fit, tmp_path)
    assert (job["start_t"], job["end_t"]) == (3, 55)
    assert (job["steady_state_end"], job["steady_state_source"]) == (12, "tv")


def test_sessions_that_cannot_be_matched_are_left_out_with_the_reason(tmp_path, capsys):
    write_fit(tmp_path, "sub-A1", "ses-01")
    write_fit(tmp_path, "sub-A2", "ses-01", blood=None)       # blood T1 taken from the map
    write_fit(tmp_path, "sub-A3", "ses-01", aif_mode="imported")
    fit_root = tmp_path / "derivatives/rocketship-testretest/python_AC"
    (fit_root / "fit_manifest.tsv").write_text(
        "subject\tsession\tstatus\nsub-A1\tses-01\tok\nsub-A2\tses-01\tok\nsub-A3\tses-01\tok\nsub-A4\tses-01\tfailed\n")
    out = tmp_path / "jobs.json"

    assert export.main(["--fit-root", str(fit_root), "--bids-root", str(tmp_path), "--rater", "AC", "--out", str(out)]) == 0

    payload = json.loads(out.read_text())
    assert [j["subject"] for j in payload["jobs"]] == ["sub-A1"]
    assert payload["rater"] == "AC" and payload["fit_root"] == "derivatives/rocketship-testretest/python_AC"
    reasons = " | ".join(payload["left_out"])
    assert "sub-A2 ses-01: the Python fit took blood T1 from the AIF T1 map" in reasons
    assert "sub-A3 ses-01: aif_mode 'imported'" in reasons

    with pytest.raises(ValueError, match="not under --bids-root"):
        export.session_job(fit_root / "sub-A1/ses-01", tmp_path / "elsewhere")
