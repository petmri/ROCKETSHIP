"""run_testretest_fits.py: per-session config resolution, and one real fit on a fixture."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import run_testretest_fits as fits  # noqa: E402

TEMPLATE = {
    "_doc": ["ignored"],
    "backend": "cpu",
    "aif_mode": "raw",
    "dynamic_files": ["dce/*_desc-bfcz_DCE.nii*"],
    "aif_files": ["dce/*_label-AIF_T1map.nii*"],
    "roi_files": ["anat/*_space-DCEref_label-brain_mask.nii*"],
    "t1map_files": ["anat/*_space-DCEref_T1map.nii*"],
    "model_flags": {"patlak": 1},
    "stage_overrides": {"write_qof_maps": True},
    "cohort_overrides": {"sub-A": {"blood_t1_ms": 1441}, "sub-AB": {"blood_t1_ms": 1664}},
}


def make_session(root: Path, subject: str, session: str = "ses-01"):
    prefix = f"{subject}_{session}"
    session_dir = root / "derivatives" / "dceprep-x" / subject / session
    staged = root / "staged" / subject / session
    for rel in (
        f"dce/{prefix}_desc-bfcz_DCE.nii.gz",
        f"dce/{prefix}_label-AIF_T1map.nii.gz",
        f"dce/{prefix}_label-AIF_T1map.json",
        f"anat/{prefix}_space-DCEref_label-brain_mask.nii.gz",
        f"anat/{prefix}_space-DCEref_T1map.nii.gz",
        f"anat/{prefix}_space-DCEref_desc-rsquared_T1map.nii.gz",
    ):
        path = session_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (staged / "dce").mkdir(parents=True)
    (staged / "dce" / f"{prefix}_DCE.json").write_text("{}")
    return session_dir, staged


@pytest.mark.unit
def test_config_names_one_file_per_input_and_keeps_inputs_and_outputs_apart(tmp_path):
    session_dir, staged = make_session(tmp_path, "sub-AB01")
    out = tmp_path / "out" / "sub-AB01" / "ses-01"

    config, reason = fits.build_session_config(TEMPLATE, session_dir, staged, out)

    assert reason == ""
    assert config["t1map_files"] == [str(session_dir / "anat/sub-AB01_ses-01_space-DCEref_T1map.nii.gz")]
    assert config["aif_files"] == [str(session_dir / "dce/sub-AB01_ses-01_label-AIF_T1map.nii.gz")]
    assert config["subject_tp_path"] == str(session_dir)
    assert config["output_dir"] == str(out)
    overrides = config["stage_overrides"]
    assert overrides["rootname"] == "sub-AB01_ses-01"
    assert overrides["dce_metadata_path"] == str(staged / "dce/sub-AB01_ses-01_DCE.json")
    assert overrides["write_qof_maps"] is True
    # the longest matching prefix wins
    assert overrides["blood_t1_ms"] == 1664
    assert "cohort_overrides" not in config and "_doc" not in config


@pytest.mark.unit
def test_several_roi_patterns_resolve_in_order_and_from_a_regions_root(tmp_path):
    session_dir, staged = make_session(tmp_path, "sub-AB01")
    regions = tmp_path / "regions"
    tumor = regions / "sub-AB01/ses-01/anat/sub-AB01_ses-01_space-DCEref_label-tumor_mask.nii.gz"
    tumor.parent.mkdir(parents=True)
    tumor.touch()
    template = {**TEMPLATE, "roi_files": [
        "anat/*_space-DCEref_label-brain_mask.nii*", "anat/*_space-DCEref_label-tumor_mask.nii*"]}

    config, reason = fits.build_session_config(template, session_dir, staged, tmp_path / "out", [regions])
    assert reason == ""
    assert config["roi_files"] == [
        str(session_dir / "anat/sub-AB01_ses-01_space-DCEref_label-brain_mask.nii.gz"), str(tumor)]

    config, reason = fits.build_session_config(template, session_dir, staged, tmp_path / "out")
    assert config is None and "label-tumor_mask" in reason

    own = session_dir / "anat/sub-AB01_ses-01_space-DCEref_label-tumor_mask.nii.gz"
    own.touch()
    config, _ = fits.build_session_config(template, session_dir, staged, tmp_path / "out", [regions])
    assert config["roi_files"][1] == str(tumor)  # the regions root overrides the session's own mask


@pytest.mark.unit
def test_shorter_prefix_applies_when_the_longer_one_does_not_match(tmp_path):
    session_dir, staged = make_session(tmp_path, "sub-A7")
    config, _ = fits.build_session_config(TEMPLATE, session_dir, staged, tmp_path / "out")
    assert config["stage_overrides"]["blood_t1_ms"] == 1441


@pytest.mark.unit
def test_a_session_that_cannot_be_resolved_is_reported_not_guessed(tmp_path):
    session_dir, staged = make_session(tmp_path, "sub-A1")
    (session_dir / "anat/sub-A1_ses-01_space-DCEref_T1map.nii.gz").unlink()
    config, reason = fits.build_session_config(TEMPLATE, session_dir, staged, tmp_path / "out")
    assert config is None and "t1map_files: 0 files" in reason

    session_dir, staged = make_session(tmp_path, "sub-A2")
    (session_dir / "dce/sub-A2_ses-01_run-02_desc-bfcz_DCE.nii.gz").touch()
    config, reason = fits.build_session_config(TEMPLATE, session_dir, staged, tmp_path / "out")
    assert config is None and "dynamic_files: 2 files" in reason

    session_dir, staged = make_session(tmp_path, "sub-Z1")
    config, reason = fits.build_session_config(TEMPLATE, session_dir, staged, tmp_path / "out")
    assert config is None and "no cohort_overrides entry" in reason


@pytest.mark.unit
def test_refuses_to_write_into_the_rater_run(tmp_path):
    session_dir, _ = make_session(tmp_path, "sub-A1")
    run = session_dir.parents[1]
    with pytest.raises(SystemExit):
        fits.main(["--dceprep-run", str(run), "--staged-root", str(tmp_path / "staged"),
                   "--output-root", str(run / "fits"), "--allow-dirty"])


@pytest.mark.integration
def test_fits_a_fixture_session_and_records_it(tmp_path):
    """sub-02downsample has the dceprep layout; fit Patlak on CPU into a separate root."""
    bids = REPO_ROOT / "tests/data/BIDS_test"
    template = {
        **TEMPLATE,
        "aif_mode": "fitted",
        "aif_files": ["dce/*label-AIF_T1map.nii*"],
        "roi_files": ["anat/*space-DCEref_label-brain_mask.nii*"],
        "t1map_files": ["anat/*space-DCEref_T1map.nii*"],
        "model_flags": {"tofts": 0, "ex_tofts": 0, "patlak": 1, "tissue_uptake": 0,
                        "two_cxm": 0, "fxr": 0, "auc": 0, "nested": 0, "FXL_rr": 0},
        "stage_overrides": {"stage_b_mode": "real", "stage_d_mode": "real",
                            "write_qof_maps": True, "start_t": 3, "relaxivity": 5.7},
        "cohort_overrides": {},
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template))
    out = tmp_path / "fits"

    code = fits.main([
        "--dceprep-run", str(bids / "derivatives"), "--staged-root", str(bids / "rawdata"),
        "--output-root", str(out), "--config", str(template_path),
        "--subject", "sub-02downsample", "--allow-dirty",
    ])

    assert code == 0
    with open(out / "fit_manifest.tsv", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(r["subject"], r["session"], r["status"]) for r in rows] == [("sub-02downsample", "ses-01", "ok")]
    session_out = out / "sub-02downsample" / "ses-01"
    assert (session_out / "sub-02downsample_ses-01_patlak_fit_Ktrans.nii.gz").exists()
    assert (session_out / "sub-02downsample_ses-01_patlak_fit_qof_reliable.nii.gz").exists()
    assert json.loads((session_out / "run_config.json").read_text())["output_dir"] == str(session_out)
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["status_counts"] == {"ok": 1, "failed": 0, "skipped": 0}
    assert provenance["rocketship"]["version"]
