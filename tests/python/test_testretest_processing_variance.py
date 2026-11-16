"""Processing-variance runner: level staging, the pooled SD, and a real run on the tiny fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

import run_testretest_processing_variance as pv  # noqa: E402
import testretest_study as study  # noqa: E402


def save(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)
    return path


def aif_map(root: Path, rater: str, voxels, baseline_end=None) -> Path:
    data = np.zeros((7, 7, 3))
    for voxel in voxels:
        data[voxel] = 1500.0
    path = save(root / rater / "sub-A1/ses-01/dce/sub-A1_ses-01_label-AIF_T1map.nii.gz", data)
    if baseline_end is not None:
        pv._sidecar(path).write_text(json.dumps({"SteadyStateEndTimeIndex": baseline_end}))
    return path


@pytest.mark.unit
def test_dilation_grows_in_plane_only_and_fills_the_blood_t1(tmp_path):
    base = aif_map(tmp_path, "AC", [(3, 3, 1)], baseline_end=12)

    path, reason = pv.stage_aif({"name": "dilated", "aif_mask": "dilate"}, base, tmp_path / "v", {}, "sub-A1", "ses-01", 1664.0)

    data = nib.load(path).get_fdata()
    assert reason == "" and int((data > 0).sum()) == 5
    assert (data[:, :, 0] == 0).all() and (data[:, :, 2] == 0).all()
    assert data[3, 3, 1] == 1500.0 and data[2, 3, 1] == 1664.0
    # the baseline end travels with the file unless the level says otherwise
    assert json.loads(pv._sidecar(path).read_text())["SteadyStateEndTimeIndex"] == 12


@pytest.mark.unit
def test_a_level_that_would_empty_the_aif_is_skipped_with_the_reason(tmp_path):
    base = aif_map(tmp_path, "AC", [(3, 3, 1)])
    path, reason = pv.stage_aif({"name": "eroded", "aif_mask": "erode"}, base, tmp_path / "v", {}, "sub-A1", "ses-01", None)
    assert path is None and reason == "erode leaves no AIF voxel"


@pytest.mark.unit
def test_rater_and_baseline_can_be_crossed(tmp_path):
    base = aif_map(tmp_path, "AC", [(3, 3, 1)])  # AC: no sidecar, baseline auto-detected
    aif_map(tmp_path, "WN", [(1, 1, 2), (1, 2, 2)], baseline_end=29)
    runs = {"AC": tmp_path / "AC", "WN": tmp_path / "WN"}

    def stage(name, **level):
        return pv.stage_aif({"name": name, **level}, base, tmp_path / name, runs, "sub-A1", "ses-01", None)

    wn_own, _ = stage("wn_own", aif_from="WN", baseline="as_drawn")
    assert int((nib.load(wn_own).get_fdata() > 0).sum()) == 2
    assert json.loads(pv._sidecar(wn_own).read_text())["SteadyStateEndTimeIndex"] == 29

    ac_with_wn_baseline, _ = stage("ac_wn", aif_from="AC", baseline={"from": "WN"})
    assert int((nib.load(ac_with_wn_baseline).get_fdata() > 0).sum()) == 1
    assert json.loads(pv._sidecar(ac_with_wn_baseline).read_text())["SteadyStateEndTimeIndex"] == 29

    wn_auto, _ = stage("wn_auto", aif_from="WN", baseline="auto")
    assert not pv._sidecar(wn_auto).exists()

    missing, reason = stage("xx", aif_from="XX")
    assert missing is None and "no --rater-run" in reason


@pytest.mark.unit
def test_processing_sd_pools_sessions_and_ignores_failed_levels():
    def row(session, level, value, status="ok", factor="start_t"):
        return {"subject": "s", "session": session, "factor": factor, "level": level, "status": status,
                "model": "patlak", "region": "WM", "parameter": "Ktrans", "value": value}

    rows = [
        row("ses-01", "1", 1.0), row("ses-01", "2", 3.0),           # variance 2
        row("ses-02", "1", 5.0), row("ses-02", "2", 5.0),           # variance 0
        row("ses-03", "1", 9.0), row("ses-03", "2", float("nan"), status="failed"),  # one level: not usable
    ]
    table = pv.processing_sd_table(rows)
    assert len(table) == 1
    assert table[0]["n_sessions"] == 2 and table[0]["n_levels"] == 2
    assert table[0]["processing_sd"] == pytest.approx(1.0)  # sqrt(mean([2, 0]))


@pytest.mark.integration
def test_runs_factors_on_the_tiny_fixture_and_keeps_the_budget_locked(tmp_path, capsys):
    tests_dir = REPO_ROOT / "tests/python"
    config = json.loads((tests_dir / "dce_run_tiny.json").read_text())
    for key in ("subject_source_path", "subject_tp_path"):
        config[key] = str((tests_dir / config[key]).resolve())
    for key in ("dynamic_files", "aif_files", "roi_files", "t1map_files", "noise_files"):
        config[key] = [str((tests_dir / p).resolve()) for p in config[key]]
    overrides = config["stage_overrides"]
    for key, value in list(overrides.items()):
        if key.endswith("_path") and isinstance(value, str):
            overrides[key] = str((tests_dir / value).resolve())
    config.pop("_doc", None)
    fit_dir = tmp_path / "fits/sub-11tiny/ses-01"
    fit_dir.mkdir(parents=True)
    (fit_dir / "run_config.json").write_text(json.dumps(config))

    factors = tmp_path / "factors.json"
    factors.write_text(json.dumps({"factors": {
        "aif_curve": [{"name": "fitted", "top": {"aif_mode": "fitted"}}, {"name": "raw", "top": {"aif_mode": "raw"}}],
        "aif_mask": [{"name": "as_drawn"}, {"name": "eroded", "aif_mask": "erode"}],
    }}))
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps({
        "regions": {"brain": "anat/*_label-brain_mask.nii"}, "fit_roi": "anat/*_label-brain_mask.nii",
        "model_parameters": {"tofts": ["Ktrans", "ve"]},
    }))
    common = ["--fit-root", str(tmp_path / "fits"), "--dceprep-run", str(REPO_ROOT / "tests/data/BIDS_test/derivatives"),
              "--factors", str(factors), "--regions-json", str(regions)]

    assert pv.main([*common, "--out", str(tmp_path / "pv")]) == 0

    levels = study.read_table(tmp_path / "pv/processing_levels.csv", numeric=("value",))
    status = {(r["factor"], r["level"]): r["status"] for r in levels}
    assert status[("aif_curve", "fitted")] == status[("aif_curve", "raw")] == status[("aif_mask", "as_drawn")] == "ok"
    sd = study.read_table(tmp_path / "pv/processing_sd.csv", numeric=("processing_sd", "n_sessions"))
    curve = [r for r in sd if r["factor"] == "aif_curve" and r["parameter"] == "Ktrans"]
    assert len(curve) == 1 and curve[0]["n_sessions"] == 1 and curve[0]["processing_sd"] >= 0
    # with one usable level the mask factor has no spread to report
    if status[("aif_mask", "eroded")] != "ok":
        assert not [r for r in sd if r["factor"] == "aif_mask"]
    assert str(curve[0]["processing_sd"]) not in capsys.readouterr().out  # values are not printed

    # relating the spread to the scan-rescan wSD needs a signed-off pre-registration
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps({"signed_off": {"by": None, "date": None}}))
    code = pv.main([*common, "--out", str(tmp_path / "pv2"), "--rescan-metrics", str(tmp_path / "metrics.csv"),
                    "--rescan-rater", "AC", "--prereg", str(prereg)])
    assert code == 2 and not (tmp_path / "pv2").exists()
