"""Test-retest study layer: collection, metadata audit, blinding, exclusions, analysis."""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

import repeatability as rp  # noqa: E402
import run_testretest_analysis as cli  # noqa: E402
import testretest_study as study  # noqa: E402

COMPLETE_PREREG = {
    "signed_off": {"by": "PI", "date": "2026-12-01"},
    "primary_endpoint": {"model": "patlak", "region": "WM", "parameter": "Ktrans", "rater": "SYN"},
    "pooling": "separate",
    "noninferiority": {"margin": 1.25, "confidence": 0.95, "interval": "pitman"},
    "exclusions": {"max_displacement_mm": 2.0, "min_aif_fit_r2_adj": "not_applied", "min_qof_reliable_fraction": 0.5},
    "split": {"development_fraction": 0.33, "seed": 7},
    "report_wcv_for": [],
    "bootstrap": {"n_boot": 200, "seed": 1},
}


def save(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)


def make_fitted_session(root: Path, subject="sub-QINGBMTR900", session="ses-01", ktrans_wm=0.002, ktrans_gm=0.004):
    """A DCEPrep session and a fit folder on a 10x10x4 grid: WM in x<5, GM in x>=5."""
    prefix = f"{subject}_{session}"
    dceprep = root / "dceprep" / subject / session
    fit = root / "fits" / subject / session
    shape = (10, 10, 4)
    brain = np.ones(shape)
    brain[:, :, 0] = 0  # the bottom slice is outside the fit ROI
    wm = np.zeros(shape)
    wm[:5] = 1
    gm = np.zeros(shape)
    gm[5:] = 1
    save(dceprep / f"anat/{prefix}_space-DCEref_label-brain_mask.nii.gz", brain)
    save(dceprep / f"anat/{prefix}_space-DCEref_label-WM_mask.nii.gz", wm)
    save(dceprep / f"anat/{prefix}_space-DCEref_label-GM_mask.nii.gz", gm)
    (dceprep / "dce").mkdir(parents=True, exist_ok=True)
    (dceprep / f"dce/{prefix}_desc-hmc_maxdisp.txt").write_text("Max displacement: 0.75mm at frame 12/60\n")
    ktrans = np.where(wm > 0, ktrans_wm, ktrans_gm).astype(float)
    ktrans[:, :, 0] = 99.0  # must be ignored: outside the fit ROI
    ktrans[0, 0, 1] = np.nan  # a failed voxel inside WM
    reliable = np.ones(shape)
    reliable[:5, :5, 1:] = 0  # half of the WM voxels inside the ROI
    for model in ("patlak", "ex_tofts"):
        save(fit / f"{prefix}_{model}_fit_Ktrans.nii.gz", ktrans)
        save(fit / f"{prefix}_{model}_fit_vp.nii.gz", np.full(shape, 0.01))
        save(fit / f"{prefix}_{model}_fit_qof_reliable.nii.gz", reliable)
    save(fit / f"{prefix}_ex_tofts_fit_ve.nii.gz", np.full(shape, 0.2))
    return fit, dceprep


# --------------------------------------------------------------------------- pre-registration


@pytest.mark.unit
def test_unblinding_is_refused_until_every_decision_is_made():
    prereg = copy.deepcopy(COMPLETE_PREREG)
    study.require_unblinded(prereg)  # complete: no error; 'not_applied' counts as decided

    prereg["noninferiority"]["margin"] = None
    prereg["signed_off"]["by"] = None
    assert study.missing_decisions(prereg) == ["noninferiority.margin", "signed_off.by"]
    with pytest.raises(study.BlindingError, match="noninferiority.margin"):
        study.require_unblinded(prereg)


@pytest.mark.unit
def test_placeholders_are_only_available_in_the_blinded_modes():
    prereg = copy.deepcopy(COMPLETE_PREREG)
    prereg["noninferiority"]["margin"] = None
    used: list = []
    assert study.decision(prereg, "noninferiority.margin", "shuffled", used) == 1.1
    assert used == ["noninferiority.margin = 1.1"]
    with pytest.raises(study.BlindingError):
        study.decision(prereg, "noninferiority.margin", "unblinded", [])
    assert study.decision(prereg, "noninferiority.confidence", "unblinded", []) == 0.95


@pytest.mark.unit
def test_the_committed_preregistration_parses_and_names_only_known_decisions():
    prereg = study.load_prereg(cli.DEFAULT_PREREG)
    for key in study.REQUIRED_DECISIONS:
        section, name = key.split(".") if "." in key else (key, None)
        assert section in prereg and (name is None or name in prereg[section]), key


# --------------------------------------------------------------------------- collection


@pytest.mark.unit
def test_collect_session_takes_the_median_inside_region_and_fit_roi(tmp_path):
    fit, dceprep = make_fitted_session(tmp_path)

    rows, qc, problems = study.collect_session(fit, dceprep, pipeline="python", rater="AC")

    assert problems == []
    by_key = {(r["model"], r["region"], r["parameter"]): r for r in rows}
    assert len(rows) == 2 * 2 + 3 * 2  # patlak: Ktrans, vp; ex_tofts: Ktrans, ve, vp; x WM, GM
    wm = by_key[("patlak", "WM", "Ktrans")]
    assert wm["value"] == pytest.approx(0.002)
    assert wm["n_voxels"] == 5 * 10 * 3  # the slice outside the fit ROI is not counted
    assert wm["n_finite"] == 5 * 10 * 3 - 1
    assert wm["qof_reliable_fraction"] == pytest.approx(0.5)
    assert by_key[("patlak", "GM", "Ktrans")]["value"] == pytest.approx(0.004)
    assert by_key[("ex_tofts", "GM", "ve")]["value"] == pytest.approx(0.2)
    assert {wm["subject"], wm["session"], wm["cohort"], wm["pipeline"], wm["rater"]} == {
        "sub-QINGBMTR900", "ses-01", "QIN-GBM", "python", "AC"}
    assert qc["max_displacement_mm"] == pytest.approx(0.75)
    assert np.isnan(qc["aif_fit_r2_adj"])  # a raw-AIF run has no AIF fit


@pytest.mark.unit
def test_collect_session_reports_missing_maps_and_grid_mismatches(tmp_path):
    fit, dceprep = make_fitted_session(tmp_path)
    next(fit.glob("*_patlak_fit_vp.nii.gz")).unlink()
    save(next(dceprep.glob("anat/*label-GM_mask.nii.gz")), np.ones((10, 10, 5)))

    rows, _, problems = study.collect_session(fit, dceprep, pipeline="python", rater="AC")

    assert any("no patlak vp map" in p for p in problems)
    assert any("GM mask grid differs" in p for p in problems)
    assert {r["region"] for r in rows} == {"WM"}


@pytest.mark.unit
def test_collect_arm_walks_the_fit_root(tmp_path):
    make_fitted_session(tmp_path, session="ses-01")
    make_fitted_session(tmp_path, session="ses-02")
    (tmp_path / "fits/sub-QINGBMTR901/ses-01").mkdir(parents=True)  # fitted folder without a DCEPrep session

    rows, qc_rows, problems = study.collect_arm(tmp_path / "fits", tmp_path / "dceprep", pipeline="python", rater="AC")

    assert len(qc_rows) == 2 and len(rows) == 20
    assert any("sub-QINGBMTR901" in p for p in problems)


# --------------------------------------------------------------------------- metadata audit


def write_run_json(fit_dir: Path, **stage_a):
    fit_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "tr_ms": 6.8, "fa_deg": 10.0, "time_resolution_min": 0.1001, "relaxivity": 3.8, "hematocrit": 0.45,
        "metadata_sources": {"tr_ms": f"json:/data/{fit_dir.name}/x_DCE.json.RepetitionTimeExcitation",
                             "time_resolution_sec": f"json:/data/{fit_dir.name}/x_DCE.json.RepetitionTime"},
        "steady_state_auto": {"method_used": "tv"},
    }
    base.update(stage_a)
    (fit_dir / "dce_pipeline_run.json").write_text(json.dumps({"stages": {"A": base, "B": {"end_time_index": 60}}}))


@pytest.mark.unit
def test_metadata_audit_flags_only_real_differences_between_sessions(tmp_path):
    fits = tmp_path / "fits"
    write_run_json(fits / "sub-A/ses-01")
    write_run_json(fits / "sub-A/ses-02", time_resolution_min=0.10010001)  # rounding, not a difference
    write_run_json(fits / "sub-B/ses-01")
    write_run_json(fits / "sub-B/ses-02", steady_state_auto={"method_used": "aif_sidecar"},
                   metadata_sources={"tr_ms": "json:/x/y_DCE.json.RepetitionTime"})
    write_run_json(fits / "sub-C/ses-01")
    staged = tmp_path / "staged"
    for session, flips in (("ses-01", (2, 5, 10, 15)), ("ses-02", (2, 5, 10, 15, 20))):
        for i, flip in enumerate(flips, 1):
            path = staged / f"sub-A/{session}/anat/sub-A_{session}_flip-{i:02d}_VFA.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"FlipAngle": flip, "RepetitionTime": 0.0073}))

    rows = {r["subject"]: r for r in study.metadata_audit(fits, staged)}

    assert "vfa_flip_angles" in rows["sub-A"]["flags"] and "time_resolution" not in rows["sub-A"]["flags"]
    assert "baseline_method" in rows["sub-B"]["flags"] and "metadata_fields" in rows["sub-B"]["flags"]
    assert rows["sub-C"]["flags"] == "1 fitted session(s)"


# --------------------------------------------------------------------------- blinding


def big_table(ratio=1.5, seed=2, n=300):
    return study.synthetic_table({"A": n, "B": n}, {"python": 0.3 * ratio, "matlab": 0.3}, seed=seed)


@pytest.mark.unit
def test_shuffle_keeps_the_values_but_no_subject_keeps_its_second_session():
    table = big_table(n=40)
    shuffled = study.shuffle_labels(table, seed=5)

    assert Counter(r["value"] for r in shuffled) == Counter(r["value"] for r in table)
    assert Counter(r["cohort"] for r in shuffled) == Counter(r["cohort"] for r in table)
    original_second = {(r["subject"], r["pipeline"]): r["value"] for r in table if r["session"] == "ses-02"}
    original_values = {r["subject"]: set() for r in table}
    for r in table:
        if r["session"] == "ses-02":
            original_values[r["subject"]].add(r["value"])
    for r in shuffled:
        if r["session"] == "ses-02":
            assert r["value"] not in original_values[r["subject"]]
    # every subject still has two sessions in each pipeline
    counts = Counter((r["subject"], r["pipeline"]) for r in shuffled)
    assert set(counts.values()) == {2} and len(counts) == len(original_second)
    assert study.shuffle_labels(table, seed=5) == shuffled
    assert study.shuffle_labels(table, seed=6) != shuffled


@pytest.mark.unit
def test_shuffle_hides_both_the_repeatability_and_the_pipeline_difference():
    table = big_table(ratio=1.5)
    kwargs = dict(margin=1.1, confidence=0.95, interval="pitman", n_boot=100, seed=1)
    real = study.analyse(table, **kwargs)
    blind = study.analyse(study.shuffle_labels(table, seed=3), **kwargs)

    def ratio(results):
        return next(r["wSD_ratio"] for r in results["noninferiority"] if r["cohort"] == "A")

    def wsd(results):
        return next(m["wSD"] for m in results["metrics"] if m["cohort"] == "A" and m["pipeline"] == "matlab")

    assert ratio(real) == pytest.approx(1.5, abs=0.1)
    assert ratio(blind) == pytest.approx(1.0, abs=0.1)
    assert wsd(real) == pytest.approx(0.3, abs=0.03)
    assert wsd(blind) > 0.9  # between-subject SD is 1.0


@pytest.mark.unit
def test_shuffle_refuses_a_cohort_it_cannot_scramble():
    table = study.synthetic_table({"A": 1}, {"python": 0.3}, seed=1)
    with pytest.raises(ValueError, match="fewer than two paired subjects"):
        study.shuffle_labels(table, seed=1)


@pytest.mark.unit
def test_shuffle_leaves_a_subject_with_one_session_alone():
    """A session that failed preprocessing leaves its subject unpaired in the real table."""
    table = study.synthetic_table({"A": 6}, {"python": 0.3}, seed=1)
    table = [r for r in table if not (r["subject"] == "sub-A002" and r["session"] == "ses-02")]

    shuffled = study.shuffle_labels(table, seed=4)

    counts = Counter((r["subject"], r["session"]) for r in shuffled)
    assert set(counts.values()) == {1}  # no (subject, session) appears twice
    lone = [r for r in shuffled if r["subject"] == "sub-A002"]
    assert lone == [r for r in table if r["subject"] == "sub-A002"]
    results = study.analyse(shuffled, margin=1.1, confidence=0.95, interval="pitman", n_boot=50, seed=1)
    assert results["metrics"][0]["n"] == 5


# --------------------------------------------------------------------------- exclusions


@pytest.mark.unit
def test_exclusions_remove_the_pair_in_every_pipeline_and_log_the_reason():
    table = study.synthetic_table({"A": 5}, {"python": 0.3, "matlab": 0.3}, seed=1)
    qc = [
        {"subject": "sub-A000", "session": "ses-02", "pipeline": "python", "rater": "SYN", "max_displacement_mm": 3.5, "aif_fit_r2_adj": float("nan")},
        {"subject": "sub-A001", "session": "ses-01", "pipeline": "python", "rater": "SYN", "max_displacement_mm": float("nan"), "aif_fit_r2_adj": float("nan")},
    ]
    for row in table:  # one pipeline's QoF fails in one session of sub-A002
        if row["subject"] == "sub-A002" and row["pipeline"] == "matlab" and row["session"] == "ses-01":
            row["qof_reliable_fraction"] = 0.2
    manual = [{"subject": "sub-A003", "session": "ses-01", "rater": "*", "reason": "missed injection"}]
    thresholds = {"max_displacement_mm": 2.0, "min_aif_fit_r2_adj": 0.9, "min_qof_reliable_fraction": 0.5}

    kept, log = study.apply_exclusions(table, qc, thresholds, manual)

    assert {r["subject"] for r in kept} == {"sub-A001", "sub-A004"}  # missing QC cannot fail a rule
    assert {(e["subject"], e["rule"]) for e in log} == {
        ("sub-A000", "motion"), ("sub-A002", "qof"), ("sub-A003", "manual")}
    assert "3.50 mm > 2" in next(e["detail"] for e in log if e["rule"] == "motion")

    kept, log = study.apply_exclusions(table, qc, {"max_displacement_mm": "not_applied", "min_qof_reliable_fraction": None})
    assert len(kept) == len(table) and log == []


# --------------------------------------------------------------------------- analysis


@pytest.mark.unit
def test_analyse_marks_the_primary_comparison_only_for_a_complete_endpoint():
    table = study.synthetic_table({"A": 30, "B": 30}, {"python": 0.3, "matlab": 0.3}, seed=4)
    kwargs = dict(margin=1.25, confidence=0.95, interval="pitman", n_boot=100, seed=1)

    undecided = study.analyse(table, primary_endpoint={"model": None, "region": None, "parameter": None, "rater": None}, **kwargs)
    assert not any(r["primary"] for r in undecided["noninferiority"])

    decided = study.analyse(table, primary_endpoint=COMPLETE_PREREG["primary_endpoint"], **kwargs)
    assert {r["cohort"] for r in decided["noninferiority"]} == {"A", "B", study.POOLED}
    assert all(r["primary"] for r in decided["noninferiority"])
    assert all(r["decision_interval"] == "pitman" and r["margin"] == 1.25 for r in decided["noninferiority"])
    pooled = next(m for m in decided["metrics"] if m["cohort"] == study.POOLED and m["pipeline"] == "python")
    assert pooled["n"] == 60 and pooled["wSD_ci_low"] < pooled["wSD"] < pooled["wSD_ci_high"]


@pytest.mark.unit
def test_wcv_is_reported_only_where_the_preregistration_asks_for_it():
    table = study.synthetic_table({"A": 30}, {"python": 0.3}, seed=4)
    kwargs = dict(margin=1.25, confidence=0.95, interval="pitman", n_boot=50, seed=1)
    assert np.isnan(study.analyse(table, **kwargs)["metrics"][0]["wCV_log"])
    listed = study.analyse(table, report_wcv_for=[{"region": "WM", "parameter": "Ktrans"}], **kwargs)
    assert listed["metrics"][0]["wCV_log"] == pytest.approx(0.3 / 5.0, rel=0.4)


@pytest.mark.unit
def test_incomplete_pairs_are_listed_not_silently_lost():
    table = study.synthetic_table({"A": 6}, {"python": 0.3}, seed=4)
    table = [r for r in table if not (r["subject"] == "sub-A000" and r["session"] == "ses-02")]
    results = study.analyse(table, margin=1.25, confidence=0.95, interval="pitman", n_boot=50, seed=1)
    assert results["metrics"][0]["n"] == 5
    assert [(d["subject"], d["reason"]) for d in results["dropped"]] == [("sub-A000", "1 session(s), need 2")]


# --------------------------------------------------------------------------- command line


def write_tables(tmp_path, table):
    path = tmp_path / "roi_medians.csv"
    study.write_table(path, table, study.TABLE_COLUMNS)
    return path


@pytest.mark.integration
def test_cli_refuses_the_unblinded_mode_and_writes_nothing(tmp_path, capsys):
    table = write_tables(tmp_path, big_table(n=10))
    prereg = copy.deepcopy(COMPLETE_PREREG)
    prereg["noninferiority"]["margin"] = None
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg))

    code = cli.main(["analyse", "--mode", "unblinded", "--table", str(table), "--prereg", str(prereg_path), "--out", str(tmp_path / "reports")])

    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "reports").exists()


@pytest.mark.integration
def test_cli_shuffled_report_is_labelled_and_the_unblinded_one_needs_sign_off(tmp_path):
    table = write_tables(tmp_path, big_table(n=20))
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(COMPLETE_PREREG))
    common = ["--table", str(table), "--prereg", str(prereg_path), "--out", str(tmp_path / "reports"), "--no-plots"]

    assert cli.main(["analyse", "--mode", "shuffled", "--n-boot", "100", *common]) == 0
    shuffled = (tmp_path / "reports/shuffled/report.md").read_text()
    assert "LABEL-SHUFFLED" in shuffled and "NOT results" in shuffled

    assert cli.main(["analyse", "--mode", "unblinded", "--subset", "held_out", *common]) == 0
    unblinded = (tmp_path / "reports/unblinded/report.md").read_text()
    assert "UNBLINDED" in unblinded and "subset: held_out" in unblinded
    rows = study.read_table(tmp_path / "reports/unblinded/noninferiority.csv", numeric=("n",))
    split = rp.stratified_split({"A": [f"sub-A{i:03d}" for i in range(20)]}, 0.33, 7)
    assert next(r for r in rows if r["cohort"] == "A")["n"] == sum(v == "held_out" for v in split.values())


@pytest.mark.integration
def test_cli_collect_and_selfcheck(tmp_path, capsys):
    make_fitted_session(tmp_path, session="ses-01")
    make_fitted_session(tmp_path, session="ses-02", ktrans_wm=0.003)
    arm = f"python:AC:{tmp_path / 'fits'}:{tmp_path / 'dceprep'}"

    assert cli.main(["collect", "--arm", arm, "--out", str(tmp_path / "tables")]) == 0
    rows = study.read_roi_table(tmp_path / "tables/roi_medians.csv")
    assert len(rows) == 20 and {r["session"] for r in rows} == {"ses-01", "ses-02"}
    assert "0.003" not in capsys.readouterr().out  # collected values are never printed

    assert cli.main(["selfcheck"]) == 0
