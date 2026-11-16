"""run_testretest_derive_rater.py: another rater's fit inputs from an existing DCEPrep run."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

import run_testretest_derive_rater as derive  # noqa: E402
import testretest_study as study  # noqa: E402

SUBJECT, SESSION = "sub-QINGBMTR900", "ses-01"
PREFIX = f"{SUBJECT}_{SESSION}"
TRIMMED_BOTTOM = 2  # the base run dropped two empty slices at the bottom


def affine(z_origin=0.0):
    out = np.diag([2.0, 2.0, 5.0, 1.0])
    out[2, 3] = z_origin
    return out


def save(path: Path, data, z_origin=0.0, dtype=np.float32) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=dtype), affine(z_origin)), path)
    return path


def make_inputs(root: Path, aif_voxels, tumor_voxels=(), baseline_end=28):
    """A base DCEPrep session on an 8x8x4 grid cut from the rater's 8x8x6 drawing grid."""
    base = root / "dceprep-AC" / SUBJECT / SESSION
    shape = (8, 8, 4)
    brain = np.zeros(shape)
    brain[1:7, 1:7, :] = 1
    dynamic = np.zeros((*shape, 5))
    dynamic[brain > 0] = 100.0  # the series is masked: no signal outside the brain
    t1 = np.where(brain > 0, 1200.0, 0.0)
    t1[2, 2, 1] = np.nan  # a brain voxel whose T1 fit failed
    z0 = 5.0 * TRIMMED_BOTTOM
    save(base / f"dce/{PREFIX}_desc-bfcz_DCE.nii.gz", dynamic, z0)
    save(base / f"anat/{PREFIX}_space-DCEref_T1map.nii.gz", t1, z0)
    save(base / f"anat/{PREFIX}_space-DCEref_label-brain_mask.nii.gz", brain, z0)
    save(base / f"anat/{PREFIX}_space-DCEref_label-WM_mask.nii.gz", brain, z0)
    (base / f"dce/{PREFIX}_desc-hmc_maxdisp.txt").write_text("Max displacement: 0.5mm at frame 3/5\n")

    rater = root / "AIFArtist" / SUBJECT / SESSION / "dce"
    for label, voxels in (("AIF", aif_voxels), ("Tumor", tumor_voxels)):
        if not voxels:
            continue
        mask = np.zeros((8, 8, 6))
        for voxel in voxels:
            mask[voxel] = 1
        save(rater / f"{PREFIX}_run-01_desc-raterSG_label-{label}_mask.nii.gz", mask, dtype=np.uint8)
    (rater / f"{PREFIX}_run-01_desc-raterSG_label-AIF_mask.json").write_text(json.dumps({"SteadyStateEndTimeIndex": baseline_end}))
    return base, rater


@pytest.mark.unit
def test_to_grid_applies_the_trim_and_refuses_what_it_cannot_map(tmp_path):
    drawn = np.zeros((8, 8, 6))
    drawn[3, 4, 2] = 1  # slice 2 of the drawing grid is slice 0 after the trim
    mask_image = nib.Nifti1Image(drawn.astype(np.uint8), affine())
    reference = nib.Nifti1Image(np.zeros((8, 8, 4, 5), dtype=np.float32), affine(10.0))

    mask, reason = derive.to_grid(mask_image, reference)
    assert reason == "" and mask.shape == (8, 8, 4) and np.argwhere(mask).tolist() == [[3, 4, 0]]

    drawn[3, 4, 1] = 1  # in a trimmed slice
    mask, reason = derive.to_grid(nib.Nifti1Image(drawn.astype(np.uint8), affine()), reference)
    assert mask is None and "1 of 2 mask voxels fall in slices the base run trimmed" in reason

    half_shift = nib.Nifti1Image(np.zeros((8, 8, 4, 5), dtype=np.float32), affine(12.5))
    mask, reason = derive.to_grid(mask_image, half_shift)
    assert mask is None and "whole-voxel shift" in reason


@pytest.mark.unit
def test_derived_session_links_the_preprocessing_and_adds_the_raters_aif(tmp_path):
    base, rater = make_inputs(tmp_path, aif_voxels=[(3, 3, 3), (2, 2, 3)], tumor_voxels=[(4, 4, 4), (4, 5, 4)])
    out = tmp_path / "derived" / SUBJECT / SESSION

    row = derive.derive_session("SG", rater, base, out, blood_t1_ms=1664.0)

    assert row["status"] == "ok" and row["aif_voxels"] == 2 and row["aif_voxels_t1_filled"] == 1
    assert row["baseline_end"] == 28 and row["tumor_voxels"] == 2
    linked = out / f"dce/{PREFIX}_desc-bfcz_DCE.nii.gz"
    assert linked.is_symlink() and linked.resolve() == (base / f"dce/{PREFIX}_desc-bfcz_DCE.nii.gz").resolve()
    assert not Path(linked.readlink()).is_absolute()  # survives a different mount point
    aif_t1 = nib.load(out / f"dce/{PREFIX}_label-AIF_T1map.nii.gz").get_fdata()
    assert int((aif_t1 > 0).sum()) == 2
    assert aif_t1[3, 3, 1] == 1200.0  # drawn slice 3 is slice 1 after the trim
    assert aif_t1[2, 2, 1] == 1664.0  # no valid T1 there: the cohort blood T1 marks the voxel
    assert json.loads((out / f"dce/{PREFIX}_label-AIF_T1map.json").read_text())["SteadyStateEndTimeIndex"] == 28
    tumor = nib.load(out / f"anat/{PREFIX}_space-DCEref_label-tumor_mask.nii.gz").get_fdata()
    assert np.argwhere(tumor > 0).tolist() == [[4, 4, 2], [4, 5, 2]]


@pytest.mark.unit
def test_an_aif_outside_the_base_runs_mask_needs_dceprep_and_writes_nothing(tmp_path):
    """The base run zeroed the series outside brain-or-AIF of its own rater."""
    base, rater = make_inputs(tmp_path, aif_voxels=[(3, 3, 3), (0, 0, 3)])
    out = tmp_path / "derived" / SUBJECT / SESSION

    row = derive.derive_session("SG", rater, base, out, blood_t1_ms=None)

    assert row["status"] == "needs_dceprep"
    assert "1 of 2 AIF voxels lie outside" in row["detail"]
    assert not out.exists()


@pytest.mark.integration
def test_command_line_writes_the_manifest_and_keeps_the_base_run_read_only(tmp_path):
    base, _ = make_inputs(tmp_path, aif_voxels=[(3, 3, 3)])
    before = sorted(p.name for p in base.rglob("*"))
    out = tmp_path / "derived"
    common = ["--rater", "SG", "--aifartist", str(tmp_path / "AIFArtist"), "--base-run", str(tmp_path / "dceprep-AC")]

    assert derive.main([*common, "--out", str(out)]) == 0

    with open(out / "derive_manifest.tsv", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(r["subject"], r["status"], r["aif_voxels"]) for r in rows] == [(SUBJECT, "ok", "1")]
    assert "Not a DCEPrep run" in (out / "DERIVED.md").read_text()
    assert sorted(p.name for p in base.rglob("*")) == before
    with pytest.raises(SystemExit):
        derive.main([*common, "--out", str(tmp_path / "dceprep-AC" / "derived")])


@pytest.mark.integration
def test_tumour_rois_are_written_even_where_the_aif_needs_dceprep(tmp_path):
    make_inputs(tmp_path, aif_voxels=[(0, 0, 3)], tumor_voxels=[(4, 4, 4)])  # AIF outside the base mask
    common = ["--rater", "SG", "--aifartist", str(tmp_path / "AIFArtist"), "--base-run", str(tmp_path / "dceprep-AC")]

    assert derive.main([*common, "--out", str(tmp_path / "derived"), "--regions-out", str(tmp_path / "regions")]) == 0

    assert not (tmp_path / "derived" / SUBJECT).exists()
    tumor = nib.load(tmp_path / "regions" / SUBJECT / SESSION / f"anat/{PREFIX}_space-DCEref_label-tumor_mask.nii.gz").get_fdata()
    assert np.argwhere(tumor > 0).tolist() == [[4, 4, 2]]
    assert "\tok\t" in (tmp_path / "regions/regions_manifest.tsv").read_text()
    with pytest.raises(SystemExit):
        derive.main(common)


@pytest.mark.unit
def test_collector_applies_a_tumour_roi_from_another_run_to_any_raters_fits(tmp_path):
    """Regions 'tumor' and 'WM minus tumor' come from the derived run via region_roots."""
    base, rater = make_inputs(tmp_path, aif_voxels=[(3, 3, 3)], tumor_voxels=[(2, 2, 3), (2, 3, 3)])
    derive.derive_session("SG", rater, base, tmp_path / "derived" / SUBJECT / SESSION, blood_t1_ms=None)
    fit = tmp_path / "fits" / SUBJECT / SESSION
    ktrans = np.full((8, 8, 4), 0.002)
    ktrans[2, 2, 1] = ktrans[2, 3, 1] = 0.05  # the tumour voxels
    save(fit / f"{PREFIX}_patlak_fit_Ktrans.nii.gz", ktrans, 10.0)
    regions = {
        "WM": "anat/*_space-DCEref_label-WM_mask.nii*",
        "tumor": "anat/*_space-DCEref_label-tumor_mask.nii*",
        "WM_nontumor": {"mask": "anat/*_space-DCEref_label-WM_mask.nii*", "minus": "anat/*_space-DCEref_label-tumor_mask.nii*"},
    }
    kwargs = dict(pipeline="python", rater="AC", regions=regions, model_parameters={"patlak": ["Ktrans"]})

    rows, _, problems = study.collect_session(fit, base, region_roots=[tmp_path / "derived"], **kwargs)
    by_region = {r["region"]: r for r in rows}
    assert problems == []
    assert by_region["tumor"]["value"] == pytest.approx(0.05) and by_region["tumor"]["n_voxels"] == 2
    assert by_region["WM_nontumor"]["n_voxels"] == by_region["WM"]["n_voxels"] - 2

    # without the other run the tumour regions are left out and reported, WM stays
    rows, _, problems = study.collect_session(fit, base, **kwargs)
    assert {r["region"] for r in rows} == {"WM"} and len(problems) == 2
