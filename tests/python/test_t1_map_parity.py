"""End-to-end T1 map parity: Python parametric pipeline vs MATLAB.

T1 mapping is the primary Python porting target, but before this test the only
T1-vs-MATLAB coverage was a single 4-point synthetic voxel fit. Here the full
Python VFA T1 pipeline is run on a small real VFA fixture and its T1 map is
compared voxel-by-voxel against a MATLAB reference produced by
``tests/matlab/generate_t1_parity_map.m`` (which loops the function-level
parity-validated ``fitParameter`` over the same fixture).

Per project guidance on noisy data: variable-flip-angle T1 is only well
conditioned where there is real signal, so parity is asserted only on voxels
whose fitted T1 is physiologically plausible in BOTH maps. Background / CSF
voxels where T1 is non-identifiable (both optimizers wander to huge values) are
excluded, and a floor on valid-voxel count keeps the masking from making the
test pass vacuously.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline  # noqa: E402

# Small VFA T1 fixture, hosted as the anat series of the BIDS_test sub-11tiny subject.
# flip-01/02/03 correspond to the 2/5/10 deg flip angles.
BIDS_ROOT = REPO_ROOT / "tests/data/BIDS_test"
T1_SUBJECT = "sub-11tiny"
T1_SESSION = "ses-01"
_T1_STEM = f"{T1_SUBJECT}_{T1_SESSION}"
FIXTURE = BIDS_ROOT / "rawdata" / T1_SUBJECT / T1_SESSION / "anat"
MATLAB_MAP = (
    BIDS_ROOT / "derivatives" / "matlabref" / T1_SUBJECT / T1_SESSION / "anat" / f"{_T1_STEM}_desc-t1fafit_T1map.nii"
)
VFA_FILES = [
    FIXTURE / f"{_T1_STEM}_flip-01_VFA.nii.gz",
    FIXTURE / f"{_T1_STEM}_flip-02_VFA.nii.gz",
    FIXTURE / f"{_T1_STEM}_flip-03_VFA.nii.gz",
]
FLIP_ANGLES_DEG = [2.0, 5.0, 10.0]
TR_MS = 8.012

# Physiologically plausible T1 (ms) at clinical field strengths. Outside this band
# the VFA fit is non-identifiable; both implementations diverge there for reasons
# unrelated to porting, so those voxels are not gated (see module docstring).
T1_PLAUSIBLE_MIN_MS = 100.0
T1_PLAUSIBLE_MAX_MS = 4000.0

# Thresholds carry generous headroom over measured same-fixture agreement
# (corr 0.99999999, MAE 0.028 ms, max rel err 0.03%, 100% within 1%), so a real
# port divergence fails while cross-MATLAB-version optimizer noise passes.
CORR_MIN = 0.999
MAE_MAX_MS = 2.0
MAX_REL_ERR = 0.01
FRAC_WITHIN_1PCT_MIN = 0.98
MIN_VALID_VOXELS = 300


def _load_map(path: Path) -> np.ndarray:
    return np.asarray(np.squeeze(nib.load(str(path)).get_fdata()), dtype=np.float64)


def _matlab_hint() -> str:
    return (
        f"Missing MATLAB T1 reference map: {MATLAB_MAP}\nGenerate it with:\n"
        "  matlab -batch \"addpath('tests/matlab'); addpath('tests/matlab/helpers'); "
        "generate_t1_parity_map('vfaFiles', "
        "{'tests/data/BIDS_test/rawdata/sub-11tiny/ses-01/anat/sub-11tiny_ses-01_flip-01_VFA.nii.gz',"
        "'tests/data/BIDS_test/rawdata/sub-11tiny/ses-01/anat/sub-11tiny_ses-01_flip-02_VFA.nii.gz',"
        "'tests/data/BIDS_test/rawdata/sub-11tiny/ses-01/anat/sub-11tiny_ses-01_flip-03_VFA.nii.gz'}, "
        "'flipAngles', [2 5 10], 'trMs', 8.012, 'fitType', 't1_fa_fit', "
        "'outputPath', 'tests/data/BIDS_test/derivatives/matlabref/sub-11tiny/ses-01/anat/"
        "sub-11tiny_ses-01_desc-t1fafit_T1map.nii', "
        "'rsquaredThreshold', 0);\""
    )


@pytest.mark.parity
@pytest.mark.integration
def test_bids_t1_map_parity_nonlinear(
    parity_summary_dir: Path | None,
) -> None:
    # Default-on: skip gracefully (not fail) if this environment lacks the fixture assets.
    missing = [str(p) for p in list(VFA_FILES) + [MATLAB_MAP] if not p.exists()]
    if missing:
        pytest.skip(f"T1 map parity fixture assets missing ({len(missing)}); first: {missing[0]}")

    with tempfile.TemporaryDirectory() as tmp:
        # from_dict, not the constructor: every unnamed setting then resolves from the
        # shipped parametric_defaults.json, which is what a real run reads.
        config = ParametricT1Config.from_dict(
            {
                "output_dir": str(Path(tmp) / "py_out"),
                "vfa_files": [str(p) for p in VFA_FILES],
                "fit_type": "t1_fa_fit",
                "flip_angles_deg": list(FLIP_ANGLES_DEG),
                "tr_ms": TR_MS,
                "backend": "cpu",
                # No r^2 masking / no fill: compare the raw fit everywhere, then gate on
                # physiological plausibility below so the mask is explicit and symmetric.
                "rsquared_threshold": 0.0,
                "invalid_fill_value": float("nan"),
            }
        )
        result = run_parametric_t1_pipeline(config)
        assert result["meta"]["status"] == "ok", result["meta"]

        t1_candidates = sorted(Path(tmp).glob("py_out/T1_map_*t1_fa_fit*.nii*"))
        assert t1_candidates, f"Python T1 map not written; outputs: {list(Path(tmp).glob('py_out/*'))}"
        py_map = _load_map(t1_candidates[0])

    matlab_map = _load_map(MATLAB_MAP)
    assert py_map.shape == matlab_map.shape, f"shape mismatch py={py_map.shape} matlab={matlab_map.shape}"

    plausible = (
        np.isfinite(py_map)
        & np.isfinite(matlab_map)
        & (py_map > T1_PLAUSIBLE_MIN_MS)
        & (py_map < T1_PLAUSIBLE_MAX_MS)
        & (matlab_map > T1_PLAUSIBLE_MIN_MS)
        & (matlab_map < T1_PLAUSIBLE_MAX_MS)
    )
    n_valid = int(np.count_nonzero(plausible))

    # Anti-vacuous guard: masking to plausible T1 must not collapse the comparison.
    assert n_valid >= MIN_VALID_VOXELS, (
        f"only {n_valid} plausible-T1 voxels (< {MIN_VALID_VOXELS}); mask collapsed, "
        "test would be vacuous. Check the fixture or fit health."
    )

    x = py_map[plausible]
    y = matlab_map[plausible]
    diff = x - y
    rel = np.abs(diff) / np.abs(y)
    corr = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(np.abs(diff)))
    max_rel = float(np.max(rel))
    frac_within_1pct = float(np.mean(rel < 0.01))

    summary = (
        f"t1_map_parity: n={n_valid} corr={corr:.8f} mae_ms={mae:.4g} "
        f"max_rel={max_rel*100:.4f}% frac_within_1pct={frac_within_1pct:.4f}"
    )
    print(f"[PARITY] {summary}", flush=True)

    if parity_summary_dir is not None:
        (parity_summary_dir / "parity_t1_map_summary.json").write_text(
            json.dumps(
                {
                    "suite": "t1-map-nonlinear",
                    "fixture": str(FIXTURE),
                    "n_valid": n_valid,
                    "corr": corr,
                    "mae_ms": mae,
                    "max_rel_err": max_rel,
                    "frac_within_1pct": frac_within_1pct,
                    "plausible_band_ms": [T1_PLAUSIBLE_MIN_MS, T1_PLAUSIBLE_MAX_MS],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    assert corr >= CORR_MIN, f"{summary} (corr_min={CORR_MIN})"
    assert mae <= MAE_MAX_MS, f"{summary} (mae_max_ms={MAE_MAX_MS})"
    assert max_rel <= MAX_REL_ERR, f"{summary} (max_rel_err={MAX_REL_ERR})"
    assert frac_within_1pct >= FRAC_WITHIN_1PCT_MIN, f"{summary} (frac_within_1pct_min={FRAC_WITHIN_1PCT_MIN})"
