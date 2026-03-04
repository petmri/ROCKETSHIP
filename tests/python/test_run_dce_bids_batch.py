"""Unit tests for DCE BIDS batch config assembly."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import BidsSession  # noqa: E402
from run_dce_bids_batch import _build_session_config  # noqa: E402


def _make_session(tmp_root: Path) -> BidsSession:
    bids_root = tmp_root / "bids"
    raw_ses = bids_root / "rawdata" / "sub-01" / "ses-01"
    deriv_ses = bids_root / "derivatives" / "dceprep" / "sub-01" / "ses-01"
    dce_deriv = deriv_ses / "dce"
    anat_deriv = deriv_ses / "anat"
    raw_dce = raw_ses / "dce"

    dce_deriv.mkdir(parents=True, exist_ok=True)
    anat_deriv.mkdir(parents=True, exist_ok=True)
    raw_dce.mkdir(parents=True, exist_ok=True)

    (dce_deriv / "sub-01_ses-01_desc-bfcz_DCE.nii.gz").write_text("")
    (dce_deriv / "sub-01_ses-01_desc-AIF_T1map.nii.gz").write_text("")
    (anat_deriv / "sub-01_ses-01_space-DCEref_desc-brain_mask.nii.gz").write_text("")
    (anat_deriv / "sub-01_ses-01_space-DCEref_T1map.nii.gz").write_text("")
    (dce_deriv / "sub-01_ses-01_DCE.json").write_text(
        '{"RepetitionTime": 0.005, "FlipAngle": 15, "TemporalResolution": 5.0}'
    )

    return BidsSession(
        bids_root=bids_root,
        subject="sub-01",
        session="ses-01",
        rawdata_path=raw_ses,
        derivatives_path=deriv_ses,
    )


def test_batch_config_strips_template_injection_windows_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        session = _make_session(Path(tmp))
        config = _build_session_config(
            session=session,
            output_dir=Path(tmp) / "out",
            backend="auto",
            models=[],
            config_template={
                "stage_overrides": {
                    "start_injection_min": 1.056,
                    "end_injection_min": 1.584,
                    "aif_curve_mode": "fitted",
                }
            },
            set_overrides={},
        )

        keys = {str(k).strip().lower() for k in config.stage_overrides}
        assert "start_injection_min" not in keys
        assert "end_injection_min" not in keys
        assert config.stage_overrides.get("aif_curve_mode") == "fitted"


def test_batch_config_keeps_injection_windows_when_explicitly_set() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        session = _make_session(Path(tmp))
        config = _build_session_config(
            session=session,
            output_dir=Path(tmp) / "out",
            backend="auto",
            models=[],
            config_template={
                "stage_overrides": {
                    "start_injection_min": 1.056,
                    "end_injection_min": 1.584,
                }
            },
            set_overrides={
                "start_injection_min": "0.75",
                "end_injection_min": "1.05",
            },
        )

        assert str(config.stage_overrides["start_injection_min"]) == "0.75"
        assert str(config.stage_overrides["end_injection_min"]) == "1.05"
