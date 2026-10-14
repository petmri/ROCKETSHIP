"""Unit tests for parametric BIDS batch config assembly and run summary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from bids_discovery import BidsSession  # noqa: E402
from run_parametric_bids_batch import _build_session_config, main  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _make_session(tmp_root: Path) -> BidsSession:
    bids_root = tmp_root / "bids"
    raw_ses = bids_root / "rawdata" / "sub-01" / "ses-01"
    deriv_ses = bids_root / "derivatives" / "t1prep" / "sub-01" / "ses-01"

    _touch(raw_ses / "anat" / "sub-01_ses-01_flip-01_VFA.nii.gz")
    _touch(raw_ses / "anat" / "sub-01_ses-01_flip-02_VFA.nii.gz")
    _touch(deriv_ses / "anat" / "B1_scaled_FAreg.nii.gz")

    return BidsSession(
        bids_root=bids_root,
        subject="sub-01",
        session="ses-01",
        rawdata_path=raw_ses,
        derivatives_path=deriv_ses,
    )


def test_build_session_config_discovers_vfa_and_b1_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        session = _make_session(Path(tmp))

        config = _build_session_config(
            session=session,
            output_dir=Path(tmp) / "out",
            fit_type="t1_fa_fit",
            config_template=None,
            set_overrides={},
        )

        assert config.fit_type == "t1_fa_fit"
        assert len(config.vfa_files) == 2
        assert config.b1_map_file is not None
        assert config.output_label == "sub-01_ses-01"


def test_main_writes_summary_for_successful_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        session = _make_session(tmp_path)
        summary_path = tmp_path / "summary.json"

        def _fake_run_pipeline(_: Any) -> Dict[str, Any]:
            return {"meta": {"status": "ok"}}

        with patch("run_parametric_bids_batch.run_parametric_t1_pipeline", side_effect=_fake_run_pipeline):
            rc = main(
                [
                    "--bids-root",
                    str(session.bids_root),
                    "--pipeline-folder",
                    "t1prep",
                    "--summary-json",
                    str(summary_path),
                    "--set",
                    "tr_ms=5.0",
                ]
            )

        assert rc == 0
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["sessions_total"] == 1
        assert summary["sessions_success"] == 1
        assert summary["sessions_failed"] == 0


def test_build_session_config_prefers_dceref_vfa_and_uses_raw_sidecars() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        session = _make_session(tmp_path)

        # Add DCEref-space VFA files in derivatives.
        _touch(session.derivatives_path / "anat" / "sub-01_ses-01_flip-01_space-DCEref_VFA.nii.gz")
        _touch(session.derivatives_path / "anat" / "sub-01_ses-01_flip-02_space-DCEref_VFA.nii.gz")
        # Add derivative variants that must not be treated as extra flip frames.
        _touch(session.derivatives_path / "anat" / "sub-01_ses-01_flip-01_space-DCEref_desc-bfc_VFA.nii.gz")
        _touch(session.derivatives_path / "anat" / "sub-01_ses-01_flip-01_space-DCEref_desc-bfcz_VFA.nii.gz")
        _touch(session.derivatives_path / "anat" / "sub-01_ses-01_flip-01_space-DCEref_VFA_RAS.nii.gz")

        # Add raw sidecars carrying acquisition metadata.
        (session.rawdata_path / "anat" / "sub-01_ses-01_flip-01_VFA.json").write_text(
            '{"FlipAngle": 2, "RepetitionTime": 0.005}',
            encoding="utf-8",
        )
        (session.rawdata_path / "anat" / "sub-01_ses-01_flip-02_VFA.json").write_text(
            '{"FlipAngle": 5, "RepetitionTime": 0.005}',
            encoding="utf-8",
        )

        config = _build_session_config(
            session=session,
            output_dir=tmp_path / "out",
            fit_type="t1_fa_fit",
            config_template=None,
            set_overrides={},
        )

        assert len(config.vfa_files) == 2
        assert all("space-DCEref_VFA" in str(path) for path in config.vfa_files)
        assert config.flip_angles_deg == [2.0, 5.0]
        assert config.tr_ms == 5.0


def test_build_session_config_prefers_preprocessed_bfczunified_vfa() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        session = _make_session(tmp_path)

        # Add preprocessed unified DCEref stack.
        unified = session.derivatives_path / "anat" / "sub-01_ses-01_space-DCEref_desc-bfczunified_VFA.nii"
        _touch(unified)

        # Add canonical per-flip DCEref files; these should be ignored when unified exists.
        _touch(session.derivatives_path / "anat" / "sub-01_ses-01_flip-01_space-DCEref_VFA.nii.gz")
        _touch(session.derivatives_path / "anat" / "sub-01_ses-01_flip-02_space-DCEref_VFA.nii.gz")

        # Raw sidecars still provide FA/TR metadata.
        (session.rawdata_path / "anat" / "sub-01_ses-01_flip-01_VFA.json").write_text(
            '{"FlipAngle": 2, "RepetitionTime": 0.005}',
            encoding="utf-8",
        )
        (session.rawdata_path / "anat" / "sub-01_ses-01_flip-02_VFA.json").write_text(
            '{"FlipAngle": 5, "RepetitionTime": 0.005}',
            encoding="utf-8",
        )

        config = _build_session_config(
            session=session,
            output_dir=tmp_path / "out",
            fit_type="t1_fa_fit",
            config_template=None,
            set_overrides={},
        )

        assert len(config.vfa_files) == 1
        assert config.vfa_files[0] == unified.resolve()
        assert config.flip_angles_deg == [2.0, 5.0]
        assert config.tr_ms == 5.0
