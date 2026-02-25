"""Reusable BIDS dataset/session discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class BidsSession:
    """Minimal BIDS subject/session pairing with raw+derivative roots."""

    bids_root: Path
    subject: str
    session: Optional[str]
    rawdata_path: Path
    derivatives_path: Path

    @property
    def id(self) -> str:
        if self.session:
            return f"{self.subject}_{self.session}"
        return self.subject


def discover_bids_sessions(bids_root: Path) -> List[BidsSession]:
    """Discover subject/session entries present in both rawdata and derivatives."""
    root = Path(bids_root).expanduser().resolve()
    raw_root = root / "rawdata"
    deriv_root = root / "derivatives"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing BIDS rawdata directory: {raw_root}")
    if not deriv_root.is_dir():
        raise FileNotFoundError(f"Missing BIDS derivatives directory: {deriv_root}")

    entries: List[BidsSession] = []
    for subject_dir in sorted(raw_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue

        session_dirs = sorted(path for path in subject_dir.glob("ses-*") if path.is_dir())
        if session_dirs:
            for session_dir in session_dirs:
                deriv_session = deriv_root / subject_dir.name / session_dir.name
                if not deriv_session.is_dir():
                    continue
                entries.append(
                    BidsSession(
                        bids_root=root,
                        subject=subject_dir.name,
                        session=session_dir.name,
                        rawdata_path=session_dir,
                        derivatives_path=deriv_session,
                    )
                )
            continue

        deriv_subject = deriv_root / subject_dir.name
        if deriv_subject.is_dir():
            entries.append(
                BidsSession(
                    bids_root=root,
                    subject=subject_dir.name,
                    session=None,
                    rawdata_path=subject_dir,
                    derivatives_path=deriv_subject,
                )
            )

    return entries

