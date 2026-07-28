"""BIDS file discovery for DCE preprocessing pipeline (dceprep)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bids_discovery import BidsSession


@dataclass(frozen=True)
class DceInputs:
    """Discovered DCE inputs for a single BIDS session.
    
    Follows dceprep BIDS convention where derivatives are organized as:
      derivatives/{pipeline_folder}/sub-*/ses-*/dce/
      derivatives/{pipeline_folder}/sub-*/ses-*/anat/
    """

    session: BidsSession
    dynamic: Path
    aif_mask: Path
    roi_mask: Path
    t1_map: Path
    noise_mask: Optional[Path] = None
    metadata_json: Optional[Path] = None

    @property
    def all_inputs_exist(self) -> bool:
        """Check that all required inputs are accessible."""
        return all(p.exists() for p in [self.dynamic, self.aif_mask, self.roi_mask, self.t1_map])


# dceprep naming convention. Exported so tooling that has to select the same files the pipeline
# would (e.g. tests/python/run_baseline_end_reliability.py) can share the pattern instead of
# keeping a copy of the string that silently stops matching if the convention changes.
DYNAMIC_PATTERN = "*desc-bfcz_DCE.nii*"
DYNAMIC_FALLBACK_PATTERN = "*DCE.nii*"
AIF_MASK_PATTERN = "*label-AIF_T1map.nii*"


def _find_one(parent: Path, pattern: str) -> Optional[Path]:
    """Find first file matching glob pattern in directory (sorted)."""
    if not parent.is_dir():
        return None
    matches = sorted(parent.glob(pattern))
    return matches[0] if matches else None


def discover_dce_inputs(session: BidsSession) -> DceInputs:
    """Discover DCE derivative inputs following dceprep naming convention.
    
    Expected file locations:
      {derivatives_path}/dce/*desc-bfcz_DCE.nii* or *DCE.nii*
      {derivatives_path}/dce/*label-AIF_T1map.nii*
      {derivatives_path}/anat/*label-brain_mask.nii*
      {derivatives_path}/anat/*space-DCEref_T1map.nii*
    
    Args:
        session: BidsSession with derivatives_path pointing to pipeline output
        
    Returns:
        DceInputs with discovered file paths
        
    Raises:
        FileNotFoundError: If any required file is missing
    """
    dce_deriv = session.derivatives_path / "dce"
    anat_deriv = session.derivatives_path / "anat"

    # Dynamic image
    dynamic = _find_one(dce_deriv, DYNAMIC_PATTERN)
    if dynamic is None:
        dynamic = _find_one(dce_deriv, DYNAMIC_FALLBACK_PATTERN)

    # AIF mask
    aif = _find_one(dce_deriv, AIF_MASK_PATTERN)

    # ROI mask
    roi = _find_one(anat_deriv, "*space-DCEref_label-brain_mask.nii*")

    # T1 map
    t1map = _find_one(anat_deriv, "*space-DCEref_T1map.nii*")

    # Noise mask (optional)
    noise = _find_one(anat_deriv, "*label-noise_mask.nii*")

    # Metadata (optional)
    metadata_json = _find_one(dce_deriv, "*DCE.json")

    missing = []
    if dynamic is None:
        missing.append("dynamic (pattern: *desc-bfcz_DCE.nii* or *DCE.nii*)")
    if aif is None:
        missing.append("aif_mask (pattern: *label-AIF_T1map.nii*)")
    if roi is None:
        missing.append("roi_mask (pattern: *label-brain_mask.nii*)")
    if t1map is None:
        missing.append("t1_map (pattern: *space-DCEref_T1map.nii*)")

    if missing:
        raise FileNotFoundError(
            f"Missing DCE derivative inputs for {session.id}:\n" + "\n  ".join([""] + missing)
        )

    return DceInputs(
        session=session,
        dynamic=dynamic,
        aif_mask=aif,
        roi_mask=roi,
        t1_map=t1map,
        noise_mask=noise,
        metadata_json=metadata_json,
    )
