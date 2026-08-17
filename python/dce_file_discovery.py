"""BIDS file discovery for DCE preprocessing pipeline (dceprep)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
ROI_MASK_PATTERN = "*space-DCEref_label-brain_mask.nii*"
T1_MAP_PATTERN = "*space-DCEref_T1map.nii*"
NOISE_MASK_PATTERN = "*label-noise_mask.nii*"
METADATA_PATTERN = "*DCE.json"

# Inputs the pipeline cannot run without; the rest are optional.
REQUIRED_INPUT_KINDS = ("dynamic", "aif_mask", "roi_mask", "t1_map")

_PATTERN_HINTS = {
    "dynamic": f"{DYNAMIC_PATTERN} or {DYNAMIC_FALLBACK_PATTERN}",
    "aif_mask": AIF_MASK_PATTERN,
    "roi_mask": ROI_MASK_PATTERN,
    "t1_map": T1_MAP_PATTERN,
}


def _find_one(parent: Path, pattern: str) -> Optional[Path]:
    """Find first file matching glob pattern in directory (sorted)."""
    if not parent.is_dir():
        return None
    matches = sorted(parent.glob(pattern))
    return matches[0] if matches else None


def discover_dce_input_paths(session: BidsSession) -> Dict[str, Optional[Path]]:
    """Locate each dceprep input, using None for anything that is missing.

    Same conventions as `discover_dce_inputs`, without its all-or-nothing contract, so
    callers that need to show partial results (the GUI's auto-fill) don't have to keep
    a second copy of the naming convention.
    """
    dce_deriv = session.derivatives_path / "dce"
    anat_deriv = session.derivatives_path / "anat"

    dynamic = _find_one(dce_deriv, DYNAMIC_PATTERN)
    if dynamic is None:
        dynamic = _find_one(dce_deriv, DYNAMIC_FALLBACK_PATTERN)

    return {
        "dynamic": dynamic,
        "aif_mask": _find_one(dce_deriv, AIF_MASK_PATTERN),
        "roi_mask": _find_one(anat_deriv, ROI_MASK_PATTERN),
        "t1_map": _find_one(anat_deriv, T1_MAP_PATTERN),
        "noise_mask": _find_one(anat_deriv, NOISE_MASK_PATTERN),
        "metadata_json": _find_one(dce_deriv, METADATA_PATTERN),
    }


def missing_required_inputs(found: Dict[str, Optional[Path]]) -> List[str]:
    """Names of the required inputs that discovery could not locate."""
    return [kind for kind in REQUIRED_INPUT_KINDS if found.get(kind) is None]


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
    found = discover_dce_input_paths(session)

    missing = [f"{kind} (pattern: {_PATTERN_HINTS[kind]})" for kind in missing_required_inputs(found)]
    if missing:
        raise FileNotFoundError(
            f"Missing DCE derivative inputs for {session.id}:\n" + "\n  ".join([""] + missing)
        )

    return DceInputs(session=session, **found)
