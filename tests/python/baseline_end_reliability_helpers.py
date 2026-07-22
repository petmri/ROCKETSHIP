"""Helpers for evaluating automatic steady-state-end (end-baseline) detectors against
human-rated ground truth carried in AIF-mask JSON sidecars (`SteadyStateEndTimeIndex`,
as produced by tools like AIFArtist)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    _glr_baseline_end,
    _legacy_sobel_baseline_end,
    _piecewise_constant_baseline_end,
    _tv_baseline_end,
)


# Registration order also drives figure legend / summary table order.
DETECTORS: Dict[str, Callable[[np.ndarray], Dict[str, Any]]] = {
    "piecewise_constant": _piecewise_constant_baseline_end,
    "legacy_sobel": _legacy_sobel_baseline_end,
    "glr": _glr_baseline_end,
    "tv": _tv_baseline_end,
}

_NON_DYNAMIC_TOKENS = ("mask", "t1map", "seg", "roi")
_SUBJECT_RE = re.compile(r"^sub-[A-Za-z0-9]+$")
_SESSION_RE = re.compile(r"^ses-[A-Za-z0-9]+$")


@dataclass(frozen=True)
class AifSidecarRecord:
    """A discovered AIF-mask sidecar carrying a human-rated SteadyStateEndTimeIndex."""

    json_path: Path
    mask_path: Path
    subject: str
    session: Optional[str]
    ground_truth_1b: int
    input_image: Optional[str]

    @property
    def id(self) -> str:
        return f"{self.subject}_{self.session}" if self.session else self.subject


@dataclass
class SessionResult:
    """Outcome of running all detectors on one session's AIF-mask curve."""

    subject: str
    session: Optional[str]
    sidecar_path: Path
    dynamic_path: Optional[Path]
    dynamic_source: str  # "bids" | "fallback" | "none"
    status: str  # "ok" | "skipped"
    reason: Optional[str]
    ground_truth_1b: Optional[int]
    n_timepoints: Optional[int]
    predictions: Dict[str, Optional[int]]
    mean_curve: Optional[np.ndarray]

    @property
    def id(self) -> str:
        return f"{self.subject}_{self.session}" if self.session else self.subject


def is_ground_truth_valid(result: SessionResult) -> bool:
    """True when the session ran and its ground truth index falls within the curve length."""
    return (
        result.status == "ok"
        and result.ground_truth_1b is not None
        and result.n_timepoints is not None
        and 1 <= result.ground_truth_1b <= result.n_timepoints
    )


def _swap_json_to_mask(json_path: Path) -> Optional[Path]:
    """Reverse of dce_pipeline._resolve_aif_sidecar_steady_state_end's nii->json swap."""
    base = str(json_path)[: -len(".json")]
    gz_candidate = Path(base + ".nii.gz")
    if gz_candidate.exists():
        return gz_candidate
    plain_candidate = Path(base + ".nii")
    if plain_candidate.exists():
        return plain_candidate
    return None


def discover_aif_sidecars(
    derivatives_root: Path, *, subjects: Optional[Iterable[str]] = None
) -> List[AifSidecarRecord]:
    """Recursively find AIF-mask JSON sidecars carrying SteadyStateEndTimeIndex.

    No pipeline-folder name is assumed -- any `*.json` under `derivatives_root` with that
    key is treated as ground truth, so this generalizes across pipeline folders (e.g. an
    "AIFArtist" and a sibling "AIFArtist_old" both get picked up as independent rows; point
    `derivatives_root` at a single pipeline folder to avoid that).
    """
    root = Path(derivatives_root).expanduser().resolve()
    subject_filter = {str(s).strip() for s in subjects} if subjects else None
    records: List[AifSidecarRecord] = []
    for json_path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: failed to parse {json_path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(payload, dict) or "SteadyStateEndTimeIndex" not in payload:
            continue

        mask_path = _swap_json_to_mask(json_path)
        if mask_path is None:
            print(f"WARNING: no paired mask file (.nii/.nii.gz) for sidecar {json_path}", file=sys.stderr)
            continue

        subject = next((part for part in json_path.parts if _SUBJECT_RE.match(part)), None)
        if subject is None:
            print(f"WARNING: could not determine BIDS subject from path {json_path}", file=sys.stderr)
            continue
        if subject_filter is not None and subject not in subject_filter:
            continue
        session = next((part for part in json_path.parts if _SESSION_RE.match(part)), None)

        try:
            # AIFArtist writes SteadyStateEndTimeIndex as a 0-based frame index; convert to
            # the 1-based convention used by the detectors' end_ss_1b outputs.
            ground_truth_1b = int(payload["SteadyStateEndTimeIndex"]) + 1
        except (TypeError, ValueError):
            print(f"WARNING: non-integer SteadyStateEndTimeIndex in {json_path}", file=sys.stderr)
            continue

        records.append(
            AifSidecarRecord(
                json_path=json_path,
                mask_path=mask_path,
                subject=subject,
                session=session,
                ground_truth_1b=ground_truth_1b,
                input_image=payload.get("InputImage"),
            )
        )
    return records


def find_dynamic_file(
    raw_root: Path,
    subject: str,
    session: Optional[str],
    input_image_fallback: Optional[str],
) -> Tuple[Optional[Path], str, Optional[str]]:
    """Locate the raw dynamic 4D DCE series for a subject/session.

    Returns (path_or_None, source in {"bids","fallback","none"}, optional warning note).
    BIDS-convention discovery under `raw_root/subject/session/dce/` is tried first;
    `input_image_fallback` (AIFArtist's recorded absolute source path) is a fallback only,
    since it can point outside any BIDS raw tree.
    """
    dce_dir = Path(raw_root) / subject / (session or "") / "dce"
    candidates = sorted(dce_dir.glob("*.nii")) + sorted(dce_dir.glob("*.nii.gz")) if dce_dir.is_dir() else []
    candidates = [c for c in candidates if not any(tok in c.name.lower() for tok in _NON_DYNAMIC_TOKENS)]

    if len(candidates) == 1:
        return candidates[0], "bids", None
    if len(candidates) > 1:
        dce_named = [c for c in candidates if "dce" in c.name.lower()]
        if len(dce_named) == 1:
            return dce_named[0], "bids", None
        chosen = sorted(candidates)[0]
        return chosen, "bids", f"ambiguous dynamic file ({len(candidates)} candidates in {dce_dir}), chose {chosen.name}"

    if input_image_fallback:
        fallback_path = Path(input_image_fallback)
        if fallback_path.exists():
            return fallback_path, "fallback", f"no BIDS dynamic file under {dce_dir}; used InputImage fallback {fallback_path}"

    return None, "none", f"no dynamic file found under {dce_dir} and no usable InputImage fallback"


def process_session(record: AifSidecarRecord, raw_root: Path, *, use_all_voxels: bool = False) -> SessionResult:
    """Load one session's data, extract the signal curve, and run all detectors.

    By default the curve is the mean over the AIF mask. With `use_all_voxels=True`, the
    mask is ignored entirely and the curve is the mean over every voxel in the dynamic
    image instead -- an either/or choice, not a combination of both.
    """

    def _skip(dyn_path: Optional[Path], source: str, reason: str) -> SessionResult:
        return SessionResult(
            subject=record.subject,
            session=record.session,
            sidecar_path=record.json_path,
            dynamic_path=dyn_path,
            dynamic_source=source,
            status="skipped",
            reason=reason,
            ground_truth_1b=record.ground_truth_1b,
            n_timepoints=None,
            predictions={},
            mean_curve=None,
        )

    dyn_path, source, note = find_dynamic_file(raw_root, record.subject, record.session, record.input_image)
    if dyn_path is None:
        return _skip(None, source, note or "no dynamic file found")
    if note:
        print(f"WARNING: {record.id}: {note}", file=sys.stderr)

    import nibabel as nib  # type: ignore  # local import mirrors dce_pipeline._load_nifti_data

    try:
        dynamic = np.asarray(nib.load(str(dyn_path)).get_fdata(), dtype=np.float64)
    except Exception as exc:
        return _skip(dyn_path, source, f"failed to load NIfTI data: {exc}")

    if dynamic.ndim != 4:
        return _skip(dyn_path, source, f"dynamic image is not 4D (shape={dynamic.shape})")

    if use_all_voxels:
        stlv = dynamic.reshape(-1, dynamic.shape[-1]).T  # (n_time, n_voxels), whole image
    else:
        try:
            mask = np.asarray(nib.load(str(record.mask_path)).get_fdata(), dtype=np.float64)
        except Exception as exc:
            return _skip(dyn_path, source, f"failed to load NIfTI data: {exc}")
        if mask.ndim == 4:
            mask = mask[..., 0]
        if mask.shape != dynamic.shape[:3]:
            return _skip(dyn_path, source, f"mask shape {mask.shape} != dynamic spatial shape {dynamic.shape[:3]}")

        # AIFArtist's `SelectedAIFLabels` is metadata about which candidate label the rater
        # picked during rating, not the voxel value written into the exported mask -- the
        # mask file itself is already a plain binary (0/1) mask of the selected voxels
        # regardless of the label ID (confirmed against real AIFArtist output: masks with
        # SelectedAIFLabels=[2] or [3] contain only 0/1, never 2 or 3), so it's not used here.
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return _skip(dyn_path, source, "AIF mask has no positive voxels")

        stlv = dynamic[mask_bool].T  # (n_time, n_voxels)
    n_timepoints = int(stlv.shape[0])
    mean_curve = np.mean(stlv, axis=1)

    predictions: Dict[str, Optional[int]] = {}
    for name, detector in DETECTORS.items():
        try:
            predictions[name] = int(detector(stlv)["end_ss_1b"])
        except Exception as exc:
            print(f"WARNING: {record.id}: detector '{name}' failed: {exc}", file=sys.stderr)
            predictions[name] = None

    return SessionResult(
        subject=record.subject,
        session=record.session,
        sidecar_path=record.json_path,
        dynamic_path=dyn_path,
        dynamic_source=source,
        status="ok",
        reason=None,
        ground_truth_1b=record.ground_truth_1b,
        n_timepoints=n_timepoints,
        predictions=predictions,
        mean_curve=mean_curve,
    )


@dataclass
class AlgorithmStats:
    name: str
    n_valid: int
    n_exact_match: int
    accuracy_pct: float
    mse: float
    n_within_tolerance: Optional[int] = None
    tolerance_pct: Optional[float] = None


def compute_algorithm_stats(
    results: Iterable[SessionResult], *, tolerance_frames: Optional[int] = None
) -> Dict[str, AlgorithmStats]:
    """Per-algorithm exact-match accuracy % and MSE (frame-index units) over valid sessions."""
    results = list(results)
    stats: Dict[str, AlgorithmStats] = {}
    for name in DETECTORS:
        errors: List[float] = []
        n_exact = 0
        n_within_tol = 0
        n_valid = 0
        for result in results:
            if not is_ground_truth_valid(result):
                continue
            pred = result.predictions.get(name)
            if pred is None:
                continue
            n_valid += 1
            gt = int(result.ground_truth_1b)  # type: ignore[arg-type]
            err = pred - gt
            errors.append(float(err))
            if pred == gt:
                n_exact += 1
            if tolerance_frames is not None and abs(err) <= tolerance_frames:
                n_within_tol += 1

        stats[name] = AlgorithmStats(
            name=name,
            n_valid=n_valid,
            n_exact_match=n_exact,
            accuracy_pct=(100.0 * n_exact / n_valid) if n_valid > 0 else float("nan"),
            mse=float(np.mean(np.square(errors))) if errors else float("nan"),
            n_within_tolerance=(n_within_tol if tolerance_frames is not None else None),
            tolerance_pct=(
                (100.0 * n_within_tol / n_valid) if (tolerance_frames is not None and n_valid > 0) else None
            ),
        )
    return stats
