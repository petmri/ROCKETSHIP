"""Helpers for evaluating automatic steady-state-end (end-baseline) detectors against
human-rated ground truth carried in AIF-mask JSON sidecars (`SteadyStateEndTimeIndex`,
as produced by tools like AIFArtist)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _aif_biexp_con,
    _biexp_fit_baseline_end,
    _glr_baseline_end,
    _legacy_sobel_baseline_end,
    _piecewise_constant_baseline_end,
    _tv_baseline_end,
)


DEFAULT_CONFIG_TEMPLATE = REPO_ROOT / "python" / "dce_default.json"

_biexp_config: Optional[DcePipelineConfig] = None


def configure_biexp_detector(config_template: Optional[Path] = None) -> DcePipelineConfig:
    """Build the `DcePipelineConfig` the `biexp_fit` detector reads its `aif_*` settings from.

    Unlike the other four detectors -- which are pure functions of the signal curve -- the
    biexponential fit is governed by the same `aif_lower_limits` / `aif_Robust` /
    `aif_peak_weight_exponent` settings as the production Stage-B fit. Those live in
    `stage_overrides`, so the harness loads them from the real config template rather than
    letting `_stage_override` fall through to its hardcoded defaults; otherwise this would
    measure a configuration nobody actually runs.

    The three required paths are placeholders: nothing in the fit path touches them (with no
    `dce_preferences_path` override, `_load_dce_preferences` never reaches the filesystem).
    """
    global _biexp_config
    template = Path(config_template) if config_template is not None else DEFAULT_CONFIG_TEMPLATE
    overrides: Dict[str, Any] = {}
    if template.exists():
        try:
            payload = json.loads(template.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: failed to parse config template {template}: {exc}", file=sys.stderr)
        else:
            if isinstance(payload, dict) and isinstance(payload.get("stage_overrides"), dict):
                overrides = dict(payload["stage_overrides"])
    else:
        print(f"WARNING: config template {template} not found; using built-in aif_* defaults", file=sys.stderr)

    _biexp_config = DcePipelineConfig(
        subject_source_path=Path("."),
        subject_tp_path=Path("."),
        output_dir=Path("."),
        stage_overrides=overrides,
    )
    return _biexp_config


def _biexp_fit_detector(stlv: np.ndarray) -> Dict[str, Any]:
    if _biexp_config is None:
        configure_biexp_detector()
    return _biexp_fit_baseline_end(stlv, _biexp_config)


def biexp_fitted_curve(details: Dict[str, Any], n_timepoints: int) -> Optional[np.ndarray]:
    """Re-evaluate the `biexp_fit` detector's fitted curve in the original signal units.

    Returns None unless the fit actually ran (`mode == "fit"`). The fit works on a
    baseline-subtracted, max-normalised curve in frame units, so this undoes that scaling to
    put the result back on the same axes as the measured mean curve.
    """
    if details.get("mode") != "fit":
        return None
    params = details.get("fit_params")
    if params is None or len(params) < 6:
        return None
    scale = float(details.get("normalization_scale", 1.0))
    baseline_mean = float(details.get("baseline_mean", 0.0))
    timer = np.arange(int(n_timepoints), dtype=np.float64)
    normalized = _aif_biexp_con(
        timer,
        float(params[0]),
        float(params[1]),
        float(params[2]),
        float(params[3]),
        float(params[4]),
        float(params[5]),
        fitting_au=False,
        baseline=0.0,
    )
    return normalized * scale + baseline_mean


# Registration order also drives figure legend / summary table order.
DETECTORS: Dict[str, Callable[[np.ndarray], Dict[str, Any]]] = {
    "piecewise_constant": _piecewise_constant_baseline_end,
    "legacy_sobel": _legacy_sobel_baseline_end,
    "glr": _glr_baseline_end,
    "tv": _tv_baseline_end,
    "biexp_fit": _biexp_fit_detector,
}

_NON_DYNAMIC_TOKENS = ("mask", "t1map", "seg", "roi")
_SUBJECT_RE = re.compile(r"^sub-[A-Za-z0-9]+$")
_SESSION_RE = re.compile(r"^ses-[A-Za-z0-9]+$")


@dataclass(frozen=True)
class AifSidecarRecord:
    """A discovered AIF mask, with a human-rated SteadyStateEndTimeIndex when one exists.

    `ground_truth_1b` is None for masks found by `discover_aif_masks` (no rating available).
    Such sessions still run every detector and still get a figure; they are simply excluded
    from the accuracy/MSE statistics, which have nothing to score against.
    """

    json_path: Path
    mask_path: Path
    subject: str
    session: Optional[str]
    ground_truth_1b: Optional[int]
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
    # Full per-detector details dicts, keyed the same as `predictions`. Only `biexp_fit`
    # currently carries anything the reporting layer uses (the fitted coefficients and the
    # fractional `end_injection_1b`), but every detector's details are kept for diagnosis.
    detector_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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


DEFAULT_AIF_MASK_PATTERN = "*desc-AIF_T1map.nii*"


def discover_aif_masks(
    derivatives_root: Path,
    *,
    subjects: Optional[Iterable[str]] = None,
    pattern: str = DEFAULT_AIF_MASK_PATTERN,
) -> List[AifSidecarRecord]:
    """Find AIF masks by filename when no human-rated sidecars exist.

    `discover_aif_sidecars` keys on `SteadyStateEndTimeIndex`, so a dataset that was never
    opened in AIFArtist yields nothing at all -- including the `RUNNER_DATA` sessions. This
    discovers the mask directly instead, which still supports everything except the accuracy
    statistics: the detectors run, the figures are drawn, and the detectors can be compared
    against each other even though there is no rating to score them on.

    The default pattern is the one the pipeline itself resolves `aif_files` from
    (`dce_file_discovery.py`), so this selects the same voxels Stage A would.
    """
    root = Path(derivatives_root).expanduser().resolve()
    subject_filter = {str(s).strip() for s in subjects} if subjects else None
    records: List[AifSidecarRecord] = []
    for mask_path in sorted(root.rglob(pattern)):
        if not mask_path.is_file():
            continue
        subject = next((part for part in mask_path.parts if _SUBJECT_RE.match(part)), None)
        if subject is None:
            print(f"WARNING: could not determine BIDS subject from path {mask_path}", file=sys.stderr)
            continue
        if subject_filter is not None and subject not in subject_filter:
            continue
        session = next((part for part in mask_path.parts if _SESSION_RE.match(part)), None)

        records.append(
            AifSidecarRecord(
                json_path=mask_path,  # no sidecar; keep the mask path so reports stay traceable
                mask_path=mask_path,
                subject=subject,
                session=session,
                ground_truth_1b=None,
                input_image=None,
            )
        )
    return records


def find_dynamic_file(
    raw_root: Path,
    subject: str,
    session: Optional[str],
    input_image_fallback: Optional[str],
    dynamic_pattern: Optional[str] = None,
) -> Tuple[Optional[Path], str, Optional[str]]:
    """Locate the dynamic 4D DCE series for a subject/session.

    Returns (path_or_None, source in {"bids","fallback","none"}, optional warning note).
    BIDS-convention discovery under `raw_root/subject/session/dce/` is tried first;
    `input_image_fallback` (AIFArtist's recorded absolute source path) is a fallback only,
    since it can point outside any BIDS raw tree.

    `dynamic_pattern` pins the series explicitly. The default heuristic assumes a *raw* tree
    holding one dynamic per session; pointed at a derivatives tree it has a dozen candidates
    (`desc-bfc`, `desc-hmc`, `desc-biases`, ...) and picks alphabetically, which is arbitrary.
    Since the detectors are meant to be judged on the curve production actually feeds them,
    pass the pipeline's own pattern (`*desc-bfcz_DCE.nii*`) when reading derivatives.
    """
    dce_dir = Path(raw_root) / subject / (session or "") / "dce"
    if dynamic_pattern:
        candidates = sorted(dce_dir.glob(dynamic_pattern)) if dce_dir.is_dir() else []
        if not candidates:
            return None, "none", f"no file matching {dynamic_pattern} under {dce_dir}"
        if len(candidates) > 1:
            chosen = candidates[0]
            return chosen, "bids", f"{len(candidates)} files match {dynamic_pattern} in {dce_dir}, chose {chosen.name}"
        return candidates[0], "bids", None

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


def process_session(
    record: AifSidecarRecord,
    raw_root: Path,
    *,
    use_all_voxels: bool = False,
    dynamic_pattern: Optional[str] = None,
) -> SessionResult:
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

    dyn_path, source, note = find_dynamic_file(
        raw_root, record.subject, record.session, record.input_image, dynamic_pattern
    )
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
    detector_details: Dict[str, Dict[str, Any]] = {}
    for name, detector in DETECTORS.items():
        try:
            details = detector(stlv)
            detector_details[name] = details
            predictions[name] = int(details["end_ss_1b"])
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
        detector_details=detector_details,
    )


@dataclass
class AgreementStats:
    """How one detector compares to a reference detector, over sessions where both ran."""

    name: str
    n_compared: int
    n_identical: int
    identical_pct: float
    mean_signed_offset: float
    max_abs_offset: float


def compute_detector_agreement(
    results: Iterable[SessionResult], *, reference: str = "biexp_fit"
) -> Dict[str, AgreementStats]:
    """Pairwise agreement against `reference`, for datasets with no human ratings.

    Without ground truth there is nothing to score accuracy on, but the detectors can still be
    compared to each other: a detector that tracks the production default everywhere tells a
    very different story from one that is systematically a frame early. Offsets are signed
    (`detector - reference`) so a consistent bias is distinguishable from scatter.
    """
    results = list(results)
    stats: Dict[str, AgreementStats] = {}
    for name in DETECTORS:
        offsets: List[float] = []
        n_identical = 0
        for result in results:
            if result.status != "ok":
                continue
            pred = result.predictions.get(name)
            ref = result.predictions.get(reference)
            if pred is None or ref is None:
                continue
            offsets.append(float(pred - ref))
            if pred == ref:
                n_identical += 1
        n_compared = len(offsets)
        stats[name] = AgreementStats(
            name=name,
            n_compared=n_compared,
            n_identical=n_identical,
            identical_pct=(100.0 * n_identical / n_compared) if n_compared > 0 else float("nan"),
            mean_signed_offset=float(np.mean(offsets)) if offsets else float("nan"),
            max_abs_offset=float(np.max(np.abs(offsets))) if offsets else float("nan"),
        )
    return stats


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
