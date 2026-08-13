"""Helpers for evaluating automatic steady-state-end (end-baseline) detectors against
human-rated ground truth carried in AIF-mask JSON sidecars (`SteadyStateEndTimeIndex`,
as produced by tools like AIFArtist)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

# The pipeline's own discovery globs, so the harness selects the files production would rather
# than a copy of the naming convention. PIPELINE_DYNAMIC_PATTERN is re-exported for the CLI's
# --dynamic-pattern help text; it has no use inside this module.
from dce_file_discovery import (  # noqa: E402,F401
    AIF_MASK_PATTERN as DEFAULT_AIF_MASK_PATTERN,
    DYNAMIC_PATTERN as PIPELINE_DYNAMIC_PATTERN,
)
from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _aif_biexp_con,
    _biexp_fit_baseline_end,
    _fit_aif_biexp,
    _glr_baseline_end,
    _legacy_sobel_baseline_end,
    _piecewise_constant_baseline_end,
    _tv_baseline_end,
)


DEFAULT_CONFIG_TEMPLATE = REPO_ROOT / "python" / "dce_default.json"


def load_biexp_config(config_template: Optional[Path] = None) -> DcePipelineConfig:
    """Load the `DcePipelineConfig` the `biexp_fit` detector reads its `aif_*` settings from.

    Unlike the other four detectors -- which are pure functions of the signal curve -- the
    biexponential fit is governed by the same `aif_lower_limits` / `aif_Robust` /
    `aif_peak_weight_exponent` settings as the production Stage-B fit. Those live in
    `stage_overrides`, so the harness loads them from the real config template; otherwise this
    would measure a configuration nobody actually runs. Values the template does not set now
    come from `python/dce_defaults.json`, the same file a real run reads.

    The template's three required paths are placeholders here: nothing in the fit path touches
    them.
    """
    template = Path(config_template) if config_template is not None else DEFAULT_CONFIG_TEMPLATE
    try:
        return DcePipelineConfig.from_dict(json.loads(template.read_text(encoding="utf-8")))
    except Exception as exc:
        print(
            f"WARNING: could not load config template {template} ({exc}); using built-in aif_* defaults",
            file=sys.stderr,
        )
        return DcePipelineConfig(
            subject_source_path=Path("."), subject_tp_path=Path("."), output_dir=Path(".")
        )


def build_detectors(
    config: DcePipelineConfig, names: Optional[Sequence[str]] = None
) -> Dict[str, Callable[[np.ndarray], Dict[str, Any]]]:
    """Detector registry for `names`, in `DETECTOR_NAMES` order.

    `biexp_fit` needs the fit settings that the other four have no use for, so the registry is
    built per-run rather than being a module constant. Binding it here keeps the config an
    explicit argument: a module-level global with a lazy default would let an importer silently
    run against different settings than the summary header reports.

    `names` defaults to `DEFAULT_DETECTOR_NAMES` -- the production detector and the seed it falls
    back to. The three signal-shape heuristics are opt-in: they are superseded, and running them
    on every session costs time and three extra lines on every figure.
    """
    registry: Dict[str, Callable[[np.ndarray], Dict[str, Any]]] = {
        "piecewise_constant": _piecewise_constant_baseline_end,
        "legacy_sobel": _legacy_sobel_baseline_end,
        "glr": _glr_baseline_end,
        "tv": _tv_baseline_end,
        "biexp_fit": lambda stlv: _biexp_fit_baseline_end(stlv, config),
    }
    selected = resolve_detector_names(names)
    return {name: registry[name] for name in selected}


def resolve_detector_names(names: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """Normalise a detector selection to registry order, rejecting unknown names."""
    if names is None:
        return DEFAULT_DETECTOR_NAMES
    requested = {str(name).strip() for name in names}
    unknown = requested - set(DETECTOR_NAMES)
    if unknown:
        raise ValueError(f"Unknown detector(s): {sorted(unknown)}. Known: {list(DETECTOR_NAMES)}")
    return tuple(name for name in DETECTOR_NAMES if name in requested)


def biexp_fitted_curve(
    details: Dict[str, Any], n_timepoints: int, *, params_key: str = "fit_params"
) -> Optional[np.ndarray]:
    """Re-evaluate a `biexp_fit` fitted curve in the original signal units.

    Returns None unless the timing fit actually ran (`mode == "fit"`); the normalisation constants
    this undoes are only recorded on that branch. The fit works on a baseline-subtracted,
    max-normalised curve in frame units, so this puts the result back on the same axes as the
    measured mean curve. `params_key` selects which fit's six coefficients to evaluate --
    `"fit_params"` for the timing pass, `"production_params"` for the probe below.
    """
    if details.get("mode") != "fit":
        return None
    params = details.get(params_key)
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


def run_production_probe(
    details: Dict[str, Any],
    mean_curve: np.ndarray,
    config: DcePipelineConfig,
    *,
    end_ss_1b: int,
    end_injection_1b: float,
    seed_label: str = "biexp_fit",
) -> None:
    """Re-fit the same curve with the Stage-B (`fit_pass="production"`) pass, recording `production_*`.

    `end_ss_1b` / `end_injection_1b` are the Stage-A answers this fit is conditioned on; the caller
    chooses which detector supplies them (`seed_label` records that choice). They are an *input* to
    the production pass, never fitted, which is what makes "same fit, different detector" a
    measurable comparison.

    The detectors only exercise the *timing* pass, so a run that changes production-pass settings
    (`aif_Robust`, `aif_peak_weight_exponent`) produces byte-identical detector output and there is
    nothing to compare. This probe makes those settings observable: `t_base_end` fixed at the
    timing pass's `end_ss`, `delta` seeded from its fractional `end_injection`, peak prior applied
    -- exactly what Stage B does.

    **It is a proxy, not Stage B.** Production fits the *concentration* curve `CpROI`, which does
    not exist until R1 maps have been built; this fits the same normalised *signal* curve the
    timing pass used. Transition times are invariant to that rescaling, but amplitudes, decay
    rates, and hence the reported R² are not directly comparable to a real Stage-B run.

    Mutates `details` in place. Skipped (with a reason) unless the timing pass reached `mode="fit"`,
    since the normalisation constants it needs are recorded only on that branch.
    """
    if details.get("mode") != "fit":
        details["production_mode"] = "skipped_no_timing_fit"
        return

    scale = float(details.get("normalization_scale", 1.0))
    baseline_mean = float(details.get("baseline_mean", 0.0))
    if not np.isfinite(scale) or scale <= 0.0:
        details["production_mode"] = "skipped_no_enhancement"
        return

    curve = np.asarray(mean_curve, dtype=np.float64).reshape(-1)
    normalized = (curve - baseline_mean) / scale
    n = int(normalized.size)
    end_ss_1b = int(max(1, min(int(end_ss_1b), n)))
    end_injection_1b = float(min(max(float(end_injection_1b), float(end_ss_1b)), float(n)))

    try:
        fit = _fit_aif_biexp(
            config,
            timer=np.arange(n, dtype=np.float64),
            curve=normalized,
            start_injection_min=float(end_ss_1b - 1),
            end_injection_min=float(end_injection_1b - 1),
            fitting_au=False,
            fit_pass="production",
        )
    except Exception as exc:
        details["production_mode"] = "fit_error"
        details["production_error"] = f"{type(exc).__name__}: {exc}"
        return

    details.update(
        {
            "production_mode": "fit" if fit["fit_success"] else "fit_not_converged",
            "production_seed": seed_label,
            "production_seed_end_ss_1b": end_ss_1b,
            "production_seed_end_injection_1b": end_injection_1b,
            "production_rsquare_adj": float(fit["rsquare_adj"]),
            "production_t0_exp_frames": float(fit["t0_exp"]),
            "production_delta_frames": float(fit["delta"]),
            "production_delta_drift": float(fit["delta_drift"]),
            "production_peak_weight": float(fit["peak_weight"]),
            "production_robust_mode": str(fit["robust_mode"]),
            "production_params": [float(v) for v in np.asarray(fit["params"], dtype=np.float64)],
        }
    )


# Registration order also drives figure legend / summary table order.
DETECTOR_NAMES = ("piecewise_constant", "legacy_sobel", "glr", "tv", "biexp_fit")
# Superseded signal-shape heuristics, kept runnable but off by default.
HEURISTIC_DETECTOR_NAMES = ("piecewise_constant", "legacy_sobel", "glr")
# The production detector plus the seed it is built on and falls back to. `tv` is not optional
# here: `biexp_fit`'s accuracy row is only readable next to the row it degrades to.
DEFAULT_DETECTOR_NAMES = tuple(n for n in DETECTOR_NAMES if n not in HEURISTIC_DETECTOR_NAMES)

_NON_DYNAMIC_TOKENS = ("mask", "t1map", "seg", "roi", "dceref")
# Preferred over any other candidate in the no-pattern heuristic below: motion correction is
# almost always what you want the detectors judged against when a derivatives tree has several
# desc-* variants of the same series side by side.
_MOTION_CORRECTED_TOKEN = "desc-hmc"
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


def _bids_entities(
    path: Path, subject_filter: Optional[set]
) -> Optional[Tuple[str, Optional[str]]]:
    """Pull `sub-*`/`ses-*` out of a path, or None when it should be skipped.

    Shared by both discovery routes so the two cannot drift on what counts as a subject.
    """
    subject = next((part for part in path.parts if _SUBJECT_RE.match(part)), None)
    if subject is None:
        print(f"WARNING: could not determine BIDS subject from path {path}", file=sys.stderr)
        return None
    if subject_filter is not None and subject not in subject_filter:
        return None
    return subject, next((part for part in path.parts if _SESSION_RE.match(part)), None)


def _subject_filter(subjects: Optional[Iterable[str]]) -> Optional[set]:
    return {str(s).strip() for s in subjects} if subjects else None


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
    subject_filter = _subject_filter(subjects)
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

        entity = _bids_entities(json_path, subject_filter)
        if entity is None:
            continue
        subject, session = entity

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
    subject_filter = _subject_filter(subjects)
    records: List[AifSidecarRecord] = []
    for mask_path in sorted(root.rglob(pattern)):
        if not mask_path.is_file():
            continue
        entity = _bids_entities(mask_path, subject_filter)
        if entity is None:
            continue
        subject, session = entity

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
    (`desc-bfc`, `desc-hmc`, `desc-biases`, ...). Since the detectors are meant to be judged on
    the curve production actually feeds them, pass the pipeline's own pattern
    (`*desc-bfcz_DCE.nii*`) when reading derivatives for real. Absent a pattern, motion-corrected
    (`desc-hmc`) files are preferred when present -- otherwise the tie-break falls through to a
    single "dce"-named file, then alphabetical, both arbitrary.
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
        hmc_named = [c for c in candidates if _MOTION_CORRECTED_TOKEN in c.name.lower()]
        if len(hmc_named) == 1:
            return hmc_named[0], "bids", None
        if len(hmc_named) > 1:
            chosen = sorted(hmc_named)[0]
            return (
                chosen,
                "bids",
                f"{len(hmc_named)} motion-corrected ({_MOTION_CORRECTED_TOKEN}) candidates in "
                f"{dce_dir}, chose {chosen.name}",
            )

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
    detectors: Dict[str, Callable[[np.ndarray], Dict[str, Any]]],
    *,
    use_all_voxels: bool = False,
    dynamic_pattern: Optional[str] = None,
    production_probe_config: Optional[DcePipelineConfig] = None,
    production_probe_seed: str = "biexp_fit",
) -> SessionResult:
    """Load one session's data, extract the signal curve, and run all detectors.

    `detectors` is the registry from `build_detectors`, passed in rather than read from module
    state so a caller cannot silently run against different settings than it reports.

    `production_probe_config` enables `run_production_probe` on the `biexp_fit` details; pass the
    same config the detectors were built from. `production_probe_seed` names the detector whose
    `end_ss` the production fit is conditioned on -- the point of the probe is to hold the fit
    fixed and vary the Stage-A answer, so it need not be `biexp_fit`.

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
    for name, detector in detectors.items():
        try:
            details = detector(stlv)
            detector_details[name] = details
            predictions[name] = int(details["end_ss_1b"])
        except Exception as exc:
            print(f"WARNING: {record.id}: detector '{name}' failed: {exc}", file=sys.stderr)
            predictions[name] = None

    if production_probe_config is not None and "biexp_fit" in detector_details:
        biexp = detector_details["biexp_fit"]
        seed_details = detector_details.get(production_probe_seed, biexp)
        seed_end_ss = int(seed_details.get("end_ss_1b", 1))
        seed_end_injection = seed_details.get("end_injection_1b")
        if seed_end_injection is None:
            # `tv` reports no injection end, so Stage A falls back to the mean per-voxel peak
            # frame -- dce_pipeline.py's `mean(peak_indices_1b)`, mirroring find_end_ss_tv.m.
            # Reproduce that here rather than substituting the mean curve's argmax, which is a
            # different statistic and would not be the number production feeds Stage B.
            seed_end_injection = float(np.mean(np.argmax(stlv, axis=0) + 1))
        run_production_probe(
            biexp,
            mean_curve,
            production_probe_config,
            end_ss_1b=seed_end_ss,
            end_injection_1b=float(seed_end_injection),
            seed_label=production_probe_seed,
        )

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
    results: Iterable[SessionResult],
    *,
    names: Optional[Sequence[str]] = None,
    reference: str = "biexp_fit",
) -> Dict[str, AgreementStats]:
    """Pairwise agreement against `reference`, for datasets with no human ratings.

    Without ground truth there is nothing to score accuracy on, but the detectors can still be
    compared to each other: a detector that tracks the production default everywhere tells a
    very different story from one that is systematically a frame early. Offsets are signed
    (`detector - reference`) so a consistent bias is distinguishable from scatter.
    """
    results = list(results)
    stats: Dict[str, AgreementStats] = {}
    for name in resolve_detector_names(names):
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
class RsquareStats:
    """Distribution of one fit pass's adjusted R² across sessions."""

    label: str
    n: int
    mean: float
    median: float
    minimum: float
    p10: float
    maximum: float
    n_below_threshold: int
    threshold: float
    # (session id, R²) for the worst sessions, ascending. Which sessions fit badly is the
    # actionable part; the aggregate only says whether to go looking.
    worst: List[Tuple[str, float]] = field(default_factory=list)


def compute_rsquare_stats(
    results: Iterable[SessionResult],
    *,
    detail_key: str,
    label: str,
    threshold: float = 0.90,
    n_worst: int = 10,
) -> RsquareStats:
    """Summarise `detail_key` from every ok session's `biexp_fit` details.

    Sessions where the fit did not run contribute nothing, so `n` is also the count of sessions
    that actually produced a fit -- read it alongside the outcome breakdown rather than assuming
    it equals the number of ok sessions.
    """
    pairs: List[Tuple[str, float]] = []
    for result in results:
        if result.status != "ok":
            continue
        value = result.detector_details.get("biexp_fit", {}).get(detail_key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            pairs.append((result.id, float(value)))

    values = np.array([v for _, v in pairs], dtype=np.float64)
    nan = float("nan")
    return RsquareStats(
        label=label,
        n=int(values.size),
        mean=float(np.mean(values)) if values.size else nan,
        median=float(np.median(values)) if values.size else nan,
        minimum=float(np.min(values)) if values.size else nan,
        p10=float(np.percentile(values, 10)) if values.size else nan,
        maximum=float(np.max(values)) if values.size else nan,
        n_below_threshold=int(np.count_nonzero(values < threshold)),
        threshold=float(threshold),
        worst=sorted(pairs, key=lambda kv: kv[1])[:n_worst],
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
    results: Iterable[SessionResult],
    *,
    names: Optional[Sequence[str]] = None,
    tolerance_frames: Optional[int] = None,
) -> Dict[str, AlgorithmStats]:
    """Per-algorithm exact-match accuracy % and MSE (frame-index units) over valid sessions."""
    results = list(results)
    stats: Dict[str, AlgorithmStats] = {}
    for name in resolve_detector_names(names):
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
