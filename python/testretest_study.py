"""Test-retest validation study: collect ROI medians, keep the analysis blinded, report.

The statistics live in repeatability.py. This module is the study around them:

- collect one tidy table of ROI medians from the fits of every pipeline arm
- audit acquisition metadata between the two sessions of each pair
- apply the pre-registered exclusion rules and log each exclusion
- run the repeatability analysis in one of three modes:
    synthetic   generated pairs with known wSD (checks the code)
    shuffled    real values with the session pairing and the pipeline labels scrambled
                (checks the code on real distributions; the numbers mean nothing)
    unblinded   the real analysis; refused until every decision in the pre-registration
                file is filled in and signed off

Plan: project-management/projects/test-retest-validation/test_retest_validation.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import repeatability as rp

TABLE_COLUMNS = (
    "subject", "session", "cohort", "pipeline", "rater", "model", "region", "parameter",
    "value", "n_voxels", "n_finite", "qof_reliable_fraction",
)
SESSION_QC_COLUMNS = (
    "subject", "session", "cohort", "pipeline", "rater", "max_displacement_mm", "aif_fit_r2_adj",
)
GROUP_KEYS = ("cohort", "pipeline", "rater", "model", "region", "parameter")
COHORT_PREFIXES = {"sub-RIDERNeuroMRI": "RIDER", "sub-QINGBMTR": "QIN-GBM"}
POOLED = "pooled"
MODES = ("synthetic", "shuffled", "unblinded")

DEFAULT_REGIONS = {
    "WM": "anat/*_space-DCEref_label-WM_mask.nii*",
    "GM": "anat/*_space-DCEref_label-GM_mask.nii*",
}
DEFAULT_FIT_ROI = "anat/*_space-DCEref_label-brain_mask.nii*"
MODEL_PARAMETERS = {"patlak": ("Ktrans", "vp"), "ex_tofts": ("Ktrans", "ve", "vp")}

# Stand-ins for decisions the pre-registration has not made yet. Used only in the synthetic
# and shuffled modes, and named in the report whenever they are.
PLACEHOLDERS = {
    "noninferiority.margin": 1.1,
    "noninferiority.confidence": 0.95,
    "noninferiority.interval": "bootstrap_percentile",
    "pooling": "separate",
}
REQUIRED_DECISIONS = (
    "primary_endpoint.model", "primary_endpoint.region", "primary_endpoint.parameter",
    "primary_endpoint.rater", "pooling", "noninferiority.margin", "noninferiority.confidence",
    "noninferiority.interval", "exclusions.max_displacement_mm",
    "exclusions.min_aif_fit_r2_adj", "exclusions.min_qof_reliable_fraction",
    "split.development_fraction", "split.seed", "signed_off.by", "signed_off.date",
)
NOT_APPLIED = "not_applied"


class BlindingError(RuntimeError):
    """The real analysis was requested before the pre-registration was complete."""


# --------------------------------------------------------------------------- pre-registration


def load_prereg(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lookup(prereg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = prereg
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, Mapping) else None
    return node


def missing_decisions(prereg: Mapping[str, Any]) -> List[str]:
    """Pre-registration entries still undecided (null). 'not_applied' counts as decided."""
    return [key for key in REQUIRED_DECISIONS if _lookup(prereg, key) is None]


def require_unblinded(prereg: Mapping[str, Any]) -> None:
    missing = missing_decisions(prereg)
    if missing:
        raise BlindingError(
            "The real analysis stays blinded until the pre-registration is complete and "
            "signed off. Still undecided: " + ", ".join(missing)
        )


def decision(prereg: Mapping[str, Any], key: str, mode: str, used: List[str]) -> Any:
    """A pre-registered value, or its placeholder in the blinded modes (recorded in `used`)."""
    value = _lookup(prereg, key)
    if value is not None:
        return value
    if mode == "unblinded":
        raise BlindingError(f"pre-registration entry {key} is undecided")
    if key in PLACEHOLDERS:
        used.append(f"{key} = {PLACEHOLDERS[key]}")
        return PLACEHOLDERS[key]
    return None


# --------------------------------------------------------------------------- tables


def write_table(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_table(path: Path, numeric: Sequence[str] = ()) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows: List[Dict[str, Any]] = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            if key in row:
                row[key] = float(row[key]) if row[key] not in ("", None) else float("nan")
    return rows


def read_roi_table(path: Path) -> List[Dict[str, Any]]:
    return read_table(path, numeric=("value", "n_voxels", "n_finite", "qof_reliable_fraction"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort_of(subject: str) -> str:
    matches = [prefix for prefix in COHORT_PREFIXES if subject.startswith(prefix)]
    return COHORT_PREFIXES[max(matches, key=len)] if matches else "unknown"


# --------------------------------------------------------------------------- collection


def _one(folder: Path, pattern: str) -> Optional[Path]:
    matches = sorted(folder.glob(pattern))
    return matches[0] if len(matches) == 1 else None


def _one_in(folders: Sequence[Path], pattern: str) -> Optional[Path]:
    """The single match in the first folder that has one."""
    for folder in folders:
        path = _one(folder, pattern)
        if path is not None:
            return path
    return None


def _load(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).get_fdata())


def _max_displacement_mm(session_dir: Path) -> float:
    path = _one(session_dir, "dce/*_desc-hmc_maxdisp.txt")
    if path is None:
        return float("nan")
    match = re.search(r"([0-9.]+(?:[eE][-+]?[0-9]+)?)\s*mm", path.read_text(encoding="utf-8", errors="replace"))
    return float(match.group(1)) if match else float("nan")


def _aif_fit_r2_adj(fit_dir: Path) -> float:
    """Stage B's adjusted R-squared of the AIF fit; NaN when the run did not fit the AIF."""
    for name in ("dce_pipeline_run.json", "reports/dce_pipeline_run.json"):
        path = fit_dir / name
        if path.exists():
            stage_b = json.loads(path.read_text(encoding="utf-8")).get("stages", {}).get("B", {})
            value = stage_b.get("fit_rsquared_cp_adj")
            return float(value) if value is not None else float("nan")
    return float("nan")


def collect_session(
    fit_dir: Path,
    dceprep_session: Path,
    *,
    pipeline: str,
    rater: str,
    regions: Mapping[str, Any] = DEFAULT_REGIONS,
    fit_roi: str = DEFAULT_FIT_ROI,
    model_parameters: Mapping[str, Sequence[str]] = MODEL_PARAMETERS,
    region_roots: Sequence[Path] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """ROI medians of one fitted session.

    Parameter maps are read from ``fit_dir`` (``*_<model>_fit_<parameter>.nii*``, the name
    both pipelines write) and region masks from the DCEPrep session, both in DCE space. A
    region is the mask intersected with the fit ROI; its value is the median of the finite
    voxels. Returns (table rows, session QC row, problems).

    A region is a glob, or ``{"mask": glob, "minus": glob}`` for a mask with another
    removed (white matter outside the tumour ROI). Masks are looked for in the DCEPrep
    session first, then under each of ``region_roots`` (``<root>/sub-*/ses-*``), so a
    region drawn by one rater can be applied to every rater's fits. A region whose mask is
    missing for a session is left out for that session and reported.
    """
    subject, session = dceprep_session.parent.name, dceprep_session.name
    base = {"subject": subject, "session": session, "cohort": cohort_of(subject), "pipeline": pipeline, "rater": rater}
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []

    roi_path = _one(dceprep_session, fit_roi)
    roi = _load(roi_path) > 0 if roi_path else None
    if roi is None:
        problems.append(f"{subject} {session}: fit ROI {fit_roi} not found or ambiguous")
    folders = [dceprep_session, *(Path(root) / subject / session for root in region_roots)]
    masks: Dict[str, np.ndarray] = {}
    for region, spec in regions.items():
        patterns = [spec] if isinstance(spec, str) else [spec["mask"], spec["minus"]]
        paths = [_one_in(folders, pattern) for pattern in patterns]
        if any(path is None for path in paths):
            missing = patterns[[path is None for path in paths].index(True)]
            problems.append(f"{subject} {session}: {region} mask {missing} not found or ambiguous")
            continue
        loaded = [_load(path) > 0 for path in paths]
        if roi is None or any(mask.shape != roi.shape for mask in loaded):
            problems.append(f"{subject} {session}: {region} mask grid differs from the fit ROI")
            continue
        mask = loaded[0] & roi
        if len(loaded) == 2:
            mask &= ~loaded[1]
        masks[region] = mask

    for model, parameters in model_parameters.items():
        reliable_path = _one(fit_dir, f"*_{model}_fit_qof_reliable.nii*")
        reliable = _load(reliable_path) > 0 if reliable_path else None
        for parameter in parameters:
            map_path = _one(fit_dir, f"*_{model}_fit_{parameter}.nii*")
            if map_path is None:
                problems.append(f"{subject} {session}: no {model} {parameter} map in {fit_dir}")
                continue
            values = _load(map_path)
            for region, mask in masks.items():
                if values.shape != mask.shape:
                    problems.append(f"{subject} {session}: {model} {parameter} map grid differs from the {region} mask")
                    continue
                inside = values[mask]
                finite = inside[np.isfinite(inside)]
                fraction = float("nan")
                if reliable is not None and reliable.shape == mask.shape and mask.any():
                    fraction = float(reliable[mask].mean())
                rows.append({
                    **base, "model": model, "region": region, "parameter": parameter,
                    "value": float(np.median(finite)) if finite.size else float("nan"),
                    "n_voxels": int(mask.sum()), "n_finite": int(finite.size),
                    "qof_reliable_fraction": fraction,
                })
    qc = {**base, "max_displacement_mm": _max_displacement_mm(dceprep_session), "aif_fit_r2_adj": _aif_fit_r2_adj(fit_dir)}
    return rows, qc, problems


def collect_arm(fit_root: Path, dceprep_run: Path, *, pipeline: str, rater: str, **kwargs: Any):
    """Collect every fitted session under ``fit_root`` (sub-*/ses-*)."""
    rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for fit_dir in sorted(p for p in Path(fit_root).glob("sub-*/ses-*") if p.is_dir()):
        dceprep_session = Path(dceprep_run) / fit_dir.parent.name / fit_dir.name
        if not dceprep_session.is_dir():
            problems.append(f"{fit_dir.parent.name} {fit_dir.name}: no DCEPrep session at {dceprep_session}")
            continue
        session_rows, qc, session_problems = collect_session(
            fit_dir, dceprep_session, pipeline=pipeline, rater=rater, **kwargs
        )
        rows.extend(session_rows)
        qc_rows.append(qc)
        problems.extend(session_problems)
    return rows, qc_rows, problems


# --------------------------------------------------------------------------- metadata audit

AUDIT_FIELDS = ("tr_ms", "fa_deg", "time_resolution_min", "relaxivity", "hematocrit")


def _session_metadata(fit_dir: Path, staged_session: Optional[Path]) -> Dict[str, Any]:
    run = json.loads((fit_dir / "dce_pipeline_run.json").read_text(encoding="utf-8"))
    stage_a = run.get("stages", {}).get("A", {})
    record: Dict[str, Any] = {field: stage_a.get(field) for field in AUDIT_FIELDS}
    # keep the sidecar field each value came from, not the session-specific path
    sources = stage_a.get("metadata_sources", {}) or {}
    record["metadata_fields"] = {k: str(v).rsplit(".", 1)[-1] for k, v in sorted(sources.items())}
    record["baseline_method"] = (stage_a.get("steady_state_auto", {}) or {}).get("method_used")
    record["n_frames"] = (run.get("stages", {}).get("B", {}) or {}).get("end_time_index")
    if staged_session is not None:
        flips = []
        for sidecar in sorted((staged_session / "anat").glob("*_VFA.json")):
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            flips.append((payload.get("FlipAngle"), payload.get("RepetitionTime")))
        record["vfa_flip_angles"] = [f for f, _ in flips]
        record["vfa_repetition_times"] = sorted({t for _, t in flips})
    return record


def metadata_audit(fit_root: Path, staged_root: Optional[Path] = None, rel_tol: float = 1e-3) -> List[Dict[str, Any]]:
    """Compare acquisition metadata between the sessions of each subject.

    One row per subject: ``flags`` lists every field that differs between its sessions --
    a different sidecar field supplying TR/FA/frame spacing, different values, a different
    baseline method, or a different VFA flip-angle set. Empty flags = matched pair.
    """
    by_subject: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fit_dir in sorted(p for p in Path(fit_root).glob("sub-*/ses-*") if (p / "dce_pipeline_run.json").exists()):
        staged = Path(staged_root) / fit_dir.parent.name / fit_dir.name if staged_root else None
        by_subject.setdefault(fit_dir.parent.name, {})[fit_dir.name] = _session_metadata(fit_dir, staged)

    rows: List[Dict[str, Any]] = []
    for subject, sessions in sorted(by_subject.items()):
        flags: List[str] = []
        names = sorted(sessions)
        if len(names) != 2:
            flags.append(f"{len(names)} fitted session(s)")
        else:
            first, second = sessions[names[0]], sessions[names[1]]
            for key in sorted(set(first) | set(second)):
                a, b = first.get(key), second.get(key)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    same = math.isclose(a, b, rel_tol=rel_tol)
                else:
                    same = a == b
                if not same:
                    flags.append(f"{key}: {json.dumps(a)} vs {json.dumps(b)}")
        rows.append({"subject": subject, "cohort": cohort_of(subject), "sessions": " ".join(names), "flags": "; ".join(flags)})
    return rows


# --------------------------------------------------------------------------- blinding


SHUFFLE_KEYS = ("cohort", "rater", "model", "region", "parameter")


def shuffle_labels(rows: Sequence[Mapping[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Scramble a real table so no repeatability result survives.

    Within each cohort x rater x model x region x parameter, every paired subject's second
    session is swapped for another paired subject's (no subject keeps its own), which
    turns within-subject differences into between-subject ones. The swap is the same for
    every pipeline, so the pipelines still see the same scans. Then each subject's pipeline
    labels are swapped with probability one half, which centres any pipeline contrast on
    no difference. Values and their marginal distributions are untouched, and every group
    keeps its number of pairs, so the analysis code sees a table shaped like the real one.

    A subject with one session in a group has no pair to scramble and is left as it is.
    Where a group has a single paired subject its pair cannot be scrambled, and that
    subject's rows are left out of the group rather than passed through.
    """
    rng = np.random.default_rng(seed)
    by_key: Dict[Tuple[Any, ...], Dict[str, List[str]]] = {}
    for row in rows:
        sessions = by_key.setdefault(tuple(row[k] for k in SHUFFLE_KEYS), {}).setdefault(str(row["subject"]), [])
        if row["session"] not in sessions:
            sessions.append(str(row["session"]))

    recipient_of: Dict[Tuple[Any, ...], Dict[str, str]] = {}
    unscrambled: Dict[Tuple[Any, ...], str] = {}
    for key in sorted(by_key, key=str):
        paired = sorted(subject for subject, sessions in by_key[key].items() if len(sessions) > 1)
        if len(paired) == 1:
            unscrambled[key] = paired[0]
            continue
        order = [paired[i] for i in rng.permutation(len(paired))]
        # a cycle: each subject's second session goes to the next, so nobody keeps their own
        recipient_of[key] = {subject: order[(i + 1) % len(order)] for i, subject in enumerate(order)}
    subjects = sorted({str(row["subject"]) for row in rows})
    pipelines = sorted({str(row["pipeline"]) for row in rows})
    swap = {subject: bool(rng.integers(0, 2)) for subject in subjects} if len(pipelines) == 2 else {}

    shuffled: List[Dict[str, Any]] = []
    for row in rows:
        key = tuple(row[k] for k in SHUFFLE_KEYS)
        subject = str(row["subject"])
        if unscrambled.get(key) == subject:
            continue
        new = dict(row)
        if subject in recipient_of.get(key, {}) and row["session"] == sorted(by_key[key][subject])[-1]:
            recipient = recipient_of[key][subject]
            new["subject"] = recipient
            new["session"] = sorted(by_key[key][recipient])[-1]
        if swap.get(str(new["subject"])):
            new["pipeline"] = pipelines[1 - pipelines.index(str(row["pipeline"]))]
        shuffled.append(new)
    return shuffled


def synthetic_table(
    n_subjects: Mapping[str, int],
    wsd_by_pipeline: Mapping[str, float],
    *,
    shared_fraction: float = 0.8,
    between_sd: float = 1.0,
    mean: float = 5.0,
    seed: int = 0,
    rater: str = "SYN",
    model: str = "patlak",
    region: str = "WM",
    parameter: str = "Ktrans",
) -> List[Dict[str, Any]]:
    """Generated ROI medians with a known wSD per pipeline, on the same simulated scans.

    ``shared_fraction`` of the smallest pipeline's rescan variance is common to all
    pipelines (biology and acquisition); the rest is each pipeline's own noise.
    """
    rng = np.random.default_rng(seed)
    shared_variance = shared_fraction * min(wsd_by_pipeline.values()) ** 2
    rows: List[Dict[str, Any]] = []
    for cohort, count in n_subjects.items():
        for i in range(count):
            subject = f"sub-{cohort}{i:03d}"
            truth = mean + between_sd * rng.standard_normal()
            common = math.sqrt(shared_variance) * rng.standard_normal(2)
            for pipeline, wsd in wsd_by_pipeline.items():
                own = math.sqrt(wsd**2 - shared_variance) * rng.standard_normal(2)
                for k, session in enumerate(("ses-01", "ses-02")):
                    rows.append({
                        "subject": subject, "session": session, "cohort": cohort, "pipeline": pipeline,
                        "rater": rater, "model": model, "region": region, "parameter": parameter,
                        "value": truth + common[k] + own[k], "n_voxels": 1000, "n_finite": 1000,
                        "qof_reliable_fraction": 1.0,
                    })
    return rows


# --------------------------------------------------------------------------- exclusions


def apply_exclusions(
    rows: Sequence[Mapping[str, Any]],
    session_qc: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    manual: Sequence[Mapping[str, Any]] = (),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drop excluded pairs and return (kept rows, exclusion log).

    Rules look only at QC measures, never at the outcome. A session that fails removes its
    subject's pair for that rater in every pipeline, so the pipelines are always compared
    on the same subjects. The QoF rule works per model and region. A threshold of None or
    'not_applied' switches its rule off; a QC value that is missing cannot fail a rule.
    """

    def threshold(name: str) -> Optional[float]:
        value = thresholds.get(name)
        return None if value in (None, NOT_APPLIED) else float(value)

    log: List[Dict[str, Any]] = []
    pair_out: Dict[Tuple[str, str], None] = {}  # (subject, rater)
    group_out: Dict[Tuple[str, str, str, str], None] = {}  # (subject, rater, model, region)

    def exclude_pair(subject: str, session: str, rater: str, rule: str, detail: str) -> None:
        pair_out[(subject, rater)] = None
        log.append({"subject": subject, "session": session, "rater": rater, "model": "", "region": "", "rule": rule, "detail": detail})

    max_disp, min_r2, min_qof = threshold("max_displacement_mm"), threshold("min_aif_fit_r2_adj"), threshold("min_qof_reliable_fraction")
    seen_motion = set()
    for qc in session_qc:
        subject, session, rater = str(qc["subject"]), str(qc["session"]), str(qc["rater"])
        displacement = float(qc.get("max_displacement_mm", float("nan")))
        if max_disp is not None and displacement > max_disp and (subject, session, rater) not in seen_motion:
            seen_motion.add((subject, session, rater))
            exclude_pair(subject, session, rater, "motion", f"max displacement {displacement:.2f} mm > {max_disp:g}")
        r2 = float(qc.get("aif_fit_r2_adj", float("nan")))
        if min_r2 is not None and r2 < min_r2:
            exclude_pair(subject, session, rater, "aif_fit", f"{qc['pipeline']} AIF fit adjusted R2 {r2:.3f} < {min_r2:g}")
    for entry in manual:
        raters = sorted({str(r["rater"]) for r in rows}) if entry.get("rater", "*") in ("*", "") else [str(entry["rater"])]
        for rater in raters:
            exclude_pair(str(entry["subject"]), str(entry.get("session", "")), rater, "manual", str(entry.get("reason", "")))
    if min_qof is not None:
        for row in rows:
            fraction = float(row.get("qof_reliable_fraction", float("nan")))
            key = (str(row["subject"]), str(row["rater"]), str(row["model"]), str(row["region"]))
            if fraction < min_qof and key not in group_out:
                group_out[key] = None
                log.append({
                    "subject": key[0], "session": row["session"], "rater": key[1], "model": key[2], "region": key[3],
                    "rule": "qof", "detail": f"{row['pipeline']} reliable fraction {fraction:.3f} < {min_qof:g}",
                })
    kept = [
        dict(row) for row in rows
        if (str(row["subject"]), str(row["rater"])) not in pair_out
        and (str(row["subject"]), str(row["rater"]), str(row["model"]), str(row["region"])) not in group_out
    ]
    return kept, log


# --------------------------------------------------------------------------- analysis


def _with_pooled(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Add a copy of every row under the 'pooled' cohort, remembering its real cohort."""
    out = [{**row, "stratum": row["cohort"]} for row in rows]
    if len({row["cohort"] for row in rows}) > 1:
        out += [{**row, "stratum": row["cohort"], "cohort": POOLED} for row in rows]
    return out


def analyse(
    rows: Sequence[Mapping[str, Any]],
    *,
    margin: float,
    confidence: float,
    interval: str,
    report_wcv_for: Sequence[Mapping[str, str]] = (),
    primary_endpoint: Optional[Mapping[str, Any]] = None,
    reference_pipeline: str = "matlab",
    new_pipeline: str = "python",
    n_boot: int = 5000,
    seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Repeatability per group, and the non-inferiority test wherever both pipelines exist.

    Returns {'metrics': [...], 'noninferiority': [...], 'dropped': [...], 'pairs': [...]}.
    Cohorts are analysed separately and pooled; pooled bootstrap intervals resample within
    cohort.
    """
    if interval not in ("bootstrap_percentile", "pitman"):
        raise ValueError(f"unknown interval '{interval}'")
    rows = _with_pooled(rows)
    stratum_of = {(row["cohort"], row["subject"]): row["stratum"] for row in rows}
    pairs, dropped = rp.pair_sessions(rows, GROUP_KEYS)
    wcv_wanted = {(e["region"], e["parameter"]) for e in report_wcv_for}
    # the primary comparison is marked only once the whole endpoint is pre-registered
    primary = {k: (primary_endpoint or {}).get(k) for k in ("model", "region", "parameter", "rater")}

    metrics: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    for group in sorted(pairs):
        labels = dict(zip(GROUP_KEYS, group))
        subjects = sorted(pairs[group])
        a = np.array([pairs[group][s][0] for s in subjects])
        b = np.array([pairs[group][s][1] for s in subjects])
        pair_rows += [{**labels, "subject": s, "first": x, "second": y} for s, x, y in zip(subjects, a, b)]
        if len(subjects) < 3:
            dropped.append({**labels, "subject": "*", "reason": f"only {len(subjects)} complete pair(s)"})
            continue
        strata = [stratum_of[(labels["cohort"], s)] for s in subjects] if labels["cohort"] == POOLED else None
        row: Dict[str, Any] = {**labels, **rp.paired_metrics(a, b)}
        if (labels["region"], labels["parameter"]) not in wcv_wanted:
            row["wCV_log"] = float("nan")  # not reported for near-zero parameters (plan, Metrics)
        for name, statistic in (("wSD", rp.within_subject_sd), ("ICC_2_1", rp.icc_2_1)):
            low, high = rp.bootstrap_ci(statistic, a, b, n_boot=n_boot, confidence=confidence, seed=seed, strata=strata)
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = low, high
        row["RC_ci_low"] = rp.REPEATABILITY_COEFFICIENT_FACTOR * row["wSD_ci_low"]
        row["RC_ci_high"] = rp.REPEATABILITY_COEFFICIENT_FACTOR * row["wSD_ci_high"]
        metrics.append(row)

    noninferiority: List[Dict[str, Any]] = []
    pipeline_index = GROUP_KEYS.index("pipeline")
    for group in sorted(pairs):
        if group[pipeline_index] != new_pipeline:
            continue
        reference_group = group[:pipeline_index] + (reference_pipeline,) + group[pipeline_index + 1:]
        if reference_group not in pairs:
            continue
        labels = {k: v for k, v in zip(GROUP_KEYS, group) if k != "pipeline"}
        subjects = sorted(set(pairs[group]) & set(pairs[reference_group]))
        if len(subjects) < 4:
            continue
        new = np.array([pairs[group][s] for s in subjects])
        ref = np.array([pairs[reference_group][s] for s in subjects])
        strata = [stratum_of[(labels["cohort"], s)] for s in subjects] if labels["cohort"] == POOLED else None
        result = rp.wsd_ratio_noninferiority(
            new[:, 0], new[:, 1], ref[:, 0], ref[:, 1],
            margin=margin, confidence=confidence, n_boot=n_boot, seed=seed, strata=strata,
        )
        upper = result.ci_high if interval == "bootstrap_percentile" else result.pitman_ci_high
        noninferiority.append({
            **labels, "n": result.n, "wSD_ratio": result.ratio,
            "bootstrap_ci_low": result.ci_low, "bootstrap_ci_high": result.ci_high,
            "pitman_ratio": result.pitman_ratio, "pitman_ci_low": result.pitman_ci_low,
            "pitman_ci_high": result.pitman_ci_high, "confidence": confidence, "margin": margin,
            "decision_interval": interval, "non_inferior": bool(np.isfinite(upper) and upper < margin),
            "primary": bool(primary) and all(v is not None and labels.get(k) == v for k, v in primary.items()),
        })
    return {"metrics": metrics, "noninferiority": noninferiority, "dropped": dropped, "pairs": pair_rows}


# --------------------------------------------------------------------------- report

METRIC_COLUMNS = (
    *GROUP_KEYS, "n", "mean", "wSD", "wSD_ci_low", "wSD_ci_high", "RC", "RC_ci_low", "RC_ci_high",
    "wCV_log", "ICC_2_1", "ICC_2_1_ci_low", "ICC_2_1_ci_high", "BA_bias", "BA_loa_low", "BA_loa_high",
    "BA_spread_vs_mean_rho", "BA_spread_vs_mean_p", "BA_trend_slope", "BA_trend_p",
)
NONINFERIORITY_COLUMNS = (
    "cohort", "rater", "model", "region", "parameter", "n", "wSD_ratio", "bootstrap_ci_low",
    "bootstrap_ci_high", "pitman_ratio", "pitman_ci_low", "pitman_ci_high", "confidence", "margin",
    "decision_interval", "non_inferior", "primary",
)
BANNERS = {
    "synthetic": "SYNTHETIC DATA -- generated pairs, not measurements.",
    "shuffled": "LABEL-SHUFFLED DATA -- session pairing and pipeline labels are scrambled. These numbers are NOT results.",
    "unblinded": "UNBLINDED ANALYSIS -- real pairs, pre-registration complete.",
}


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_format(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def plot_pairs(pair_rows: Sequence[Mapping[str, Any]], out_dir: Path, mode: str) -> List[Path]:
    """One figure per group: Bland-Altman next to the per-subject paired values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_group: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in pair_rows:
        by_group.setdefault(tuple(row[k] for k in GROUP_KEYS), []).append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for group, members in sorted(by_group.items()):
        if len(members) < 3:
            continue
        a = np.array([m["first"] for m in members])
        b = np.array([m["second"] for m in members])
        ba = rp.bland_altman(a, b)
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
        left.scatter((a + b) / 2, b - a, s=18)
        for level, style in ((ba.bias, "-"), (ba.loa_low, "--"), (ba.loa_high, "--")):
            left.axhline(level, color="0.3", linestyle=style, linewidth=1)
        left.set_xlabel("mean of the two sessions")
        left.set_ylabel("second - first")
        left.set_title("Bland-Altman")
        right.plot([0, 1], np.vstack([a, b]), color="0.5", linewidth=0.8, marker="o", markersize=3)
        right.set_xticks([0, 1], ["first", "second"])
        right.set_title("paired values")
        title = " / ".join(str(g) for g in group)
        fig.suptitle(title if mode == "unblinded" else f"{title}   [{mode.upper()} -- not a result]")
        fig.tight_layout()
        path = out_dir / ("pairs_" + "_".join(str(g) for g in group).replace("/", "-") + ".png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def write_report(
    out_dir: Path,
    results: Mapping[str, List[Dict[str, Any]]],
    *,
    mode: str,
    exclusions: Sequence[Mapping[str, Any]] = (),
    notes: Sequence[str] = (),
    plots: bool = True,
) -> Path:
    """Write the CSV tables, the figures and report.md into ``out_dir``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(out_dir / "metrics.csv", results["metrics"], METRIC_COLUMNS)
    write_table(out_dir / "noninferiority.csv", results["noninferiority"], NONINFERIORITY_COLUMNS)
    write_table(out_dir / "dropped.csv", results["dropped"], (*GROUP_KEYS, "subject", "reason"))
    write_table(out_dir / "exclusions.csv", exclusions, ("subject", "session", "rater", "model", "region", "rule", "detail"))
    figures = plot_pairs(results["pairs"], out_dir / "figures", mode) if plots else []

    short = (*GROUP_KEYS, "n", "wSD", "wSD_ci_low", "wSD_ci_high", "RC", "wCV_log", "ICC_2_1", "BA_bias", "BA_spread_vs_mean_p")
    text = [
        "# Test-retest repeatability report", "", f"**{BANNERS[mode]}**", "",
        *[f"- {note}" for note in notes], "",
        "## Non-inferiority: wSD(python) / wSD(matlab)", "",
        _markdown_table(results["noninferiority"], NONINFERIORITY_COLUMNS),
        "## Repeatability by group", "", _markdown_table(results["metrics"], short),
        f"## Exclusions ({len(exclusions)})", "",
        _markdown_table(exclusions, ("subject", "session", "rater", "model", "region", "rule", "detail")),
        f"## Pairs left out as incomplete ({len(results['dropped'])})", "",
        _markdown_table(results["dropped"], (*GROUP_KEYS, "subject", "reason")),
        f"Figures: {len(figures)} in figures/. Full tables: metrics.csv, noninferiority.csv.", "",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path
