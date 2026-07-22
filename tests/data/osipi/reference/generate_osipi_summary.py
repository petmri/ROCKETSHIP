"""Generate the OSIPI accuracy summary (markdown + figures).

What this produces
------------------
- ``docs/project-management/projects/osipi-verification/osipi_summary.md`` -- a plain
  markdown report: data provenance, a per-backend accuracy table (ROCKETSHIP error vs
  the OSIPI gate and the peer spread), and per-case ground-truth-vs-fit tables.
- ``tests/data/osipi/reference/figures/*.png`` -- comparison figures.

Fitting backends
----------------
ROCKETSHIP has four fitting routines: MATLAB, python (pure-CPU scipy), cpufit (pyCpufit)
and gpufit (pyGpufit). This report verifies the three non-MATLAB backends against OSIPI,
where each is available on the machine that runs it:

- **python** -- always run; the DCE reference functions the reliability tests gate on
  (``model_tofts_fit`` etc.). Also the only backend for T1 mapping.
- **cpufit** -- run when ``pyCpufit`` imports. Accelerated Stage-D fit for the five DCE
  models.
- **gpufit** -- run only when a CUDA ``pyGpufit`` backend is available; otherwise noted as
  unavailable.

Two reference limits (per backend)
----------------------------------
1. OSIPI official acceptance tolerances (``osipi_official_tolerances.json``): round,
   method-agnostic pass/fail bars transcribed from the OSIPI test suite. The gate.
2. Peer-implementation error spread (``osipi_peer_error_summary.json``): how the published
   contributor implementations scatter around ground truth. Context only.

Run: ``.venv/bin/python tests/data/osipi/reference/generate_osipi_summary.py``
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from rocketship import (  # noqa: E402
    model_2cxm_fit,
    model_extended_tofts_fit,
    model_patlak_fit,
    model_tissue_uptake_fit,
    model_tofts_fit,
    t1_fa_linear_fit,
)
from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    _apply_model_specific_prefs,
    _fit_stage_d_model_accelerated,
    _stage_d_fit_prefs,
    probe_acceleration_backend,
)

OSIPI_ROOT = REPO_ROOT / "tests" / "data" / "osipi"
DCE_DATA_DIR = OSIPI_ROOT / "dce_models"
T1_DATA_DIR = OSIPI_ROOT / "t1_mapping"
REFERENCE_DIR = OSIPI_ROOT / "reference"
FIG_DIR = REFERENCE_DIR / "figures"
PEER_SUMMARY_JSON = REFERENCE_DIR / "osipi_peer_error_summary.json"
OFFICIAL_TOL_JSON = REFERENCE_DIR / "osipi_official_tolerances.json"
SUMMARY_MD = (
    REPO_ROOT / "docs" / "project-management" / "projects" / "osipi-verification" / "osipi_summary.md"
)

SOURCE_COMMIT = "23d3714797045d8103d5b5fa4f4c016840094dc0"
SOURCE_REPO = "https://github.com/OSIPI/DCE-DSC-MRI_TestResults"

_BASE_CONFIG = DcePipelineConfig(
    subject_source_path=REPO_ROOT, subject_tp_path=REPO_ROOT, output_dir=REPO_ROOT, backend="cpu"
)
_BASE_STAGE_D_PREFS = _stage_d_fit_prefs(_BASE_CONFIG)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(raw: str) -> List[float]:
    return [float(x) for x in str(raw).split()]


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = (len(vals) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(vals[lo])
    frac = idx - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _ps_per_min(ktrans_per_sec: float, fp_per_sec: float) -> float:
    if abs(fp_per_sec - ktrans_per_sec) < 1e-12:
        return float("inf")
    return (ktrans_per_sec * fp_per_sec / (fp_per_sec - ktrans_per_sec)) * 60.0


def _fnum(x: float, param: str) -> str:
    a = abs(x)
    if not math.isfinite(x):
        return "nan"
    if a != 0 and a < 1e-3:
        return f"{x:.2e}"
    if param == "fp":
        return f"{x:.3f}"
    return f"{x:.4f}"


def _ferr(x: float) -> str:
    a = abs(x)
    if not math.isfinite(a):
        return "nan"
    if a == 0:
        return "0"
    if a < 1e-3:
        return f"{a:.1e}"
    if a < 1:
        return f"{a:.4f}"
    return f"{a:.3g}"


# --------------------------------------------------------------------------- #
# per-model ground-truth extraction from a raw fit vector (shared by all backends)
# --------------------------------------------------------------------------- #
def _ex_tofts(f: np.ndarray, row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    return {"Ktrans": (float(row["Ktrans"]), float(f[0]) * 60.0),
            "ve": (float(row["ve"]), float(f[1]))}


def _ex_etofts(f: np.ndarray, row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    return {"Ktrans": (float(row["Ktrans"]), float(f[0]) * 60.0),
            "ve": (float(row["ve"]), float(f[1])),
            "vp": (float(row["vp"]), float(f[2]))}


def _ex_patlak(f: np.ndarray, row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    return {"ps": (float(row["ps"]), float(f[0]) * 60.0),
            "vp": (float(row["vp"]), float(f[1]))}


def _ex_2cxm(f: np.ndarray, row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    kt, ve, vp, fp = float(f[0]), float(f[1]), float(f[2]), float(f[3])
    return {"ve": (float(row["ve"]), ve),
            "vp": (float(row["vp"]), vp),
            "fp": (float(row["fp"]), fp * 6000.0),
            "ps": (float(row["ps"]), _ps_per_min(kt, fp))}


def _ex_2cum(f: np.ndarray, row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    kt, fp, vp = float(f[0]), float(f[1]), float(f[2])
    return {"vp": (float(row["vp"]), vp),
            "fp": (float(row["fp"]), fp * 6000.0),
            "ps": (float(row["ps"]), _ps_per_min(kt, fp))}


class ModelSpec:
    def __init__(self, key: str, peer_method: str, accel_name: str, params: List[str],
                 delay0: str, delay5: Optional[str], sig_col: str, aif_col: str, t_col: str,
                 py_func: Callable[..., Any],
                 extract: Callable[[np.ndarray, Dict[str, str]], Dict[str, Tuple[float, float]]]):
        self.key = key
        self.peer_method = peer_method
        self.accel_name = accel_name
        self.params = params
        self.delay0 = delay0
        self.delay5 = delay5
        self.sig_col = sig_col
        self.aif_col = aif_col
        self.t_col = t_col
        self.py_func = py_func
        self.extract = extract


DCE_SPECS = [
    ModelSpec("tofts", "tofts", "tofts", ["Ktrans", "ve"],
              "dce_DRO_data_tofts.csv", None, "C", "ca", "t", model_tofts_fit, _ex_tofts),
    ModelSpec("etofts", "etofts", "ex_tofts", ["Ktrans", "ve", "vp"],
              "dce_DRO_data_extended_tofts.csv", None, "C", "ca", "t", model_extended_tofts_fit, _ex_etofts),
    ModelSpec("patlak", "patlak", "patlak", ["ps", "vp"],
              "patlak_sd_0.02_delay_0.csv", "patlak_sd_0.02_delay_5.csv", "C_t", "cp_aif", "t",
              model_patlak_fit, _ex_patlak),
    ModelSpec("2cxm", "2CXM", "2cxm", ["ve", "vp", "fp", "ps"],
              "2cxm_sd_0.001_delay_0.csv", "2cxm_sd_0.001_delay_5.csv", "C_t", "cp_aif", "t",
              model_2cxm_fit, _ex_2cxm),
    ModelSpec("2cum", "2CUM", "tissue_uptake", ["vp", "fp", "ps"],
              "2cum_sd_0.0025_delay_0.csv", "2cum_sd_0.0025_delay_5.csv", "C_t", "cp_aif", "t",
              model_tissue_uptake_fit, _ex_2cum),
]
MODEL_ORDER = [s.key for s in DCE_SPECS]
BACKEND_ORDER = ["python", "cpufit", "gpufit"]


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def available_backends() -> Tuple[List[Tuple[str, str, Optional[str]]], Optional[str]]:
    """Return ([(label, kind, backend_id)], gpufit_note). kind is 'python' or 'accel'."""
    probe_acceleration_backend.cache_clear()
    probe = probe_acceleration_backend()
    backends: List[Tuple[str, str, Optional[str]]] = [("python", "python", None)]
    if bool(probe.get("pycpufit_imported", False)):
        backends.append(("cpufit", "accel", "cpufit_cpu"))
    gpu_note = None
    if str(probe.get("backend", "")) == "gpufit_cuda":
        backends.append(("gpufit", "accel", "gpufit_cuda"))
    elif bool(probe.get("pygpufit_imported", False)):
        gpu_note = ("gpufit: pyGpufit is installed but no CUDA GPU backend was available on the "
                    "machine that generated this report, so gpufit was not run.")
    else:
        gpu_note = f"gpufit: pyGpufit not importable ({probe.get('pygpufit_error')}); not run."
    return backends, gpu_note


def _accel_prefs(accel_name: str) -> Dict[str, Any]:
    prefs = dict(_BASE_STAGE_D_PREFS)
    if accel_name in {"2cxm", "tissue_uptake"}:
        return _apply_model_specific_prefs(prefs, accel_name)
    return prefs


def _fit_vector(spec: ModelSpec, row: Dict[str, str], kind: str, backend_id: Optional[str]) -> Optional[np.ndarray]:
    sig = _series(row[spec.sig_col])
    cp = _series(row[spec.aif_col])
    timer = _series(row[spec.t_col])
    try:
        if kind == "python":
            return np.asarray(spec.py_func(sig, cp, timer), dtype=np.float64)
        out = _fit_stage_d_model_accelerated(
            model_name=spec.accel_name,
            ct=np.asarray(sig, dtype=np.float64).reshape(-1, 1),
            cp_use=np.asarray(cp, dtype=np.float64),
            timer=np.asarray(timer, dtype=np.float64),
            prefs=_accel_prefs(spec.accel_name),
            acceleration_backend=backend_id,
        )
    except Exception:
        return None
    if out is None or np.asarray(out).shape[0] == 0:
        return None
    return np.asarray(out[0], dtype=np.float64)


def _fit_dataset(spec: ModelSpec, csv_name: str, kind: str, backend_id: Optional[str]
                 ) -> Tuple[List[str], Dict[str, List[Tuple[float, float]]]]:
    labels: List[str] = []
    out: Dict[str, List[Tuple[float, float]]] = {p: [] for p in spec.params}
    nan_vec = np.full(4, float("nan"))
    for row in _rows(DCE_DATA_DIR / csv_name):
        labels.append(row["label"])
        vec = _fit_vector(spec, row, kind, backend_id)
        pairs = spec.extract(vec if vec is not None else nan_vec, row)
        for p in spec.params:
            out[p].append(pairs[p])
    return labels, out


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def _stats(pairs: List[Tuple[float, float]], official: Optional[Dict[str, float]],
           peer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    errs = [abs(fit - ref) for ref, fit in pairs if math.isfinite(fit)]
    n_fail = len(pairs) - len(errs)
    cleaned = errs if errs else [float("inf")]
    row: Dict[str, Any] = {
        "n": len(pairs),
        "n_fail": n_fail,
        "our_max": max(cleaned),
        "our_mae": statistics.mean(cleaned),
        "our_p95": _pct(cleaned, 0.95),
    }
    if official is not None:
        a_tol, r_tol = official["a_tol"], official["r_tol"]
        worst = 0.0
        passed = True
        for ref, fit in pairs:
            eff = a_tol + r_tol * abs(ref)
            ratio = abs(fit - ref) / eff if eff > 0 else float("inf")
            worst = max(worst, ratio)
            if not (math.isfinite(fit) and abs(fit - ref) <= eff):
                passed = False
        row.update({"a_tol": a_tol, "r_tol": r_tol, "official_worst_frac": worst, "official_pass": passed})
    if peer is not None:
        row["peer_max"] = float(peer["max_abs_error"])
        pm = row["peer_max"]
        row["our_over_peer_max"] = row["our_max"] / pm if pm > 0 else float("inf")
    return row


def compute() -> Dict[str, Any]:
    official = json.loads(OFFICIAL_TOL_JSON.read_text())["DCEmodels"]
    peer = json.loads(PEER_SUMMARY_JSON.read_text())["metrics"]
    backends, gpu_note = available_backends()

    result: Dict[str, Any] = {
        "backends": [b[0] for b in backends],
        "gpu_note": gpu_note,
        "gated": [],       # per (backend, model, param), delay=0
        "gap": [],         # python delay=5
        "percase": {},     # python per-case, per model
        "t1": None,
    }

    for label, kind, backend_id in backends:
        for spec in DCE_SPECS:
            off = official.get(spec.peer_method, {})
            pr = peer["DCEmodels"].get(spec.peer_method, {})
            labels, pairs = _fit_dataset(spec, spec.delay0, kind, backend_id)
            if label == "python":
                result["percase"][spec.key] = {"labels": labels, "params": spec.params, "pairs": pairs}
            for p in spec.params:
                r = _stats(pairs[p], off.get(p), pr.get(p))
                r.update({"backend": label, "model": spec.key, "param": p})
                result["gated"].append(r)
            if label == "python" and spec.delay5 is not None:
                _, pairs5 = _fit_dataset(spec, spec.delay5, kind, backend_id)
                for p in spec.params:
                    r = _stats(pairs5[p], off.get(p), pr.get(p))
                    r.update({"backend": label, "model": spec.key, "param": p})
                    result["gap"].append(r)

    result["t1"] = _compute_t1(peer)
    return result


def _compute_t1(peer: Dict[str, Any]) -> Dict[str, Any]:
    r1_pairs: List[Tuple[float, float]] = []
    for dataset_name, csv_name in [("brain", "t1_brain_data.csv"),
                                   ("quiba", "t1_quiba_data.csv"),
                                   ("prostate", "t1_prostate_data.csv")]:
        for row in _rows(T1_DATA_DIR / csv_name):
            fa = _series(row["FA"])
            signal = _series(row["s"])
            if dataset_name == "prostate":
                tr_ms = float(str(row["TR"]).split()[0])
                r1_ref = 1000.0 / float(row[" T1 nonlinear"])
            elif dataset_name == "quiba":
                tr_ms = float(str(row["TR"]).split()[0]) * 1000.0
                r1_ref = float(row["R1"]) * 1000.0
            else:
                tr_ms = float(str(row["TR"]).split()[0]) * 1000.0
                r1_ref = float(row["R1"])
            t1_ms = float(t1_fa_linear_fit(fa, signal, tr_ms)[0])
            r1_pairs.append((r1_ref, 1000.0 / t1_ms))
    r = _stats(r1_pairs, None, peer["T1mapping"]["linear"]["r1"])
    r.update({"backend": "python", "model": "t1_linear", "param": "r1"})
    return r


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plot_backends(res: Dict[str, Any], keys: List[str], *, title: str, outfile: Path) -> None:
    """Per DCE parameter, plot each backend's max error vs the OSIPI a_tol."""
    idx = [(r["model"], r["param"]) for r in res["gated"]
           if r["backend"] == "python" and r["model"] in keys]
    labels = [f"{m}\n{p}" for m, p in idx]
    x = np.arange(len(idx), dtype=float)
    by = {(r["backend"], r["model"], r["param"]): r for r in res["gated"]}

    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(idx)), 5.0))
    markers = {"python": ("o", "#1f77b4"), "cpufit": ("s", "#d62728"), "gpufit": ("^", "#9467bd")}
    for be in res["backends"]:
        ys = [by.get((be, m, p), {}).get("our_max", np.nan) for m, p in idx]
        m_, c_ = markers.get(be, ("x", "#333333"))
        ax.scatter(x, ys, marker=m_, color=c_, s=55, label=f"{be} max", zorder=3)
    official = [by[("python", m, p)].get("a_tol", np.nan) for m, p in idx]
    ax.scatter(x, official, marker="_", color="#2ca02c", s=340, linewidths=2.2, label="OSIPI a_tol (gate)")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Max absolute error (log scale)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def _plot_single(rows: List[Dict[str, Any]], *, title: str, outfile: Path) -> None:
    labels = [f"{r['model']}\n{r['param']}" for r in rows]
    x = np.arange(len(rows), dtype=float)
    our_max = np.array([r["our_max"] for r in rows], dtype=float)
    peer_max = np.array([r.get("peer_max", np.nan) for r in rows], dtype=float)
    official = np.array([r.get("a_tol", np.nan) for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(rows)), 5.0))
    ax.scatter(x, our_max, marker="o", color="#1f77b4", s=60, label="python max", zorder=3)
    ax.scatter(x, peer_max, marker="_", color="#8a3f00", s=260, linewidths=2.2, label="peer max (context)")
    ax.scatter(x, official, marker="D", facecolors="none", edgecolors="#2ca02c", s=70,
               linewidths=1.8, label="OSIPI a_tol (gate)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Absolute error (log scale)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def _write_figures(res: Dict[str, Any]) -> List[Path]:
    out = []
    _plot_backends(res, ["tofts", "etofts", "2cxm", "2cum"],
                   title="OSIPI DROs: max fit error by backend vs OSIPI gate",
                   outfile=FIG_DIR / "osipi_accuracy_dros.png")
    out.append(FIG_DIR / "osipi_accuracy_dros.png")
    _plot_backends(res, ["patlak"],
                   title="OSIPI Patlak (delay 0): max fit error by backend vs OSIPI gate",
                   outfile=FIG_DIR / "osipi_accuracy_patlak_delay.png")
    out.append(FIG_DIR / "osipi_accuracy_patlak_delay.png")
    _plot_single([res["t1"]], title="OSIPI T1 linear (python): error vs peer spread",
                 outfile=FIG_DIR / "osipi_accuracy_t1.png")
    out.append(FIG_DIR / "osipi_accuracy_t1.png")
    return out


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _provenance_lines(res: Dict[str, Any]) -> List[str]:
    ran = ", ".join(f"`{b}`" for b in res["backends"])
    lines = [
        "## Fitting backends verified",
        "",
        f"ROCKETSHIP has four fitting routines (MATLAB, python, cpufit, gpufit). This report "
        f"verifies the three non-MATLAB backends against OSIPI. Backends run for this report: {ran}.",
        "",
    ]
    if res.get("gpu_note"):
        lines += [f"> {res['gpu_note']}", ""]
    lines += [
        "- **python** — the pure-CPU scipy fit (`model_*_fit`), the DCE reference the reliability "
        "tests gate on, and the only backend for T1 mapping.",
        "- **cpufit / gpufit** — the accelerated (float32) Stage-D fit for the five DCE models. "
        "Reliable for `tofts`/`etofts`/`patlak` and, via a backend-agnostic random multi-start that "
        "escapes the wrong-Fp-basin degenerate minimum, for `2cum`. The stiff `2cxm` fit still misses "
        "a few low-flow (Fp=5) cases where vp is weakly identifiable (see the FAIL cells below) -- not "
        "a precision issue; the float64 python backend, which fits the extraction fraction E=Ktrans/Fp, "
        "is the reference for `2cxm`.",
        "",
        "## Where these numbers come from",
        "",
        "**Ground-truth data (fully verified).** The DCE digital reference objects under "
        "`tests/data/osipi/dce_models/` are byte-identical (MD5) to the OSIPI source at "
        f"[`{SOURCE_COMMIT[:10]}`]({SOURCE_REPO}/tree/{SOURCE_COMMIT}) "
        "(`test/DCEmodels/data/`). Per the source docstrings the concentration curves were "
        "generated by M. Thrippleton with [mjt320/DCE-functions](https://github.com/mjt320/DCE-functions); "
        "each row's `vp/ve/fp/ps` (or `Ktrans/ve/vp`) are the *true parameters used to generate the data*. "
        "Published in **Manning et al., Magnetic Resonance in Medicine, 2021** "
        "([doi:10.1002/mrm.28833](https://doi.org/10.1002/mrm.28833)).",
        "",
        "**OSIPI official acceptance tolerances (the gate).** `osipi_official_tolerances.json` is "
        "transcribed verbatim from the OSIPI test suite (`test/DCEmodels/DCEmodels_data.py`). Per the "
        "OSIPI paper these tolerances are deliberately *wide validity checks* -- \"not intended to "
        "indicate an acceptable level of accuracy\" -- so passing them means a backend has no "
        "gross/unit errors.",
        "",
        "**Peer-implementation spread (reproducible; context).** `osipi_peer_error_summary.json` pools "
        "the deviations of every published contributor implementation in the OSIPI DCE-DSC-MRI testing "
        "framework (**van Houdt et al., MRM 2023**, "
        "[doi:10.1002/mrm.29826](https://doi.org/10.1002/mrm.29826)); `generate_peer_error_summary.py` "
        "recomputes it from the committed result CSVs. Reported for context, not gated: the pool "
        "includes the LEK/Edinburgh implementation ROCKETSHIP's python `2cxm`/`tissue_uptake` fits "
        "reproduce, so `peer max` tracks the python error there.",
        "",
    ]
    return lines


def _accuracy_table(res: Dict[str, Any]) -> List[str]:
    by = {(r["backend"], r["model"], r["param"]): r for r in res["gated"]}
    order = [(r["model"], r["param"]) for r in res["gated"] if r["backend"] == "python"]
    # de-dup preserving order
    seen = set()
    order = [mp for mp in order if not (mp in seen or seen.add(mp))]

    backends = res["backends"]
    head = "| Model | Param | " + " | ".join(backends) + " | peer max |"
    sep = "| --- | --- | " + " | ".join(["---"] * len(backends)) + " | ---: |"
    lines = [
        "## Accuracy by backend",
        "",
        "Each backend cell is `max |GT − fit|` over all cases and its worst-case error as a % of the "
        "OSIPI tolerance (`a_tol + r_tol·|ref|`); `ok` if every case is within tolerance, `FAIL` "
        "otherwise. `peer max` is the published-implementation spread (context only). T1 mapping is "
        "python-only.",
        "",
        head, sep,
    ]

    def cell(be: str, m: str, p: str) -> str:
        r = by.get((be, m, p))
        if r is None:
            return "—"
        frac = r.get("official_worst_frac")
        pct = "n/a" if frac is None else f"{frac * 100:.0f}%"
        verdict = "" if "official_pass" not in r else (" ok" if r["official_pass"] else " **FAIL**")
        return f"{r['our_max']:.3g} · {pct}{verdict}"

    for (m, p) in order:
        cells = [cell(be, m, p) for be in backends]
        peer_row = by.get(("python", m, p), {})
        pmax = f"{peer_row['peer_max']:.4g}" if "peer_max" in peer_row else "—"
        lines.append(f"| {m} | {p} | " + " | ".join(cells) + f" | {pmax} |")
    # T1 row (python only)
    t1 = res["t1"]
    t1cells = []
    for be in backends:
        t1cells.append(f"{t1['our_max']:.3g} · peer-ref" if be == "python" else "—")
    lines.append(f"| t1_linear | r1 | " + " | ".join(t1cells) + f" | {t1['peer_max']:.4g} |")
    lines.append("")

    # delay=5 gap (python only)
    lines += ["### Delay=5 (arterial-delay fitting not implemented — python, gap visibility, not gated)", "",
              "| Model | Param | N | python max | peer max |",
              "| --- | --- | ---: | ---: | ---: |"]
    for r in res["gap"]:
        pmax = f"{r['peer_max']:.4g}" if "peer_max" in r else "—"
        lines.append(f"| {r['model']} | {r['param']} | {r['n']} | {r['our_max']:.4g} | {pmax} |")
    lines.append("")
    return lines


def _percase_tables(res: Dict[str, Any]) -> List[str]:
    lines = ["## Per-case ground truth vs fit — python (delay=0)", "",
             "Each row is one DRO case. `GT` = generating parameter, `fit` = ROCKETSHIP **python** fit, "
             "`Δ` = |GT − fit|. Units: v_e, v_p fractional; K^trans, PS per min; F_p mL/100mL/min.", ""]
    for spec in DCE_SPECS:
        pc = res["percase"][spec.key]
        params = pc["params"]
        header = "| case | " + " | ".join(f"{p} GT | {p} fit | {p} Δ" for p in params) + " |"
        sep = "| --- | " + " | ".join(["---: | ---: | ---:"] * len(params)) + " |"
        lines += [f"### {spec.key}", "", header, sep]
        for i, label in enumerate(pc["labels"]):
            cells = [label.replace("case_", "#")]
            for p in params:
                ref, fit = pc["pairs"][p][i]
                cells += [_fnum(ref, p), _fnum(fit, p), _ferr(abs(fit - ref))]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def write_markdown(res: Dict[str, Any], figures: List[Path]) -> None:
    lines = ["# OSIPI Accuracy Summary", "",
             "ROCKETSHIP DCE/T1 fits against the OSIPI digital reference objects, per fitting backend, "
             "gated on OSIPI's own published acceptance tolerances. Regenerate with "
             "`.venv/bin/python tests/data/osipi/reference/generate_osipi_summary.py`.", ""]
    lines += _provenance_lines(res)
    lines += _accuracy_table(res)
    lines += ["## Figures", ""]
    for f in figures:
        lines.append(f"- `{f.relative_to(REPO_ROOT)}`")
    lines.append("")
    lines += _percase_tables(res)
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    res = compute()
    figures = _write_figures(res)
    write_markdown(res, figures)
    print(f"wrote {SUMMARY_MD}")
    for f in figures:
        print(f"wrote {f}")
    print(f"backends run: {', '.join(res['backends'])}")
    for r in res["gated"]:
        if not r.get("official_pass", True):
            print(f"  gate FAIL: {r['backend']} {r['model']}.{r['param']} "
                  f"({r['official_worst_frac'] * 100:.0f}% of tol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
