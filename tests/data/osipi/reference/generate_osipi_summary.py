"""Generate the OSIPI accuracy summary (markdown + figures).

What this produces
------------------
- ``docs/project-management/projects/osipi-verification/osipi_summary.md`` -- a plain
  markdown report: data provenance, a dual-gate accuracy table, and per-case
  ground-truth-vs-fit tables for every gated DCE model.
- ``tests/data/osipi/reference/figures/*.png`` -- ROCKETSHIP error vs the OSIPI
  official acceptance tolerance and vs the imported peer-implementation spread.

Fit path
--------
Every DCE model is fit with the same reference functions the reliability tests
gate on (``model_tofts_fit``, ``model_extended_tofts_fit``, ``model_patlak_fit``,
``model_2cxm_fit``, ``model_tissue_uptake_fit``) using default preferences -- NOT
the accelerated backend -- so the numbers here match the pytest gates exactly.

Two reference limits
---------------------
1. OSIPI official acceptance tolerances (``osipi_official_tolerances.json``):
   round, method-agnostic pass/fail bars transcribed verbatim from the OSIPI test
   suite. This is the hard gate.
2. Imported peer-implementation error spread (``osipi_peer_error_summary.json``):
   how the published contributor implementations scatter around ground truth.
   Reported for context only -- it is NOT reproducible in-repo (see the summary's
   provenance section) and for LEK-derived models our error tracks its maximum.

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
    """Human-friendly fixed/scientific formatting for a parameter value."""
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
    if a == 0:
        return "0"
    if a < 1e-3:
        return f"{a:.1e}"
    if a < 1:
        return f"{a:.4f}"
    return f"{a:.3f}"


# --------------------------------------------------------------------------- #
# per-model fit + ground-truth extraction (comparison units)
# --------------------------------------------------------------------------- #
def _fit_tofts(row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    f = model_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))
    return {"Ktrans": (float(row["Ktrans"]), float(f[0]) * 60.0),
            "ve": (float(row["ve"]), float(f[1]))}


def _fit_etofts(row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    f = model_extended_tofts_fit(_series(row["C"]), _series(row["ca"]), _series(row["t"]))
    return {"Ktrans": (float(row["Ktrans"]), float(f[0]) * 60.0),
            "ve": (float(row["ve"]), float(f[1])),
            "vp": (float(row["vp"]), float(f[2]))}


def _fit_patlak(row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    f = model_patlak_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))
    return {"ps": (float(row["ps"]), float(f[0]) * 60.0),
            "vp": (float(row["vp"]), float(f[1]))}


def _fit_2cxm(row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    f = model_2cxm_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))
    kt, ve, vp, fp = float(f[0]), float(f[1]), float(f[2]), float(f[3])
    return {"ve": (float(row["ve"]), ve),
            "vp": (float(row["vp"]), vp),
            "fp": (float(row["fp"]), fp * 6000.0),
            "ps": (float(row["ps"]), _ps_per_min(kt, fp))}


def _fit_2cum(row: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    f = model_tissue_uptake_fit(_series(row["C_t"]), _series(row["cp_aif"]), _series(row["t"]))
    kt, fp, vp = float(f[0]), float(f[1]), float(f[2])
    return {"vp": (float(row["vp"]), vp),
            "fp": (float(row["fp"]), fp * 6000.0),
            "ps": (float(row["ps"]), _ps_per_min(kt, fp))}


class ModelSpec:
    def __init__(self, key: str, peer_method: str, params: List[str],
                 delay0: str, delay5: Optional[str],
                 fitter: Callable[[Dict[str, str]], Dict[str, Tuple[float, float]]]):
        self.key = key
        self.peer_method = peer_method
        self.params = params
        self.delay0 = delay0
        self.delay5 = delay5
        self.fitter = fitter


DCE_SPECS = [
    ModelSpec("tofts", "tofts", ["Ktrans", "ve"], "dce_DRO_data_tofts.csv", None, _fit_tofts),
    ModelSpec("etofts", "etofts", ["Ktrans", "ve", "vp"], "dce_DRO_data_extended_tofts.csv", None, _fit_etofts),
    ModelSpec("patlak", "patlak", ["ps", "vp"], "patlak_sd_0.02_delay_0.csv", "patlak_sd_0.02_delay_5.csv", _fit_patlak),
    ModelSpec("2cxm", "2CXM", ["ve", "vp", "fp", "ps"], "2cxm_sd_0.001_delay_0.csv", "2cxm_sd_0.001_delay_5.csv", _fit_2cxm),
    ModelSpec("2cum", "2CUM", ["vp", "fp", "ps"], "2cum_sd_0.0025_delay_0.csv", "2cum_sd_0.0025_delay_5.csv", _fit_2cum),
]


# --------------------------------------------------------------------------- #
# computation
# --------------------------------------------------------------------------- #
def _fit_dataset(spec: ModelSpec, csv_name: str) -> Tuple[List[str], Dict[str, List[Tuple[float, float]]]]:
    """Return (labels, {param: [(ref, fit), ...]}) for one dataset."""
    labels: List[str] = []
    out: Dict[str, List[Tuple[float, float]]] = {p: [] for p in spec.params}
    for row in _rows(DCE_DATA_DIR / csv_name):
        labels.append(row["label"])
        got = spec.fitter(row)
        for p in spec.params:
            out[p].append(got[p])
    return labels, out


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
    # dual gate 1: OSIPI official acceptance tolerance (per-case atol + rtol*|ref|)
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
        row.update({"a_tol": a_tol, "r_tol": r_tol,
                    "official_worst_frac": worst, "official_pass": passed})
    # informational: peer spread
    if peer is not None:
        row["peer_max"] = float(peer["max_abs_error"])
        row["peer_mae"] = float(peer["mae"])
        row["peer_p95"] = float(peer["p95_abs_error"])
        pm = row["peer_max"]
        row["our_over_peer_max"] = row["our_max"] / pm if pm > 0 else float("inf")
    return row


def compute() -> Dict[str, Any]:
    official = json.loads(OFFICIAL_TOL_JSON.read_text())["DCEmodels"]
    peer = json.loads(PEER_SUMMARY_JSON.read_text())["metrics"]

    result: Dict[str, Any] = {"gated": [], "gap": [], "percase": {}, "t1": None}

    for spec in DCE_SPECS:
        off = official.get(spec.peer_method, {})
        pr = peer["DCEmodels"].get(spec.peer_method, {})

        labels, pairs = _fit_dataset(spec, spec.delay0)
        result["percase"][spec.key] = {"labels": labels, "params": spec.params, "pairs": pairs}
        for p in spec.params:
            row = _stats(pairs[p], off.get(p), pr.get(p))
            row.update({"model": spec.key, "param": p, "slice": "delay=0"})
            result["gated"].append(row)

        if spec.delay5 is not None:
            _, pairs5 = _fit_dataset(spec, spec.delay5)
            for p in spec.params:
                row = _stats(pairs5[p], off.get(p), pr.get(p))
                row.update({"model": spec.key, "param": p, "slice": "delay=5",
                            "note": "delay fitting not implemented; shown for gap visibility"})
                result["gap"].append(row)

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
    pr = peer["T1mapping"]["linear"]["r1"]
    row = _stats(r1_pairs, None, pr)
    row.update({"model": "t1_linear", "param": "r1", "slice": "brain+quiba+prostate"})
    return row


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plot(rows: List[Dict[str, Any]], *, title: str, outfile: Path) -> None:
    labels = [f"{r['model']}\n{r['param']}" for r in rows]
    x = np.arange(len(rows), dtype=float)

    our_max = np.array([r["our_max"] for r in rows], dtype=float)
    our_mae = np.array([r["our_mae"] for r in rows], dtype=float)
    peer_max = np.array([r.get("peer_max", np.nan) for r in rows], dtype=float)
    official = np.array([r.get("a_tol", np.nan) + r.get("r_tol", 0.0) * 0 for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(rows)), 5.0))
    ax.bar(x, our_mae, 0.5, color="#1f77b4", alpha=0.85, label="ROCKETSHIP MAE")
    ax.scatter(x, our_max, marker="x", color="#0b3a62", s=70, label="ROCKETSHIP max")
    ax.scatter(x, peer_max, marker="_", color="#8a3f00", s=260, linewidths=2.2,
               label="Peer max (imported, informational)")
    # OSIPI official acceptance tolerance (the hard gate); atol component shown.
    ax.scatter(x, official, marker="D", facecolors="none", edgecolors="#2ca02c",
               s=70, linewidths=1.8, label="OSIPI official a_tol (gate)")

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
    gated = res["gated"]
    dro = [r for r in gated if r["model"] in {"tofts", "etofts", "2cxm", "2cum"}]
    patlak = [r for r in gated if r["model"] == "patlak"] + [r for r in res["gap"] if r["model"] == "patlak"]
    t1 = [res["t1"]]
    out = []
    for rows, title, name in [
        (dro, "OSIPI DROs: ROCKETSHIP error vs OSIPI gate and peer spread", "osipi_accuracy_dros.png"),
        (patlak, "OSIPI Patlak (delay 0 gated, delay 5 gap): ROCKETSHIP vs gate/peer", "osipi_accuracy_patlak_delay.png"),
        (t1, "OSIPI T1 linear: ROCKETSHIP vs peer spread", "osipi_accuracy_t1.png"),
    ]:
        _plot(rows, title=title, outfile=FIG_DIR / name)
        out.append(FIG_DIR / name)
    return out


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _provenance_lines() -> List[str]:
    return [
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
        "**OSIPI official acceptance tolerances (the hard gate).** `osipi_official_tolerances.json` is "
        "transcribed verbatim from the OSIPI test suite (`test/DCEmodels/DCEmodels_data.py`), where every "
        "contributor implementation is asserted with "
        "`np.testing.assert_allclose(measured, reference, atol=a_tol, rtol=r_tol)`. Per the OSIPI paper "
        "these tolerances are deliberately *wide validity checks* -- \"not intended to indicate an "
        "acceptable level of accuracy\" -- so passing them means ROCKETSHIP has no gross/unit errors, "
        "which is exactly what the OSIPI framework itself gates on.",
        "",
        "**Peer-implementation spread (reproducible; informational).** `osipi_peer_error_summary.json` "
        "holds the pooled error spread (mae / p90 / p95 / max of |measured - reference|) of every published "
        "contributor implementation in the OSIPI DCE-DSC-MRI testing framework "
        "(**van Houdt et al., Magnetic Resonance in Medicine, 2023**, "
        "[doi:10.1002/mrm.29826](https://doi.org/10.1002/mrm.29826)). Every per-contributor result CSV is "
        "committed under `tests/data/osipi/reference/{dce_models_results,t1_mapping_results,"
        "si_to_conc_results,dsc_models_results}/`, and `generate_peer_error_summary.py` recomputes the JSON "
        "from them exactly. It is reported for context, not gated, because the pool *includes* the "
        "LEK/Edinburgh implementation that ROCKETSHIP ports (`2cxm`, `tissue_uptake`): our fit reproduces "
        "LEK, so `peer max` tracks our own error to ~4 significant figures (see `our/peer` near 1.0) -- a "
        "near-circular limit.",
        "",
    ]


def _accuracy_table(res: Dict[str, Any]) -> List[str]:
    lines = ["## Accuracy vs OSIPI gate (hard) and peer spread (informational)", "",
             "`gate %` is the worst case's error as a fraction of its OSIPI tolerance "
             "(`a_tol + r_tol*|ref|`); < 100% passes. `our/peer` is ROCKETSHIP max error over the "
             "imported peer max -- values near 1.0 flag the near-circular limit discussed above.", "",
             "| Model | Param | Slice | N | our max | our MAE | our p95 | OSIPI a_tol | OSIPI r_tol | gate % | gate | peer max | our/peer |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |"]
    for r in res["gated"] + [res["t1"]]:
        atol = f"{r['a_tol']:g}" if "a_tol" in r else "--"
        rtol = f"{r['r_tol']:g}" if "r_tol" in r else "--"
        gpct = f"{r['official_worst_frac']*100:.1f}%" if "official_worst_frac" in r else "--"
        gate = ("PASS" if r["official_pass"] else "FAIL") if "official_pass" in r else "--"
        pmax = f"{r['peer_max']:.4g}" if "peer_max" in r else "--"
        oop = f"{r['our_over_peer_max']:.3f}" if "our_over_peer_max" in r else "--"
        lines.append(
            f"| {r['model']} | {r['param']} | {r['slice']} | {r['n']} | {r['our_max']:.4g} | "
            f"{r['our_mae']:.4g} | {r['our_p95']:.4g} | {atol} | {rtol} | {gpct} | {gate} | {pmax} | {oop} |"
        )
    lines.append("")
    # gap rows
    lines += ["### Delay=5 (arterial-delay fitting not implemented -- gap visibility, not gated)", "",
              "| Model | Param | N | our max | our MAE | peer max | our/peer |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in res["gap"]:
        pmax = f"{r['peer_max']:.4g}" if "peer_max" in r else "--"
        oop = f"{r['our_over_peer_max']:.3f}" if "our_over_peer_max" in r else "--"
        lines.append(f"| {r['model']} | {r['param']} | {r['n']} | {r['our_max']:.4g} | {r['our_mae']:.4g} | {pmax} | {oop} |")
    lines.append("")
    return lines


def _percase_tables(res: Dict[str, Any]) -> List[str]:
    lines = ["## Per-case ground truth vs fit (delay=0)", "",
             "Each row is one DRO case. `GT` = generating parameter, `fit` = ROCKETSHIP fit, "
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
             "ROCKETSHIP DCE/T1 fits against the OSIPI digital reference objects, gated on OSIPI's own "
             "published acceptance tolerances. Regenerate with "
             "`.venv/bin/python tests/data/osipi/reference/generate_osipi_summary.py`.", ""]
    lines += _provenance_lines()
    lines += _accuracy_table(res)
    lines += ["## Figures", ""]
    for f in figures:
        rel = f.relative_to(REPO_ROOT)
        lines.append(f"- `{rel}`")
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
    # console gate summary
    fails = [r for r in res["gated"] if not r.get("official_pass", True)]
    print(f"gated params: {len(res['gated'])}, official-tolerance failures: {len(fails)}")
    for r in fails:
        print(f"  FAIL {r['model']}.{r['param']} gate%={r['official_worst_frac']*100:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
