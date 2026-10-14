"""Regenerate ``osipi_peer_error_summary.json`` from the committed OSIPI result CSVs.

The "peer" summary is the pooled error spread (mae / p90 / p95 / max of
|measured - reference|) of every published contributor implementation in the OSIPI
DCE-DSC-MRI testing framework, aggregated per (category, method, parameter).

Provenance
----------
All inputs are the per-implementation result CSVs exported by that framework and
published in the OSIPI ``DCE-DSC-MRI_TestResults`` repository @ commit ``23d3714``
(see van Houdt et al., *Magnetic Resonance in Medicine*, 2023,
`doi:10.1002/mrm.29826 <https://doi.org/10.1002/mrm.29826>`_). They are committed here
under ``reference/{dce_models_results,t1_mapping_results,si_to_conc_results,dsc_models_results}/``,
so this summary is fully reproducible.

Note: these peer *spread* numbers are reported for context only. The DCE reliability
tests gate on OSIPI's official acceptance tolerances (``osipi_official_tolerances.json``);
see ``README.md``.

Run: ``.venv/bin/python tests/data/osipi/reference/generate_peer_error_summary.py``
Add ``--check`` to verify it reproduces the committed JSON without writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

REFERENCE_DIR = Path(__file__).resolve().parent
OUT_JSON = REFERENCE_DIR / "osipi_peer_error_summary.json"

SOURCE_REPO = "https://github.com/OSIPI/DCE-DSC-MRI_TestResults"
SOURCE_COMMIT = "23d3714797045d8103d5b5fa4f4c016840094dc0"


def _dce_method(name: str) -> str:
    low = name.lower()
    if "2cum" in low:
        return "2CUM"
    if "2cxm" in low:
        return "2CXM"
    if "etofts" in low:
        return "etofts"
    if "patlak" in low:
        return "patlak"
    if "tofts" in low:
        return "tofts"
    raise ValueError(f"Cannot map DCE model for {name!r}")


def _t1_method(name: str) -> str:
    low = name.lower()
    if "2fa" in low:
        return "two-FA"
    if "nonlin" in low or "novifast" in low:
        return "nonlinear"
    if "lin" in low:
        return "linear"
    raise ValueError(f"Cannot map T1 method for {name!r}")


# (subdirectory, category, method-resolver) — method None means single "" bucket.
SOURCES = [
    ("dce_models_results", "DCEmodels", _dce_method),
    ("t1_mapping_results", "T1mapping", _t1_method),
    ("si_to_conc_results", "SI_to_Conc", None),
    ("dsc_models_results", "DSCmodels", None),
]


def _pct(values: List[float], p: float) -> float:
    """Linear-interpolated percentile (numpy 'linear' method)."""
    vals = sorted(values)
    idx = (len(vals) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - idx) + vals[hi] * (idx - lo))


def collect() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    # errs[category][method][param] -> list of abs errors
    errs: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for subdir, category, resolver in SOURCES:
        d = REFERENCE_DIR / subdir
        if not d.is_dir():
            continue
        for csv_path in sorted(d.glob("*.csv")):
            method = "" if resolver is None else resolver(csv_path.name)
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                cols = reader.fieldnames or []
                pairs: List[Tuple[str, str]] = []  # (param, meas_col)
                for col in cols:
                    if col.endswith("_ref"):
                        param = col[:-4]
                        for suffix in ("_meas", "_measured"):
                            if param + suffix in cols:
                                pairs.append((param, param + suffix))
                                break
                for row in reader:
                    for param, meas_col in pairs:
                        ref_raw = row.get(param + "_ref", "")
                        meas_raw = row.get(meas_col, "")
                        try:
                            e = abs(float(meas_raw) - float(ref_raw))
                        except (TypeError, ValueError):
                            continue
                        errs.setdefault(category, {}).setdefault(method, {}).setdefault(param, []).append(e)

    metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for category, methods in errs.items():
        for method, params in methods.items():
            for param, values in params.items():
                metrics.setdefault(category, {}).setdefault(method, {})[param] = {
                    "mae": sum(values) / len(values),
                    "max_abs_error": max(values),
                    "n": len(values),
                    "p90_abs_error": _pct(values, 0.90),
                    "p95_abs_error": _pct(values, 0.95),
                }
    return metrics


def build() -> Dict[str, object]:
    return {
        "metrics": collect(),
        "source": {"commit": SOURCE_COMMIT, "repo": SOURCE_REPO},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify reproduction without writing")
    args = ap.parse_args()

    payload = build()
    new_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.check:
        old = json.loads(OUT_JSON.read_text())
        if old == payload:
            print("OK: regenerated peer summary matches the committed JSON exactly.")
            return 0
        print("MISMATCH: regenerated peer summary differs from committed JSON.")
        return 1

    OUT_JSON.write_text(new_text)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
