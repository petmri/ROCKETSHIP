"""Synthetic stability sweeps for DCE tissue uptake and 2CXM inverse fits.

This script helps tune constraints/initial guesses on low-noise synthetic curves
without relying on MATLAB maps as the primary reference.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Callable, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_models import (  # noqa: E402
    model_2cxm_cfit,
    model_2cxm_fit,
    model_tissue_uptake_cfit,
    model_tissue_uptake_fit,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep synthetic stability for tissue_uptake/2cxm fits.")
    parser.add_argument("--model", choices=["tissue_uptake", "2cxm", "both"], default="both")
    parser.add_argument("--curves", type=int, default=120, help="Curves per model/noise/profile.")
    parser.add_argument(
        "--noise-std",
        default="0,2e-4,5e-4",
        help="Comma-separated Gaussian noise std levels applied to Ct.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _cp_curve(timer: List[float]) -> List[float]:
    out: List[float] = []
    for t in timer:
        t_pos = max(0.0, t - 0.03)
        arterial = 5.5 * (t_pos**2.1) * math.exp(-t_pos / 0.22)
        recirc = 0.22 * math.exp(-max(0.0, t - 0.8) / 1.35)
        out.append(max(0.0, arterial + recirc))
    peak = max(out) if out else 1.0
    if peak <= 0.0:
        return [0.0 for _ in out]
    return [v / peak for v in out]


def _timer() -> List[float]:
    return [
        0.0,
        0.03,
        0.06,
        0.1,
        0.14,
        0.2,
        0.28,
        0.38,
        0.5,
        0.68,
        0.9,
        1.2,
        1.6,
        2.1,
        2.7,
        3.4,
    ]


def _safe_corr(a: List[float], b: List[float]) -> float:
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    da = [x - mean_a for x in a]
    db = [x - mean_b for x in b]
    num = sum(x * y for x, y in zip(da, db))
    den_a = math.sqrt(sum(x * x for x in da))
    den_b = math.sqrt(sum(y * y for y in db))
    den = den_a * den_b
    if den <= 0.0:
        return float("nan")
    return num / den


def _median_rel_err(truth: List[float], pred: List[float]) -> float:
    errs: List[float] = []
    for t, p in zip(truth, pred):
        scale = max(abs(float(t)), 1e-8)
        errs.append(abs(float(p) - float(t)) / scale)
    if not errs:
        return float("nan")
    return float(statistics.median(errs))


def _profile_library(model_name: str) -> Dict[str, Dict[str, float | str]]:
    if model_name == "tissue_uptake":
        return {
            "baseline": {},
            "tight_tp": {
                "upper_limit_tp": 1.5,
                "initial_value_tp": 0.12,
                "initial_value_fp": 0.35,
                "max_nfev": 120,
            },
            "broad_iter": {
                "max_nfev": 300,
                "tol_x": 1e-7,
                "initial_value_fp": 0.45,
                "initial_value_tp": 0.2,
            },
            "robust_bisquare": {
                "robust": "bisquare",
                "max_nfev": 220,
                "initial_value_fp": 0.4,
                "initial_value_tp": 0.15,
            },
        }
    return {
        "baseline": {},
        "tight_ve": {
            "lower_limit_ve": 0.05,
            "initial_value_ve": 0.18,
            "upper_limit_fp": 20.0,
            "initial_value_fp": 0.35,
            "max_nfev": 140,
        },
        "broad_iter": {
            "max_nfev": 300,
            "tol_x": 1e-7,
            "initial_value_ktrans": 5e-4,
            "initial_value_fp": 0.4,
        },
        "robust_bisquare": {
            "robust": "bisquare",
            "max_nfev": 220,
            "initial_value_fp": 0.4,
            "initial_value_ve": 0.18,
        },
    }


def _sample_tissue_truth(rng: random.Random) -> Tuple[float, float, float, float]:
    ktrans = rng.uniform(0.005, 0.12)
    fp = rng.uniform(ktrans * 1.4, 1.5)
    tp = rng.uniform(0.03, 0.5)
    ps = ktrans * fp / (fp - ktrans)
    vp = (fp + ps) * tp
    return ktrans, fp, tp, vp


def _sample_2cxm_truth(rng: random.Random) -> Tuple[float, float, float, float]:
    ktrans = rng.uniform(0.003, 0.08)
    ve = rng.uniform(0.06, 0.65)
    vp = rng.uniform(0.005, 0.22)
    fp = rng.uniform(max(0.06, ktrans * 1.6), 1.8)
    return ktrans, ve, vp, fp


def _run_model(
    model_name: str,
    curves: int,
    noise_levels: Iterable[float],
    rng: random.Random,
) -> None:
    timer = _timer()
    cp = _cp_curve(timer)
    profiles = _profile_library(model_name)

    print(f"\n[SYNTH] model={model_name} curves={curves} profiles={list(profiles.keys())}")
    for noise_std in noise_levels:
        print(f"[SYNTH] noise_std={noise_std:.2e}")
        for profile_name, prefs in profiles.items():
            ok = 0
            truths_by_param: List[List[float]] = []
            preds_by_param: List[List[float]] = []

            if model_name == "tissue_uptake":
                truths_by_param = [[], [], []]
                preds_by_param = [[], [], []]
                for _ in range(curves):
                    k, fp, tp, vp = _sample_tissue_truth(rng)
                    ct_clean = model_tissue_uptake_cfit(k, fp, tp, cp, timer)
                    ct_noisy = [float(v + rng.gauss(0.0, noise_std)) for v in ct_clean]
                    fit = model_tissue_uptake_fit(ct_noisy, cp, timer, prefs)
                    fit_k, fit_fp, fit_vp = float(fit[0]), float(fit[1]), float(fit[2])
                    if not (math.isfinite(fit_k) and math.isfinite(fit_fp) and math.isfinite(fit_vp)):
                        continue
                    ok += 1
                    truths_by_param[0].append(k)
                    truths_by_param[1].append(fp)
                    truths_by_param[2].append(vp)
                    preds_by_param[0].append(fit_k)
                    preds_by_param[1].append(fit_fp)
                    preds_by_param[2].append(fit_vp)

                names = ["ktrans", "fp", "vp"]
            else:
                truths_by_param = [[], [], [], []]
                preds_by_param = [[], [], [], []]
                for _ in range(curves):
                    k, ve, vp, fp = _sample_2cxm_truth(rng)
                    ct_clean = model_2cxm_cfit(k, ve, vp, fp, cp, timer)
                    ct_noisy = [float(v + rng.gauss(0.0, noise_std)) for v in ct_clean]
                    fit = model_2cxm_fit(ct_noisy, cp, timer, prefs)
                    fit_k, fit_ve, fit_vp, fit_fp = float(fit[0]), float(fit[1]), float(fit[2]), float(fit[3])
                    if not (math.isfinite(fit_k) and math.isfinite(fit_ve) and math.isfinite(fit_vp) and math.isfinite(fit_fp)):
                        continue
                    ok += 1
                    truths_by_param[0].append(k)
                    truths_by_param[1].append(ve)
                    truths_by_param[2].append(vp)
                    truths_by_param[3].append(fp)
                    preds_by_param[0].append(fit_k)
                    preds_by_param[1].append(fit_ve)
                    preds_by_param[2].append(fit_vp)
                    preds_by_param[3].append(fit_fp)

                names = ["ktrans", "ve", "vp", "fp"]

            fail_rate = 1.0 - (ok / max(1, curves))
            corr_vals: List[float] = []
            med_rel_vals: List[float] = []
            for idx in range(len(names)):
                corr_vals.append(_safe_corr(truths_by_param[idx], preds_by_param[idx]))
                med_rel_vals.append(_median_rel_err(truths_by_param[idx], preds_by_param[idx]))

            corr_text = " ".join(
                f"{n}:{c:.3f}" if math.isfinite(c) else f"{n}:nan" for n, c in zip(names, corr_vals)
            )
            mre_text = " ".join(
                f"{n}:{m:.3f}" if math.isfinite(m) else f"{n}:nan" for n, m in zip(names, med_rel_vals)
            )
            print(
                f"  - profile={profile_name:<16} ok={ok:>4}/{curves:<4} fail={fail_rate:>6.1%} "
                f"corr[{corr_text}] mre[{mre_text}]"
            )


def main() -> int:
    args = _parse_args()
    noise_levels = [float(token.strip()) for token in args.noise_std.split(",") if token.strip()]
    rng = random.Random(int(args.seed))

    targets = [args.model] if args.model != "both" else ["tissue_uptake", "2cxm"]
    for model_name in targets:
        _run_model(model_name, curves=max(1, int(args.curves)), noise_levels=noise_levels, rng=rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
