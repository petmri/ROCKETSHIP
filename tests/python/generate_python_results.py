#!/usr/bin/env python3
"""Generate Python model outputs in parity-runner JSON format."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("tests/contracts/baselines/matlab_reference_v1.json"),
        help="MATLAB baseline JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/python/python_results.json"),
        help="Output JSON path for Python results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))

    from rocketship import (  # pylint: disable=import-outside-toplevel
        dsc_convolution_ssvd,
        import_aif,
        model_2cxm_cfit,
        model_2cxm_fit,
        model_extended_tofts_cfit,
        model_fxr_cfit,
        model_fxr_fit,
        model_patlak_cfit,
        model_patlak_linear,
        model_tissue_uptake_cfit,
        model_tissue_uptake_fit,
        model_tofts_cfit,
        model_tofts_fit,
        model_vp_cfit,
        model_vp_fit,
        previous_aif,
        t1_fa_linear_fit,
        t2_linear_fast,
    )
    from dsc_helpers import matlab_reshape_linspace  # pylint: disable=import-outside-toplevel

    baseline = json.loads(args.baseline.read_text())

    timer = baseline["dce"]["forward"]["timer"]
    cp = baseline["dce"]["forward"]["Cp"]

    # Use MATLAB-fit values from baseline to avoid hard-coding fixture internals.
    tofts_fit = baseline["dce"]["inverse"]["tofts_fit"]
    ktrans = float(tofts_fit[0])
    ve = float(tofts_fit[1])

    ex_tofts_fit = baseline["dce"]["inverse"]["extended_tofts_fit"]
    ex_ktrans = float(ex_tofts_fit[0])
    ex_ve = float(ex_tofts_fit[1])
    ex_vp = float(ex_tofts_fit[2])

    patlak_fit = baseline["dce"]["inverse"]["patlak_linear"]
    patlak_ktrans = float(patlak_fit[0])
    patlak_vp = float(patlak_fit[1])

    patlak_forward = model_patlak_cfit(patlak_ktrans, patlak_vp, cp, timer)

    dce_params = baseline["dce"].get("params", {})
    dce_ktrans = float(dce_params.get("ktrans", patlak_ktrans))
    dce_ve = float(dce_params.get("ve", float(tofts_fit[1])))
    dce_vp = float(dce_params.get("vp", patlak_vp))
    tissue_uptake_fp = float(dce_params.get("fp", 0.15))
    tissue_uptake_tp = float(dce_params.get("tp", dce_vp / tissue_uptake_fp))
    dce_tau = float(dce_params.get("tau", 0.08))
    dce_r1o = float(dce_params.get("R1o", 1.3))
    dce_r1i = float(dce_params.get("R1i", 0.65))
    dce_r1 = float(dce_params.get("r1", 3.4))
    dce_fw = float(dce_params.get("fw", 0.8))

    # Match the synthetic DSC fixture used in MATLAB export_parity_baseline.m
    mean_aif = [0.0 + (1.1 / 13.0) * i for i in range(14)]
    bolus_time = 3  # MATLAB 1-based index semantics
    time_vect = [0.0 + 0.1 * i for i in range(19)]
    concentration_array = matlab_reshape_linspace(0.05, 0.6, 2 * 2 * len(time_vect), (2, 2, len(time_vect)))

    import_aif_out = import_aif(mean_aif, bolus_time, time_vect, concentration_array, 3.4, 0.03)
    previous_aif_out = previous_aif(
        import_aif_out[0], import_aif_out[3], bolus_time, import_aif_out[1], import_aif_out[2]
    )

    # Match synthetic DSC deconvolution fixture from MATLAB export_parity_baseline.m
    time_index = list(range(10))
    ssvd_concentration = []
    for ix in range(2):
        row = []
        for iy in range(2):
            trace = [math.exp(-(((t - (2 + (ix + 1) + (iy + 1) / 2.0)) ** 2) / 6.0)) for t in time_index]
            row.append(trace)
        ssvd_concentration.append(row)
    ssvd_aif = [math.exp(-(((t - 2) ** 2) / 4.0)) for t in time_index]
    ssvd_out = dsc_convolution_ssvd(ssvd_concentration, ssvd_aif, 0.1, 0.73, 1.04, 20, 1)

    # Match synthetic parametric fixture from MATLAB export_parity_baseline.m
    te = [10.0, 20.0, 40.0, 60.0]
    true_t2 = 85.0
    rho = 900.0
    si_t2 = [rho * math.exp(-t / true_t2) for t in te]

    fa = [2.0, 5.0, 10.0, 15.0]
    tr = 8.0
    true_t1 = 1300.0
    m0 = 1100.0
    theta = [f * (math.pi / 180.0) for f in fa]
    si_t1 = [
        m0
        * ((1.0 - math.exp(-tr / true_t1)) * math.sin(th))
        / (1.0 - math.exp(-tr / true_t1) * math.cos(th))
        for th in theta
    ]

    results = {
        "meta": {
            "source": "generate_python_results.py",
            "baseline": str(args.baseline),
            "models": [
                "model_tofts_cfit",
                "model_extended_tofts_cfit",
                "model_patlak_cfit",
                "model_patlak_linear",
                "model_tofts_fit",
                "model_vp_cfit",
                "model_tissue_uptake_cfit",
                "model_2cxm_cfit",
                "model_vp_fit",
                "model_tissue_uptake_fit",
                "model_2cxm_fit",
                "model_fxr_cfit",
                "model_fxr_fit",
                "dsc_convolution_ssvd",
                "t2_linear_fast",
                "t1_fa_linear_fit",
            ],
        },
        "results": {
            "tofts_forward": model_tofts_cfit(ktrans, ve, cp, timer),
            "extended_tofts_forward": model_extended_tofts_cfit(ex_ktrans, ex_ve, ex_vp, cp, timer),
            "patlak_forward": patlak_forward,
            "vp_forward": model_vp_cfit(dce_vp, cp, timer),
            "tissue_uptake_forward": model_tissue_uptake_cfit(
                dce_ktrans, tissue_uptake_fp, tissue_uptake_tp, cp, timer
            ),
            "twocxm_forward": model_2cxm_cfit(
                dce_ktrans,
                dce_ve,
                dce_vp,
                tissue_uptake_fp,
                cp,
                timer,
            ),
            "fxr_forward": model_fxr_cfit(
                dce_ktrans,
                dce_ve,
                dce_tau,
                cp,
                timer,
                dce_r1o,
                dce_r1i,
                dce_r1,
                dce_fw,
            ),
            "patlak_linear_inverse": model_patlak_linear(patlak_forward, cp, timer),
            "tofts_fit_inverse": model_tofts_fit(
                baseline["dce"]["forward"]["tofts"],
                cp,
                timer,
            ),
            "vp_fit_inverse": model_vp_fit(
                baseline["dce"]["forward"]["vp"],
                cp,
                timer,
            ),
            "tissue_uptake_fit_inverse": model_tissue_uptake_fit(
                baseline["dce"]["forward"]["tissue_uptake"],
                cp,
                timer,
            ),
            "twocxm_fit_inverse": model_2cxm_fit(
                baseline["dce"]["forward"]["twocxm"],
                cp,
                timer,
            ),
            "fxr_fit_inverse": model_fxr_fit(
                baseline["dce"]["forward"]["fxr"],
                cp,
                timer,
                dce_r1o,
                dce_r1i,
                dce_r1,
                dce_fw,
            ),
            "import_aif_truncation": {
                "meanAIF_adjusted": import_aif_out[0],
                "time_vect": import_aif_out[1],
                "concentration_array": import_aif_out[2],
                "meanSignal": import_aif_out[3],
            },
            "previous_aif_truncation": {
                "meanAIF_adjusted": previous_aif_out[0],
                "time_vect": previous_aif_out[1],
                "concentration_array": previous_aif_out[2],
            },
            "ssvd_deconvolution": {
                "CBF": ssvd_out[0],
                "CBV": ssvd_out[1],
                "MTT": ssvd_out[2],
            },
            "t2_linear_fast": t2_linear_fast(te, si_t2),
            "t1_fa_linear_fit": t1_fa_linear_fit(fa, si_t1, tr),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote Python results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
