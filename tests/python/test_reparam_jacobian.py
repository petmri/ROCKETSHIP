"""Analytic-Jacobian guard for the reparameterized accelerated 2CXM / 2CUM kernels.

The cpufit/gpufit kernels fit ``E = Ktrans/Fp`` and emit analytic Jacobians built from the
O(N) exponential-recurrence convolution primitive. This test mirrors that exact math in
float64 Python and checks every Jacobian column against a central difference of the *same
discrete forward model* on the real OSIPI low-flow DRO grid. It guards the derivation: if the
kernel formulas (or this mirror) drift, the columns stop matching.

The compiled kernels themselves are exercised end-to-end by the OSIPI cpufit/gpufit sweeps
(``test_osipi_pycpufit.py`` / ``test_osipi_pygpufit.py``). Full derivation:
``docs/project-management/projects/osipi-verification/STATUS.md`` and
``docs/project-management/projects/osipi-verification/verify_analytic_jac.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from osipi_fast_backend_helpers import DCE_DATA_DIR, FAST_BACKEND_CASES, _rows, _series

# Real low-flow 2CXM DRO grid (600 pts, dt=0.5s) -- the regime where the old fixed-step
# numerical Jacobian was corrupted near the Ktrans=Fp pole.
_row = _rows(DCE_DATA_DIR / FAST_BACKEND_CASES["2cxm"]["dataset"])[0]
T = np.asarray(_series(_row["t"]), dtype=np.float64)
CP = np.asarray(_series(_row["cp_aif"]), dtype=np.float64)
DT = np.diff(T)


def _conv_G_Gp(kappa: float):
    """G[k] = discrete trapezoid conv of Cp with exp(-kappa*(t_k - tau)); Gp[k] = dG[k]/dkappa."""
    n = T.size
    g = np.zeros(n)
    gp = np.zeros(n)
    for k in range(1, n):
        d = DT[k - 1]
        decay = math.exp(-kappa * d)
        gp[k] = decay * (gp[k - 1] - d * g[k - 1] - 0.5 * d * d * CP[k - 1])
        g[k] = decay * g[k - 1] + 0.5 * d * (CP[k - 1] * decay + CP[k])
    return g, gp


def _cum_U():
    u = np.zeros(T.size)
    u[1:] = np.cumsum(0.5 * DT * (CP[1:] + CP[:-1]))
    return u


# ---- 2CXM (E, ve, vp, Fp) ----
def _cxm_scalars(E, ve, vp, Fp):
    one_me = 1.0 - E
    ps = Fp * E / one_me
    rp = (ps + Fp) / vp
    re = ps / ve
    rb = Fp / vp
    a = rp + re
    dr = math.sqrt(max(a * a - 4.0 * re * rb, 0.0))
    kpos = 0.5 * (a + dr)
    kneg = 0.5 * (a - dr)
    dr_safe = dr if dr > 1e-12 else 1e-12
    eneg = (kpos - rb) / dr_safe
    return dict(ps=ps, rp=rp, re=re, rb=rb, a=a, dr=dr_safe, kpos=kpos, kneg=kneg, eneg=eneg)


def _cxm_forward(E, ve, vp, Fp):
    s = _cxm_scalars(E, ve, vp, Fp)
    gpos, _ = _conv_G_Gp(s["kpos"])
    gneg, _ = _conv_G_Gp(s["kneg"])
    return Fp * ((1.0 - s["eneg"]) * gpos + s["eneg"] * gneg)


def _cxm_jacobian(E, ve, vp, Fp):
    s = _cxm_scalars(E, ve, vp, Fp)
    gpos, gppos = _conv_G_Gp(s["kpos"])
    gneg, gpneg = _conv_G_Gp(s["kneg"])
    ps, rp, re, rb, a, dr, kpos, eneg = (
        s["ps"], s["rp"], s["re"], s["rb"], s["a"], s["dr"], s["kpos"], s["eneg"]
    )
    one_me = 1.0 - E
    dps = {"E": Fp / one_me**2, "ve": 0.0, "vp": 0.0, "Fp": E / one_me}
    cols = {}
    for p, is_ve, is_vp, is_fp in (("E", 0, 0, 0), ("ve", 1, 0, 0), ("vp", 0, 1, 0), ("Fp", 0, 0, 1)):
        drp = dps[p] / vp + is_fp / vp - is_vp * (ps + Fp) / vp**2
        dre = dps[p] / ve - is_ve * ps / ve**2
        drb = is_fp / vp - is_vp * Fp / vp**2
        da = drp + dre
        dc = dre * rb + re * drb
        ddr = (a * da - 2.0 * dc) / dr
        dkpos = 0.5 * (da + ddr)
        dkneg = 0.5 * (da - ddr)
        deneg = ((dkpos - drb) * dr - (kpos - rb) * ddr) / dr**2
        col = Fp * (deneg * (gneg - gpos) + (1.0 - eneg) * gppos * dkpos + eneg * gpneg * dkneg)
        if p == "Fp":
            col = col + ((1.0 - eneg) * gpos + eneg * gneg)
        cols[p] = col
    return cols


# ---- 2CUM (E, vp, Fp) ----
def _cum_forward(E, vp, Fp):
    rp = Fp / (vp * (1.0 - E))
    g, _ = _conv_G_Gp(rp)
    return E * Fp * _cum_U() + Fp * (1.0 - E) * g


def _cum_jacobian(E, vp, Fp):
    one_me = 1.0 - E
    rp = Fp / (vp * one_me)
    g, gp = _conv_G_Gp(rp)
    u = _cum_U()
    return {
        "E": Fp * u - Fp * g + Fp * rp * gp,
        "vp": -Fp * one_me * rp * gp / vp,
        "Fp": E * u + one_me * g + one_me * rp * gp,
    }


def _central(forward, base: dict, key: str) -> np.ndarray:
    h = 1e-6 * abs(base[key]) + 1e-12
    hi = dict(base); hi[key] = base[key] + h
    lo = dict(base); lo[key] = base[key] - h
    return (forward(**hi) - forward(**lo)) / (2.0 * h)


def _l2_rel(analytic: np.ndarray, numeric: np.ndarray) -> float:
    denom = float(np.linalg.norm(numeric))
    return float(np.linalg.norm(analytic - numeric) / (denom if denom > 0 else 1.0))


@pytest.mark.parametrize("E", [0.30, 0.60, 0.85])
def test_cxm_analytic_jacobian_matches_central(E: float) -> None:
    base = dict(E=E, ve=0.15, vp=0.03, Fp=8.333e-4)
    cols = _cxm_jacobian(**base)
    for key in base:
        num = _central(_cxm_forward, base, key)
        assert _l2_rel(cols[key], num) < 1e-6, f"2CXM dC/d{key} at E={E}"


@pytest.mark.parametrize("E", [0.30, 0.60, 0.85])
def test_cum_analytic_jacobian_matches_central(E: float) -> None:
    base = dict(E=E, vp=0.03, Fp=8.333e-4)
    cols = _cum_jacobian(**base)
    for key in base:
        num = _central(_cum_forward, base, key)
        assert _l2_rel(cols[key], num) < 1e-6, f"2CUM dC/d{key} at E={E}"
