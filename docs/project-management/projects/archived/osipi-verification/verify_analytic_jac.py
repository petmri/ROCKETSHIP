"""Derive + verify analytic Jacobians for reparameterized 2CXM and 2CUM.

Reparam: E = Ktrans/Fp (extraction fraction, in (0,1)), so Ktrans = E*Fp and
PS = Fp*E/(1-E).  The forward model is the *discrete trapezoidal* convolution the
kernel actually fits, so we differentiate the discrete recurrence (not the continuous
integral) -- that is what LM's Jacobian should be. Verify each column against a
high-accuracy central difference of the same forward model in float64.

This standalone script is the human-readable derivation. The enforced CI guard is
``tests/python/test_reparam_jacobian.py`` (same math, as assertions).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np

# Repo root = five parents up from docs/project-management/projects/osipi-verification/.
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tests" / "python"))
from osipi_fast_backend_helpers import FAST_BACKEND_CASES, DCE_DATA_DIR, _rows, _series

row = _rows(DCE_DATA_DIR / FAST_BACKEND_CASES["2cxm"]["dataset"])[0]
T = np.array(_series(row["t"]), dtype=np.float64)
Cp = np.array(_series(row["cp_aif"]), dtype=np.float64)
dT = np.diff(T)


# ---------- O(N) recurrences (discrete trapezoid) ----------
def conv_G_Gp(kappa):
    """G_k = trapz conv of Cp with exp(-kappa*(t_k - tau)); Gp_k = dG_k/dkappa. O(N)."""
    N = T.size
    G = np.zeros(N); Gp = np.zeros(N)
    for k in range(1, N):
        d = dT[k - 1]
        decay = math.exp(-kappa * d)
        s = 0.5 * d * (Cp[k - 1] * decay + Cp[k])
        G[k] = decay * G[k - 1] + s
        # dG_k/dkappa
        Gp[k] = decay * (Gp[k - 1] - d * G[k - 1] - 0.5 * d * d * Cp[k - 1])
    return G, Gp


def cum_U():
    U = np.zeros(T.size)
    U[1:] = np.cumsum(0.5 * dT * (Cp[1:] + Cp[:-1]))
    return U


# ================= 2CXM =================
def cxm_internals(E, ve, vp, Fp):
    oneME = 1.0 - E
    PS = Fp * E / oneME
    dPS_dE = Fp / oneME**2
    dPS_dFp = E / oneME
    rp = (PS + Fp) / vp; re = PS / ve; rb = Fp / vp
    d = {}  # partials keyed by param
    d["rp"] = {"E": dPS_dE / vp, "ve": 0.0, "vp": -(PS + Fp) / vp**2, "Fp": (dPS_dFp + 1.0) / vp}
    d["re"] = {"E": dPS_dE / ve, "ve": -PS / ve**2, "vp": 0.0, "Fp": dPS_dFp / ve}
    d["rb"] = {"E": 0.0, "ve": 0.0, "vp": -Fp / vp**2, "Fp": 1.0 / vp}
    a = rp + re; c = re * rb
    da = {p: d["rp"][p] + d["re"][p] for p in ("E", "ve", "vp", "Fp")}
    dc = {p: d["re"][p] * rb + re * d["rb"][p] for p in ("E", "ve", "vp", "Fp")}
    disc = a * a - 4.0 * c
    Dr = math.sqrt(disc)
    dDr = {p: (a * da[p] - 2.0 * dc[p]) / Dr for p in ("E", "ve", "vp", "Fp")}
    Kpos = 0.5 * (a + Dr); Kneg = 0.5 * (a - Dr)
    dKpos = {p: 0.5 * (da[p] + dDr[p]) for p in da}
    dKneg = {p: 0.5 * (da[p] - dDr[p]) for p in da}
    Eneg = (Kpos - rb) / Dr
    dEneg = {p: ((dKpos[p] - d["rb"][p]) * Dr - (Kpos - rb) * dDr[p]) / Dr**2 for p in da}
    return dict(Kpos=Kpos, Kneg=Kneg, Eneg=Eneg, dKpos=dKpos, dKneg=dKneg, dEneg=dEneg, disc=disc)


def cxm_forward(E, ve, vp, Fp):
    it = cxm_internals(E, ve, vp, Fp)
    Gp_pos, _ = conv_G_Gp(it["Kpos"])
    Gn, _ = conv_G_Gp(it["Kneg"])
    return Fp * ((1.0 - it["Eneg"]) * Gp_pos + it["Eneg"] * Gn)


def cxm_jac_analytic(E, ve, vp, Fp):
    it = cxm_internals(E, ve, vp, Fp)
    G_pos, Gp_pos = conv_G_Gp(it["Kpos"])
    G_neg, Gp_neg = conv_G_Gp(it["Kneg"])
    Eneg = it["Eneg"]
    cols = {}
    for p in ("E", "ve", "vp", "Fp"):
        col = Fp * (it["dEneg"][p] * (G_neg - G_pos)
                    + (1.0 - Eneg) * Gp_pos * it["dKpos"][p]
                    + Eneg * Gp_neg * it["dKneg"][p])
        if p == "Fp":
            col = col + ((1.0 - Eneg) * G_pos + Eneg * G_neg)
        cols[p] = col
    return cols


# ================= 2CUM (tissue uptake) =================
def cum_forward(E, vp, Fp):
    rp = Fp / (vp * (1.0 - E))
    G, _ = conv_G_Gp(rp)
    U = cum_U()
    return E * Fp * U + Fp * (1.0 - E) * G


def cum_jac_analytic(E, vp, Fp):
    oneME = 1.0 - E
    rp = Fp / (vp * oneME)
    G, Gp = conv_G_Gp(rp)
    U = cum_U()
    return {
        "E":  Fp * U - Fp * G + Fp * rp * Gp,          # drp/dE = rp/(1-E)
        "vp": -Fp * oneME * rp * Gp / vp,              # drp/dvp = -rp/vp
        "Fp": E * U + oneME * G + oneME * rp * Gp,      # drp/dFp = rp/Fp
    }


def central(fwd, base, key, rel=1e-6):
    x = dict(base); h = rel * abs(base[key]) + 1e-12
    x[key] = base[key] + h; fp = fwd(**x)
    x[key] = base[key] - h; fm = fwd(**x)
    return (fp - fm) / (2.0 * h)


def report(name, params, jac_fn, fwd_fn):
    print(f"\n=== {name}  params={params} ===")
    ana = jac_fn(**params)
    for k in params:
        num = central(fwd_fn, params, k)
        a = ana[k]
        denom = np.maximum(np.abs(num), 1e-12)
        rel = np.max(np.abs(a - num) / denom)
        print(f"  dC/d{k:<3}  max|analytic-central|/|central| = {rel:.2e}   "
              f"(||col||={np.linalg.norm(a):.4g})")


for E in (0.30, 0.60, 0.85):
    report(f"2CXM E={E}", dict(E=E, ve=0.15, vp=0.03, Fp=8.333e-4),
           cxm_jac_analytic, cxm_forward)
for E in (0.30, 0.60, 0.85):
    report(f"2CUM E={E}", dict(E=E, vp=0.03, Fp=8.333e-4),
           cum_jac_analytic, cum_forward)
print("\n(analytic-vs-central agreement ~1e-6 or better => Jacobian formulas verified)")
