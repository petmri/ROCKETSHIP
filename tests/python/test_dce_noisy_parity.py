"""Python-vs-MATLAB DCE parity on NOISY data (the real-data gold standard).

Recovering ground truth on noise-free synthetic curves proves little: any
converging optimizer does that. What matters is that the Python port reproduces
MATLAB's fit of the SAME noisy curve, because real data is noisy.

``export_parity_baseline.m`` stores, per primary model and noise level, several
deterministic noisy realizations together with MATLAB's fit of each
(``baseline.dce.noisy``). Here the Python fitter is run on the identical stored
curve and compared to MATLAB's stored fit.

The important subtlety (per project guidance): noisy fits can be unstable, and a
parameter that is not identifiable from the data at a given noise level will make
MATLAB and Python disagree for reasons that are not porting bugs (2CXM is the
worst offender, which is why it is excluded from the stored fixtures entirely).
So parity is gated per parameter on whether MATLAB *itself* recovered that
parameter near ground truth. Non-identifiable parameters are reported as
diagnostics, never asserted. A floor on the number of gated comparisons keeps the
test from passing vacuously if the gate were ever to exclude everything.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import (  # noqa: E402
    model_extended_tofts_fit,
    model_patlak_linear,
    model_tofts_fit,
)

BASELINE = REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json"
TOLERANCES = REPO_ROOT / "tests/contracts/tolerance_profiles.json"

# A parameter counts as identifiable at a given noise level when MATLAB's own fit
# lands within this relative band of ground truth. Chosen to keep well-conditioned
# parameters in-scope at the committed 1%/3% levels while excluding any parameter
# that has become noise-dominated.
IDENTIFIABILITY_REL_BAND = 0.25
IDENTIFIABILITY_ABS_FLOOR = 1e-6

_FITTERS = {
    "tofts": model_tofts_fit,
    "ex_tofts": model_extended_tofts_fit,
    "patlak": model_patlak_linear,
}


def is_identifiable(matlab_value: float, ground_truth: float, *, rel_band: float = IDENTIFIABILITY_REL_BAND,
                    abs_floor: float = IDENTIFIABILITY_ABS_FLOOR) -> bool:
    """True if MATLAB recovered the parameter close enough to ground truth to be a
    meaningful parity reference (i.e. the parameter is determined by the data)."""
    band = max(abs_floor, rel_band * abs(float(ground_truth)))
    return abs(float(matlab_value) - float(ground_truth)) <= band


def _within(actual: float, expected: float, atol: float, rtol: float) -> bool:
    return abs(float(actual) - float(expected)) <= (atol + rtol * abs(float(expected)))


@pytest.mark.unit
@pytest.mark.parity
def test_noisy_data_parity_primary_models() -> None:
    baseline = json.loads(BASELINE.read_text())
    tol = json.loads(TOLERANCES.read_text())["noisy_parity"]
    atol, rtol = float(tol["atol"]), float(tol["rtol"])

    noisy = baseline["dce"].get("noisy")
    assert noisy, "baseline.dce.noisy missing; regenerate with export_parity_baseline.m"

    cp = baseline["dce"]["forward"]["Cp"]
    timer = baseline["dce"]["forward"]["timer"]

    total = 0
    gated = 0
    failures: list[str] = []
    diagnostics: list[str] = []

    for entry in noisy:
        model = entry["model"]
        fitter = _FITTERS.get(model)
        assert fitter is not None, f"no Python fitter mapped for model {model!r}"

        py_fit = fitter(entry["Ct"], cp, timer)
        matlab_fit = entry["fit"]
        names = entry["param_names"]
        gt = entry["ground_truth"]
        sigma = entry["sigma_frac"]
        real = entry.get("realization", "?")

        for i, name in enumerate(names):
            total += 1
            matlab_val = float(matlab_fit[i])
            py_val = float(py_fit[i])
            gt_val = float(gt[i])
            tag = f"{model} sigma={sigma} r{real} {name}"

            if not is_identifiable(matlab_val, gt_val):
                diagnostics.append(
                    f"{tag}: NOT identifiable (matlab={matlab_val:.5g} vs gt={gt_val:.5g}); parity not gated"
                )
                continue

            gated += 1
            if not _within(py_val, matlab_val, atol, rtol):
                abs_err = abs(py_val - matlab_val)
                failures.append(
                    f"{tag}: python={py_val:.8g} matlab={matlab_val:.8g} abs_err={abs_err:.3e} "
                    f"(tol atol={atol:g} rtol={rtol:g})"
                )

    print(f"[NOISY-PARITY] comparisons={total} gated={gated} diagnostics={len(diagnostics)} failures={len(failures)}")
    for line in diagnostics:
        print(f"[NOISY-PARITY][diag] {line}")

    # Anti-vacuous guard: the gate must actually be exercising real parity checks,
    # not silently excluding everything (which would make this test always pass).
    assert total > 0, "no noisy comparisons found in baseline"
    assert gated >= max(1, total // 2), (
        f"only {gated}/{total} noisy comparisons were gated; identifiability gate is "
        "excluding too much to be a meaningful parity test"
    )

    if failures:
        pytest.fail(
            "Python DCE fits diverged from MATLAB on identical noisy curves:\n  "
            + "\n  ".join(failures)
        )


@pytest.mark.unit
def test_identifiability_gate_excludes_unstable_parameters() -> None:
    """The gate must admit near-ground-truth recoveries and reject noise-dominated
    ones, so the parity test above cannot be fooled by an unstable parameter."""
    # Within 25% of ground truth -> identifiable.
    assert is_identifiable(0.031, 0.03)
    assert is_identifiable(0.036, 0.03)  # +20%
    assert is_identifiable(0.25 * 1.2, 0.25)
    # Beyond the band -> excluded (this is the 2CXM/high-noise failure mode).
    assert not is_identifiable(0.05, 0.03)  # +67%
    assert not is_identifiable(0.0, 0.04)   # collapsed to lower bound
    assert not is_identifiable(0.5, 0.25)   # +100%
