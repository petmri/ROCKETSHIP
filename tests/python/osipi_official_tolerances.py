"""OSIPI official acceptance tolerances -- the hard gate for OSIPI DCE tests.

These are OSIPI's own published per-parameter pass/fail tolerances, transcribed
verbatim into ``tests/data/osipi/reference/osipi_official_tolerances.json`` from the
OSIPI test suite (``test/DCEmodels/DCEmodels_data.py`` @ commit ``23d3714``). Every
contributor implementation in the OSIPI suite is asserted against them with
``np.testing.assert_allclose(measured, reference, atol=a_tol, rtol=r_tol)``.

We gate on these (round, method-agnostic, reproducible) rather than on the imported
peer-error *spread* (``osipi_peer_error_summary.json``), whose DCE maximum is
near-circular for the LEK-derived models ROCKETSHIP ports (2cxm, tissue_uptake).
The peer spread is reported as a non-gating signal by ``run_osipi_reliability.py`` and
the ``osipi_summary.md`` generator.

Method keys: ``tofts``, ``etofts``, ``patlak``, ``2CXM``, ``2CUM``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = json.loads(
    (REPO_ROOT / "tests" / "data" / "osipi" / "reference" / "osipi_official_tolerances.json").read_text()
)["DCEmodels"]


def official_tolerance(method: str, param: str) -> Tuple[float, float]:
    """Return (a_tol, r_tol) for a model/parameter."""
    entry = _OFFICIAL[method][param]
    return float(entry["a_tol"]), float(entry["r_tol"])


def official_abs_tol(method: str, param: str, reference: float) -> float:
    """Effective absolute tolerance for one case: ``a_tol + r_tol * |reference|``.

    Matches the ``assert_allclose(atol, rtol)`` pass criterion OSIPI uses.
    """
    a_tol, r_tol = official_tolerance(method, param)
    return a_tol + r_tol * abs(float(reference))
