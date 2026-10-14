"""OSIPI-labeled SI-to-concentration reliability tests."""

from __future__ import annotations

import pytest

from osipi_si_to_conc_helpers import compute_si_to_conc_metrics, evaluate_si_to_conc_gate, peer_si_to_conc_metrics


@pytest.mark.osipi
def test_osipi_si_to_conc_error_distribution_matches_peer_results() -> None:
    ours = compute_si_to_conc_metrics()
    peer = peer_si_to_conc_metrics()
    passed, checks, limits = evaluate_si_to_conc_gate(ours, peer, epsilon=1e-12)

    assert passed, (
        "OSIPI SI-to-concentration gate failed: "
        f"ours(mae={ours['mae']:.8g}, p95={ours['p95_abs_error']:.8g}, max={ours['max_abs_error']:.8g}) "
        f"limits(mae<={limits['mae']:.8g}, p95<={limits['p95_abs_error']:.8g}, max<={limits['max_abs_error']:.8g}) "
        f"checks={checks}"
    )
