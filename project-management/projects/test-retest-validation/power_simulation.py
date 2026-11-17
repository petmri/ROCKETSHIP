#!/usr/bin/env python3
"""Power of the wSD-ratio non-inferiority test at the cohort sizes (proposal 6 of the plan).

Data-free: two pipelines measure the same simulated scans, with a share of the rescan
variance in common (biology, acquisition) and the rest their own. Prints, per cohort size
and shared fraction, how often the 95% Pitman upper limit falls below the margin when the
true wSD ratio is 1.00 and 1.05. Seeded; reproduces the table in the plan.

    .venv/bin/python project-management/projects/test-retest-validation/power_simulation.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

import repeatability as rp  # noqa: E402

PAIRS = (19, 24, 43)  # RIDER, QIN-GBM, pooled
SHARED = (0.8, 0.95, 0.99, 0.999)
MARGINS = (1.1, 1.25)
RATIOS = (1.0, 1.05)
REPLICATES = 1500


def two_pipelines(n, true_ratio, shared, seed):
    """Test-retest pairs of two pipelines on the same subjects (as in test_repeatability.py)."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n, 2))
    own_ref = rng.standard_normal((n, 2))
    own_new = rng.standard_normal((n, 2))
    truth = 5.0 + rng.standard_normal(n)[:, None]
    ref = truth + math.sqrt(shared) * common + math.sqrt(1.0 - shared) * own_ref
    new = truth + math.sqrt(shared) * common + math.sqrt(true_ratio**2 - shared) * own_new
    return new[:, 0], new[:, 1], ref[:, 0], ref[:, 1]


def main() -> int:
    print("P(conclude non-inferior): Pitman 95% upper limit < margin; true ratio 1.00 [1.05]")
    for n in PAIRS:
        for shared in SHARED:
            cells = []
            for margin in MARGINS:
                rates = []
                for ratio in RATIOS:
                    hits = 0
                    for rep in range(REPLICATES):
                        new_a, new_b, ref_a, ref_b = two_pipelines(n, ratio, shared, seed=90000 + rep)
                        _, _, high = rp.pitman_sd_ratio_ci(new_b - new_a, ref_b - ref_a)
                        hits += high < margin
                    rates.append(hits / REPLICATES)
                cells.append(f"margin {margin}: {rates[0]:.2f} [{rates[1]:.2f}]")
            print(f"n={n:2d} shared={shared:<5}  " + "   ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
