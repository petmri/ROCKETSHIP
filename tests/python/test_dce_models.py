"""Unit tests for Python DCE model ports."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rocketship import (  # noqa: E402
    model_2cxm_cfit,
    model_2cxm_fit,
    model_extended_tofts_cfit,
    model_fxr_cfit,
    model_fxr_fit,
    model_patlak_cfit,
    model_patlak_fit,
    model_patlak_linear,
    model_tissue_uptake_cfit,
    model_tissue_uptake_fit,
    model_tofts_cfit,
    model_tofts_fit,
    model_vp_cfit,
    model_vp_fit,
)


def _within_tol(actual: float, expected: float, atol: float, rtol: float) -> bool:
    return abs(actual - expected) <= (atol + rtol * abs(expected))


class TestDceModels(unittest.TestCase):
    def test_tofts_zero_cp_returns_zero(self) -> None:
        timer = [0.0, 0.1, 0.2, 0.3]
        cp = [0.0, 0.0, 0.0, 0.0]
        out = model_tofts_cfit(0.03, 0.25, cp, timer)
        self.assertEqual(len(out), len(timer))
        self.assertTrue(all(v == 0.0 for v in out))

    def test_tofts_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["forward_exact"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        expected = baseline["dce"]["forward"]["tofts"]

        # Pull parameters from baseline inverse fit output so this test remains
        # aligned with baseline fixture generation.
        ktrans = float(baseline["dce"]["inverse"]["tofts_fit"][0])
        ve = float(baseline["dce"]["inverse"]["tofts_fit"][1])

        actual = model_tofts_cfit(ktrans, ve, cp, timer)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_patlak_zero_cp_returns_zero(self) -> None:
        timer = [0.0, 0.1, 0.2, 0.3]
        cp = [0.0, 0.0, 0.0, 0.0]
        out = model_patlak_cfit(0.03, 0.05, cp, timer)
        self.assertEqual(len(out), len(timer))
        self.assertTrue(all(v == 0.0 for v in out))

    def test_patlak_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["forward_exact"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        expected = baseline["dce"]["forward"]["patlak"]

        # Pull parameters from baseline inverse fit output so this test remains
        # aligned with baseline fixture generation.
        ktrans = float(baseline["dce"]["inverse"]["patlak_linear"][0])
        vp = float(baseline["dce"]["inverse"]["patlak_linear"][1])

        actual = model_patlak_cfit(ktrans, vp, cp, timer)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_extended_tofts_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["forward_exact"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        expected = baseline["dce"]["forward"]["extended_tofts"]

        fit_vals = baseline["dce"]["inverse"]["extended_tofts_fit"]
        ktrans = float(fit_vals[0])
        ve = float(fit_vals[1])
        vp = float(fit_vals[2])

        actual = model_extended_tofts_cfit(ktrans, ve, vp, cp, timer)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_vp_forward_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["forward_exact"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        vp = float(baseline["dce"]["params"]["vp"])
        expected = baseline["dce"]["forward"]["vp"]

        actual = model_vp_cfit(vp, cp, timer)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_tissue_uptake_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["forward_exact"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        params = baseline["dce"]["params"]
        ktrans = float(params["ktrans"])
        fp = float(params["fp"])
        tp = float(params["tp"])
        expected = baseline["dce"]["forward"]["tissue_uptake"]

        actual = model_tissue_uptake_cfit(ktrans, fp, tp, cp, timer)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_twocxm_forward_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["forward_exact"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        params = baseline["dce"]["params"]
        ktrans = float(params["ktrans"])
        ve = float(params["ve"])
        vp = float(params["vp"])
        fp = float(params["fp"])
        expected = baseline["dce"]["forward"]["twocxm"]

        actual = model_2cxm_cfit(ktrans, ve, vp, fp, cp, timer)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_fxr_forward_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["forward_exact"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        params = baseline["dce"]["params"]
        expected = baseline["dce"]["forward"]["fxr"]

        actual = model_fxr_cfit(
            float(params["ktrans"]),
            float(params["ve"]),
            float(params["tau"]),
            cp,
            timer,
            float(params["R1o"]),
            float(params["R1i"]),
            float(params["r1"]),
            float(params["fw"]),
        )
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_patlak_linear_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["fit_recovery"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        ct = baseline["dce"]["forward"]["patlak"]
        expected = baseline["dce"]["inverse"]["patlak_linear"]

        actual = model_patlak_linear(ct, cp, timer)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_patlak_fit_recovers_forward_params_and_improves_sse(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        ct = baseline["dce"]["forward"]["patlak"]
        params = baseline["dce"]["params"]
        ktrans_true = float(params["ktrans"])
        vp_true = float(params["vp"])

        fit_nonlinear = model_patlak_fit(ct, cp, timer)
        fit_linear = model_patlak_linear(ct, cp, timer)

        self.assertTrue(_within_tol(float(fit_nonlinear[0]), ktrans_true, atol=5e-4, rtol=0.05))
        self.assertTrue(_within_tol(float(fit_nonlinear[1]), vp_true, atol=5e-3, rtol=0.1))
        self.assertLessEqual(float(fit_nonlinear[2]), float(fit_linear[2]) + 1e-12)

    def test_tofts_fit_inverse_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["fit_recovery"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        ct = baseline["dce"]["forward"]["tofts"]
        expected = baseline["dce"]["inverse"]["tofts_fit"]

        actual = model_tofts_fit(ct, cp, timer)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_vp_fit_inverse_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["fit_recovery"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        ct = baseline["dce"]["forward"]["vp"]
        expected = baseline["dce"]["inverse"]["vp_fit"]

        actual = model_vp_fit(ct, cp, timer)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_tissue_uptake_fit_inverse_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["fit_recovery"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        ct = baseline["dce"]["forward"]["tissue_uptake"]
        expected = baseline["dce"]["inverse"]["tissue_uptake_fit"]

        actual = model_tissue_uptake_fit(ct, cp, timer)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_twocxm_fit_inverse_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["fit_recovery"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        ct = baseline["dce"]["forward"]["twocxm"]
        expected = baseline["dce"]["inverse"]["twocxm_fit"]

        actual = model_2cxm_fit(ct, cp, timer)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_fxr_fit_inverse_matches_matlab_baseline_profile(self) -> None:
        baseline = json.loads((REPO_ROOT / "tests/contracts/baselines/matlab_reference_v1.json").read_text())
        tolerances = json.loads((REPO_ROOT / "tests/contracts/tolerance_profiles.json").read_text())
        tol = tolerances["fit_recovery"]

        timer = baseline["dce"]["forward"]["timer"]
        cp = baseline["dce"]["forward"]["Cp"]
        ct = baseline["dce"]["forward"]["fxr"]
        params = baseline["dce"]["params"]
        expected = baseline["dce"]["inverse"]["fxr_fit"]

        actual = model_fxr_fit(
            ct,
            cp,
            timer,
            float(params["R1o"]),
            float(params["R1i"]),
            float(params["r1"]),
            float(params["fw"]),
        )

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(
                _within_tol(float(a), float(e), float(tol["atol"]), float(tol["rtol"])),
                msg=f"Mismatch: actual={a} expected={e}",
            )

    def test_tissue_uptake_fit_recovers_synthetic_no_and_low_noise(self) -> None:
        timer = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.2]
        cp = [0.0, 0.4, 0.9, 1.2, 1.0, 0.82, 0.63, 0.49, 0.35, 0.25, 0.18]
        ktrans_true = 0.045
        fp_true = 0.36
        tp_true = 0.18

        clean = model_tissue_uptake_cfit(ktrans_true, fp_true, tp_true, cp, timer)
        vp_true = (fp_true + (ktrans_true * fp_true / (fp_true - ktrans_true))) * tp_true

        rng = random.Random(7)
        for noise_std in (0.0, 2.5e-4):
            noisy = [float(v + rng.gauss(0.0, noise_std)) for v in clean]
            fit = model_tissue_uptake_fit(noisy, cp, timer)

            self.assertTrue(_within_tol(float(fit[0]), ktrans_true, atol=2e-3, rtol=0.1))
            self.assertTrue(_within_tol(float(fit[1]), fp_true, atol=3e-2, rtol=0.15))
            self.assertTrue(_within_tol(float(fit[2]), vp_true, atol=2e-2, rtol=0.2))

    def test_twocxm_fit_recovers_synthetic_no_and_low_noise(self) -> None:
        timer = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.2]
        cp = [0.0, 0.4, 0.9, 1.2, 1.0, 0.82, 0.63, 0.49, 0.35, 0.25, 0.18]
        ktrans_true = 0.03
        ve_true = 0.24
        vp_true = 0.045
        fp_true = 0.31

        clean = model_2cxm_cfit(ktrans_true, ve_true, vp_true, fp_true, cp, timer)

        rng = random.Random(17)
        for noise_std in (0.0, 2.5e-4):
            noisy = [float(v + rng.gauss(0.0, noise_std)) for v in clean]
            fit = model_2cxm_fit(noisy, cp, timer)

            self.assertTrue(_within_tol(float(fit[0]), ktrans_true, atol=4e-3, rtol=0.2))
            self.assertTrue(_within_tol(float(fit[1]), ve_true, atol=3e-2, rtol=0.2))
            self.assertTrue(_within_tol(float(fit[2]), vp_true, atol=1.5e-2, rtol=0.25))
            self.assertTrue(_within_tol(float(fit[3]), fp_true, atol=3e-2, rtol=0.2))


if __name__ == "__main__":
    unittest.main()
