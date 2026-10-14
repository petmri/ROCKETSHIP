"""Backend-equivalence gate: CPU vs accelerated Stage-D on frozen real-data arrays.

The parity suite grades Python against MATLAB but never required `cpu` and `cpufit` to
agree with *each other*, so a backend-specific defect could only be found by noticing a
map looked wrong. This gates it directly, on Stage-B arrays frozen from
``RUNNER_DATA/sub-1101743/ses-01`` (see ``freeze_stage_b_backend_fixture.py``) so both
backends provably see identical inputs and CI needs no network drive.

**What is and is not required.** Raw whole-sample correlation is *not* a usable gate here:
this is a low-enhancement BBB dataset where a large minority of voxels pin a parameter at
its bound, and on that flat objective two optimizers stop at different points of an equally
good plateau. Measured on the fixture, ex_tofts Ktrans reads corr 0.23 over all voxels and
0.9998 once bound-pinned voxels are dropped, while the two backends' SSE values agree to
~1e-6 relative — i.e. they find equally good optima, not different answers. So the gate is
three-part:

1. **Identifiable subset** -- voxels where neither backend pinned a core parameter. Backends
   must agree tightly here; this is the real equivalence requirement.
2. **Objective agreement** -- SSE over *all* voxels, including pinned ones. Backends may stop
   at different plateau points but must not find genuinely worse optima.
3. **Bound-hit symmetry** -- each backend must pin core parameters at comparable rates. A
   divergence here is a bound-handling difference, which no other test would catch.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_pipeline import (  # noqa: E402
    DcePipelineConfig,
    MODEL_LAYOUTS,
    _apply_model_specific_prefs,
    _fit_stage_d_model,
    _fit_stage_d_model_accelerated,
    _stage_d_fit_prefs,
    probe_acceleration_backend,
)

FIXTURE = REPO_ROOT / "tests/data/stage_b_frozen/sub-1101743_ses-01_stage_b_backend.npz"
MODELS = ("patlak", "tofts", "ex_tofts")
CORE_PARAMS = ("Ktrans", "ve", "vp")

# Thresholds from measured agreement on the fixture (worst case in parentheses), with
# margin for float32 accumulation order across accelerated builds.
IDENTIFIABLE_CORR_MIN = 0.98      # observed >= 0.9919
# RMS-relative scatter, per parameter rather than one global bound: Ktrans and vp agree an
# order of magnitude better than ve, and a shared threshold loose enough for ve would let a
# real Ktrans regression through. ve is genuinely weakly determined on a low-enhancement BBB
# dataset even away from its bounds, so its looser bound is the data, not a concession.
IDENTIFIABLE_NRMSE_MAX = {
    "Ktrans": 0.06,   # observed <= 2.71e-2
    "ve": 0.15,       # observed <= 8.80e-2
    "vp": 0.02,       # observed <= 3.91e-3
}
# Orthogonal to scatter: catches a systematic scale/offset error, which correlation cannot
# see at all and RMS scatter only sees once it exceeds the natural spread.
IDENTIFIABLE_BIAS_MAX = 0.02
BIAS_MAGNITUDE_FLOOR = 1e-4       # ignore near-zero references when forming ratios
SSE_CORR_MIN = 0.99               # observed >= 0.9967
SSE_REL_MEDIAN_MAX = 1e-3         # observed <= 8.72e-5
BOUND_FRACTION_SKEW_MAX = 0.05    # observed <= 6.5e-3
# A subset this small would mean the "identifiable" gate stopped comparing anything real.
IDENTIFIABLE_FRACTION_MIN = 0.25  # observed >= 0.53
BOUND_REL_TOL = 1e-6


def _log(message: str) -> None:
    print(f"[BACKEND-EQUIV] {message}", flush=True)


def _require_backend(backend: str) -> None:
    probe_acceleration_backend.cache_clear()
    probe = probe_acceleration_backend()
    if backend == "cpufit_cpu":
        if not bool(probe.get("pycpufit_imported", False)):
            pytest.skip(f"pycpufit unavailable on this platform: {probe.get('pycpufit_error')}")
    elif backend == "gpufit_cuda":
        if not bool(probe.get("pygpufit_imported", False)):
            pytest.skip(f"pygpufit unavailable on this platform: {probe.get('pygpufit_error')}")
        if str(probe.get("backend", "")) != "gpufit_cuda":
            pytest.skip("pygpufit imported but no CUDA device; gpufit equivalence needs hardware.")
    else:  # pragma: no cover - guarded by the parametrization
        raise KeyError(f"Unsupported accelerated backend '{backend}'.")


@pytest.fixture(scope="module")
def frozen_arrays() -> dict:
    if not FIXTURE.exists():
        pytest.skip(
            f"Frozen Stage-B fixture missing: {FIXTURE}. "
            "Regenerate with tests/python/freeze_stage_b_backend_fixture.py (needs RUNNER_DATA)."
        )
    data = np.load(FIXTURE, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    return {
        "ct": np.asarray(data["Ct"], dtype=np.float64),
        "cp_use": np.asarray(data["Cp_use"], dtype=np.float64).ravel(),
        "timer": np.asarray(data["timer"], dtype=np.float64).ravel(),
        "meta": meta,
    }


@pytest.fixture(scope="module")
def fit_prefs() -> dict:
    config = DcePipelineConfig(
        subject_source_path=Path("."), subject_tp_path=Path("."), output_dir=Path("."),
    )
    return _stage_d_fit_prefs(config)


_CPU_REFERENCE_CACHE: dict[str, np.ndarray] = {}


def _cpu_reference(model_name: str, frozen_arrays: dict, fit_prefs: dict) -> np.ndarray:
    """Pure-Python (scipy) Stage-D fit -- the reference every backend is graded against.

    Cached across backends rather than computed in a fixture: these are the slow fits (~2 min
    for all three models), and on a machine with no accelerated backend every test skips, so
    a fixture would pay the full cost to compare nothing.
    """
    if model_name not in _CPU_REFERENCE_CACHE:
        _log(f"fitting cpu reference: {model_name}")
        _CPU_REFERENCE_CACHE[model_name] = np.asarray(
            _fit_stage_d_model(
                model_name=model_name,
                ct=frozen_arrays["ct"],
                cp_use=frozen_arrays["cp_use"],
                timer=frozen_arrays["timer"],
                prefs=fit_prefs,
                r1o=None,
                relaxivity=float(frozen_arrays["meta"]["acquisition"]["relaxivity"]),
                fw=0.8,
                stlv_use=None,
                sttum=None,
                start_injection_min=float(frozen_arrays["timer"][0]),
                sss=None,
                ssstum=None,
                acceleration_backend="none",
            ),
            dtype=np.float64,
        )
    return _CPU_REFERENCE_CACHE[model_name]


def _at_bound(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = max(hi - lo, 1e-12)
    return (np.abs(values - lo) <= BOUND_REL_TOL * span) | (np.abs(values - hi) <= BOUND_REL_TOL * span)


def _param_bounds(prefs: dict, name: str) -> tuple[float, float]:
    return float(prefs[f"lower_limit_{name.lower()}"]), float(prefs[f"upper_limit_{name.lower()}"])


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.parametrize("backend", ["cpufit_cpu", "gpufit_cuda"])
def test_accelerated_backend_matches_cpu(
    backend: str, frozen_arrays: dict, fit_prefs: dict
) -> None:
    _require_backend(backend)

    meta = frozen_arrays["meta"]
    _log(
        f"backend={backend} fixture={meta['source']} "
        f"voxels={meta['n_voxels_frozen']} frames={meta['n_time']}"
    )

    failures: list[str] = []

    for model_name in MODELS:
        cpu = _cpu_reference(model_name, frozen_arrays, fit_prefs)
        prefs_model = _apply_model_specific_prefs(fit_prefs, model_name)
        accel = _fit_stage_d_model_accelerated(
            model_name=model_name,
            ct=frozen_arrays["ct"],
            cp_use=frozen_arrays["cp_use"],
            timer=frozen_arrays["timer"],
            prefs=prefs_model,
            acceleration_backend=backend,
        )
        # Deliberately NOT _fit_stage_d_model: that falls back to the python path when a
        # backend fails, which would silently grade cpu against itself and always pass.
        if accel is None:
            failures.append(f"{model_name}: backend {backend} returned no result")
            continue
        accel = np.asarray(accel, dtype=np.float64)
        if accel.shape[0] != cpu.shape[0]:
            failures.append(f"{model_name}: {backend} returned {accel.shape[0]} rows, cpu returned {cpu.shape[0]}")
            continue

        names = list(MODEL_LAYOUTS[model_name]["param_names"])
        core = [n for n in names if n in CORE_PARAMS]

        # (3) Bound-hit symmetry, and the identifiable mask both other checks use.
        identifiable = np.ones(cpu.shape[0], dtype=bool)
        for name in core:
            col = names.index(name)
            lo, hi = _param_bounds(prefs_model, name)
            pinned_cpu = _at_bound(cpu[:, col], lo, hi)
            pinned_acc = _at_bound(accel[:, col], lo, hi)
            identifiable &= ~pinned_cpu & ~pinned_acc
            frac_cpu = float(np.mean(pinned_cpu))
            frac_acc = float(np.mean(pinned_acc))
            skew = abs(frac_cpu - frac_acc)
            _log(
                f"{model_name}.{name}: bound-pinned cpu={frac_cpu:.4f} {backend}={frac_acc:.4f} "
                f"skew={skew:.4f}"
            )
            if skew > BOUND_FRACTION_SKEW_MAX:
                failures.append(
                    f"{model_name}.{name}: bound-pinned fraction differs by {skew:.4f} "
                    f"(cpu={frac_cpu:.4f} {backend}={frac_acc:.4f}, max {BOUND_FRACTION_SKEW_MAX})"
                )

        frac_identifiable = float(np.mean(identifiable))
        _log(f"{model_name}: identifiable voxels {int(identifiable.sum())}/{identifiable.size} ({frac_identifiable:.3f})")
        if frac_identifiable < IDENTIFIABLE_FRACTION_MIN:
            failures.append(
                f"{model_name}: only {frac_identifiable:.3f} of voxels identifiable "
                f"(min {IDENTIFIABLE_FRACTION_MIN}); the equivalence check has nothing solid to compare"
            )
            continue

        # (1) Equivalence on the identifiable subset.
        for name in core:
            col = names.index(name)
            x = cpu[identifiable, col]
            y = accel[identifiable, col]
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < 2:
                failures.append(f"{model_name}.{name}: fewer than 2 comparable identifiable voxels")
                continue
            x, y = x[valid], y[valid]
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
            # Normalize by the reference RMS, not its max: these parameters are heavily
            # right-skewed, and dividing by the max lets a uniform scale error hide (a 20%
            # Ktrans bias measured 0.02 against a 0.05 max-normalized bound). Against the RMS
            # a proportional error reads as itself, so the threshold means what it says.
            scale = max(float(np.sqrt(np.mean(x**2))), 1e-12)
            nrmse = float(np.sqrt(np.mean((x - y) ** 2))) / scale
            above_floor = np.abs(x) > BIAS_MAGNITUDE_FLOOR
            bias = float(np.median(y[above_floor] / x[above_floor]) - 1.0) if above_floor.sum() >= 2 else 0.0
            nrmse_max = IDENTIFIABLE_NRMSE_MAX[name]
            _log(
                f"{model_name}.{name}: identifiable n={x.size} corr={corr:.6f} "
                f"nrmse={nrmse:.3e} bias={bias:+.3e}"
            )
            if not (corr >= IDENTIFIABLE_CORR_MIN):
                failures.append(f"{model_name}.{name}: identifiable corr={corr:.6f} < {IDENTIFIABLE_CORR_MIN}")
            if not (nrmse <= nrmse_max):
                failures.append(f"{model_name}.{name}: identifiable nrmse={nrmse:.3e} > {nrmse_max}")
            if not (abs(bias) <= IDENTIFIABLE_BIAS_MAX):
                failures.append(
                    f"{model_name}.{name}: median relative bias {bias:+.3e} exceeds "
                    f"+/-{IDENTIFIABLE_BIAS_MAX} (systematic offset, not scatter)"
                )

        # (2) Objective agreement over every voxel, pinned ones included.
        sse_col = names.index("sse")
        sse_cpu = cpu[:, sse_col]
        sse_acc = accel[:, sse_col]
        valid = np.isfinite(sse_cpu) & np.isfinite(sse_acc)
        if int(valid.sum()) < 2:
            failures.append(f"{model_name}: fewer than 2 comparable SSE values")
            continue
        a, b = sse_cpu[valid], sse_acc[valid]
        sse_corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
        sse_rel_median = float(np.median(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)))
        _log(f"{model_name}.sse: n={a.size} corr={sse_corr:.6f} rel_median={sse_rel_median:.3e}")
        if not (sse_corr >= SSE_CORR_MIN):
            failures.append(f"{model_name}.sse: corr={sse_corr:.6f} < {SSE_CORR_MIN} (backends found different optima)")
        if not (sse_rel_median <= SSE_REL_MEDIAN_MAX):
            failures.append(
                f"{model_name}.sse: median relative difference {sse_rel_median:.3e} > {SSE_REL_MEDIAN_MAX:.0e}"
            )

    assert not failures, f"{backend} vs cpu Stage-D equivalence failed:\n  " + "\n  ".join(failures)
