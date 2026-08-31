"""Shared gpufit/cpufit acceleration-backend detection and selection.

The DCE and parametric pipelines both offer the same `auto`/`cpu`/`gpufit`
choice and resolve it identically. This module owns that logic so a change to
import detection, a new fallback state, or a reworded reason string lands once
instead of twice -- the two copies had no test tying them together, so drift
would have been silent.

`resolve_backend_selection` takes the probe as a callable rather than calling
`probe_acceleration_backend` directly: each pipeline passes a thunk that reads
its own module-level name, which keeps `patch("dce_pipeline.probe_acceleration_backend")`
working as a test seam. The callable is also why `backend="cpu"` still resolves
without importing pygpufit at all.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Optional

ALLOWED_BACKENDS = {"auto", "cpu", "gpufit"}


@lru_cache(maxsize=1)
def probe_acceleration_backend() -> Dict[str, Any]:
    """Detect available acceleration backend in priority order."""

    pygpufit_module: Any = None
    pycpufit_module: Any = None
    pygpufit_error: Optional[str] = None
    pycpufit_error: Optional[str] = None
    cuda_available = False

    try:
        import pygpufit.gpufit as gf  # type: ignore

        pygpufit_module = gf
    except Exception as exc:
        pygpufit_error = str(exc)

    if pygpufit_module is not None:
        try:
            cuda_available = bool(pygpufit_module.cuda_available())
        except Exception:
            cuda_available = False

    try:
        import pycpufit.cpufit as cf  # type: ignore

        pycpufit_module = cf
    except Exception as exc:
        pycpufit_error = str(exc)

    if cuda_available:
        return {
            "backend": "gpufit_cuda",
            "reason": "pygpufit imported and CUDA is available",
            "cuda_available": True,
            "pygpufit_imported": pygpufit_module is not None,
            "pycpufit_imported": pycpufit_module is not None,
            "pygpufit_error": pygpufit_error,
            "pycpufit_error": pycpufit_error,
        }

    if pycpufit_module is not None:
        return {
            "backend": "cpufit_cpu",
            "reason": "using pycpufit CPU backend",
            "cuda_available": cuda_available,
            "pygpufit_imported": pygpufit_module is not None,
            "pycpufit_imported": True,
            "pygpufit_error": pygpufit_error,
            "pycpufit_error": pycpufit_error,
        }

    if pygpufit_module is not None:
        return {
            "backend": "gpufit_cpu_fallback",
            "reason": "pygpufit imported without CUDA and pycpufit unavailable; using pygpufit fallback path",
            "cuda_available": cuda_available,
            "pygpufit_imported": True,
            "pycpufit_imported": False,
            "pygpufit_error": pygpufit_error,
            "pycpufit_error": pycpufit_error,
        }

    return {
        "backend": "none",
        "reason": "no pygpufit/pycpufit backend detected",
        "cuda_available": False,
        "pygpufit_imported": False,
        "pycpufit_imported": False,
        "pygpufit_error": pygpufit_error,
        "pycpufit_error": pycpufit_error,
    }


def resolve_backend_selection(
    requested_backend: str, probe_fn: Callable[[], Dict[str, Any]]
) -> Dict[str, str]:
    """Map a requested backend onto a concrete acceleration backend.

    `probe_fn` is only invoked for the `auto` and `gpufit` paths, so `cpu`
    resolves without triggering backend imports.
    """
    backend = requested_backend.strip().lower()
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(f"Unsupported backend '{requested_backend}'. Allowed: {sorted(ALLOWED_BACKENDS)}")

    if backend == "cpu":
        return {
            "requested_backend": backend,
            "selected_backend": "cpu",
            "acceleration_backend": "none",
            "reason": "backend=cpu forces pure CPU fitting path",
        }

    probe = probe_fn()
    probe_backend = str(probe.get("backend", "none"))
    probe_reason = str(probe.get("reason", ""))
    pygpufit_imported = bool(probe.get("pygpufit_imported", False))

    if backend == "gpufit":
        if not pygpufit_imported:
            raise RuntimeError("GPUfit backend requested but pygpufit could not be imported")
        acceleration_backend = probe_backend if probe_backend != "none" else "gpufit_cpu_fallback"
        return {
            "requested_backend": backend,
            "selected_backend": "gpufit",
            "acceleration_backend": acceleration_backend,
            "reason": f"backend=gpufit selected acceleration backend '{acceleration_backend}' ({probe_reason})",
        }

    # auto
    if probe_backend in {"gpufit_cuda", "cpufit_cpu", "gpufit_cpu_fallback"}:
        return {
            "requested_backend": backend,
            "selected_backend": "gpufit",
            "acceleration_backend": probe_backend,
            "reason": f"backend=auto selected acceleration backend '{probe_backend}' ({probe_reason})",
        }
    return {
        "requested_backend": backend,
        "selected_backend": "cpu",
        "acceleration_backend": "none",
        "reason": "backend=auto fell back to pure CPU fitting path",
    }


def load_fit_module_for_acceleration(acceleration_backend: str) -> Any:
    if acceleration_backend == "cpufit_cpu":
        import pycpufit.cpufit as fit_module  # type: ignore

        return fit_module
    import pygpufit.gpufit as fit_module  # type: ignore

    return fit_module
