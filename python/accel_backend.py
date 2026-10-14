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
from importlib.metadata import PackageNotFoundError, version as _dist_version
from typing import Any, Callable, Dict, Optional

ALLOWED_BACKENDS = {"auto", "cpu", "gpufit"}


def _module_version(module: Any, dist_name: str) -> Optional[str]:
    """Version of an imported pygpufit/pycpufit module, for audit trails in run output.

    Tries the installed distribution's metadata first (what `pip show` reports),
    then falls back to the module's own `<pkg>.version.__version__` (present in
    both packages) for non-pip installs (e.g. built from source on PATH).
    """
    try:
        return _dist_version(dist_name)
    except PackageNotFoundError:
        pass
    try:
        return str(module.version.__version__)
    except Exception:
        return None


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

    pygpufit_version = _module_version(pygpufit_module, "pyGpufit") if pygpufit_module is not None else None
    pycpufit_version = _module_version(pycpufit_module, "pyCpufit") if pycpufit_module is not None else None

    if cuda_available:
        return {
            "backend": "gpufit_cuda",
            "reason": "pygpufit imported and CUDA is available",
            "cuda_available": True,
            "pygpufit_imported": pygpufit_module is not None,
            "pycpufit_imported": pycpufit_module is not None,
            "pygpufit_error": pygpufit_error,
            "pycpufit_error": pycpufit_error,
            "pygpufit_version": pygpufit_version,
            "pycpufit_version": pycpufit_version,
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
            "pygpufit_version": pygpufit_version,
            "pycpufit_version": pycpufit_version,
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
            "pygpufit_version": pygpufit_version,
            "pycpufit_version": pycpufit_version,
        }

    return {
        "backend": "none",
        "reason": "no pygpufit/pycpufit backend detected",
        "cuda_available": False,
        "pygpufit_imported": False,
        "pycpufit_imported": False,
        "pygpufit_error": pygpufit_error,
        "pycpufit_error": pycpufit_error,
        "pygpufit_version": None,
        "pycpufit_version": None,
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
            "acceleration_backend_version": None,
            "reason": "backend=cpu forces pure CPU fitting path",
        }

    probe = probe_fn()
    probe_backend = str(probe.get("backend", "none"))
    probe_reason = str(probe.get("reason", ""))
    pygpufit_imported = bool(probe.get("pygpufit_imported", False))

    def _version_for(acceleration_backend: str) -> Optional[str]:
        # cpufit_cpu is pyCpufit's own backend; every gpufit_* variant (including the
        # CPU fallback path) is pyGpufit's, since gpufit_cpu_fallback still runs
        # through the pygpufit binding rather than pycpufit's.
        if acceleration_backend == "cpufit_cpu":
            return probe.get("pycpufit_version")
        return probe.get("pygpufit_version")

    if backend == "gpufit":
        if not pygpufit_imported:
            raise RuntimeError("GPUfit backend requested but pygpufit could not be imported")
        acceleration_backend = probe_backend if probe_backend != "none" else "gpufit_cpu_fallback"
        return {
            "requested_backend": backend,
            "selected_backend": "gpufit",
            "acceleration_backend": acceleration_backend,
            "acceleration_backend_version": _version_for(acceleration_backend),
            "reason": f"backend=gpufit selected acceleration backend '{acceleration_backend}' ({probe_reason})",
        }

    # auto
    if probe_backend in {"gpufit_cuda", "cpufit_cpu", "gpufit_cpu_fallback"}:
        return {
            "requested_backend": backend,
            "selected_backend": "gpufit",
            "acceleration_backend": probe_backend,
            "acceleration_backend_version": _version_for(probe_backend),
            "reason": f"backend=auto selected acceleration backend '{probe_backend}' ({probe_reason})",
        }
    return {
        "requested_backend": backend,
        "selected_backend": "cpu",
        "acceleration_backend": "none",
        "acceleration_backend_version": None,
        "reason": "backend=auto fell back to pure CPU fitting path",
    }


def load_fit_module_for_acceleration(acceleration_backend: str) -> Any:
    if acceleration_backend == "cpufit_cpu":
        import pycpufit.cpufit as fit_module  # type: ignore

        return fit_module
    import pygpufit.gpufit as fit_module  # type: ignore

    return fit_module
