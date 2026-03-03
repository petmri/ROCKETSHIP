"""In-memory DCE A->B->D pipeline with non-GUI algorithm implementations."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from dce_models import (
    model_2cxm_cfit,
    model_2cxm_fit,
    model_extended_tofts_cfit,
    model_extended_tofts_fit,
    model_fxr_cfit,
    model_fxr_fit,
    model_patlak_cfit,
    model_patlak_fit,
    model_patlak_linear,
    model_tissue_uptake_cfit,
    model_tissue_uptake_fit,
    model_tofts_cfit,
    model_tofts_fit,
)


ALLOWED_BACKENDS = {"auto", "cpu", "gpufit"}
ALLOWED_AIF_MODES = {"auto", "fitted", "raw", "imported"}
ALLOWED_STAGE_A_MODES = {"real", "scaffold"}
ALLOWED_STAGE_B_MODES = {"real", "scaffold", "auto"}
ALLOWED_STAGE_D_MODES = {"real", "scaffold", "auto"}
ALLOWED_STEADY_STATE_AUTO_METHODS = {"none", "legacy_sobel", "piecewise_constant", "glr", "tv"}
PIECEWISE_CONSTANT_BASELINE_FORWARD_DELTA_FRACTION = 0.01

MODEL_SELECTION_ORDER = [
    ("tofts", "tofts"),
    ("ex_tofts", "ex_tofts"),
    ("fxr", "fxr"),
    ("nested", "nested"),
    ("patlak", "patlak"),
    ("tissue_uptake", "tissue_uptake"),
    ("two_cxm", "2cxm"),
    ("auc", "auc"),
    ("FXL_rr", "FXL_rr"),
]

MODEL_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "tofts": {
        "headings": [
            "ROI path",
            "ROI",
            "Ktrans",
            "Ve",
            "SSE",
            "Ktrans 95% low",
            "Ktrans 95% high",
            "Ve 95% low",
            "Ve 95% high",
        ],
        "param_names": ["Ktrans", "ve", "sse", "ktrans_ci_low", "ktrans_ci_high", "ve_ci_low", "ve_ci_high"],
    },
    "ex_tofts": {
        "headings": [
            "ROI path",
            "ROI",
            "Ktrans",
            "Ve",
            "Vp",
            "SSE",
            "Ktrans 95% low",
            "Ktrans 95% high",
            "Ve 95% low",
            "Ve 95% high",
            "Vp 95% low",
            "Vp 95% high",
        ],
        "param_names": [
            "Ktrans",
            "ve",
            "vp",
            "sse",
            "ktrans_ci_low",
            "ktrans_ci_high",
            "ve_ci_low",
            "ve_ci_high",
            "vp_ci_low",
            "vp_ci_high",
        ],
    },
    "patlak": {
        "headings": [
            "ROI path",
            "ROI",
            "Ktrans",
            "Vp",
            "SSE",
            "Ktrans 95% low",
            "Ktrans 95% high",
            "Vp 95% low",
            "Vp 95% high",
        ],
        "param_names": ["Ktrans", "vp", "sse", "ktrans_ci_low", "ktrans_ci_high", "vp_ci_low", "vp_ci_high"],
    },
    "tissue_uptake": {
        "headings": [
            "ROI path",
            "ROI",
            "Ktrans",
            "Fp",
            "Vp",
            "SSE",
            "Ktrans 95% low",
            "Ktrans 95% high",
            "Fp 95% low",
            "Fp 95% high",
            "Vp 95% low",
            "Vp 95% high",
        ],
        "param_names": [
            "Ktrans",
            "fp",
            "vp",
            "sse",
            "ktrans_ci_low",
            "ktrans_ci_high",
            "fp_ci_low",
            "fp_ci_high",
            "vp_ci_low",
            "vp_ci_high",
        ],
    },
    "2cxm": {
        "headings": [
            "ROI path",
            "ROI",
            "Ktrans",
            "Ve",
            "Vp",
            "Fp",
            "SSE",
            "Ktrans 95% low",
            "Ktrans 95% high",
            "Ve 95% low",
            "Ve 95% high",
            "Vp 95% low",
            "Vp 95% high",
            "Fp 95% low",
            "Fp 95% high",
        ],
        "param_names": [
            "Ktrans",
            "ve",
            "vp",
            "fp",
            "sse",
            "ktrans_ci_low",
            "ktrans_ci_high",
            "ve_ci_low",
            "ve_ci_high",
            "vp_ci_low",
            "vp_ci_high",
            "fp_ci_low",
            "fp_ci_high",
        ],
    },
    "fxr": {
        "headings": [
            "ROI path",
            "ROI",
            "Ktrans",
            "Ve",
            "Tau",
            "SSE",
            "Ktrans 95% low",
            "Ktrans 95% high",
            "Ve 95% low",
            "Ve 95% high",
            "Tau 95% low",
            "Tau 95% high",
        ],
        "param_names": [
            "Ktrans",
            "ve",
            "tau",
            "sse",
            "ktrans_ci_low",
            "ktrans_ci_high",
            "ve_ci_low",
            "ve_ci_high",
            "tau_ci_low",
            "tau_ci_high",
        ],
    },
    "auc": {
        "headings": ["ROI path", "ROI", "AUC conc", "AUC sig", "NAUC conc", "NAUC sig"],
        "param_names": ["AUCc", "AUCs", "NAUCc", "NAUCs"],
    },
}

SUPPORTED_STAGE_D_MODELS = {"tofts", "ex_tofts", "patlak", "tissue_uptake", "2cxm", "fxr", "auc"}
ACCELERATED_STAGE_D_MODELS = {"tofts", "ex_tofts", "patlak", "tissue_uptake", "2cxm"}
PREFERENCE_NUMERIC_CHARS = set("0123456789eE.+-*/^() ")


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return default
        return float(value) != 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return default


def _eval_numeric_expr(expr: str) -> float:
    text = expr.strip()
    if not text:
        raise ValueError("Empty expression")
    if any(ch not in PREFERENCE_NUMERIC_CHARS for ch in text):
        raise ValueError(f"Unsafe numeric expression: {expr}")

    node = ast.parse(text.replace("^", "**"), mode="eval")

    def _eval(node_in: ast.AST) -> float:
        if isinstance(node_in, ast.Expression):
            return _eval(node_in.body)
        if isinstance(node_in, ast.Constant):
            if isinstance(node_in.value, (int, float)):
                return float(node_in.value)
            raise ValueError(f"Unsupported constant: {node_in.value!r}")
        if isinstance(node_in, ast.UnaryOp) and isinstance(node_in.op, (ast.UAdd, ast.USub)):
            inner = _eval(node_in.operand)
            return inner if isinstance(node_in.op, ast.UAdd) else -inner
        if isinstance(node_in, ast.BinOp):
            left = _eval(node_in.left)
            right = _eval(node_in.right)
            if isinstance(node_in.op, ast.Add):
                return left + right
            if isinstance(node_in.op, ast.Sub):
                return left - right
            if isinstance(node_in.op, ast.Mult):
                return left * right
            if isinstance(node_in.op, ast.Div):
                return left / right
            if isinstance(node_in.op, ast.Pow):
                return left**right
        raise ValueError(f"Unsupported expression syntax: {expr}")

    out = float(_eval(node))
    if not math.isfinite(out):
        raise ValueError(f"Non-finite numeric expression: {expr}")
    return out


def _parse_numeric_token(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        if math.isfinite(out):
            return out
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return _eval_numeric_expr(text)
    except Exception:
        return None


@lru_cache(maxsize=32)
def _parse_preference_file(path_str: str, mtime_ns: int) -> Dict[str, str]:
    del mtime_ns  # cache key only
    path = Path(path_str)
    if not path.exists():
        return {}

    prefs: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        if not key:
            continue
        value = value.split("%", 1)[0].strip()
        prefs[key] = value
    return prefs


def _resolve_dce_preferences_path(config: "DcePipelineConfig") -> Optional[Path]:
    use_prefs = _to_bool(config.stage_overrides.get("use_dce_preferences", True), True)
    if not use_prefs:
        return None

    explicit = config.stage_overrides.get("dce_preferences_path")
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"dce_preferences file not found: {path}")
        return path

    repo_default = Path(__file__).resolve().parents[2] / "dce" / "dce_preferences.txt"
    if repo_default.exists():
        return repo_default

    cwd_default = Path.cwd() / "dce_preferences.txt"
    if cwd_default.exists():
        return cwd_default
    return None


def _load_dce_preferences(config: "DcePipelineConfig") -> Dict[str, str]:
    path = _resolve_dce_preferences_path(config)
    if path is None:
        return {}
    stat = path.stat()
    return _parse_preference_file(str(path), int(stat.st_mtime_ns))


def _scipy_loss_from_robust(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode in {"", "off", "none", "linear"}:
        return "linear"
    if mode == "lar":
        return "soft_l1"
    if mode == "bisquare":
        return "cauchy"
    return "linear"


def _to_path_list(values: Optional[List[str]]) -> List[Path]:
    if not values:
        return []
    return [Path(v).expanduser().resolve() for v in values]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64, np.float16)):
        return float(value)
    if isinstance(value, (np.int32, np.int64, np.int16, np.int8)):
        return int(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


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


def is_gpufit_available() -> bool:
    """Return whether pygpufit is importable."""
    probe = probe_acceleration_backend()
    return bool(probe.get("pygpufit_imported", False))


def _resolve_backend_selection(requested_backend: str) -> Dict[str, str]:
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

    probe = probe_acceleration_backend()
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


def resolve_backend(requested_backend: str) -> str:
    """Resolve high-level backend choice (`cpu` or `gpufit`)."""
    return _resolve_backend_selection(requested_backend)["selected_backend"]


def _resolve_stage_d_backend(config: "DcePipelineConfig") -> Dict[str, str]:
    requested_backend = str(config.backend).strip().lower()
    force_cpu = _to_bool(_stage_override(config, "force_cpu", 0), False)
    if requested_backend == "auto" and force_cpu:
        return {
            "requested_backend": "auto",
            "selected_backend": "cpu",
            "acceleration_backend": "none",
            "reason": "force_cpu=1 overrides backend=auto to pure CPU fitting path",
        }
    return _resolve_backend_selection(requested_backend)


@dataclass
class DcePipelineConfig:
    """Configuration for a single end-to-end DCE CLI run."""

    subject_source_path: Path
    subject_tp_path: Path
    output_dir: Path
    backend: str = "auto"
    checkpoint_dir: Optional[Path] = None
    write_xls: bool = True
    aif_mode: str = "auto"
    imported_aif_path: Optional[Path] = None
    dynamic_files: List[Path] = field(default_factory=list)
    aif_files: List[Path] = field(default_factory=list)
    roi_files: List[Path] = field(default_factory=list)
    t1map_files: List[Path] = field(default_factory=list)
    noise_files: List[Path] = field(default_factory=list)
    drift_files: List[Path] = field(default_factory=list)
    model_flags: Dict[str, int] = field(default_factory=dict)
    stage_overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DcePipelineConfig":
        return cls(
            subject_source_path=Path(data["subject_source_path"]).expanduser().resolve(),
            subject_tp_path=Path(data["subject_tp_path"]).expanduser().resolve(),
            output_dir=Path(data["output_dir"]).expanduser().resolve(),
            backend=str(data.get("backend", "auto")),
            checkpoint_dir=Path(data["checkpoint_dir"]).expanduser().resolve()
            if data.get("checkpoint_dir")
            else None,
            write_xls=bool(data.get("write_xls", True)),
            aif_mode=str(data.get("aif_mode", "auto")),
            imported_aif_path=Path(data["imported_aif_path"]).expanduser().resolve()
            if data.get("imported_aif_path")
            else None,
            dynamic_files=_to_path_list(data.get("dynamic_files")),
            aif_files=_to_path_list(data.get("aif_files")),
            roi_files=_to_path_list(data.get("roi_files")),
            t1map_files=_to_path_list(data.get("t1map_files")),
            noise_files=_to_path_list(data.get("noise_files")),
            drift_files=_to_path_list(data.get("drift_files")),
            model_flags=dict(data.get("model_flags", {})),
            stage_overrides=dict(data.get("stage_overrides", {})),
        )

    def validate(self) -> None:
        backend = self.backend.strip().lower()
        if backend not in ALLOWED_BACKENDS:
            raise ValueError(f"Unsupported backend '{self.backend}'. Allowed: {sorted(ALLOWED_BACKENDS)}")

        override_import_path = None
        for override_key, override_val in self.stage_overrides.items():
            key_lc = str(override_key).strip().lower()
            if key_lc in {"import_aif_path", "imported_aif_path"} and str(override_val).strip():
                override_import_path = str(override_val).strip()
                break

        mode = self.aif_mode.strip().lower()
        if mode not in ALLOWED_AIF_MODES:
            raise ValueError(f"Unsupported aif_mode '{self.aif_mode}'. Allowed: {sorted(ALLOWED_AIF_MODES)}")
        if mode == "imported" and self.imported_aif_path is None and not override_import_path:
            raise ValueError("aif_mode=imported requires imported_aif_path")

        stage_a_mode = str(self.stage_overrides.get("stage_a_mode", "real")).strip().lower()
        if stage_a_mode not in ALLOWED_STAGE_A_MODES:
            raise ValueError(f"Unsupported stage_a_mode '{stage_a_mode}'. Allowed: {sorted(ALLOWED_STAGE_A_MODES)}")
        stage_b_mode = str(self.stage_overrides.get("stage_b_mode", "auto")).strip().lower()
        if stage_b_mode not in ALLOWED_STAGE_B_MODES:
            raise ValueError(f"Unsupported stage_b_mode '{stage_b_mode}'. Allowed: {sorted(ALLOWED_STAGE_B_MODES)}")
        stage_d_mode = str(self.stage_overrides.get("stage_d_mode", "auto")).strip().lower()
        if stage_d_mode not in ALLOWED_STAGE_D_MODES:
            raise ValueError(f"Unsupported stage_d_mode '{stage_d_mode}'. Allowed: {sorted(ALLOWED_STAGE_D_MODES)}")
        aif_curve_mode = str(self.stage_overrides.get("aif_curve_mode", "")).strip().lower()
        if aif_curve_mode and aif_curve_mode not in ALLOWED_AIF_MODES:
            raise ValueError(
                f"Unsupported stage_overrides.aif_curve_mode '{aif_curve_mode}'. Allowed: {sorted(ALLOWED_AIF_MODES)}"
            )
        if aif_curve_mode == "imported" and self.imported_aif_path is None and not override_import_path:
            raise ValueError("stage_overrides.aif_curve_mode=imported requires imported_aif_path")

        # Port scope decision: ImageJ ROI input is intentionally not supported.
        for roi_path in self.roi_files:
            if roi_path.suffix.lower() == ".roi":
                raise ValueError("ImageJ ROI (.roi) input is out of scope for the Python DCE CLI port")

        if not self.dynamic_files:
            raise ValueError("dynamic_files is required (non-empty)")
        if not self.aif_files:
            raise ValueError(
                "aif_files is required (non-empty). Automatic AIF discovery is not available in the Python DCE pipeline yet; "
                "provide a dedicated AIF ROI mask."
            )
        if self.roi_files:
            roi_set = {Path(p).expanduser().resolve() for p in self.roi_files}
            overlap = [p for p in self.aif_files if Path(p).expanduser().resolve() in roi_set]
            if overlap:
                raise ValueError(
                    "AIF mask must be a dedicated vascular ROI and cannot reuse ROI mask file(s): "
                    + ", ".join(str(Path(p).expanduser().resolve()) for p in overlap)
                )
        if not self.t1map_files:
            raise ValueError("t1map_files is required (non-empty)")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return json.loads(json.dumps(payload, default=_json_default))


class DceStageRunner(Protocol):
    """Execution contract for in-memory A/B/D stages."""

    def run_a(
        self, config: DcePipelineConfig, event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        ...

    def run_b(
        self,
        config: DcePipelineConfig,
        stage_a: Dict[str, Any],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        ...

    def run_d(
        self,
        config: DcePipelineConfig,
        stage_a: Dict[str, Any],
        stage_b: Dict[str, Any],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        ...


def _stage_override(config: DcePipelineConfig, key: str, default: Any) -> Any:
    if key in config.stage_overrides:
        return config.stage_overrides[key]

    key_lc = key.lower()
    for override_key, override_val in config.stage_overrides.items():
        if str(override_key).lower() == key_lc:
            return override_val

    prefs = _load_dce_preferences(config)
    if key_lc in prefs:
        return prefs[key_lc]
    return default


def _override_value_is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def _explicit_stage_override(config: DcePipelineConfig, keys: Tuple[str, ...]) -> Tuple[Optional[Any], Optional[str]]:
    for key in keys:
        if key in config.stage_overrides:
            value = config.stage_overrides[key]
            if _override_value_is_set(value):
                return value, str(key)
        key_lc = key.lower()
        for override_key, override_val in config.stage_overrides.items():
            if str(override_key).lower() == key_lc and _override_value_is_set(override_val):
                return override_val, str(override_key)
    return None, None


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _load_nifti_data(path: Path) -> np.ndarray:
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:
        raise RuntimeError("nibabel is required for Stage-A NIfTI loading") from exc

    image = nib.load(str(path))
    return np.asarray(image.get_fdata(), dtype=np.float64)


def _resolve_dynamic_metadata(config: DcePipelineConfig, n_timepoints: int) -> Dict[str, Any]:
    metadata_path, _ = _explicit_stage_override(config, ("dce_metadata_path",))
    candidates: List[Path] = []
    if metadata_path is not None:
        candidates.append(Path(str(metadata_path)).expanduser().resolve())

    for dynamic in config.dynamic_files:
        dynamic_text = str(dynamic)
        if dynamic_text.endswith(".nii.gz"):
            candidates.append(Path(dynamic_text[:-7] + ".json"))
        elif dynamic.suffix.lower() == ".nii":
            candidates.append(dynamic.with_suffix(".json"))

    candidates.extend(sorted((config.subject_source_path / "dce").glob("*DCE.json")))

    payload: Dict[str, Any] = {}
    metadata_source_path: Optional[str] = None
    for candidate in candidates:
        if candidate.exists():
            payload = _load_json(candidate)
            metadata_source_path = str(candidate)
            break

    tr_sec_raw, tr_sec_key = _explicit_stage_override(config, ("tr_sec",))
    tr_ms_raw, tr_ms_key = _explicit_stage_override(config, ("tr_ms", "tr"))
    time_resolution_raw, time_resolution_key = _explicit_stage_override(
        config, ("time_resolution_sec", "time_resolution")
    )
    fa_raw, fa_key = _explicit_stage_override(config, ("fa_deg", "fa"))

    if tr_sec_raw is not None and tr_ms_raw is not None:
        raise ValueError("Specify only one of stage_overrides.tr_sec and stage_overrides.tr_ms/tr")

    manual_tr = tr_sec_raw is not None or tr_ms_raw is not None
    manual_fa = fa_raw is not None
    manual_time = time_resolution_raw is not None
    manual_any = manual_tr or manual_fa or manual_time
    manual_all = manual_tr and manual_fa and manual_time

    if metadata_source_path is None and not manual_all:
        raise ValueError(
            "DCE metadata JSON not found. Provide stage_overrides.tr_ms/tr_sec, "
            "stage_overrides.fa_deg/fa, and stage_overrides.time_resolution_sec/time_resolution."
        )
    if metadata_source_path is not None and manual_any and not manual_all:
        raise ValueError(
            "Partial manual DCE metadata override is not allowed when metadata JSON is present. "
            "Provide all of tr_ms/tr_sec + fa_deg/fa + time_resolution_sec/time_resolution, or provide none."
        )

    tr_ms: Optional[float] = None
    time_resolution_sec: Optional[float] = None
    fa_deg: Optional[float] = None
    metadata_sources: Dict[str, str] = {}

    if tr_sec_raw is not None:
        tr_ms = float(tr_sec_raw) * 1000.0
        metadata_sources["tr_ms"] = f"stage_overrides.{tr_sec_key}"
    elif tr_ms_raw is not None:
        tr_ms = float(tr_ms_raw)
        metadata_sources["tr_ms"] = f"stage_overrides.{tr_ms_key}"

    if time_resolution_raw is not None:
        time_resolution_sec = float(time_resolution_raw)
        metadata_sources["time_resolution_sec"] = f"stage_overrides.{time_resolution_key}"

    if fa_raw is not None:
        fa_deg = float(fa_raw)
        metadata_sources["fa_deg"] = f"stage_overrides.{fa_key}"

    source_prefix = f"json:{metadata_source_path}" if metadata_source_path else "json"

    def _payload_lookup(keys: Tuple[str, ...]) -> Tuple[Optional[Any], Optional[str]]:
        for raw_key in keys:
            if "." not in raw_key:
                if raw_key in payload:
                    return payload[raw_key], raw_key
                continue
            cur: Any = payload
            ok = True
            for part in raw_key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    ok = False
                    break
                cur = cur[part]
            if ok:
                return cur, raw_key
        return None, None

    if tr_ms is None:
        if "tr_ms" in payload:
            tr_ms = float(payload["tr_ms"])
            metadata_sources["tr_ms"] = f"{source_prefix}.tr_ms"
        elif "tr_sec" in payload:
            tr_ms = float(payload["tr_sec"]) * 1000.0
            metadata_sources["tr_ms"] = f"{source_prefix}.tr_sec"
        elif "RepetitionTimeExcitation" in payload:
            tr_ms = float(payload["RepetitionTimeExcitation"]) * 1000.0
            metadata_sources["tr_ms"] = f"{source_prefix}.RepetitionTimeExcitation"
        elif "RepetitionTime" in payload:
            tr_ms = float(payload["RepetitionTime"]) * 1000.0
            metadata_sources["tr_ms"] = f"{source_prefix}.RepetitionTime"

    if time_resolution_sec is None:
        if "time_resolution_sec" in payload:
            time_resolution_sec = float(payload["time_resolution_sec"])
            metadata_sources["time_resolution_sec"] = f"{source_prefix}.time_resolution_sec"
        elif "TemporalResolution" in payload:
            time_resolution_sec = float(payload["TemporalResolution"])
            metadata_sources["time_resolution_sec"] = f"{source_prefix}.TemporalResolution"

    if (
        time_resolution_sec is not None
        and "NumberOfAverages" in payload
        and metadata_sources.get("time_resolution_sec", "").startswith(source_prefix)
    ):
        time_resolution_sec = float(time_resolution_sec) * float(payload["NumberOfAverages"])
        metadata_sources["time_resolution_sec"] = metadata_sources["time_resolution_sec"] + "*NumberOfAverages"

    if fa_deg is None:
        if "fa_deg" in payload:
            fa_deg = float(payload["fa_deg"])
            metadata_sources["fa_deg"] = f"{source_prefix}.fa_deg"
        elif "FlipAngle" in payload:
            fa_deg = float(payload["FlipAngle"])
            metadata_sources["fa_deg"] = f"{source_prefix}.FlipAngle"

    relaxivity_val = None
    relaxivity_key = None
    relaxivity_val, relaxivity_key = _payload_lookup(
        (
            "relaxivity",
            "Relaxivity_per_mM_per_s",
            "contrast_relaxivity",
            "ContrastRelaxivity_per_mM_per_s",
            "SyntheticPhantom.relaxivity",
            "SyntheticPhantom.Relaxivity_per_mM_per_s",
        )
    )
    relaxivity = None
    if relaxivity_val is not None:
        relaxivity = float(relaxivity_val)
        metadata_sources["relaxivity"] = f"{source_prefix}.{relaxivity_key}"

    hematocrit_val = None
    hematocrit_key = None
    hematocrit_val, hematocrit_key = _payload_lookup(
        (
            "hematocrit",
            "Hematocrit",
            "SyntheticPhantom.hematocrit",
            "SyntheticPhantom.Hematocrit",
            "SyntheticPhantom.RecommendedROCKETSHIPHematocrit",
        )
    )
    hematocrit = None
    if hematocrit_val is not None:
        hematocrit = float(hematocrit_val)
        metadata_sources["hematocrit"] = f"{source_prefix}.{hematocrit_key}"

    aif_kind_val, aif_kind_key = _payload_lookup(
        (
            "AIFConcentrationKind",
            "SyntheticPhantom.AIFConcentrationKind",
        )
    )
    aif_concentration_kind = str(aif_kind_val).strip().lower() if aif_kind_val is not None else None
    if aif_kind_val is not None and aif_kind_key is not None:
        metadata_sources["aif_concentration_kind"] = f"{source_prefix}.{aif_kind_key}"

    if tr_ms is None:
        raise ValueError("Unable to determine TR; set stage_overrides.tr_ms or provide DCE metadata JSON")
    if time_resolution_sec is None:
        raise ValueError(
            "Unable to determine DCE frame spacing (time resolution); set stage_overrides.time_resolution_sec "
            "or provide DCE metadata JSON with TemporalResolution/time_resolution_sec."
        )
    if fa_deg is None:
        raise ValueError("Unable to determine flip angle; set stage_overrides.fa_deg or provide DCE metadata JSON")
    if float(tr_ms) <= 0.0:
        raise ValueError(f"Resolved TR must be positive, got {tr_ms}")
    if float(time_resolution_sec) <= 0.0:
        raise ValueError(f"Resolved time resolution must be positive, got {time_resolution_sec}")
    if float(fa_deg) <= 0.0:
        raise ValueError(f"Resolved flip angle must be positive, got {fa_deg}")
    if relaxivity is not None and float(relaxivity) <= 0.0:
        raise ValueError(f"Resolved relaxivity must be positive, got {relaxivity}")
    if hematocrit is not None and not (0.0 <= float(hematocrit) < 1.0):
        raise ValueError(f"Resolved hematocrit must be in [0, 1), got {hematocrit}")

    return {
        "tr_ms": float(tr_ms),
        "time_resolution_sec": float(time_resolution_sec),
        "time_resolution_min": float(time_resolution_sec) / 60.0,
        "fa_deg": float(fa_deg),
        "relaxivity": (float(relaxivity) if relaxivity is not None else None),
        "hematocrit": (float(hematocrit) if hematocrit is not None else None),
        "aif_concentration_kind": aif_concentration_kind,
        "metadata_source_keys": sorted(payload.keys()),
        "metadata_source_path": metadata_source_path,
        "metadata_sources": metadata_sources,
    }


def _baseline_window(config: DcePipelineConfig, n_timepoints: int) -> Tuple[int, int]:
    start_raw = _stage_override(config, "steady_state_start", 1)
    end_raw = _stage_override(config, "steady_state_end", min(2, n_timepoints))
    if not _override_value_is_set(start_raw):
        start_raw = 1
    if not _override_value_is_set(end_raw):
        end_raw = min(2, n_timepoints)
    start_1b = int(start_raw)
    end_1b = int(end_raw)
    start_1b = max(1, min(start_1b, n_timepoints))
    end_1b = max(start_1b, min(end_1b, n_timepoints))
    return start_1b - 1, end_1b


def _moving_average_smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    window = max(1, int(window))
    if window == 1 or arr.size == 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    sigma = float(sigma)
    if sigma <= 0.0 or arr.size == 1:
        return arr.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    padded = np.pad(arr, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _normalize_zero_one(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float64)
    filled = arr.copy()
    if not np.all(finite):
        idx = np.arange(arr.size, dtype=np.float64)
        filled[~finite] = np.interp(idx[~finite], idx[finite], filled[finite])
    vmin = float(np.min(filled))
    vmax = float(np.max(filled))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(filled, dtype=np.float64)
    return (filled - vmin) / (vmax - vmin)


def _linear_fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_center = x - x_mean
    denom = float(np.sum(x_center * x_center))
    if denom <= 0.0:
        return 0.0
    slope = float(np.sum(x_center * (y - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 1e-15 else 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(max(-1.0, min(1.0, r2)))


def _normalize_steady_state_auto_method(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).strip().lower()
    if text in {"", "none", "off", "disabled", "manual", "false", "0"}:
        return "none"
    aliases = {
        "legacy": "legacy_sobel",
        "legacy_sobel": "legacy_sobel",
        "sobel": "legacy_sobel",
        "matlab_legacy": "legacy_sobel",
        "dce_auto_aif": "legacy_sobel",
        "piecewise": "piecewise_constant",
        "piecewise_constant": "piecewise_constant",
        "piecewise-constant": "piecewise_constant",
        "find_end_ss": "piecewise_constant",
        "bruteforce_piecewise": "piecewise_constant",
        "glr": "glr",
        "edge": "glr",
        "glr_edge": "glr",
        "find_end_ss_edge": "glr",
        "tv": "tv",
        "tv_denoise": "tv",
        "tv_denoising": "tv",
        "find_end_ss_tv": "tv",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(
        f"Unsupported stage_overrides.steady_state_auto_method '{value}'. "
        f"Allowed: {sorted(ALLOWED_STEADY_STATE_AUTO_METHODS)}"
    )


def _legacy_sobel_baseline_end(stlv: np.ndarray) -> Dict[str, Any]:
    curves = np.asarray(stlv, dtype=np.float64)
    if curves.ndim == 1:
        curves = curves[:, np.newaxis]
    if curves.ndim != 2 or curves.shape[0] < 2:
        return {
            "method": "legacy_sobel",
            "end_ss_1b": 1,
            "global_edge_index_1b": 1,
            "slope_at_edge": 0.0,
            "linefit_r2_threshold": 0.95,
        }

    global_dyn = np.mean(curves, axis=1)
    global_smooth = _moving_average_smooth_1d(global_dyn, window=9)
    global_smooth = _normalize_zero_one(global_smooth)
    if global_smooth.size < 2:
        return {
            "method": "legacy_sobel",
            "end_ss_1b": 1,
            "global_edge_index_1b": 1,
            "slope_at_edge": 0.0,
            "linefit_r2_threshold": 0.95,
        }

    # Approximate MATLAB's Sobel edge response on a 1D curve.
    sobel = np.array([1.0, 0.0, -1.0], dtype=np.float64) / 2.0
    bx = np.convolve(np.pad(global_smooth, (1, 1), mode="edge"), sobel, mode="valid")
    i0 = int(np.argmin(bx))  # MATLAB uses min(bx) for the strongest rising edge.
    i0 = max(1, min(i0, global_smooth.size - 1))
    end_ss_1b = i0  # i (1-based) minus 1 => Python 0-based index i0 corresponds to 1-based i0

    r2_threshold = 0.95
    if i0 >= 2:
        # MATLAB: for n=2:(i-1); fit((n:i)', y(n:i), 'poly1'); if rsquare>0.95, end_ss=n; break
        x_all = np.arange(1, global_smooth.size + 1, dtype=np.float64)
        i_1b = i0 + 1
        for n_1b in range(2, i_1b):
            seg = slice(n_1b - 1, i_1b)
            r2 = _linear_fit_r2(x_all[seg], global_smooth[seg])
            if r2 > r2_threshold:
                end_ss_1b = n_1b
                break

    end_ss_1b = max(1, min(int(end_ss_1b), global_smooth.size))
    return {
        "method": "legacy_sobel",
        "end_ss_1b": int(end_ss_1b),
        "global_edge_index_1b": int(i0 + 1),
        "slope_at_edge": float(bx[i0]) if bx.size > 0 else 0.0,
        "linefit_r2_threshold": float(r2_threshold),
    }


def _glr_baseline_end(stlv: np.ndarray) -> Dict[str, Any]:
    curves = np.asarray(stlv, dtype=np.float64)
    if curves.ndim == 1:
        curves = curves[:, np.newaxis]
    if curves.ndim != 2 or curves.shape[0] < 2:
        return {
            "method": "glr",
            "end_ss_1b": 1,
            "mode": "fallback_short_signal",
        }

    global_time_curve = np.mean(curves, axis=1)
    mean_si = float(np.mean(global_time_curve))
    n = int(global_time_curve.size)
    min_before = 3
    min_after = 5

    # Early-jump guard for immediate contrast arrival (ported from end_baseline_detect.py).
    early_threshold = None
    early_detected_0b = None
    if n >= 3:
        first_jumps = np.diff(global_time_curve[:3])
        if n > 10:
            tail_segment = global_time_curve[max(0, n - 10) : n]
            tail_jumps = np.diff(tail_segment)
            baseline_jump_std = float(np.std(tail_jumps)) if tail_jumps.size > 0 else float(np.std(tail_segment))
        else:
            baseline_jump_std = float(np.std(first_jumps)) if first_jumps.size > 1 else float(np.std(global_time_curve[:3]))
        if baseline_jump_std < 1e-10:
            baseline_jump_std = 0.01 * (float(np.mean(global_time_curve[:3])) + 1e-10)
        early_threshold = 3.0 * baseline_jump_std
        for idx, jump in enumerate(first_jumps):
            if float(jump) > early_threshold:
                early_detected_0b = int(idx)
                return {
                    "method": "glr",
                    "end_ss_1b": int(early_detected_0b + 1),
                    "mode": "early_jump",
                    "early_jump_index_1b": int(early_detected_0b + 1),
                    "early_jump_threshold": float(early_threshold),
                    "early_jump_value": float(jump),
                }

    max_score = -math.inf
    best_changepoint_0b = max(0, min_before - 1)
    best_k_1b = min_before
    for k in range(min_before, n - min_after + 1):
        before = global_time_curve[:k]
        after = global_time_curve[k : k + min_after]
        if before.size == 0 or after.size == 0:
            continue
        mean_before = float(np.mean(before))
        mean_after = float(np.mean(after))
        if mean_after <= mean_before:
            continue
        var_before = float(np.var(before))
        baseline_flatness = var_before / (mean_before + mean_si * 0.001)
        mean_increase = mean_after - mean_before
        score = mean_increase / (baseline_flatness + mean_si * 0.001)
        if k < 10:
            score *= 0.6 + (k / 10.0)
        if score > max_score:
            max_score = score
            best_k_1b = int(k)
            best_changepoint_0b = int(k - 1)  # last baseline point

    end_ss_1b = int(max(1, min(best_changepoint_0b + 1, n)))
    return {
        "method": "glr",
        "end_ss_1b": end_ss_1b,
        "mode": "glr_score",
        "best_split_k_1b": int(best_k_1b),
        "glr_score": (float(max_score) if math.isfinite(max_score) else None),
        "min_before": int(min_before),
        "min_after": int(min_after),
        "early_jump_threshold": (float(early_threshold) if early_threshold is not None else None),
    }


def _tv_baseline_end(stlv: np.ndarray) -> Dict[str, Any]:
    curves = np.asarray(stlv, dtype=np.float64)
    if curves.ndim == 1:
        curves = curves[:, np.newaxis]
    if curves.ndim != 2 or curves.shape[0] < 2:
        return {
            "method": "tv",
            "end_ss_1b": 1,
            "mode": "fallback_short_signal",
        }

    global_time_curve = np.mean(curves, axis=1).astype(np.float64, copy=False)
    n = int(global_time_curve.size)
    if n < 3:
        return {
            "method": "tv",
            "end_ss_1b": 1,
            "mode": "fallback_short_signal",
        }

    diff = np.diff(global_time_curve)
    mad = float(np.median(np.abs(diff - np.median(diff)))) if diff.size > 0 else 0.0
    lambda_tv = 2.0 * mad
    if mad < 1e-6:
        lambda_tv = 0.1

    x = global_time_curve.copy()
    n_iter = 0
    for iteration in range(50):
        n_iter = iteration + 1
        x_old = x.copy()
        d = np.diff(x)
        d_thresh = np.sign(d) * np.maximum(np.abs(d) - lambda_tv / n, 0.0)
        x[0] = global_time_curve[0]
        for i in range(1, n):
            x[i] = 0.5 * (x[i - 1] + d_thresh[i - 1]) + 0.5 * global_time_curve[i]
        if float(np.max(np.abs(x - x_old))) < 1e-6:
            break

    jumps = np.diff(x)
    baseline_len = min(n, max(5, int(0.2 * n)))
    baseline_segment = x[:baseline_len]
    baseline_jumps = np.diff(baseline_segment)
    if baseline_jumps.size > 0:
        baseline_jump_mad = float(np.median(np.abs(baseline_jumps - np.median(baseline_jumps))))
        baseline_jump_median = float(np.median(baseline_jumps))
    else:
        baseline_jump_mad = 0.0
        baseline_jump_median = 0.0
    if baseline_jump_mad < 1e-6:
        baseline_jump_mad = 0.01 * float(np.std(global_time_curve[:baseline_len]))
        if baseline_jump_mad < 1e-6:
            baseline_jump_mad = 0.01

    jump_threshold = baseline_jump_median + 3.5 * baseline_jump_mad
    significant_jumps = np.where(jumps > jump_threshold)[0]
    valid_jumps: List[int] = []
    for idx in significant_jumps:
        i = int(idx)
        if i < jumps.size - 1:
            next_jump = float(jumps[i + 1])
            if next_jump > -baseline_jump_mad or float(jumps[i]) > 2.0 * jump_threshold:
                valid_jumps.append(i)
        else:
            if float(jumps[i]) > 1.5 * jump_threshold:
                valid_jumps.append(i)

    if len(valid_jumps) == 0:
        end_ss_0b = 0
        detected_jump = 0.0
    else:
        end_ss_0b = int(valid_jumps[0])  # index before the first accepted jump
        detected_jump = float(jumps[end_ss_0b])

    end_ss_1b = int(max(1, min(end_ss_0b + 1, n)))
    baseline_noise = (
        float(np.std(global_time_curve[: min(10, max(1, n // 4))])) if end_ss_0b > 0 else float(np.std(global_time_curve))
    )
    if baseline_noise < 1e-6:
        baseline_noise = 1e-6
    raw_strength = detected_jump / baseline_noise
    strength = float(np.clip(1.0 - np.exp(-raw_strength / 2.0), 0.0, 1.0))

    return {
        "method": "tv",
        "end_ss_1b": end_ss_1b,
        "mode": "tv_jump",
        "lambda_tv": float(lambda_tv),
        "tv_iterations": int(n_iter),
        "baseline_len": int(baseline_len),
        "baseline_jump_mad": float(baseline_jump_mad),
        "jump_threshold": float(jump_threshold),
        "detected_jump": float(detected_jump),
        "strength": float(strength),
        "valid_jump_count": int(len(valid_jumps)),
    }


def _piecewise_constant_baseline_end(stlv: np.ndarray) -> Dict[str, Any]:
    curves = np.asarray(stlv, dtype=np.float64)
    if curves.ndim == 1:
        curves = curves[:, np.newaxis]
    if curves.ndim != 2 or curves.shape[0] < 3:
        return {
            "method": "piecewise_constant",
            "end_ss_1b": 1,
            "transition_index_1b": 1,
            "local_max_index_1b": 1,
        }

    x = np.mean(curves, axis=1)
    x = _gaussian_smooth_1d(x, sigma=1.0)
    n = x.size
    best_t_1b = 2
    best_mse = math.inf

    # MATLAB branch brute-force split: two piecewise constants with transition at t.
    for t_1b in range(2, n):
        if t_1b >= n:
            continue
        before = x[: t_1b - 1]
        after = x[t_1b:]
        if before.size == 0 or after.size == 0:
            continue
        before_mean = float(np.mean(before))
        after_mean = float(np.mean(after))
        pred = np.empty_like(x)
        pred[: t_1b - 1] = before_mean
        pred[t_1b - 1 :] = after_mean
        mse = float(np.mean((x - pred) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_t_1b = t_1b

    local_min_1b = best_t_1b
    while local_min_1b > 1 and x[local_min_1b - 1] >= x[local_min_1b - 2]:
        local_min_1b -= 1

    # Flat/noiseless baselines can make the backtrack run all the way to frame 1.
    # Step forward again while values remain within a small tolerance of the local minimum.
    mean_aif = float(np.mean(x)) if x.size > 0 else 0.0
    delta_abs = abs(mean_aif) * float(PIECEWISE_CONSTANT_BASELINE_FORWARD_DELTA_FRACTION)
    anchor_value = float(x[local_min_1b - 1])
    local_min_forward_1b = local_min_1b
    while local_min_forward_1b < best_t_1b:
        next_value = float(x[local_min_forward_1b])  # 1-based -> next sample
        if abs(next_value - anchor_value) <= delta_abs:
            local_min_forward_1b += 1
            continue
        break
    local_min_1b = local_min_forward_1b

    local_max_1b = best_t_1b
    while local_max_1b < n and x[local_max_1b - 1] <= x[local_max_1b]:
        local_max_1b += 1

    return {
        "method": "piecewise_constant",
        "end_ss_1b": int(max(1, min(local_min_1b, n))),
        "transition_index_1b": int(max(1, min(best_t_1b, n))),
        "local_max_index_1b": int(max(1, min(local_max_1b, n))),
        "transition_mse": (float(best_mse) if math.isfinite(best_mse) else None),
        "flat_forward_delta_fraction": float(PIECEWISE_CONSTANT_BASELINE_FORWARD_DELTA_FRACTION),
        "flat_forward_delta_abs": float(delta_abs),
    }


def _resolve_baseline_window(
    config: DcePipelineConfig,
    n_timepoints: int,
    stlv: Optional[np.ndarray] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    sentinel = object()
    start_raw = _stage_override(config, "steady_state_start", sentinel)
    end_raw = _stage_override(config, "steady_state_end", sentinel)
    auto_method_raw = _stage_override(config, "steady_state_auto_method", None)
    auto_method_requested = _normalize_steady_state_auto_method(auto_method_raw)

    start_is_set = start_raw is not sentinel and _override_value_is_set(start_raw)
    end_is_set = end_raw is not sentinel and _override_value_is_set(end_raw)

    start_1b = 1 if not start_is_set else int(start_raw)
    used_method = "manual" if end_is_set else "default"
    auto_details: Optional[Dict[str, Any]] = None
    end_source: str
    if end_is_set:
        end_1b = int(end_raw)
        end_source = "steady_state_end"
    else:
        auto_method = auto_method_requested if auto_method_requested != "none" else "legacy_sobel"
        if stlv is None:
            raise ValueError(
                "stage_overrides.steady_state_auto_method requires Stage-A AIF signal data to estimate baseline end"
            )
        detector_map: Dict[str, Callable[[np.ndarray], Dict[str, Any]]] = {
            "legacy_sobel": _legacy_sobel_baseline_end,
            "piecewise_constant": _piecewise_constant_baseline_end,
            "glr": _glr_baseline_end,
            "tv": _tv_baseline_end,
        }
        detector = detector_map[auto_method]
        auto_details = detector(stlv)
        end_1b = int(auto_details["end_ss_1b"])
        used_method = auto_method
        if auto_method_requested == "none":
            end_source = "default_auto_method:legacy_sobel"
        else:
            end_source = f"steady_state_auto_method:{auto_method}"

    start_1b = max(1, min(start_1b, n_timepoints))
    end_1b = max(start_1b, min(end_1b, n_timepoints))
    info = {
        "method_requested": auto_method_requested,
        "method_used": used_method,
        "start_1b": int(start_1b),
        "end_1b": int(end_1b),
        "source": end_source,
    }
    if auto_details is not None:
        info["auto_details"] = auto_details
    return start_1b - 1, end_1b, info


def _clean_ab(
    ab: np.ndarray,
    t1_values: np.ndarray,
    roi_indices: np.ndarray,
    threshold_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time = ab.shape[0]
    keep = np.ones(ab.shape[1], dtype=bool)
    cleaned = ab.copy()

    for col in range(ab.shape[1]):
        vec = cleaned[:, col]
        # MATLAB cleanAB.m uses TEST < 1 as the bad-point criterion.
        bad = np.where(vec < 1.0)[0]
        if bad.size > threshold_fraction * n_time:
            keep[col] = False
            continue
        if bad.size == 0:
            continue
        for idx in bad:
            start = max(0, int(idx) - 5)
            end = min(n_time, int(idx) + 6)
            neighbors = np.arange(start, end, dtype=np.int64)
            neighbors = neighbors[neighbors != int(idx)]
            neighbors = neighbors[~np.isin(neighbors, bad)]
            if neighbors.size < 2:
                keep[col] = False
                break
            rel = neighbors - int(idx)
            if not np.any(rel > 0) or not np.any(rel < 0):
                keep[col] = False
                break
            vec[int(idx)] = float(np.interp(float(idx), neighbors.astype(np.float64), vec[neighbors].astype(np.float64)))
        if keep[col]:
            cleaned[:, col] = vec

    return cleaned[:, keep], t1_values[keep], roi_indices[keep]


def _clean_r1(
    r1: np.ndarray,
    t1_values: np.ndarray,
    roi_indices: np.ndarray,
    threshold_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time = r1.shape[0]
    keep = np.ones(r1.shape[1], dtype=bool)
    cleaned = r1.copy()

    for col in range(r1.shape[1]):
        vec = cleaned[:, col]
        imag_bad = np.where(np.imag(vec) > 0)[0]
        if imag_bad.size > threshold_fraction * n_time:
            keep[col] = False
            continue
        if imag_bad.size > 0:
            vec = np.real(vec)

        bad = np.where((~np.isfinite(vec)) | (vec > 100.0))[0]
        if bad.size > threshold_fraction * n_time:
            keep[col] = False
            continue
        for idx in bad:
            start = max(0, int(idx) - 3)
            end = min(n_time, int(idx) + 4)
            neighbors = np.arange(start, end, dtype=np.int64)
            neighbors = neighbors[neighbors != int(idx)]
            if neighbors.size < 2:
                keep[col] = False
                break
            replacement = float(np.interp(float(idx), neighbors.astype(np.float64), np.asarray(vec[neighbors], dtype=np.float64)))
            vec[int(idx)] = replacement
            if not math.isfinite(replacement):
                keep[col] = False
                break
        if keep[col]:
            cleaned[:, col] = np.asarray(vec, dtype=np.float64)

    return cleaned[:, keep], t1_values[keep], roi_indices[keep]


def _save_stage_a_qc_figures(
    output_dir: Path,
    dynamic: np.ndarray,
    roi_mask: np.ndarray,
    aif_mask: np.ndarray,
    noise_mask: np.ndarray,
    ct: np.ndarray,
    cp: np.ndarray,
    r1_toi: np.ndarray,
    r1_lv: np.ndarray,
) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for QC figure saving") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, str] = {}

    timecurves_png = output_dir / "dce_timecurves.png"
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.plot(np.median(r1_toi, axis=1), "r")
    ax1.set_title("R1 ROI")
    ax2.plot(np.median(r1_lv, axis=1), "b")
    ax2.set_title("R1 AIF")
    ax3.plot(np.mean(ct, axis=1), "r")
    ax3.set_title("Ct mean")
    ax4.plot(np.mean(cp, axis=1), "b")
    ax4.set_title("Cp mean")
    fig.tight_layout()
    fig.savefig(timecurves_png, dpi=150)
    plt.close(fig)
    figure_paths["timecurves_png"] = str(timecurves_png)

    roi_png = output_dir / "dce_roi_overview.png"
    z_mid = dynamic.shape[2] // 2
    base = dynamic[:, :, z_mid, 0]
    fig2 = plt.figure(figsize=(8, 8))
    ax = fig2.add_subplot(1, 1, 1)
    ax.imshow(base.T, cmap="gray", origin="lower")
    ax.contour(roi_mask[:, :, z_mid].T.astype(float), levels=[0.5], colors=["red"], linewidths=0.8)
    ax.contour(aif_mask[:, :, z_mid].T.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
    ax.contour(noise_mask[:, :, z_mid].T.astype(float), levels=[0.5], colors=["yellow"], linewidths=0.8)
    ax.set_title("ROI(red) AIF(cyan) Noise(yellow)")
    fig2.tight_layout()
    fig2.savefig(roi_png, dpi=150)
    plt.close(fig2)
    figure_paths["roi_overview_png"] = str(roi_png)

    return figure_paths


def _run_stage_a_real(config: DcePipelineConfig) -> Dict[str, Any]:
    dynamic = _load_nifti_data(config.dynamic_files[0])
    if dynamic.ndim != 4:
        raise ValueError(f"Expected 4D dynamic input, got shape {dynamic.shape}")

    if not config.roi_files:
        raise ValueError("Stage-A real mode requires at least one ROI mask file")
    aif_mask_img = _load_nifti_data(config.aif_files[0])
    roi_mask_img = _load_nifti_data(config.roi_files[0])
    t1map_img = _load_nifti_data(config.t1map_files[0])

    if aif_mask_img.ndim == 4:
        aif_mask_img = aif_mask_img[..., 0]
    if roi_mask_img.ndim == 4:
        roi_mask_img = roi_mask_img[..., 0]
    if t1map_img.ndim == 4:
        t1map_img = t1map_img[..., 0]

    # Accept 2D inputs for single-slice dynamics.
    if dynamic.shape[2] == 1:
        if aif_mask_img.ndim == 2:
            aif_mask_img = aif_mask_img[..., np.newaxis]
        if roi_mask_img.ndim == 2:
            roi_mask_img = roi_mask_img[..., np.newaxis]
        if t1map_img.ndim == 2:
            t1map_img = t1map_img[..., np.newaxis]

    spatial = dynamic.shape[:3]
    if aif_mask_img.shape != spatial:
        raise ValueError(f"AIF mask shape {aif_mask_img.shape} does not match dynamic spatial shape {spatial}")
    if roi_mask_img.shape != spatial:
        raise ValueError(f"ROI mask shape {roi_mask_img.shape} does not match dynamic spatial shape {spatial}")
    if t1map_img.shape != spatial:
        raise ValueError(f"T1 map shape {t1map_img.shape} does not match dynamic spatial shape {spatial}")

    # MATLAB A_make_R1maps_func converts T1 maps from ms->s when values are large.
    # Keep the same unit behavior to avoid 1000x concentration scaling errors.
    t1_pos = t1map_img[np.isfinite(t1map_img) & (t1map_img > 0)]
    if t1_pos.size > 0 and float(np.median(t1_pos)) > 20.0:
        t1map_img = t1map_img / 1000.0

    aif_mask = aif_mask_img > 0
    roi_mask = roi_mask_img > 0
    if not np.any(aif_mask):
        raise ValueError("AIF mask has no positive voxels")
    if not np.any(roi_mask):
        raise ValueError("ROI mask has no positive voxels")
    if np.array_equal(aif_mask, roi_mask):
        raise ValueError("AIF mask must be a dedicated vascular ROI and cannot be identical to ROI mask")

    if config.noise_files:
        noise_mask_img = _load_nifti_data(config.noise_files[0])
        if noise_mask_img.ndim == 4:
            noise_mask_img = noise_mask_img[..., 0]
        if dynamic.shape[2] == 1 and noise_mask_img.ndim == 2:
            noise_mask_img = noise_mask_img[..., np.newaxis]
        if noise_mask_img.shape != spatial:
            raise ValueError(f"Noise mask shape {noise_mask_img.shape} does not match dynamic spatial shape {spatial}")
        noise_mask = noise_mask_img > 0
    else:
        noise_mask = np.zeros(spatial, dtype=bool)
        noise_size = int(_stage_override(config, "noise_pixsize", 5))
        noise_mask[:noise_size, :noise_size, :] = True

    # Match MATLAB linear indexing (column-major) for voxel ordering.
    dyn_2d = dynamic.reshape((-1, dynamic.shape[3]), order="F").T
    t1_flat = t1map_img.reshape(-1, order="F")

    lvind = np.where(aif_mask.reshape(-1, order="F"))[0]
    tumind = np.where(roi_mask.reshape(-1, order="F"))[0]
    noiseind = np.where(noise_mask.reshape(-1, order="F"))[0]
    if noiseind.size == 0:
        raise ValueError("Noise mask has no positive voxels")

    stlv = dyn_2d[:, lvind]
    sttum = dyn_2d[:, tumind]
    dynam_noise = np.std(dyn_2d[:, noiseind], axis=1)
    noise_mean = float(np.mean(dynam_noise))
    if noise_mean < 0:
        raise ValueError("Noise estimate is negative (internal error)")

    snr_filter = float(_stage_override(config, "snr_filter", 0.0))
    
    # Skip SNR filtering if noise is effectively zero (e.g., synthetic/perfect data)
    if noise_mean > 1e-12 and snr_filter > 0:
        voxel_snr = np.mean(stlv, axis=0) / noise_mean
        keep_snr = voxel_snr >= snr_filter
        if not np.any(keep_snr):
            raise ValueError("SNR filter removed all AIF voxels; lower snr_filter or noise threshold")
        lvind = lvind[keep_snr]
        stlv = stlv[:, keep_snr]
    
    t1_lv = t1_flat[lvind].astype(np.float64)
    blood_t1_override = _stage_override(config, "blood_t1_ms", None)
    if blood_t1_override is None:
        blood_t1_override = _stage_override(config, "blood_t1_sec", None)
    if blood_t1_override is None:
        blood_t1_override = _stage_override(config, "blood_t1", None)
    blood_t1_override_sec: Optional[float] = None
    if blood_t1_override is not None:
        blood_t1_override_sec = float(blood_t1_override)
        # Mirror MATLAB conventions where blood_t1 is often entered in ms.
        if blood_t1_override_sec > 20.0:
            blood_t1_override_sec /= 1000.0
        if blood_t1_override_sec <= 0.0:
            raise ValueError("blood_t1 override must be positive")
        t1_lv = np.full_like(t1_lv, blood_t1_override_sec, dtype=np.float64)

    t1_tum = t1_flat[tumind].astype(np.float64)

    n_time = dynamic.shape[3]
    timing = _resolve_dynamic_metadata(config, n_time)
    tr_ms = float(timing["tr_ms"])
    tr_sec = tr_ms / 1000.0
    fa_deg = float(timing["fa_deg"])
    time_resolution_min = float(timing["time_resolution_min"])

    relaxivity_override, _ = _explicit_stage_override(config, ("relaxivity",))
    hematocrit_override, _ = _explicit_stage_override(config, ("hematocrit",))
    relaxivity = float(
        relaxivity_override
        if relaxivity_override is not None
        else (timing.get("relaxivity") if timing.get("relaxivity") is not None else 3.4)
    )
    hematocrit = float(
        hematocrit_override
        if hematocrit_override is not None
        else (timing.get("hematocrit") if timing.get("hematocrit") is not None else 0.45)
    )
    if relaxivity <= 0.0:
        raise ValueError(f"relaxivity must be positive, got {relaxivity}")
    if not (0.0 <= hematocrit < 1.0):
        raise ValueError(f"hematocrit must be in [0, 1), got {hematocrit}")

    ss_start, ss_end, baseline_info = _resolve_baseline_window(config, n_time, stlv=stlv)
    baseline_slice = slice(ss_start, ss_end)

    # AIF path to R1
    sss = np.mean(stlv[baseline_slice, :], axis=0)
    
    # Filter out voxels with zero/near-zero baseline signal to prevent divide-by-zero
    valid_baseline = sss > 1e-10
    if not np.any(valid_baseline):
        raise ValueError("All AIF voxels have zero baseline signal")
    stlv = stlv[:, valid_baseline]
    t1_lv = t1_lv[valid_baseline]
    lvind = lvind[valid_baseline]
    sss = sss[valid_baseline]
    
    sstar_lv = (1.0 - np.exp(-tr_sec / t1_lv)) / (1.0 - np.cos(np.deg2rad(fa_deg)) * np.exp(-tr_sec / t1_lv))
    a = 1.0 - np.cos(np.deg2rad(fa_deg)) * sstar_lv[np.newaxis, :] * stlv / sss[np.newaxis, :]
    b = 1.0 - sstar_lv[np.newaxis, :] * stlv / sss[np.newaxis, :]
    ab_lv = a / b
    ab_lv, t1_lv, lvind = _clean_ab(ab_lv, t1_lv, lvind, threshold_fraction=0.05)
    if t1_lv.size == 0:
        raise ValueError("All AIF voxels removed after AB cleaning")

    r1_lv = (1.0 / tr_sec) * np.log(ab_lv)
    r1_lv, t1_lv, lvind = _clean_r1(r1_lv, t1_lv, lvind, threshold_fraction=0.005)
    if t1_lv.size == 0:
        raise ValueError("All AIF voxels removed after R1 cleaning")

    for j in range(t1_lv.size):
        scale = (1.0 / t1_lv[j]) - np.mean(r1_lv[baseline_slice, j])
        r1_lv[:, j] = r1_lv[:, j] + scale

    # ROI path to R1
    ss_tum = np.mean(sttum[baseline_slice, :], axis=0)
    
    # Filter out voxels with zero/near-zero baseline signal to prevent divide-by-zero
    valid_baseline_tum = ss_tum > 1e-10
    if not np.any(valid_baseline_tum):
        raise ValueError("All ROI voxels have zero baseline signal")
    sttum = sttum[:, valid_baseline_tum]
    t1_tum = t1_tum[valid_baseline_tum]
    tumind = tumind[valid_baseline_tum]
    ss_tum = ss_tum[valid_baseline_tum]
    
    sstar_tum = (1.0 - np.exp(-tr_sec / t1_tum)) / (1.0 - np.cos(np.deg2rad(fa_deg)) * np.exp(-tr_sec / t1_tum))
    a_tum = 1.0 - np.cos(np.deg2rad(fa_deg)) * sstar_tum[np.newaxis, :] * sttum / ss_tum[np.newaxis, :]
    b_tum = 1.0 - sstar_tum[np.newaxis, :] * sttum / ss_tum[np.newaxis, :]
    ab_tum = a_tum / b_tum
    ab_tum, t1_tum, tumind = _clean_ab(ab_tum, t1_tum, tumind, threshold_fraction=0.7)
    if t1_tum.size == 0:
        raise ValueError("All ROI voxels removed after AB cleaning")

    r1_toi = (1.0 / tr_sec) * np.log(ab_tum)
    r1_toi, t1_tum, tumind = _clean_r1(r1_toi, t1_tum, tumind, threshold_fraction=0.7)
    if t1_tum.size == 0:
        raise ValueError("All ROI voxels removed after R1 cleaning")

    for j in range(t1_tum.size):
        scale = (1.0 / t1_tum[j]) - np.mean(r1_toi[baseline_slice, j])
        r1_toi[:, j] = r1_toi[:, j] + scale

    cp = (r1_lv - (1.0 / t1_lv)[np.newaxis, :]) / (relaxivity * (1.0 - hematocrit))
    ct = (r1_toi - (1.0 / t1_tum)[np.newaxis, :]) / relaxivity

    delta_r1_lv = r1_lv - np.mean(r1_lv[baseline_slice, :], axis=0)[np.newaxis, :]
    delta_r1_toi = r1_toi - np.mean(r1_toi[baseline_slice, :], axis=0)[np.newaxis, :]

    timer = np.arange(n_time, dtype=np.float64) * time_resolution_min
    cp_mean = np.mean(cp, axis=1)
    cp_delta = np.diff(cp_mean, prepend=cp_mean[0])
    start_injection = int(np.argmax(cp_delta)) + 1  # 1-based to match MATLAB conventions
    injection_duration_frames = int(
        _stage_override(
            config,
            "injection_duration_frames",
            max(1, int(round(float(_stage_override(config, "injection_duration", 1.0))))),
        )
    )
    end_injection = min(n_time, start_injection + injection_duration_frames)

    figures = _save_stage_a_qc_figures(
        config.output_dir,
        dynamic,
        roi_mask,
        aif_mask,
        noise_mask,
        ct,
        cp,
        r1_toi,
        r1_lv,
    )

    return {
        "stage": "A",
        "status": "ok",
        "impl": "real",
        "rootname": str(_stage_override(config, "rootname", "python_dce")),
        "image_shape": [int(v) for v in spatial],
        "quant": True,
        "tr_ms": tr_ms,
        "fa_deg": fa_deg,
        "time_resolution_min": time_resolution_min,
        "relaxivity": relaxivity,
        "hematocrit": hematocrit,
        "aif_concentration_kind": timing.get("aif_concentration_kind"),
        "blood_t1_override_sec": blood_t1_override_sec,
        "steady_state_time": [ss_start + 1, ss_end],
        "steady_state_auto": baseline_info,
        "start_injection": start_injection,
        "end_injection": end_injection,
        "figure_paths": figures,
        "arrays": {
            "Cp": cp,
            "Ct": ct,
            "Stlv": stlv,
            "Sttum": sttum,
            "Sss": sss,
            "Ssstum": ss_tum,
            "R1tLV": r1_lv,
            "R1tTOI": r1_toi,
            "deltaR1LV": delta_r1_lv,
            "deltaR1TOI": delta_r1_toi,
            "T1LV": t1_lv,
            "T1TUM": t1_tum,
            "timer": timer,
            "lvind": lvind.astype(np.int64),
            "tumind": tumind.astype(np.int64),
            "noiseind": noiseind.astype(np.int64),
        },
    }


def _as_time_by_voxel(data: Any, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D/2D, got shape {arr.shape}")
    return arr


def _as_1d_float(data: Any, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _nearest_index(timer: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(timer - float(value))))


def _fill_nonfinite_1d(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if np.all(finite):
        return out
    if not np.any(finite):
        return np.zeros_like(out)
    idx = np.arange(out.size, dtype=np.float64)
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


def _load_vector_from_path(path: Path, key: str = "timer") -> np.ndarray:
    suffix = path.suffix.lower()
    if path.name.lower().endswith(".nii.gz"):
        suffix = ".nii.gz"

    if suffix == ".json":
        payload = _load_json(path)
        if key in payload:
            return _as_1d_float(payload[key], key)
        raise ValueError(f"JSON file {path} missing key '{key}'")

    if suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "," if suffix == ".csv" else None
        data = np.loadtxt(path, delimiter=delimiter)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            return _as_1d_float(data, key)
        return _as_1d_float(data[:, 0], key)

    if suffix == ".npy":
        return _as_1d_float(np.load(path), key)

    if suffix == ".npz":
        with np.load(path) as payload:
            if key in payload:
                return _as_1d_float(payload[key], key)
            if len(payload.files) == 1:
                return _as_1d_float(payload[payload.files[0]], key)
            raise ValueError(f"NPZ file {path} missing key '{key}'")

    if suffix == ".mat":
        try:
            from scipy.io import loadmat  # type: ignore
        except Exception as exc:
            raise RuntimeError("scipy is required to read MATLAB .mat files") from exc

        payload = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        if key in payload:
            return _as_1d_float(payload[key], key)
        raise ValueError(f"MAT file {path} missing variable '{key}'")

    raise ValueError(f"Unsupported vector file format for {path}")


def _resolve_stage_b_timer(config: DcePipelineConfig, stage_a: Dict[str, Any], n_time: int) -> np.ndarray:
    time_vector_path = _stage_override(config, "time_vector_path", None)
    timer_path = _stage_override(config, "timer_path", None)
    legacy_timevectpath = _stage_override(config, "timevectpath", None)
    use_legacy_timevect = True
    timevectyn = _stage_override(config, "timevectyn", None)
    if timevectyn is not None:
        use_legacy_timevect = _to_bool(timevectyn, False)

    path_value = time_vector_path or timer_path
    if path_value is None and use_legacy_timevect:
        path_value = legacy_timevectpath

    timer: Optional[np.ndarray] = None
    if path_value:
        timer = _load_vector_from_path(Path(path_value).expanduser().resolve(), key="timer")
    elif isinstance(stage_a.get("arrays"), dict) and "timer" in stage_a["arrays"]:
        timer = _as_1d_float(stage_a["arrays"]["timer"], "timer")

    if timer is None:
        time_resolution = _stage_override(config, "time_resolution_min", stage_a.get("time_resolution_min", None))
        if time_resolution is None:
            time_resolution_sec = _stage_override(
                config,
                "time_resolution_sec",
                _stage_override(config, "time_resolution", None),
            )
            if time_resolution_sec is not None:
                time_resolution = float(time_resolution_sec) / 60.0
        if time_resolution is None:
            raise ValueError("Stage-B requires timer data; set stage_overrides.time_resolution_min")
        timer = np.arange(n_time, dtype=np.float64) * float(time_resolution)

    if timer.size > n_time:
        timer = timer[:n_time]
    elif timer.size < n_time:
        if timer.size == 0:
            raise ValueError("Resolved timer vector is empty")
        if timer.size == 1:
            step = float(_stage_override(config, "time_resolution_min", stage_a.get("time_resolution_min", 1.0)))
            if not math.isfinite(step) or step <= 0:
                time_resolution_sec = _stage_override(
                    config,
                    "time_resolution_sec",
                    _stage_override(config, "time_resolution", None),
                )
                if time_resolution_sec is not None:
                    step = float(time_resolution_sec) / 60.0
        else:
            step = float(timer[-1] - timer[-2])
            if not math.isfinite(step) or step <= 0:
                step = float(_stage_override(config, "time_resolution_min", stage_a.get("time_resolution_min", 1.0)))
                if not math.isfinite(step) or step <= 0:
                    time_resolution_sec = _stage_override(
                        config,
                        "time_resolution_sec",
                        _stage_override(config, "time_resolution", None),
                    )
                    if time_resolution_sec is not None:
                        step = float(time_resolution_sec) / 60.0
        extension = timer[-1] + step * np.arange(1, n_time - timer.size + 1, dtype=np.float64)
        timer = np.concatenate([timer, extension])

    if np.nanmax(timer) > 100.0:
        timer = timer / 60.0
    return _fill_nonfinite_1d(timer)


def _resolve_stage_b_limits(config: DcePipelineConfig) -> Tuple[float, float]:
    start_time = float(_stage_override(config, "start_time_min", _stage_override(config, "start_time", 0.0)))
    end_time = float(_stage_override(config, "end_time_min", _stage_override(config, "end_time", 0.0)))
    return start_time, end_time


def _restrict_timer_window(timer: np.ndarray, start_time: float, end_time: float) -> Tuple[int, int]:
    if timer.size < 2:
        raise ValueError("Timer vector must have at least 2 elements for Stage-B")

    start_idx = _nearest_index(timer, start_time) if start_time > 0 else 0
    end_idx = _nearest_index(timer, end_time) + 1 if end_time > 0 else timer.size
    start_idx = max(0, min(start_idx, timer.size - 1))
    end_idx = max(start_idx + 1, min(end_idx, timer.size))
    return start_idx, end_idx


def _resolve_stage_b_aif_mode(config: DcePipelineConfig) -> str:
    mode_raw = _stage_override(config, "aif_curve_mode", None)
    if mode_raw is None or str(mode_raw).strip() == "":
        mode_raw = _stage_override(config, "aif_type", None)
    if mode_raw is None or str(mode_raw).strip() == "":
        mode_raw = config.aif_mode
    mode = str(mode_raw).strip().lower()
    if mode in {"1", "fitted", "fit"}:
        mode = "fitted"
    elif mode in {"2", "raw"}:
        mode = "raw"
    elif mode in {"3", "import", "imported"}:
        mode = "imported"
    if mode == "auto":
        mode = "imported" if _resolve_imported_aif_path(config) is not None else "fitted"
    if mode not in {"fitted", "raw", "imported"}:
        raise ValueError(f"Unsupported Stage-B AIF mode '{mode}'")
    if mode == "imported" and _resolve_imported_aif_path(config) is None:
        raise ValueError("Stage-B imported mode requires imported_aif_path")
    return mode


def _resolve_imported_aif_path(config: DcePipelineConfig) -> Optional[Path]:
    if config.imported_aif_path is not None:
        return config.imported_aif_path.expanduser().resolve()
    path_val = _stage_override(config, "import_aif_path", _stage_override(config, "imported_aif_path", None))
    if path_val is None:
        return None
    path_text = str(path_val).strip()
    if not path_text:
        return None
    return Path(path_text).expanduser().resolve()


def _resolve_stage_b_injection_window(
    config: DcePipelineConfig,
    stage_a: Dict[str, Any],
    timer_full: np.ndarray,
) -> Tuple[float, float]:
    start_override = _stage_override(config, "start_injection_min", _stage_override(config, "start_injection", None))
    end_override = _stage_override(config, "end_injection_min", _stage_override(config, "end_injection", None))

    if start_override is not None:
        start_val = float(start_override)
    else:
        source = float(stage_a.get("start_injection", 1.0))
        if abs(source - round(source)) < 1e-8 and 1 <= int(round(source)) <= timer_full.size:
            start_val = float(timer_full[int(round(source)) - 1])
        else:
            start_val = source

    if end_override is not None:
        end_val = float(end_override)
    else:
        source = float(stage_a.get("end_injection", start_val))
        if abs(source - round(source)) < 1e-8 and 1 <= int(round(source)) <= timer_full.size:
            end_val = float(timer_full[int(round(source)) - 1])
        else:
            end_val = source

    if end_val < start_val:
        end_val = start_val
    return start_val, end_val


def _aif_biexp_con(
    timer: np.ndarray,
    step: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
    fitting_au: bool,
    baseline: float,
) -> np.ndarray:
    time = np.asarray(timer, dtype=np.float64).reshape(-1)
    stepv = np.asarray(step, dtype=np.float64).reshape(-1)
    if time.size != stepv.size:
        raise ValueError("timer and step must have same length")

    on = np.flatnonzero(stepv > 0)
    out = np.zeros(time.size, dtype=np.float64)
    base = float(baseline) if fitting_au else 0.0
    if on.size == 0:
        out.fill(base)
        return out

    start_idx = int(on[0])
    end_idx = int(on[-1])

    idx = np.arange(time.size)
    pre = idx < start_idx
    slope = (idx >= start_idx) & (idx < end_idx)
    post = idx >= end_idx

    out[pre] = base

    if np.any(slope):
        t_start = float(time[start_idx])
        t_end = float(time[end_idx])
        duration = max(1e-12, t_end - t_start)
        frac = (time[slope] - t_start) / duration
        out[slope] = base + ((a - base) + (b - base)) * frac

    if np.any(post):
        dt = np.maximum(0.0, time[post] - float(time[end_idx]))
        out[post] = a * np.exp(-c * dt) + b * np.exp(-d * dt)

    return out


def _parse_4float_override(config: DcePipelineConfig, key: str, default: List[float]) -> np.ndarray:
    value = _stage_override(config, key, default)
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        parsed: List[float] = []
        for token in parts:
            num = _parse_numeric_token(token)
            if num is None:
                raise ValueError(f"{key} contains non-numeric token '{token}'")
            parsed.append(float(num))
        arr = np.array(parsed, dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{key} must contain exactly 4 numeric values")
    return arr


def _adjusted_rsquare(y_true: np.ndarray, y_fit: np.ndarray, n_params: int) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    yhat = np.asarray(y_fit, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if np.count_nonzero(mask) <= (n_params + 1):
        return float("nan")

    y = y[mask]
    yhat = yhat[mask]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 1.0
    r2 = 1.0 - (ss_res / ss_tot)
    n = float(y.size)
    p = float(n_params)
    return 1.0 - (1.0 - r2) * ((n - 1.0) / max(1.0, (n - p - 1.0)))


def _fit_aif_biexp(
    config: DcePipelineConfig,
    timer: np.ndarray,
    curve: np.ndarray,
    start_injection_min: float,
    end_injection_min: float,
    fitting_au: bool,
) -> Dict[str, Any]:
    try:
        from scipy.optimize import curve_fit  # type: ignore
    except Exception as exc:
        raise RuntimeError("scipy is required for Stage-B fitted AIF mode") from exc

    timer = _as_1d_float(timer, "timer")
    curve = _fill_nonfinite_1d(_as_1d_float(curve, "curve"))

    start_idx = _nearest_index(timer, start_injection_min)
    end_idx = _nearest_index(timer, end_injection_min)
    if end_idx < start_idx:
        end_idx = start_idx

    step = np.zeros(timer.size, dtype=np.float64)
    step[start_idx : end_idx + 1] = 1.0

    weighted = curve * step
    max_idx = int(np.argmax(weighted))
    fit_step = step.copy()
    fit_step[max_idx + 1 :] = 0.0
    if np.count_nonzero(fit_step > 0) == 0:
        fit_step = step.copy()

    onset = np.flatnonzero(fit_step > 0)
    baseline = float(np.mean(curve[: onset[0] + 1])) if onset.size > 0 else float(curve[0])
    maxer = float(curve[max_idx])
    if not math.isfinite(maxer) or maxer <= 0:
        maxer = float(np.max(curve))
        max_idx = int(np.argmax(curve))

    lower = _parse_4float_override(config, "aif_lower_limits", [0.0, 0.0, 0.0, 0.0])
    upper = _parse_4float_override(config, "aif_upper_limits", [5.0, 5.0, 50.0, 50.0])
    initial = _parse_4float_override(config, "aif_initial_values", [1.0, 1.0, 1.0, 0.01])

    upper[0] = max(1e-12, maxer * 2.0)
    upper[1] = max(1e-12, maxer * 2.0)
    initial[0] = max(1e-12, maxer * 0.5)
    initial[1] = max(1e-12, maxer * 0.5)
    initial = np.minimum(np.maximum(initial, lower + 1e-12), upper - 1e-12)

    aif_maxiter = int(_safe_float(_stage_override(config, "aif_MaxIter", 1000), 1000))
    max_nfev = int(_safe_float(_stage_override(config, "aif_MaxFunEvals", aif_maxiter), aif_maxiter))
    aif_tol_fun = max(_safe_float(_stage_override(config, "aif_TolFun", 1e-20), 1e-20), np.finfo(np.float64).eps)
    aif_tol_x = max(_safe_float(_stage_override(config, "aif_TolX", 1e-23), 1e-23), np.finfo(np.float64).eps)
    aif_loss = _scipy_loss_from_robust(_stage_override(config, "aif_Robust", "off"))

    def fit_fn(tvals: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        return _aif_biexp_con(tvals, fit_step, a, b, c, d, fitting_au=fitting_au, baseline=baseline)

    fit_success = True
    params = initial.copy()
    try:
        fit_kwargs: Dict[str, Any] = {
            "method": "trf",
            "max_nfev": max_nfev,
            "ftol": aif_tol_fun,
            "xtol": aif_tol_x,
        }
        if aif_loss != "linear":
            fit_kwargs["loss"] = aif_loss
        try:
            params, _ = curve_fit(
                fit_fn,
                timer,
                curve,
                p0=initial,
                bounds=(lower, upper),
                **fit_kwargs,
            )
        except TypeError:
            # Compatibility fallback for older SciPy builds.
            fallback_kwargs: Dict[str, Any] = {
                "method": "trf",
                "maxfev": max_nfev,
                "ftol": aif_tol_fun,
                "xtol": aif_tol_x,
            }
            params, _ = curve_fit(
                fit_fn,
                timer,
                curve,
                p0=initial,
                bounds=(lower, upper),
                **fallback_kwargs,
            )
    except Exception:
        fit_success = False

    fitted = fit_fn(timer, float(params[0]), float(params[1]), float(params[2]), float(params[3]))
    return {
        "curve": fitted,
        "params": np.asarray(params, dtype=np.float64),
        "step": fit_step,
        "baseline": baseline,
        "max_index": max_idx,
        "rsquare_adj": _adjusted_rsquare(curve, fitted, n_params=4),
        "fit_success": fit_success,
    }


def _load_imported_aif(path: Path) -> Dict[str, Any]:
    p = path.expanduser().resolve()
    suffix = p.suffix.lower()

    def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("Bdata", "bdata"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break

        cp_use = payload.get("Cp_use", payload.get("cp_use"))
        stlv_use = payload.get("Stlv_use", payload.get("stlv_use"))
        timer = payload.get("timer", payload.get("Timer"))
        start_injection = payload.get("start_injection", payload.get("start_injection_min"))
        return {
            "Cp_use": _as_1d_float(cp_use, "Cp_use") if cp_use is not None else None,
            "Stlv_use": _as_1d_float(stlv_use, "Stlv_use") if stlv_use is not None else None,
            "timer": _as_1d_float(timer, "timer") if timer is not None else None,
            "start_injection": float(start_injection) if start_injection is not None else None,
        }

    if suffix == ".json":
        payload = _load_json(p)
        out = _normalize_payload(payload)
        if out["Cp_use"] is None:
            raise ValueError(f"Imported AIF JSON {p} missing Cp_use")
        return out

    if suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "," if suffix == ".csv" else None
        data = np.loadtxt(p, delimiter=delimiter)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            return {"Cp_use": _as_1d_float(data, "Cp_use"), "Stlv_use": None, "timer": None, "start_injection": None}
        if data.shape[1] == 1:
            return {
                "Cp_use": _as_1d_float(data[:, 0], "Cp_use"),
                "Stlv_use": None,
                "timer": None,
                "start_injection": None,
            }
        stlv_col = data[:, 2] if data.shape[1] > 2 else None
        return {
            "Cp_use": _as_1d_float(data[:, 1], "Cp_use"),
            "Stlv_use": _as_1d_float(stlv_col, "Stlv_use") if stlv_col is not None else None,
            "timer": _as_1d_float(data[:, 0], "timer"),
            "start_injection": None,
        }

    if suffix == ".npy":
        arr = np.asarray(np.load(p), dtype=np.float64)
        if arr.ndim == 1:
            return {"Cp_use": _as_1d_float(arr, "Cp_use"), "Stlv_use": None, "timer": None, "start_injection": None}
        if arr.ndim == 2 and arr.shape[1] >= 2:
            stlv_col = arr[:, 2] if arr.shape[1] > 2 else None
            return {
                "Cp_use": _as_1d_float(arr[:, 1], "Cp_use"),
                "Stlv_use": _as_1d_float(stlv_col, "Stlv_use") if stlv_col is not None else None,
                "timer": _as_1d_float(arr[:, 0], "timer"),
                "start_injection": None,
            }
        raise ValueError(f"Unsupported NPY AIF array shape {arr.shape}")

    if suffix == ".npz":
        with np.load(p) as payload:
            cp_key = next((k for k in ("Cp_use", "cp_use", "Cp", "cp") if k in payload), None)
            if cp_key is None:
                if len(payload.files) == 1:
                    cp_key = payload.files[0]
                else:
                    raise ValueError(f"NPZ imported AIF {p} missing Cp_use")
            timer_key = next((k for k in ("timer", "Timer", "time") if k in payload), None)
            stlv_key = next((k for k in ("Stlv_use", "stlv_use", "Stlv", "stlv") if k in payload), None)
            start_key = next((k for k in ("start_injection", "start_injection_min") if k in payload), None)
            start_injection = float(payload[start_key]) if start_key is not None else None
            return {
                "Cp_use": _as_1d_float(payload[cp_key], "Cp_use"),
                "Stlv_use": _as_1d_float(payload[stlv_key], "Stlv_use") if stlv_key is not None else None,
                "timer": _as_1d_float(payload[timer_key], "timer") if timer_key is not None else None,
                "start_injection": start_injection,
            }

    if suffix == ".mat":
        try:
            from scipy.io import loadmat  # type: ignore
        except Exception as exc:
            raise RuntimeError("scipy is required to read imported AIF .mat files") from exc

        raw = loadmat(str(p), squeeze_me=True, struct_as_record=False)
        payload: Dict[str, Any] = {}
        for key, value in raw.items():
            if key.startswith("__"):
                continue
            payload[key] = value
        if "Bdata" in payload and not isinstance(payload["Bdata"], dict):
            bdata = payload["Bdata"]
            for key in ("Cp_use", "Stlv_use", "timer", "start_injection"):
                if hasattr(bdata, key):
                    payload[key] = getattr(bdata, key)
        out = _normalize_payload(payload)
        if out["Cp_use"] is None:
            raise ValueError(f"Imported AIF MAT {p} missing Cp_use")
        return out

    raise ValueError(f"Unsupported imported AIF format: {p}")


def _resample_or_pad_curve(curve: np.ndarray, dst_timer: np.ndarray, src_timer: Optional[np.ndarray]) -> np.ndarray:
    target = np.asarray(dst_timer, dtype=np.float64)
    source = _as_1d_float(curve, "curve")

    if src_timer is not None:
        src = _as_1d_float(src_timer, "src_timer")
        if src.size == source.size:
            order = np.argsort(src)
            src_sorted = src[order]
            source_sorted = source[order]
            return np.interp(target, src_sorted, source_sorted, left=source_sorted[0], right=source_sorted[-1])

    if source.size == target.size:
        return source
    if source.size > target.size:
        return source[: target.size]

    out = np.empty(target.size, dtype=np.float64)
    out[: source.size] = source
    out[source.size :] = source[-1]
    return out


def _align_imported_curve(
    curve: np.ndarray,
    import_timer: Optional[np.ndarray],
    import_start: Optional[float],
    timer: np.ndarray,
    data_start: float,
) -> Tuple[np.ndarray, int]:
    out = _as_1d_float(curve, "curve").copy()
    if import_timer is None or import_start is None or out.size != timer.size:
        return out, 0

    aif_start_idx = _nearest_index(import_timer, import_start)
    data_start_idx = _nearest_index(timer, data_start)
    shift = int(data_start_idx - aif_start_idx)
    if shift == 0:
        return out, 0

    shifted = np.roll(out, shift)
    n = shifted.size
    if shift < 0:
        pad = -shift
        if pad >= n:
            shifted[:] = shifted[0]
        else:
            fill = shifted[n - pad - 1]
            shifted[n - pad :] = fill
    else:
        if shift >= n:
            shifted[:] = 0.0
        else:
            shifted[:shift] = 0.0
    return shifted, shift


def _save_stage_b_qc_figure(
    output_dir: Path,
    timer: np.ndarray,
    cp_roi: np.ndarray,
    cp_use: np.ndarray,
    stlv_roi: np.ndarray,
    stlv_use: np.ndarray,
) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for Stage-B QC figure saving") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(timer, cp_roi, "r.", label="Original Plasma Curve")
    ax1.plot(timer, cp_use, "b", label="Selected Curve")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Concentration (mM)")
    ax1.legend(loc="best")

    ax2.plot(timer, stlv_roi, "r.", label="Original Plasma Curve: Raw data")
    ax2.plot(timer, stlv_use, "b", label="Selected Curve")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Signal (a.u)")
    ax2.legend(loc="best")

    fig.tight_layout()
    out_path = output_dir / "dce_aif_fitting.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"aif_fitting_png": str(out_path)}


def _run_stage_b_real(config: DcePipelineConfig, stage_a: Dict[str, Any]) -> Dict[str, Any]:
    arrays = stage_a.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Stage-B real mode requires Stage-A arrays")

    cp_all = _as_time_by_voxel(arrays["Cp"], "Cp")
    ct_all = _as_time_by_voxel(arrays["Ct"], "Ct")
    stlv_all = _as_time_by_voxel(arrays["Stlv"], "Stlv")
    sttum_all = _as_time_by_voxel(arrays["Sttum"], "Sttum")

    n_time = cp_all.shape[0]
    if ct_all.shape[0] != n_time or stlv_all.shape[0] != n_time or sttum_all.shape[0] != n_time:
        raise ValueError("Stage-A arrays Cp/Ct/Stlv/Sttum must have matching time dimension")

    timer_full = _resolve_stage_b_timer(config, stage_a, n_time)
    start_time_min, end_time_min = _resolve_stage_b_limits(config)
    start_idx, end_idx = _restrict_timer_window(timer_full, start_time_min, end_time_min)
    timer = timer_full[start_idx:end_idx]

    cp = cp_all[start_idx:end_idx, :]
    ct = ct_all[start_idx:end_idx, :]
    stlv = stlv_all[start_idx:end_idx, :]
    sttum = sttum_all[start_idx:end_idx, :]
    cp_roi = np.mean(cp, axis=1)
    stlv_roi = np.mean(stlv, axis=1)

    start_injection_min, end_injection_min = _resolve_stage_b_injection_window(config, stage_a, timer_full)
    start_injection_min = float(np.clip(start_injection_min, timer[0], timer[-1]))
    end_injection_min = float(np.clip(end_injection_min, start_injection_min, timer[-1]))

    aif_mode = _resolve_stage_b_aif_mode(config)
    fit_info: Dict[str, Any] = {}
    import_shift = 0

    if aif_mode == "fitted":
        fit_cp = _fit_aif_biexp(
            config,
            timer=timer,
            curve=cp_roi,
            start_injection_min=start_injection_min,
            end_injection_min=end_injection_min,
            fitting_au=False,
        )
        fit_stlv = _fit_aif_biexp(
            config,
            timer=timer,
            curve=stlv_roi,
            start_injection_min=start_injection_min,
            end_injection_min=end_injection_min,
            fitting_au=True,
        )
        cp_use = fit_cp["curve"]
        stlv_use = fit_stlv["curve"]
        fit_info = {
            "fit_success_cp": bool(fit_cp["fit_success"]),
            "fit_success_stlv": bool(fit_stlv["fit_success"]),
            "fit_rsquared_cp_adj": float(fit_cp["rsquare_adj"]),
            "fit_rsquared_stlv_adj": float(fit_stlv["rsquare_adj"]),
            "fit_params_cp": fit_cp["params"],
            "fit_params_stlv": fit_stlv["params"],
        }
        aif_name = "fitted"
    elif aif_mode == "raw":
        cp_use = cp_roi.copy()
        stlv_use = stlv_roi.copy()
        aif_name = "raw"
    else:
        imported_path = _resolve_imported_aif_path(config)
        imported = _load_imported_aif(imported_path if imported_path else Path(""))
        cp_use = _resample_or_pad_curve(imported["Cp_use"], timer, imported["timer"])
        imported_stlv = imported["Stlv_use"] if imported["Stlv_use"] is not None else imported["Cp_use"]
        stlv_use = _resample_or_pad_curve(imported_stlv, timer, imported["timer"])
        cp_use, import_shift = _align_imported_curve(
            cp_use,
            import_timer=imported["timer"],
            import_start=imported["start_injection"],
            timer=timer,
            data_start=start_injection_min,
        )
        stlv_use, _ = _align_imported_curve(
            stlv_use,
            import_timer=imported["timer"],
            import_start=imported["start_injection"],
            timer=timer,
            data_start=start_injection_min,
        )
        aif_name = "imported"

    figure_paths = _save_stage_b_qc_figure(
        output_dir=config.output_dir,
        timer=timer,
        cp_roi=cp_roi,
        cp_use=cp_use,
        stlv_roi=stlv_roi,
        stlv_use=stlv_use,
    )

    step = np.array([start_injection_min, end_injection_min], dtype=np.float64)
    time_resolution_min = float(stage_a.get("time_resolution_min", np.median(np.diff(timer_full))))
    result: Dict[str, Any] = {
        "stage": "B",
        "status": "ok",
        "impl": "real",
        "aif_mode": aif_mode,
        "aif_name": aif_name,
        "quant": bool(stage_a.get("quant", True)),
        "start_time_index": int(start_idx + 1),  # 1-based for MATLAB compatibility
        "end_time_index": int(end_idx),  # 1-based inclusive end
        "start_injection_min": start_injection_min,
        "end_injection_min": end_injection_min,
        "time_resolution_min": time_resolution_min,
        "numvoxels": int(ct.shape[1]),
        "import_shift": int(import_shift),
        "figure_paths": figure_paths,
        "arrays": {
            "CpROI": cp_roi,
            "Cp_use": cp_use,
            "StlvROI": stlv_roi,
            "Stlv_use": stlv_use,
            "timer": timer,
            "Ct": ct,
            "Sttum": sttum,
            "step": step,
        },
    }

    if fit_info:
        result.update(fit_info)

    for name in ("R1tTOI", "R1tLV", "deltaR1LV", "deltaR1TOI"):
        if name in arrays:
            result["arrays"][name] = _as_time_by_voxel(arrays[name], name)[start_idx:end_idx, :]
    for name in ("T1TUM", "tumind", "Sss", "Ssstum"):
        if name in arrays:
            result["arrays"][name] = np.asarray(arrays[name])

    return result


def _sanitize_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return out.strip("_") or "roi"


def _safe_float(value: Any, default: float) -> float:
    parsed = _parse_numeric_token(value)
    if parsed is not None and math.isfinite(parsed):
        return float(parsed)
    return float(default)


def _stage_d_fit_prefs(config: DcePipelineConfig) -> Dict[str, Any]:
    voxel_max_iter = int(_safe_float(_stage_override(config, "voxel_MaxIter", 50), 50))
    voxel_max_nfev = int(_safe_float(_stage_override(config, "voxel_MaxFunEvals", voxel_max_iter), voxel_max_iter))
    return {
        "lower_limit_ktrans": _safe_float(_stage_override(config, "voxel_lower_limit_ktrans", 1e-7), 1e-7),
        "upper_limit_ktrans": _safe_float(_stage_override(config, "voxel_upper_limit_ktrans", 2.0), 2.0),
        "initial_value_ktrans": _safe_float(_stage_override(config, "voxel_initial_value_ktrans", 2e-4), 2e-4),
        "lower_limit_ve": _safe_float(_stage_override(config, "voxel_lower_limit_ve", 0.02), 0.02),
        "upper_limit_ve": _safe_float(_stage_override(config, "voxel_upper_limit_ve", 1.0), 1.0),
        "initial_value_ve": _safe_float(_stage_override(config, "voxel_initial_value_ve", 0.2), 0.2),
        "lower_limit_vp": _safe_float(_stage_override(config, "voxel_lower_limit_vp", 1e-3), 1e-3),
        "upper_limit_vp": _safe_float(_stage_override(config, "voxel_upper_limit_vp", 1.0), 1.0),
        "initial_value_vp": _safe_float(_stage_override(config, "voxel_initial_value_vp", 0.02), 0.02),
        "lower_limit_fp": _safe_float(_stage_override(config, "voxel_lower_limit_fp", 1e-3), 1e-3),
        "upper_limit_fp": _safe_float(_stage_override(config, "voxel_upper_limit_fp", 100.0), 100.0),
        "initial_value_fp": _safe_float(_stage_override(config, "voxel_initial_value_fp", 0.2), 0.2),
        "lower_limit_tp": _safe_float(_stage_override(config, "voxel_lower_limit_tp", 0.0), 0.0),
        "upper_limit_tp": _safe_float(_stage_override(config, "voxel_upper_limit_tp", 1e6), 1e6),
        "initial_value_tp": _safe_float(_stage_override(config, "voxel_initial_value_tp", 0.05), 0.05),
        "lower_limit_tau": _safe_float(_stage_override(config, "voxel_lower_limit_tau", 0.0), 0.0),
        "upper_limit_tau": _safe_float(_stage_override(config, "voxel_upper_limit_tau", 100.0), 100.0),
        "initial_value_tau": _safe_float(_stage_override(config, "voxel_initial_value_tau", 0.01), 0.01),
        "lower_limit_ktrans_rr": _safe_float(_stage_override(config, "voxel_lower_limit_ktrans_RR", 1e-7), 1e-7),
        "upper_limit_ktrans_rr": _safe_float(_stage_override(config, "voxel_upper_limit_ktrans_RR", 2.0), 2.0),
        "initial_value_ktrans_rr": _safe_float(_stage_override(config, "voxel_initial_value_ktrans_RR", 0.1), 0.1),
        "value_ve_rr": _safe_float(_stage_override(config, "voxel_value_ve_RR", 0.08), 0.08),
        "tol_fun": _safe_float(_stage_override(config, "voxel_TolFun", 1e-12), 1e-12),
        "tol_x": _safe_float(_stage_override(config, "voxel_TolX", 1e-6), 1e-6),
        "max_iter": int(voxel_max_iter),
        "max_nfev": int(voxel_max_nfev),
        "robust": str(_stage_override(config, "voxel_Robust", "off")).strip(),
        "gpu_tolerance": _safe_float(_stage_override(config, "gpu_tolerance", 1e-6), 1e-6),
        "gpu_max_n_iterations": int(_safe_float(_stage_override(config, "gpu_max_n_iterations", 200), 200)),
        "gpu_initial_value_ktrans": _safe_float(_stage_override(config, "gpu_initial_value_ktrans", 2e-4), 2e-4),
        "gpu_initial_value_ve": _safe_float(_stage_override(config, "gpu_initial_value_ve", 0.2), 0.2),
        "gpu_initial_value_vp": _safe_float(_stage_override(config, "gpu_initial_value_vp", 0.02), 0.02),
        "gpu_initial_value_fp": _safe_float(_stage_override(config, "gpu_initial_value_fp", 0.2), 0.2),
        "fxr_fw": _safe_float(_stage_override(config, "fxr_fw", 0.8), 0.8),
        # Optional model-specific overrides to tune unstable models without impacting others.
        "2cxm_lower_limit_ktrans": _stage_override(config, "voxel_lower_limit_ktrans_2cxm", 1e-7),
        "2cxm_upper_limit_ktrans": _stage_override(config, "voxel_upper_limit_ktrans_2cxm", 2.0),
        "2cxm_initial_value_ktrans": _stage_override(config, "voxel_initial_value_ktrans_2cxm", 2e-4),
        "2cxm_lower_limit_ve": _stage_override(config, "voxel_lower_limit_ve_2cxm", 0.05),
        "2cxm_upper_limit_ve": _stage_override(config, "voxel_upper_limit_ve_2cxm", 1.0),
        "2cxm_initial_value_ve": _stage_override(config, "voxel_initial_value_ve_2cxm", 0.15),
        "2cxm_lower_limit_vp": _stage_override(config, "voxel_lower_limit_vp_2cxm", 1e-3),
        "2cxm_upper_limit_vp": _stage_override(config, "voxel_upper_limit_vp_2cxm", 1.0),
        "2cxm_initial_value_vp": _stage_override(config, "voxel_initial_value_vp_2cxm", 0.02),
        "2cxm_lower_limit_fp": _stage_override(config, "voxel_lower_limit_fp_2cxm", 1e-3),
        "2cxm_upper_limit_fp": _stage_override(config, "voxel_upper_limit_fp_2cxm", 20.0),
        "2cxm_initial_value_fp": _stage_override(config, "voxel_initial_value_fp_2cxm", 0.35),
        "2cxm_max_nfev": _stage_override(config, "voxel_MaxFunEvals_2cxm", 140),
        "2cxm_max_iter": _stage_override(config, "voxel_MaxIter_2cxm", 140),
        "2cxm_robust": _stage_override(config, "voxel_Robust_2cxm", None),
        "tissue_uptake_lower_limit_ktrans": _stage_override(config, "voxel_lower_limit_ktrans_tissue_uptake", 1e-7),
        "tissue_uptake_upper_limit_ktrans": _stage_override(config, "voxel_upper_limit_ktrans_tissue_uptake", 2.0),
        "tissue_uptake_initial_value_ktrans": _stage_override(config, "voxel_initial_value_ktrans_tissue_uptake", 2e-4),
        "tissue_uptake_lower_limit_vp": _stage_override(config, "voxel_lower_limit_vp_tissue_uptake", 1e-3),
        "tissue_uptake_upper_limit_vp": _stage_override(config, "voxel_upper_limit_vp_tissue_uptake", 1.0),
        "tissue_uptake_initial_value_vp": _stage_override(config, "voxel_initial_value_vp_tissue_uptake", 0.02),
        "tissue_uptake_lower_limit_fp": _stage_override(config, "voxel_lower_limit_fp_tissue_uptake", 1e-3),
        "tissue_uptake_upper_limit_fp": _stage_override(config, "voxel_upper_limit_fp_tissue_uptake", 20.0),
        "tissue_uptake_initial_value_fp": _stage_override(config, "voxel_initial_value_fp_tissue_uptake", 0.35),
        "tissue_uptake_lower_limit_tp": _stage_override(config, "voxel_lower_limit_tp_tissue_uptake", 0.0),
        "tissue_uptake_upper_limit_tp": _stage_override(config, "voxel_upper_limit_tp_tissue_uptake", 1.5),
        "tissue_uptake_initial_value_tp": _stage_override(config, "voxel_initial_value_tp_tissue_uptake", 0.12),
        "tissue_uptake_max_nfev": _stage_override(config, "voxel_MaxFunEvals_tissue_uptake", 120),
        "tissue_uptake_max_iter": _stage_override(config, "voxel_MaxIter_tissue_uptake", 120),
        "tissue_uptake_robust": _stage_override(config, "voxel_Robust_tissue_uptake", None),
    }


def _apply_model_specific_prefs(prefs: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    out = dict(prefs)
    prefix = f"{model_name}_"
    for key, raw in prefs.items():
        if not key.startswith(prefix):
            continue
        base = key[len(prefix) :]
        if raw is None:
            continue
        if base in {"max_iter", "max_nfev", "gpu_max_n_iterations"}:
            try:
                out[base] = int(float(raw))
            except Exception:
                continue
        else:
            parsed = _parse_numeric_token(raw)
            out[base] = float(parsed) if parsed is not None else raw
    return out


def _stage_d_selected_models(config: DcePipelineConfig) -> Tuple[List[str], List[str]]:
    selected: List[str] = []
    for flag_key, model_name in MODEL_SELECTION_ORDER:
        if int(config.model_flags.get(flag_key, 0)) == 1 and model_name not in selected:
            selected.append(model_name)
    if not selected:
        selected = ["tofts"]

    supported: List[str] = []
    skipped: List[str] = []
    for model in selected:
        if model in SUPPORTED_STAGE_D_MODELS:
            supported.append(model)
        else:
            skipped.append(model)
    return supported, skipped


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _smooth_time_matrix(data: np.ndarray, mode: str, window: int) -> np.ndarray:
    if mode == "none" or window <= 1:
        return np.asarray(data, dtype=np.float64)

    source = np.asarray(data, dtype=np.float64)
    smoothed = np.empty_like(source)
    for col in range(source.shape[1]):
        smoothed[:, col] = _moving_average_1d(source[:, col], window)
    return smoothed


def _try_load_reference_nifti(config: DcePipelineConfig) -> Optional[Dict[str, Any]]:
    try:
        import nibabel as nib  # type: ignore
    except Exception:
        return None

    candidates = config.dynamic_files + config.t1map_files + config.aif_files
    for path in candidates:
        try:
            image = nib.load(str(path))
            shape = tuple(int(v) for v in image.shape[:3])
            header = image.header.copy()
            return {"shape": shape, "affine": image.affine, "header": header}
        except Exception:
            continue
    return None


def _write_tsv_xls(path: Path, rows: List[List[Any]]) -> None:
    lines: List[str] = []
    for row in rows:
        cells: List[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.10g}")
            else:
                cells.append(str(value))
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n")


def _write_param_maps(
    config: DcePipelineConfig,
    rootname: str,
    model_name: str,
    param_names: List[str],
    fit_values: np.ndarray,
    tumind: np.ndarray,
    spatial_shape: Optional[Tuple[int, int, int]],
) -> Dict[str, str]:
    write_maps = bool(_stage_override(config, "write_param_maps", True))
    if not write_maps or spatial_shape is None:
        return {}

    reference = _try_load_reference_nifti(config)
    paths: Dict[str, str] = {}
    coords = np.unravel_index(tumind.astype(np.int64), spatial_shape, order="F")

    for idx, param in enumerate(param_names):
        if idx >= fit_values.shape[1]:
            break
        volume = np.zeros(spatial_shape, dtype=np.float32)
        volume[coords] = fit_values[:, idx].astype(np.float32)

        out_base = f"{rootname}_{model_name}_fit_{param}"
        if reference is not None:
            try:
                import nibabel as nib  # type: ignore

                header = reference["header"].copy()
                header.set_data_dtype(np.float32)
                out_path = config.output_dir / f"{out_base}.nii.gz"
                nii = nib.Nifti1Image(volume, reference["affine"], header)
                nib.save(nii, str(out_path))
                paths[param] = str(out_path)
                continue
            except Exception:
                pass

        out_path = config.output_dir / f"{out_base}.npy"
        np.save(out_path, volume)
        paths[param] = str(out_path)

    return paths


def _fit_model_curve(
    model_name: str,
    ct: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    prefs: Dict[str, Any],
    r1o: Optional[float],
    relaxivity: float,
    fw: float,
) -> np.ndarray:
    ct_list = [float(v) for v in ct]
    cp_list = [float(v) for v in cp]
    timer_list = [float(v) for v in timer]

    if model_name == "tofts":
        prefs_local = _apply_model_specific_prefs(prefs, "tofts")
        return np.asarray(model_tofts_fit(ct_list, cp_list, timer_list, prefs_local), dtype=np.float64)
    if model_name == "ex_tofts":
        prefs_local = _apply_model_specific_prefs(prefs, "ex_tofts")
        return np.asarray(model_extended_tofts_fit(ct_list, cp_list, timer_list, prefs_local), dtype=np.float64)
    if model_name == "patlak":
        prefs_local = _apply_model_specific_prefs(prefs, "patlak")
        return np.asarray(model_patlak_fit(ct_list, cp_list, timer_list, prefs_local), dtype=np.float64)
    if model_name == "tissue_uptake":
        prefs_local = _apply_model_specific_prefs(prefs, "tissue_uptake")
        # MATLAB CPU path seeds tissue-uptake fits with a quick Patlak estimate per voxel.
        try:
            patlak_estimate = model_patlak_linear(ct_list, cp_list, timer_list)
            ktrans_guess = float(patlak_estimate[0])
            vp_guess = float(patlak_estimate[1])
            if math.isfinite(ktrans_guess):
                lo = float(prefs_local.get("lower_limit_ktrans", 1e-7))
                hi = float(prefs_local.get("upper_limit_ktrans", 2.0))
                prefs_local["initial_value_ktrans"] = min(max(ktrans_guess, lo), hi)
            if math.isfinite(vp_guess):
                lo_vp = float(prefs_local.get("lower_limit_vp", 1e-3))
                hi_vp = float(prefs_local.get("upper_limit_vp", 1.0))
                prefs_local["initial_value_vp"] = min(max(vp_guess, lo_vp), hi_vp)
                # Use Patlak vp together with seeded ktrans/fp to initialize Tp.
                k_seed = float(prefs_local.get("initial_value_ktrans", ktrans_guess))
                fp_seed = float(prefs_local.get("initial_value_fp", 0.2))
                fp_seed = max(fp_seed, k_seed * 1.25)
                denom = fp_seed
                if abs(fp_seed - k_seed) > 1e-12:
                    ps_seed = (k_seed * fp_seed) / (fp_seed - k_seed)
                    denom = fp_seed + ps_seed
                if math.isfinite(denom) and abs(denom) > 1e-12:
                    tp_guess = vp_guess / denom
                    if math.isfinite(tp_guess) and tp_guess > 0.0:
                        lo_tp = float(prefs_local.get("lower_limit_tp", 0.0))
                        hi_tp = float(prefs_local.get("upper_limit_tp", 1e6))
                        prefs_local["initial_value_tp"] = min(max(tp_guess, lo_tp), hi_tp)
        except Exception:
            pass
        return np.asarray(model_tissue_uptake_fit(ct_list, cp_list, timer_list, prefs_local), dtype=np.float64)
    if model_name == "2cxm":
        prefs_local = _apply_model_specific_prefs(prefs, "2cxm")
        return np.asarray(model_2cxm_fit(ct_list, cp_list, timer_list, prefs_local), dtype=np.float64)
    if model_name == "fxr":
        if r1o is None:
            raise ValueError("FXR fitting requires R1 baseline values")
        return np.asarray(
            model_fxr_fit(
                ct_list,
                cp_list,
                timer_list,
                float(r1o),
                float(r1o),
                float(relaxivity),
                float(fw),
                prefs,
            ),
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported model '{model_name}'")


def _predict_curve_from_fit_row(
    model_name: str,
    fit_row: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    *,
    r1o: Optional[float],
    relaxivity: float,
    fw: float,
) -> Optional[np.ndarray]:
    row = np.asarray(fit_row, dtype=np.float64).reshape(-1)
    if row.size == 0 or np.any(~np.isfinite(row[: min(4, row.size)])):
        return None
    cp_list = [float(v) for v in cp]
    timer_list = [float(v) for v in timer]

    try:
        if model_name == "tofts":
            return np.asarray(model_tofts_cfit(float(row[0]), float(row[1]), cp_list, timer_list), dtype=np.float64)
        if model_name == "ex_tofts":
            return np.asarray(
                model_extended_tofts_cfit(float(row[0]), float(row[1]), float(row[2]), cp_list, timer_list),
                dtype=np.float64,
            )
        if model_name == "patlak":
            return np.asarray(model_patlak_cfit(float(row[0]), float(row[1]), cp_list, timer_list), dtype=np.float64)
        if model_name == "tissue_uptake":
            ktrans = float(row[0])
            fp = float(row[1])
            vp = float(row[2])
            if not (math.isfinite(ktrans) and math.isfinite(fp) and math.isfinite(vp)):
                return None
            if abs(fp) <= 1e-12:
                return None
            tp = vp / fp
            if not math.isfinite(tp) or tp <= 0.0:
                return None
            return np.asarray(model_tissue_uptake_cfit(ktrans, fp, tp, cp_list, timer_list), dtype=np.float64)
        if model_name == "2cxm":
            return np.asarray(
                model_2cxm_cfit(float(row[0]), float(row[1]), float(row[2]), float(row[3]), cp_list, timer_list),
                dtype=np.float64,
            )
        if model_name == "fxr":
            if r1o is None or not math.isfinite(float(r1o)):
                return None
            return np.asarray(
                model_fxr_cfit(
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    cp_list,
                    timer_list,
                    float(r1o),
                    float(r1o),
                    float(relaxivity),
                    float(fw),
                ),
                dtype=np.float64,
            )
    except Exception:
        return None
    return None


def _compute_fit_residuals(
    *,
    model_name: str,
    ct: np.ndarray,
    fit_results: np.ndarray,
    cp: np.ndarray,
    timer: np.ndarray,
    r1o_vector: Optional[np.ndarray],
    relaxivity: float,
    fw: float,
) -> Optional[np.ndarray]:
    if model_name == "auc":
        return None
    ct_use = np.asarray(ct, dtype=np.float64)
    fit_use = np.asarray(fit_results, dtype=np.float64)
    if ct_use.ndim != 2 or fit_use.ndim != 2:
        return None
    if ct_use.shape[1] != fit_use.shape[0]:
        return None

    residuals = np.full_like(ct_use, np.nan, dtype=np.float64)
    for idx in range(fit_use.shape[0]):
        r1o = None
        if r1o_vector is not None and idx < r1o_vector.size:
            r1o = float(r1o_vector[idx])
        pred = _predict_curve_from_fit_row(
            model_name,
            fit_use[idx, :],
            cp,
            timer,
            r1o=r1o,
            relaxivity=relaxivity,
            fw=fw,
        )
        if pred is None:
            continue
        if pred.shape[0] != ct_use.shape[0]:
            continue
        if np.any(~np.isfinite(pred)):
            continue
        residuals[:, idx] = ct_use[:, idx] - pred
    return residuals


def _write_postfit_arrays(
    *,
    config: DcePipelineConfig,
    rootname: str,
    model_name: str,
    param_names: List[str],
    timer: np.ndarray,
    cp_use: np.ndarray,
    ct_voxel: np.ndarray,
    voxel_results: np.ndarray,
    tumind: np.ndarray,
    spatial_shape: Optional[Tuple[int, int, int]],
    roi_names: List[str],
    roi_curve: Optional[np.ndarray],
    roi_results: np.ndarray,
    r1o_voxel: Optional[np.ndarray],
    r1o_roi: Optional[np.ndarray],
    relaxivity: float,
    fw: float,
) -> Optional[str]:
    write_arrays = _to_bool(_stage_override(config, "write_postfit_arrays", False), False)
    if not write_arrays:
        return None

    results_base = config.output_dir / f"{rootname}_{model_name}_fit"
    out_path = Path(str(results_base) + "_postfit_arrays.npz")

    payload: Dict[str, Any] = {
        "model_name": np.asarray(model_name),
        "param_names": np.asarray(param_names, dtype="<U64"),
        "timer_min": np.asarray(timer, dtype=np.float64),
        "cp_mM": np.asarray(cp_use, dtype=np.float64),
        "ct_voxel_mM": np.asarray(ct_voxel, dtype=np.float64),
        "voxel_results": np.asarray(voxel_results, dtype=np.float64),
        "tumind_0based": np.asarray(tumind, dtype=np.int64),
        "tumind_1based": np.asarray(tumind, dtype=np.int64) + 1,
    }
    if spatial_shape is not None:
        payload["dimensions_xyz"] = np.asarray(spatial_shape, dtype=np.int64)

    voxel_residuals = _compute_fit_residuals(
        model_name=model_name,
        ct=np.asarray(ct_voxel, dtype=np.float64),
        fit_results=np.asarray(voxel_results, dtype=np.float64),
        cp=np.asarray(cp_use, dtype=np.float64),
        timer=np.asarray(timer, dtype=np.float64),
        r1o_vector=r1o_voxel,
        relaxivity=relaxivity,
        fw=fw,
    )
    if voxel_residuals is not None:
        payload["voxel_residuals"] = voxel_residuals

    if roi_results.shape[0] > 0 and roi_curve is not None:
        payload["roi_ct_mM"] = np.asarray(roi_curve, dtype=np.float64)
        payload["roi_results"] = np.asarray(roi_results, dtype=np.float64)
        payload["roi_names"] = np.asarray(roi_names, dtype="<U128")
        roi_residuals = _compute_fit_residuals(
            model_name=model_name,
            ct=np.asarray(roi_curve, dtype=np.float64),
            fit_results=np.asarray(roi_results, dtype=np.float64),
            cp=np.asarray(cp_use, dtype=np.float64),
            timer=np.asarray(timer, dtype=np.float64),
            r1o_vector=r1o_roi,
            relaxivity=relaxivity,
            fw=fw,
        )
        if roi_residuals is not None:
            payload["roi_residuals"] = roi_residuals

    np.savez_compressed(out_path, **payload)
    return str(out_path)


def _load_fit_module_for_acceleration(acceleration_backend: str) -> Any:
    if acceleration_backend == "cpufit_cpu":
        import pycpufit.cpufit as fit_module  # type: ignore

        return fit_module
    import pygpufit.gpufit as fit_module  # type: ignore

    return fit_module


@lru_cache(maxsize=1)
def _cpufit_import_available() -> bool:
    try:
        import pycpufit.cpufit as _  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _gpufit_import_available() -> bool:
    try:
        import pygpufit.gpufit as _  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _acceleration_backend_attempt_order(acceleration_backend: str) -> List[str]:
    if acceleration_backend == "none":
        return []
    backends = [acceleration_backend]
    if acceleration_backend.startswith("gpufit") and _cpufit_import_available():
        backends.append("cpufit_cpu")
    if acceleration_backend == "cpufit_cpu" and _gpufit_import_available():
        backends.append("gpufit_cpu_fallback")
    return backends


def _accelerated_output_has_usable_primary_params(model_name: str, output: np.ndarray) -> bool:
    """Return whether accelerated output has any finite rows for core model parameters."""
    arr = np.asarray(output, dtype=np.float64)
    if arr.ndim != 2:
        return False
    if arr.shape[0] == 0:
        return True

    param_cols = {
        "tofts": (0, 1),
        "ex_tofts": (0, 1, 2),
        "patlak": (0, 1),
        "tissue_uptake": (0, 1, 2),
        "2cxm": (0, 1, 2, 3),
    }
    cols = param_cols.get(model_name, tuple(range(arr.shape[1])))
    max_col = max(cols) if cols else -1
    if max_col >= arr.shape[1]:
        return False

    core = arr[:, cols]
    finite_rows = np.all(np.isfinite(core), axis=1)
    return bool(np.any(finite_rows))


def _fit_stage_d_model_accelerated(
    model_name: str,
    ct: np.ndarray,
    cp_use: np.ndarray,
    timer: np.ndarray,
    prefs: Dict[str, Any],
    acceleration_backend: str,
) -> Optional[np.ndarray]:
    if acceleration_backend == "none":
        return None
    if model_name not in ACCELERATED_STAGE_D_MODELS:
        return None

    fit_module = _load_fit_module_for_acceleration(acceleration_backend)
    n_fits = int(ct.shape[1])
    if n_fits == 0:
        return np.zeros((0, len(MODEL_LAYOUTS[model_name]["param_names"])), dtype=np.float64)

    data = np.ascontiguousarray(np.asarray(ct.T, dtype=np.float32))
    timer_f32 = np.ascontiguousarray(np.asarray(timer, dtype=np.float32).reshape(-1))
    cp_f32 = np.ascontiguousarray(np.asarray(cp_use, dtype=np.float32).reshape(-1))
    user_info = np.ascontiguousarray(np.concatenate([timer_f32, cp_f32], axis=0), dtype=np.float32)

    if model_name == "tofts":
        model_id_name = "TOFTS"
        initial_row = np.array(
            [
                float(prefs["initial_value_ktrans"]),
                float(prefs["initial_value_ve"]),
            ],
            dtype=np.float32,
        )
        bounds_row = np.array(
            [
                float(prefs["lower_limit_ktrans"]),
                float(prefs["upper_limit_ktrans"]),
                float(prefs["lower_limit_ve"]),
                float(prefs["upper_limit_ve"]),
            ],
            dtype=np.float32,
        )
    elif model_name == "ex_tofts":
        model_id_name = "TOFTS_EXTENDED"
        initial_row = np.array(
            [
                float(prefs["initial_value_ktrans"]),
                float(prefs["initial_value_ve"]),
                float(prefs["initial_value_vp"]),
            ],
            dtype=np.float32,
        )
        bounds_row = np.array(
            [
                float(prefs["lower_limit_ktrans"]),
                float(prefs["upper_limit_ktrans"]),
                float(prefs["lower_limit_ve"]),
                float(prefs["upper_limit_ve"]),
                float(prefs["lower_limit_vp"]),
                float(prefs["upper_limit_vp"]),
            ],
            dtype=np.float32,
        )
    elif model_name == "patlak":
        model_id_name = "PATLAK"
        initial_row = np.array(
            [
                float(prefs["initial_value_ktrans"]),
                float(prefs["initial_value_vp"]),
            ],
            dtype=np.float32,
        )
        bounds_row = np.array(
            [
                float(prefs["lower_limit_ktrans"]),
                float(prefs["upper_limit_ktrans"]),
                float(prefs["lower_limit_vp"]),
                float(prefs["upper_limit_vp"]),
            ],
            dtype=np.float32,
        )
    elif model_name == "tissue_uptake":
        # GpuFit/CpuFit tissue uptake parameter order is [Ktrans, vp, fp].
        model_id_name = "TISSUE_UPTAKE"
        initial_row = np.array(
            [
                float(prefs["initial_value_ktrans"]),
                float(prefs["initial_value_vp"]),
                float(prefs["initial_value_fp"]),
            ],
            dtype=np.float32,
        )
        bounds_row = np.array(
            [
                float(prefs["lower_limit_ktrans"]),
                float(prefs["upper_limit_ktrans"]),
                float(prefs["lower_limit_vp"]),
                float(prefs["upper_limit_vp"]),
                float(prefs["lower_limit_fp"]),
                float(prefs["upper_limit_fp"]),
            ],
            dtype=np.float32,
        )
    else:
        model_id_name = "TWO_COMPARTMENT_EXCHANGE"
        initial_row = np.array(
            [
                float(prefs["initial_value_ktrans"]),
                float(prefs["initial_value_ve"]),
                float(prefs["initial_value_vp"]),
                float(prefs["initial_value_fp"]),
            ],
            dtype=np.float32,
        )
        bounds_row = np.array(
            [
                float(prefs["lower_limit_ktrans"]),
                float(prefs["upper_limit_ktrans"]),
                float(prefs["lower_limit_ve"]),
                float(prefs["upper_limit_ve"]),
                float(prefs["lower_limit_vp"]),
                float(prefs["upper_limit_vp"]),
                float(prefs["lower_limit_fp"]),
                float(prefs["upper_limit_fp"]),
            ],
            dtype=np.float32,
        )

    try:
        model_id = int(getattr(fit_module.ModelID, model_id_name))
    except AttributeError as exc:
        raise RuntimeError(f"Acceleration backend does not expose ModelID.{model_id_name}") from exc

    n_params = int(initial_row.size)
    initial_parameters = np.ascontiguousarray(np.tile(initial_row[None, :], (n_fits, 1)), dtype=np.float32)
    constraints = np.ascontiguousarray(np.tile(bounds_row[None, :], (n_fits, 1)), dtype=np.float32)
    constraint_types = np.ascontiguousarray(
        np.full((n_params,), int(fit_module.ConstraintType.LOWER_UPPER), dtype=np.int32)
    )
    tolerance = float(prefs.get("gpu_tolerance", 1e-6))
    max_iterations = int(prefs.get("gpu_max_n_iterations", 200))

    parameters, states, chi_squares, _, _ = fit_module.fit_constrained(
        data=data,
        weights=None,
        model_id=model_id,
        initial_parameters=initial_parameters,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=tolerance,
        max_number_iterations=max_iterations,
        parameters_to_fit=None,
        estimator_id=int(fit_module.EstimatorID.LSE),
        user_info=user_info,
    )

    params = np.asarray(parameters, dtype=np.float64)
    states_arr = np.asarray(states, dtype=np.int32).reshape(-1)
    chi = np.asarray(chi_squares, dtype=np.float64).reshape(-1)
    failed = states_arr != 0
    if np.any(failed):
        params[failed, :] = np.nan
        chi[failed] = np.nan

    if model_name == "tofts":
        out = np.full((n_fits, len(MODEL_LAYOUTS["tofts"]["param_names"])), np.nan, dtype=np.float64)
        out[:, 0] = params[:, 0]
        out[:, 1] = params[:, 1]
        out[:, 2] = chi
        out[:, 3] = params[:, 0]
        out[:, 4] = params[:, 0]
        out[:, 5] = params[:, 1]
        out[:, 6] = params[:, 1]
        return out

    if model_name == "ex_tofts":
        out = np.full((n_fits, len(MODEL_LAYOUTS["ex_tofts"]["param_names"])), np.nan, dtype=np.float64)
        out[:, 0] = params[:, 0]
        out[:, 1] = params[:, 1]
        out[:, 2] = params[:, 2]
        out[:, 3] = chi
        out[:, 4] = params[:, 0]
        out[:, 5] = params[:, 0]
        out[:, 6] = params[:, 1]
        out[:, 7] = params[:, 1]
        out[:, 8] = params[:, 2]
        out[:, 9] = params[:, 2]
        return out

    if model_name == "patlak":
        out = np.full((n_fits, len(MODEL_LAYOUTS["patlak"]["param_names"])), -1.0, dtype=np.float64)
        out[:, 0] = params[:, 0]
        out[:, 1] = params[:, 1]
        out[:, 2] = chi
        return out

    if model_name == "tissue_uptake":
        out = np.full((n_fits, len(MODEL_LAYOUTS["tissue_uptake"]["param_names"])), np.nan, dtype=np.float64)
        out[:, 0] = params[:, 0]
        out[:, 1] = params[:, 2]
        out[:, 2] = params[:, 1]
        out[:, 3] = chi
        out[:, 4] = params[:, 0]
        out[:, 5] = params[:, 0]
        out[:, 6] = params[:, 2]
        out[:, 7] = params[:, 2]
        out[:, 8] = params[:, 1]
        out[:, 9] = params[:, 1]
        return out

    out = np.full((n_fits, len(MODEL_LAYOUTS["2cxm"]["param_names"])), np.nan, dtype=np.float64)
    out[:, 0] = params[:, 0]
    out[:, 1] = params[:, 1]
    out[:, 2] = params[:, 2]
    out[:, 3] = params[:, 3]
    out[:, 4] = chi
    out[:, 5] = params[:, 0]
    out[:, 6] = params[:, 0]
    out[:, 7] = params[:, 1]
    out[:, 8] = params[:, 1]
    out[:, 9] = params[:, 2]
    out[:, 10] = params[:, 2]
    out[:, 11] = params[:, 3]
    out[:, 12] = params[:, 3]
    return out


def _fit_auc_matrix(
    timer: np.ndarray,
    cp_use: np.ndarray,
    ct: np.ndarray,
    stlv_use: np.ndarray,
    sttum: np.ndarray,
    start_injection_min: float,
    sss: Optional[np.ndarray],
    ssstum: Optional[np.ndarray],
) -> np.ndarray:
    start_idx = _nearest_index(timer, start_injection_min)
    t = timer[start_idx:]
    cp = cp_use[start_idx:]
    ct_use = ct[start_idx:, :]
    sttum_use = sttum[start_idx:, :]
    stlv = stlv_use[start_idx:]

    if ssstum is not None and ssstum.size == sttum_use.shape[1]:
        sttum_use = sttum_use - ssstum[np.newaxis, :]
    if sss is not None and sss.size > 0:
        stlv = stlv - float(np.mean(sss))

    auc_cp = float(np.trapz(cp, t))
    auc_sp = float(np.trapz(stlv, t))

    out = np.zeros((ct_use.shape[1], 4), dtype=np.float64)
    for i in range(ct_use.shape[1]):
        auc_c = float(np.trapz(ct_use[:, i], t))
        auc_s = float(np.trapz(sttum_use[:, i], t))
        nauc_c = auc_c / auc_cp if abs(auc_cp) > 1e-12 else float("nan")
        nauc_s = auc_s / auc_sp if abs(auc_sp) > 1e-12 else float("nan")
        out[i, :] = [auc_c, auc_s, nauc_c, nauc_s]
    return out


def _load_roi_columns(
    config: DcePipelineConfig,
    tumind: np.ndarray,
    spatial_shape: Optional[Tuple[int, int, int]],
) -> Tuple[List[Path], List[str], List[np.ndarray]]:
    if not config.roi_files or spatial_shape is None:
        return [], [], []

    roi_paths: List[Path] = []
    roi_names: List[str] = []
    roi_columns: List[np.ndarray] = []
    tumind = np.asarray(tumind, dtype=np.int64).reshape(-1)

    for roi_path in config.roi_files:
        if roi_path.suffix.lower() == ".roi":
            continue
        roi_img = _load_nifti_data(roi_path)
        if roi_img.ndim == 4:
            roi_img = roi_img[..., 0]
        if tuple(roi_img.shape) != tuple(spatial_shape):
            continue

        roi_idx = np.flatnonzero(roi_img.reshape(-1, order="F") > 0)
        if roi_idx.size == 0:
            continue
        _, _, tum_pos = np.intersect1d(roi_idx, tumind, assume_unique=False, return_indices=True)
        if tum_pos.size == 0:
            continue

        roi_paths.append(roi_path)
        roi_names.append(_sanitize_name(roi_path.stem))
        roi_columns.append(np.asarray(tum_pos, dtype=np.int64))

    return roi_paths, roi_names, roi_columns


def _fit_stage_d_model(
    model_name: str,
    ct: np.ndarray,
    cp_use: np.ndarray,
    timer: np.ndarray,
    prefs: Dict[str, Any],
    r1o: Optional[np.ndarray],
    relaxivity: float,
    fw: float,
    stlv_use: Optional[np.ndarray],
    sttum: Optional[np.ndarray],
    start_injection_min: float,
    sss: Optional[np.ndarray],
    ssstum: Optional[np.ndarray],
    acceleration_backend: str,
) -> np.ndarray:
    layout = MODEL_LAYOUTS[model_name]
    row_len = len(layout["param_names"])

    if model_name == "auc":
        if stlv_use is None or sttum is None:
            raise ValueError("AUC fitting requires Stlv_use and Sttum arrays")
        return _fit_auc_matrix(timer, cp_use, ct, stlv_use, sttum, start_injection_min, sss, ssstum)

    if acceleration_backend != "none" and model_name in ACCELERATED_STAGE_D_MODELS:
        candidates = _acceleration_backend_attempt_order(acceleration_backend)
        for idx, backend_candidate in enumerate(candidates):
            try:
                accelerated = _fit_stage_d_model_accelerated(
                    model_name=model_name,
                    ct=ct,
                    cp_use=cp_use,
                    timer=timer,
                    prefs=prefs,
                    acceleration_backend=backend_candidate,
                )
                if accelerated is not None:
                    if _accelerated_output_has_usable_primary_params(model_name, accelerated):
                        return accelerated
                    if idx + 1 < len(candidates):
                        print(
                            f"[DCE] Stage-D {model_name}: acceleration backend '{backend_candidate}' produced "
                            "non-finite core parameter output; trying fallback acceleration backend "
                            f"'{candidates[idx + 1]}'.",
                            flush=True,
                        )
                    else:
                        print(
                            f"[DCE] Stage-D {model_name}: acceleration backend '{backend_candidate}' produced "
                            "non-finite core parameter output; falling back to pure CPU.",
                            flush=True,
                        )
                    continue
                if idx + 1 < len(candidates):
                    print(
                        f"[DCE] Stage-D {model_name}: acceleration backend '{backend_candidate}' returned no result; "
                        f"trying fallback acceleration backend '{candidates[idx + 1]}'.",
                        flush=True,
                    )
            except Exception as exc:
                if idx + 1 < len(candidates):
                    print(
                        f"[DCE] Stage-D {model_name}: acceleration backend '{backend_candidate}' unavailable "
                        f"({exc}); trying fallback acceleration backend '{candidates[idx + 1]}'.",
                        flush=True,
                    )
                else:
                    print(
                        f"[DCE] Stage-D {model_name}: acceleration backend '{backend_candidate}' unavailable "
                        f"({exc}); falling back to pure CPU.",
                        flush=True,
                    )

    out = np.full((ct.shape[1], row_len), np.nan, dtype=np.float64)
    for i in range(ct.shape[1]):
        try:
            r1o_val = float(r1o[i]) if r1o is not None and i < r1o.size else None
            row = _fit_model_curve(model_name, ct[:, i], cp_use, timer, prefs, r1o_val, relaxivity, fw)
            n_copy = min(row_len, row.shape[0])
            out[i, :n_copy] = row[:n_copy]
        except Exception:
            continue
    return out


def _run_stage_d_real(
    config: DcePipelineConfig,
    stage_a: Dict[str, Any],
    stage_b: Dict[str, Any],
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    backend_info = _resolve_stage_d_backend(config)
    selected_backend = backend_info["selected_backend"]
    acceleration_backend = backend_info["acceleration_backend"]
    backend_reason = backend_info["reason"]
    print(
        f"[DCE] Stage-D backend selection: requested={backend_info['requested_backend']} "
        f"selected={selected_backend} acceleration={acceleration_backend} reason={backend_reason}",
        flush=True,
    )

    arrays = stage_b.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Stage-D real mode requires Stage-B arrays")

    cp_use = _as_1d_float(arrays["Cp_use"], "Cp_use")
    timer = _as_1d_float(arrays["timer"], "timer")
    ct_main = _as_time_by_voxel(arrays["Ct"], "Ct")
    tumind = np.asarray(arrays.get("tumind", stage_a.get("arrays", {}).get("tumind", [])), dtype=np.int64).reshape(-1)
    if tumind.size != ct_main.shape[1]:
        if tumind.size == 0:
            tumind = np.arange(ct_main.shape[1], dtype=np.int64)
        else:
            raise ValueError("tumind size does not match Ct voxel dimension")

    if cp_use.size != timer.size or ct_main.shape[0] != timer.size:
        raise ValueError("Stage-D expects Cp_use, Ct, timer to share time length")

    time_smoothing = str(_stage_override(config, "time_smoothing", "none")).strip().lower()
    time_smoothing_window = int(_safe_float(_stage_override(config, "time_smoothing_window", 0), 0))
    ct_main = _smooth_time_matrix(ct_main, time_smoothing, time_smoothing_window)

    sttum = _as_time_by_voxel(arrays["Sttum"], "Sttum") if "Sttum" in arrays else None
    stlv_use = _as_1d_float(arrays["Stlv_use"], "Stlv_use") if "Stlv_use" in arrays else None
    if sttum is not None:
        sttum = _smooth_time_matrix(sttum, time_smoothing, time_smoothing_window)

    relaxivity = float(stage_a.get("relaxivity", _stage_override(config, "relaxivity", 3.4)))
    prefs = _stage_d_fit_prefs(config)
    fw = float(prefs["fxr_fw"])

    spatial_shape: Optional[Tuple[int, int, int]] = None
    ref_meta = _try_load_reference_nifti(config)
    if ref_meta is not None:
        spatial_shape = tuple(ref_meta["shape"])
    elif "image_shape" in stage_a:
        shape = stage_a["image_shape"]
        if isinstance(shape, list) and len(shape) == 3:
            spatial_shape = (int(shape[0]), int(shape[1]), int(shape[2]))

    roi_paths, roi_names, roi_columns = _load_roi_columns(config, tumind, spatial_shape)
    selected_models, skipped_models = _stage_d_selected_models(config)

    rootname = str(stage_a.get("rootname", _stage_override(config, "rootname", "python_dce")))
    start_injection_min = float(stage_b.get("start_injection_min", timer[0]))
    sss = np.asarray(arrays["Sss"], dtype=np.float64).reshape(-1) if "Sss" in arrays else None
    ssstum = np.asarray(arrays["Ssstum"], dtype=np.float64).reshape(-1) if "Ssstum" in arrays else None

    model_outputs: Dict[str, Any] = {}
    stage_arrays: Dict[str, np.ndarray] = {}
    total_models = len(selected_models)
    for model_index, model_name in enumerate(selected_models, start=1):
        _emit_progress(
            event_callback,
            "model_start",
            stage="D",
            model=model_name,
            model_index=model_index,
            model_total=total_models,
        )
        layout = MODEL_LAYOUTS[model_name]
        param_names = list(layout["param_names"])

        ct_source = ct_main
        r1o = None
        if model_name == "fxr":
            if "R1tTOI" not in arrays or "T1TUM" not in arrays:
                raise ValueError("FXR model requires R1tTOI and T1TUM arrays from Stage-B")
            ct_source = _as_time_by_voxel(arrays["R1tTOI"], "R1tTOI")
            if ct_source.shape[0] != timer.size:
                raise ValueError("R1tTOI time dimension mismatch for FXR")
            t1tum = np.asarray(arrays["T1TUM"], dtype=np.float64).reshape(-1)
            if t1tum.size != ct_source.shape[1]:
                raise ValueError("T1TUM size mismatch for FXR")
            r1o = 1.0 / t1tum

        voxel_results = _fit_stage_d_model(
            model_name=model_name,
            ct=ct_source,
            cp_use=cp_use,
            timer=timer,
            prefs=prefs,
            r1o=r1o,
            relaxivity=relaxivity,
            fw=fw,
            stlv_use=stlv_use,
            sttum=sttum,
            start_injection_min=start_injection_min,
            sss=sss,
            ssstum=ssstum,
            acceleration_backend=acceleration_backend,
        )

        roi_results = np.empty((0, voxel_results.shape[1]), dtype=np.float64)
        roi_curve: Optional[np.ndarray] = None
        roi_r1o: Optional[np.ndarray] = None
        if roi_columns:
            roi_curve = np.stack([np.mean(ct_source[:, cols], axis=1) for cols in roi_columns], axis=1)
            if model_name == "fxr" and r1o is not None:
                roi_r1o = np.asarray([float(np.mean(r1o[cols])) for cols in roi_columns], dtype=np.float64)
            roi_sttum = None
            if sttum is not None:
                roi_sttum = np.stack([np.mean(sttum[:, cols], axis=1) for cols in roi_columns], axis=1)
            roi_results = _fit_stage_d_model(
                model_name=model_name,
                ct=roi_curve,
                cp_use=cp_use,
                timer=timer,
                prefs=prefs,
                r1o=roi_r1o,
                relaxivity=relaxivity,
                fw=fw,
                stlv_use=stlv_use,
                sttum=roi_sttum,
                start_injection_min=start_injection_min,
                sss=sss,
                ssstum=np.asarray([float(np.mean(ssstum[cols])) for cols in roi_columns], dtype=np.float64)
                if ssstum is not None and ssstum.size == ct_source.shape[1]
                else None,
                acceleration_backend=acceleration_backend,
            )

        map_paths = _write_param_maps(
            config=config,
            rootname=rootname,
            model_name=model_name,
            param_names=param_names,
            fit_values=voxel_results,
            tumind=tumind,
            spatial_shape=spatial_shape,
        )

        xls_path: Optional[str] = None
        if config.write_xls and roi_results.shape[0] > 0:
            results_base = config.output_dir / f"{rootname}_{model_name}_fit"
            xls_target = Path(str(results_base) + "_rois.xls")
            rows: List[List[Any]] = [list(layout["headings"])]
            for i, roi_name in enumerate(roi_names):
                row: List[Any] = [str(roi_paths[i]), roi_name]
                row.extend(float(v) for v in roi_results[i, :])
                rows.append(row)
            _write_tsv_xls(xls_target, rows)
            xls_path = str(xls_target)

        postfit_arrays_path = _write_postfit_arrays(
            config=config,
            rootname=rootname,
            model_name=model_name,
            param_names=param_names,
            timer=timer,
            cp_use=cp_use,
            ct_voxel=ct_source,
            voxel_results=voxel_results,
            tumind=tumind,
            spatial_shape=spatial_shape,
            roi_names=roi_names,
            roi_curve=roi_curve,
            roi_results=roi_results,
            r1o_voxel=r1o,
            r1o_roi=roi_r1o,
            relaxivity=relaxivity,
            fw=fw,
        )

        stage_arrays[f"{model_name}_voxel_results"] = voxel_results
        if roi_results.shape[0] > 0:
            stage_arrays[f"{model_name}_roi_results"] = roi_results

        model_outputs[model_name] = {
            "param_names": param_names,
            "voxel_result_shape": list(voxel_results.shape),
            "roi_result_shape": list(roi_results.shape),
            "map_paths": map_paths,
            "xls_path": xls_path,
            "postfit_arrays_path": postfit_arrays_path,
        }
        _emit_progress(
            event_callback,
            "model_done",
            stage="D",
            model=model_name,
            model_index=model_index,
            model_total=total_models,
            voxel_count=int(voxel_results.shape[0]),
            roi_count=int(roi_results.shape[0]),
        )

    backend_used = "cpu" if selected_backend == "cpu" else acceleration_backend
    return {
        "stage": "D",
        "status": "ok",
        "impl": "real",
        "selected_backend": selected_backend,
        "acceleration_backend": acceleration_backend,
        "backend_reason": backend_reason,
        "backend_used": backend_used,
        "write_xls": bool(config.write_xls),
        "time_smoothing": time_smoothing,
        "time_smoothing_window": int(time_smoothing_window),
        "models_requested": selected_models + skipped_models,
        "models_run": selected_models,
        "models_skipped": skipped_models,
        "model_outputs": model_outputs,
        "arrays": stage_arrays,
    }


class HybridStageRunner:
    """Runner with real Stage-A and Stage-B support plus scaffold fallbacks."""

    def run_a(
        self, config: DcePipelineConfig, event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        del event_callback
        mode = str(_stage_override(config, "stage_a_mode", "real")).strip().lower()
        if mode == "scaffold":
            return {
                "stage": "A",
                "status": "scaffold",
                "impl": "scaffold",
                "dynamic_file_count": len(config.dynamic_files),
                "aif_file_count": len(config.aif_files),
                "roi_file_count": len(config.roi_files),
                "t1map_file_count": len(config.t1map_files),
                "noise_file_count": len(config.noise_files),
                "drift_file_count": len(config.drift_files),
                "subject_tp_path": str(config.subject_tp_path),
            }
        return _run_stage_a_real(config)

    def run_b(
        self,
        config: DcePipelineConfig,
        stage_a: Dict[str, Any],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        del event_callback
        mode = str(_stage_override(config, "stage_b_mode", "auto")).strip().lower()
        if mode == "scaffold":
            return {
                "stage": "B",
                "status": "scaffold",
                "impl": "scaffold",
                "aif_mode": config.aif_mode.strip().lower(),
                "imported_aif_path": str(config.imported_aif_path) if config.imported_aif_path else None,
                "manual_click_aif_enabled": False,
            }

        if mode == "auto":
            if not isinstance(stage_a.get("arrays"), dict):
                return {
                    "stage": "B",
                    "status": "scaffold",
                    "impl": "scaffold",
                    "aif_mode": config.aif_mode.strip().lower(),
                    "imported_aif_path": str(config.imported_aif_path) if config.imported_aif_path else None,
                    "manual_click_aif_enabled": False,
                    "reason": "stage_a_arrays_missing",
                }
            return _run_stage_b_real(config, stage_a)

        if mode == "real":
            return _run_stage_b_real(config, stage_a)

        raise ValueError(f"Unsupported stage_b_mode '{mode}'")

    def run_d(
        self,
        config: DcePipelineConfig,
        stage_a: Dict[str, Any],
        stage_b: Dict[str, Any],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        mode = str(_stage_override(config, "stage_d_mode", "auto")).strip().lower()
        if mode == "scaffold":
            backend_info = _resolve_stage_d_backend(config)
            return {
                "stage": "D",
                "status": "scaffold",
                "impl": "scaffold",
                "selected_backend": backend_info["selected_backend"],
                "acceleration_backend": backend_info["acceleration_backend"],
                "backend_reason": backend_info["reason"],
                "write_xls": bool(config.write_xls),
                "model_flags": dict(config.model_flags),
            }

        if mode == "auto":
            if not isinstance(stage_b.get("arrays"), dict):
                backend_info = _resolve_stage_d_backend(config)
                return {
                    "stage": "D",
                    "status": "scaffold",
                    "impl": "scaffold",
                    "selected_backend": backend_info["selected_backend"],
                    "acceleration_backend": backend_info["acceleration_backend"],
                    "backend_reason": backend_info["reason"],
                    "write_xls": bool(config.write_xls),
                    "model_flags": dict(config.model_flags),
                    "reason": "stage_b_arrays_missing",
                }
            return _run_stage_d_real(config, stage_a, stage_b, event_callback=event_callback)

        if mode == "real":
            return _run_stage_d_real(config, stage_a, stage_b, event_callback=event_callback)

        raise ValueError(f"Unsupported stage_d_mode '{mode}'")


def _array_shapes(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = list(value.shape)
    return out


def _emit_progress(event_callback: Optional[Callable[[Dict[str, Any]], None]], event_type: str, **payload: Any) -> None:
    if event_callback is None:
        return
    event: Dict[str, Any] = {
        "type": event_type,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    event.update(payload)
    event_callback(event)


def _stage_summary(stage_data: Dict[str, Any]) -> Dict[str, Any]:
    summary = {}
    for key, value in stage_data.items():
        if key == "arrays" and isinstance(value, dict):
            summary["array_shapes"] = _array_shapes(value)
            continue
        if isinstance(value, np.ndarray):
            summary[key] = {"shape": list(value.shape)}
            continue
        summary[key] = value
    return summary


def _write_stage_checkpoint(checkpoint_dir: Path, stage_name: str, data: Dict[str, Any]) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata = dict(data)
    arrays = metadata.pop("arrays", None)
    if isinstance(arrays, dict):
        npz_path = checkpoint_dir / f"{stage_name.lower()}_out_arrays.npz"
        serializable_arrays = {k: v for k, v in arrays.items() if isinstance(v, np.ndarray)}
        if serializable_arrays:
            np.savez_compressed(npz_path, **serializable_arrays)
            metadata["array_npz"] = str(npz_path)
            metadata["array_shapes"] = _array_shapes(serializable_arrays)

    target = checkpoint_dir / f"{stage_name.lower()}_out.json"
    target.write_text(json.dumps(metadata, indent=2, default=_json_default))
    return target


def _emit_stage_artifacts(
    event_callback: Optional[Callable[[Dict[str, Any]], None]], stage_name: str, stage_data: Dict[str, Any]
) -> None:
    figure_paths = stage_data.get("figure_paths")
    if isinstance(figure_paths, dict):
        for name, path in figure_paths.items():
            _emit_progress(
                event_callback,
                "artifact_written",
                stage=stage_name,
                artifact_type="figure",
                name=str(name),
                path=str(path),
            )

    if stage_name != "D":
        return
    model_outputs = stage_data.get("model_outputs")
    if not isinstance(model_outputs, dict):
        return
    for model_name, output in model_outputs.items():
        if not isinstance(output, dict):
            continue
        map_paths = output.get("map_paths")
        if isinstance(map_paths, dict):
            for param, path in map_paths.items():
                _emit_progress(
                    event_callback,
                    "artifact_written",
                    stage="D",
                    model=str(model_name),
                    artifact_type="map",
                    name=str(param),
                    path=str(path),
                )
        xls_path = output.get("xls_path")
        if isinstance(xls_path, str) and xls_path:
            _emit_progress(
                event_callback,
                "artifact_written",
                stage="D",
                model=str(model_name),
                artifact_type="roi_xls",
                name="roi_xls",
                path=xls_path,
            )
        postfit_arrays_path = output.get("postfit_arrays_path")
        if isinstance(postfit_arrays_path, str) and postfit_arrays_path:
            _emit_progress(
                event_callback,
                "artifact_written",
                stage="D",
                model=str(model_name),
                artifact_type="postfit_arrays",
                name="postfit_arrays",
                path=postfit_arrays_path,
            )


def run_dce_pipeline(
    config: DcePipelineConfig,
    runner: Optional[DceStageRunner] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run DCE stages A->B->D in memory with optional checkpoints."""
    import time

    start_time = time.perf_counter()
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.checkpoint_dir:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    active_runner = runner or HybridStageRunner()
    prefs_path = _resolve_dce_preferences_path(config)
    _emit_progress(
        event_callback,
        "run_start",
        output_dir=str(config.output_dir),
        checkpoint_dir=str(config.checkpoint_dir) if config.checkpoint_dir else None,
        backend=str(config.backend),
        dce_preferences_path=str(prefs_path) if prefs_path else None,
    )

    current_stage = "A"
    try:
        _emit_progress(event_callback, "stage_start", stage="A")
        try:
            stage_a = active_runner.run_a(config, event_callback=event_callback)
        except TypeError:
            stage_a = active_runner.run_a(config)  # type: ignore[misc]
        if config.checkpoint_dir:
            checkpoint_a = _write_stage_checkpoint(config.checkpoint_dir, "A", stage_a)
            _emit_progress(event_callback, "checkpoint_written", stage="A", path=str(checkpoint_a))
        _emit_progress(
            event_callback,
            "stage_done",
            stage="A",
            status=str(stage_a.get("status", "")),
            impl=str(stage_a.get("impl", "")),
            array_shapes=_array_shapes(stage_a.get("arrays", {})) if isinstance(stage_a.get("arrays"), dict) else {},
        )
        _emit_stage_artifacts(event_callback, "A", stage_a)

        current_stage = "B"
        _emit_progress(event_callback, "stage_start", stage="B")
        try:
            stage_b = active_runner.run_b(config, stage_a, event_callback=event_callback)
        except TypeError:
            stage_b = active_runner.run_b(config, stage_a)  # type: ignore[misc]
        if config.checkpoint_dir:
            checkpoint_b = _write_stage_checkpoint(config.checkpoint_dir, "B", stage_b)
            _emit_progress(event_callback, "checkpoint_written", stage="B", path=str(checkpoint_b))
        _emit_progress(
            event_callback,
            "stage_done",
            stage="B",
            status=str(stage_b.get("status", "")),
            impl=str(stage_b.get("impl", "")),
            array_shapes=_array_shapes(stage_b.get("arrays", {})) if isinstance(stage_b.get("arrays"), dict) else {},
        )
        _emit_stage_artifacts(event_callback, "B", stage_b)

        current_stage = "D"
        _emit_progress(event_callback, "stage_start", stage="D")
        try:
            stage_d = active_runner.run_d(config, stage_a, stage_b, event_callback=event_callback)
        except TypeError:
            stage_d = active_runner.run_d(config, stage_a, stage_b)  # type: ignore[misc]
        if config.checkpoint_dir:
            checkpoint_d = _write_stage_checkpoint(config.checkpoint_dir, "D", stage_d)
            _emit_progress(event_callback, "checkpoint_written", stage="D", path=str(checkpoint_d))
        _emit_progress(
            event_callback,
            "stage_done",
            stage="D",
            status=str(stage_d.get("status", "")),
            impl=str(stage_d.get("impl", "")),
            array_shapes=_array_shapes(stage_d.get("arrays", {})) if isinstance(stage_d.get("arrays"), dict) else {},
            models_run=list(stage_d.get("models_run", [])),
            models_skipped=list(stage_d.get("models_skipped", [])),
        )
        _emit_stage_artifacts(event_callback, "D", stage_d)
    except Exception as exc:
        _emit_progress(
            event_callback,
            "run_error",
            stage=current_stage,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


    # Calculate execution time
    duration_sec = time.perf_counter() - start_time

    # Extract backend from stage D
    backend_used = str(stage_d.get("impl", "cpu"))

    # Build provenance
    provenance = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_sec": duration_sec,
        "inputs": {
            "dynamic": [str(p) for p in config.dynamic_files],
            "aif_mask": [str(p) for p in config.aif_files],
            "roi_mask": [str(p) for p in config.roi_files],
            "t1_map": [str(p) for p in config.t1map_files],
            "noise_mask": [str(p) for p in config.noise_files] if config.noise_files else None,
        },
        "backend_requested": config.backend,
        "backend_used": backend_used,
    }

    summary = {
        "meta": {
            "pipeline": "dce_cli_in_memory",
            "status": "ok",
            "single_process": True,
            "dce_preferences_path": str(prefs_path) if prefs_path else None,
            "duration_sec": duration_sec,
        },
        "provenance": provenance,
        "config": config.to_dict(),
        "stages": {
            "A": _stage_summary(stage_a),
            "B": _stage_summary(stage_b),
            "D": _stage_summary(stage_d),
        },
    }

    summary_path = config.output_dir / "dce_pipeline_run.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    summary["meta"]["summary_path"] = str(summary_path)
    _emit_progress(event_callback, "run_done", summary_path=str(summary_path), status="ok")
    return summary

