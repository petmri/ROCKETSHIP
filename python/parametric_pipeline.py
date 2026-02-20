"""Parametric T1 mapping pipeline (VFA fitting)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from parametric_models import t1_fa_nonlinear_fit, t1_fa_two_point_fit


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Type {type(value)} is not JSON serializable")


def _resolve_path(raw: str | Path, base_dir: Optional[Path]) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _to_path_list(values: Any, base_dir: Optional[Path]) -> List[Path]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        return [_resolve_path(values, base_dir)]
    return [_resolve_path(item, base_dir) for item in values]


@dataclass
class ParametricT1Config:
    """Configuration for a single parametric T1 VFA run."""

    output_dir: Path
    vfa_files: List[Path] = field(default_factory=list)
    fit_type: str = "t1_fa_linear_fit"
    output_basename: str = "T1_map"
    output_label: str = ""
    rsquared_threshold: float = 0.6
    tr_ms: Optional[float] = None
    flip_angles_deg: List[float] = field(default_factory=list)
    write_r_squared: bool = True
    write_rho_map: bool = False
    invalid_fill_value: float = -1.0
    mask_file: Optional[Path] = None
    b1_map_file: Optional[Path] = None
    script_preferences_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> "ParametricT1Config":
        vfa_values = data.get("vfa_files")
        if vfa_values is None:
            vfa_values = data.get("file_list")
        flip_values = data.get("flip_angles_deg")
        if flip_values is None:
            flip_values = data.get("parameters", [])
        tr_value = data.get("tr_ms")
        if tr_value is None and data.get("tr") is not None:
            tr_value = data.get("tr")

        return cls(
            output_dir=_resolve_path(data["output_dir"], base_dir),
            vfa_files=_to_path_list(vfa_values, base_dir),
            fit_type=str(data.get("fit_type", "t1_fa_linear_fit")),
            output_basename=str(data.get("output_basename", "T1_map")),
            output_label=str(data.get("output_label", "")),
            rsquared_threshold=float(data.get("rsquared_threshold", 0.6)),
            tr_ms=float(tr_value) if tr_value is not None else None,
            flip_angles_deg=[float(v) for v in flip_values],
            write_r_squared=bool(data.get("write_r_squared", True)),
            write_rho_map=bool(data.get("write_rho_map", False)),
            invalid_fill_value=float(data.get("invalid_fill_value", -1.0)),
            mask_file=_resolve_path(data["mask_file"], base_dir) if data.get("mask_file") else None,
            b1_map_file=_resolve_path(data["b1_map_file"], base_dir) if data.get("b1_map_file") else None,
            script_preferences_path=(
                _resolve_path(data["script_preferences_path"], base_dir) if data.get("script_preferences_path") else None
            ),
        )

    def validate(self) -> None:
        fit_type = self.fit_type.strip().lower()
        if fit_type not in {"t1_fa_linear_fit", "t1_fa_fit", "t1_fa_two_point_fit"}:
            raise ValueError(
                "Unsupported fit_type. Expected one of: "
                "'t1_fa_linear_fit', 't1_fa_fit', 't1_fa_two_point_fit'"
            )

        if not self.vfa_files:
            raise ValueError("vfa_files must be non-empty")

        if not (0.0 <= self.rsquared_threshold <= 1.0):
            raise ValueError("rsquared_threshold must be between 0 and 1")

        for path in self.vfa_files:
            if not path.exists():
                raise FileNotFoundError(f"VFA file not found: {path}")

        if self.mask_file is not None and not self.mask_file.exists():
            raise FileNotFoundError(f"mask_file not found: {self.mask_file}")
        if self.b1_map_file is not None and not self.b1_map_file.exists():
            raise FileNotFoundError(f"b1_map_file not found: {self.b1_map_file}")
        if self.script_preferences_path is not None and not self.script_preferences_path.exists():
            raise FileNotFoundError(f"script_preferences_path not found: {self.script_preferences_path}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return json.loads(json.dumps(payload, default=_json_default))


def _emit_event(event_callback: Optional[Callable[[Dict[str, Any]], None]], event_type: str, **payload: Any) -> None:
    if event_callback is None:
        return
    event_callback({"type": event_type, **payload})


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sidecar_path(vfa_path: Path) -> Path:
    name = vfa_path.name
    if name.endswith(".nii.gz"):
        return vfa_path.with_name(name[:-7] + ".json")
    if vfa_path.suffix.lower() == ".nii":
        return vfa_path.with_suffix(".json")
    return vfa_path.with_suffix(vfa_path.suffix + ".json")


def _load_nifti(path: Path) -> Tuple[np.ndarray, Any, Any]:
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for parametric T1 pipeline") from exc

    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float64)
    return data, image.affine, image.header


def _save_nifti(path: Path, array: np.ndarray, affine: Any, header: Any) -> None:
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for parametric T1 pipeline") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    out_header = header.copy()
    out_header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine, out_header)
    nib.save(image, str(path))


def _load_vfa_data(config: ParametricT1Config) -> Tuple[np.ndarray, Any, Any]:
    first_data, affine, header = _load_nifti(config.vfa_files[0])

    if len(config.vfa_files) == 1:
        array = np.asarray(first_data, dtype=np.float64)
        if array.ndim == 4:
            if array.shape[-1] < 2:
                raise ValueError("Single VFA file with 4D input must have at least 2 flip-angle frames")
            return array, affine, header
        if array.ndim == 3 and config.flip_angles_deg and len(config.flip_angles_deg) == array.shape[-1]:
            # Allow stacked 3D where final axis is flip angle when explicitly specified.
            return array, affine, header
        raise ValueError(
            "Single VFA input must be 4D (or 3D with explicit flip_angles_deg matching last axis)"
        )

    spatial_shape = np.squeeze(first_data).shape
    if len(spatial_shape) not in {2, 3}:
        raise ValueError("Each VFA input in vfa_files must be 2D or 3D when multiple files are provided")

    frames: List[np.ndarray] = []
    for path in config.vfa_files:
        frame, _, _ = _load_nifti(path)
        frame = np.squeeze(frame)
        if frame.shape != spatial_shape:
            raise ValueError(f"VFA frame shape mismatch for {path}: {frame.shape} != {spatial_shape}")
        frames.append(np.asarray(frame, dtype=np.float64))

    stacked = np.stack(frames, axis=-1)
    return stacked, affine, header


def _parse_preference_file(path: Path) -> Dict[str, str]:
    prefs: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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


def _resolve_script_preferences_path(config: ParametricT1Config) -> Optional[Path]:
    if config.script_preferences_path is not None:
        return config.script_preferences_path

    repo_default = Path(__file__).resolve().parents[1] / "script_preferences.txt"
    if repo_default.exists():
        return repo_default

    cwd_default = Path.cwd() / "script_preferences.txt"
    if cwd_default.exists():
        return cwd_default

    return None


def _load_script_preference_tr_ms(config: ParametricT1Config) -> Optional[float]:
    prefs_path = _resolve_script_preferences_path(config)
    if prefs_path is None:
        return None

    prefs = _parse_preference_file(prefs_path)
    tr_raw = prefs.get("tr")
    if tr_raw is None:
        return None

    try:
        tr_ms = float(tr_raw)
    except ValueError:
        return None
    if tr_ms <= 0.0 or not math.isfinite(tr_ms):
        return None
    return tr_ms


def _resolve_flip_angles_and_tr_ms(
    config: ParametricT1Config, n_flips: int
) -> Tuple[np.ndarray, float, List[Path], str]:
    sidecars: List[Path] = []
    sidecar_payloads: List[Dict[str, Any]] = []
    for vfa_path in config.vfa_files:
        candidate = _sidecar_path(vfa_path)
        if candidate.exists():
            payload = _load_json(candidate)
            sidecars.append(candidate)
            sidecar_payloads.append(payload)

    if config.flip_angles_deg:
        flip_angles = np.asarray(config.flip_angles_deg, dtype=np.float64)
        if flip_angles.shape[0] != n_flips:
            raise ValueError(
                f"flip_angles_deg length {flip_angles.shape[0]} does not match number of flip frames {n_flips}"
            )
    elif len(sidecar_payloads) == len(config.vfa_files) and len(config.vfa_files) == n_flips:
        flip_angles = np.asarray([float(payload["FlipAngle"]) for payload in sidecar_payloads], dtype=np.float64)
    elif len(sidecar_payloads) == 1 and "FlipAngles" in sidecar_payloads[0]:
        flip_angles = np.asarray([float(v) for v in sidecar_payloads[0]["FlipAngles"]], dtype=np.float64)
        if flip_angles.shape[0] != n_flips:
            raise ValueError(
                f"FlipAngles sidecar length {flip_angles.shape[0]} does not match number of flip frames {n_flips}"
            )
    elif n_flips == 5:
        # MATLAB fallback in T1mapping_fit.m when no VFA JSON sidecars are present.
        flip_angles = np.asarray([2.0, 5.0, 10.0, 12.0, 15.0], dtype=np.float64)
    else:
        raise ValueError(
            "Unable to infer flip angles: provide flip_angles_deg explicitly or VFA sidecars with FlipAngle metadata"
        )

    if np.any(~np.isfinite(flip_angles)):
        raise ValueError("flip angles must be finite")

    if config.tr_ms is not None:
        tr_ms = float(config.tr_ms)
        tr_source = "config"
    else:
        tr_values_ms = [float(payload["RepetitionTime"]) * 1000.0 for payload in sidecar_payloads if "RepetitionTime" in payload]
        if tr_values_ms:
            tr_ms = float(tr_values_ms[0])
            for value in tr_values_ms[1:]:
                if not math.isclose(float(value), tr_ms, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("RepetitionTime values across VFA sidecars are inconsistent")
            tr_source = "sidecar"
        else:
            pref_tr_ms = _load_script_preference_tr_ms(config)
            if pref_tr_ms is None:
                raise ValueError(
                    "tr_ms is required when RepetitionTime sidecar metadata is unavailable and script_preferences.txt has no valid tr"
                )
            tr_ms = float(pref_tr_ms)
            tr_source = "script_preferences"

    if tr_ms <= 0.0 or not math.isfinite(tr_ms):
        raise ValueError("tr_ms must be a positive finite value")

    return flip_angles, tr_ms, sidecars, tr_source


def _autodetect_b1_map_path(config: ParametricT1Config) -> Optional[Path]:
    """Return the default MATLAB-style B1 map path when present."""
    candidate_names = ("B1_scaled_FAreg.nii", "B1_scaled_FAreg.nii.gz")
    seen_dirs: set[Path] = set()
    for vfa_path in config.vfa_files:
        parent = vfa_path.parent.resolve()
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        for name in candidate_names:
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None


def _resolve_b1_scale_map(config: ParametricT1Config, spatial_shape: Tuple[int, ...]) -> Tuple[Optional[np.ndarray], Optional[Path]]:
    if config.b1_map_file is not None:
        b1_path = config.b1_map_file
        b1_mode = "explicit"
    else:
        b1_path = _autodetect_b1_map_path(config)
        b1_mode = "auto"

    if b1_path is None:
        return None, None

    b1_data, _, _ = _load_nifti(b1_path)
    b1_map = np.squeeze(np.asarray(b1_data, dtype=np.float64))
    if b1_map.shape != spatial_shape:
        raise ValueError(
            f"{b1_mode} b1_map_file shape {b1_map.shape} does not match VFA spatial shape {spatial_shape}"
        )
    return b1_map, b1_path


def _fit_t1_fa_linear_map(
    vfa_data: np.ndarray,
    flip_angles_deg: np.ndarray,
    tr_ms: float,
    rsquared_threshold: float,
    invalid_fill_value: float,
    mask: Optional[np.ndarray],
    b1_scale_map: Optional[np.ndarray],
) -> Dict[str, Any]:
    spatial_shape = vfa_data.shape[:-1]
    n_voxels = int(np.prod(spatial_shape))
    n_flips = int(vfa_data.shape[-1])

    flat_signal = np.reshape(vfa_data, (n_voxels, n_flips)).astype(np.float64, copy=False)
    b1_valid = np.ones((n_voxels,), dtype=bool)
    invalid_flip_geom = np.zeros((n_voxels,), dtype=bool)

    if b1_scale_map is None:
        flip_rad = np.deg2rad(flip_angles_deg)
        sin_flip = np.sin(flip_rad)
        tan_flip = np.tan(flip_rad)
        if np.any(np.isclose(sin_flip, 0.0)) or np.any(np.isclose(tan_flip, 0.0)):
            raise ValueError("flip angles cannot contain values that produce zero sine/tangent")
        with np.errstate(divide="ignore", invalid="ignore"):
            y_lin = flat_signal / sin_flip
            x_lin = flat_signal / tan_flip
    else:
        flat_b1 = np.reshape(np.asarray(b1_scale_map, dtype=np.float64), (n_voxels,))
        b1_valid = np.isfinite(flat_b1) & (flat_b1 > 0.0)
        flip_rad = np.deg2rad(flip_angles_deg)[None, :] * flat_b1[:, None]
        sin_flip = np.sin(flip_rad)
        tan_flip = np.tan(flip_rad)
        invalid_flip_geom = np.any(
            (~np.isfinite(sin_flip))
            | (~np.isfinite(tan_flip))
            | np.isclose(sin_flip, 0.0)
            | np.isclose(tan_flip, 0.0),
            axis=1,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            y_lin = flat_signal / sin_flip
            x_lin = flat_signal / tan_flip

    x_bar = np.mean(x_lin, axis=1)
    y_bar = np.mean(y_lin, axis=1)
    x_centered = x_lin - x_bar[:, None]
    y_centered = y_lin - y_bar[:, None]

    sum_x2 = np.sum(x_centered * x_centered, axis=1)
    sum_y2 = np.sum(y_centered * y_centered, axis=1)
    sum_xy = np.sum(x_centered * y_centered, axis=1)

    slope = np.zeros_like(sum_xy)
    np.divide(sum_xy, sum_x2, out=slope, where=sum_x2 != 0.0)
    intercept = y_bar - slope * x_bar

    corr = np.zeros_like(sum_xy)
    denom = np.sqrt(sum_x2 * sum_y2)
    np.divide(sum_xy, denom, out=corr, where=denom != 0.0)
    r_squared = corr * corr
    r_squared[~np.isfinite(r_squared)] = 0.0

    sse = (1.0 - r_squared) * sum_y2

    t1_fit = np.empty_like(slope)
    t1_fit.fill(np.nan)

    neg_mask = slope < 0.0
    one_mask = slope == 1.0
    std_mask = (~neg_mask) & (~one_mask)

    t1_fit[neg_mask] = -0.4
    t1_fit[one_mask] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        t1_fit[std_mask] = -tr_ms / np.log(slope[std_mask])
    t1_fit[t1_fit < 0.0] = -0.5

    finite_signal = np.all(np.isfinite(flat_signal), axis=1)
    finite_outputs = np.isfinite(intercept) & np.isfinite(t1_fit)

    if mask is not None:
        mask_flat = np.reshape(mask > 0, (n_voxels,))
    else:
        mask_flat = np.ones((n_voxels,), dtype=bool)

    threshold_failed = r_squared < rsquared_threshold
    bad_fit = (
        (~mask_flat)
        | (~finite_signal)
        | (~finite_outputs)
        | (~b1_valid)
        | invalid_flip_geom
        | threshold_failed
    )

    t1_fit[bad_fit] = invalid_fill_value
    intercept[bad_fit] = invalid_fill_value

    t1_map = np.reshape(t1_fit, spatial_shape)
    rho_map = np.reshape(intercept, spatial_shape)
    r_squared_map = np.reshape(r_squared, spatial_shape)
    sse_map = np.reshape(sse, spatial_shape)

    valid_mask = (~bad_fit) & mask_flat

    return {
        "t1_map": t1_map,
        "rho_map": rho_map,
        "r_squared_map": r_squared_map,
        "sse_map": sse_map,
        "metrics": {
            "total_voxels": int(n_voxels),
            "mask_voxels": int(mask_flat.sum()),
            "valid_fits": int(valid_mask.sum()),
            "threshold_failed": int((threshold_failed & mask_flat).sum()),
            "finite_signal_voxels": int((finite_signal & mask_flat).sum()),
            "b1_valid_voxels": int((b1_valid & mask_flat).sum()),
            "t1_mean_ms": float(np.mean(t1_fit[valid_mask])) if np.any(valid_mask) else float("nan"),
            "t1_median_ms": float(np.median(t1_fit[valid_mask])) if np.any(valid_mask) else float("nan"),
            "r2_mean": float(np.mean(r_squared[mask_flat])) if np.any(mask_flat) else float("nan"),
            "r2_median": float(np.median(r_squared[mask_flat])) if np.any(mask_flat) else float("nan"),
        },
    }


def _fit_t1_fa_model_map(
    *,
    vfa_data: np.ndarray,
    flip_angles_deg: np.ndarray,
    tr_ms: float,
    rsquared_threshold: float,
    invalid_fill_value: float,
    mask: Optional[np.ndarray],
    b1_scale_map: Optional[np.ndarray],
    fit_fn: Callable[[List[float], List[float], float], List[float]],
) -> Dict[str, Any]:
    spatial_shape = vfa_data.shape[:-1]
    n_voxels = int(np.prod(spatial_shape))
    n_flips = int(vfa_data.shape[-1])

    flat_signal = np.reshape(vfa_data, (n_voxels, n_flips)).astype(np.float64, copy=False)
    finite_signal = np.all(np.isfinite(flat_signal), axis=1)

    if mask is not None:
        mask_flat = np.reshape(mask > 0, (n_voxels,))
    else:
        mask_flat = np.ones((n_voxels,), dtype=bool)

    t1_fit = np.full((n_voxels,), float(invalid_fill_value), dtype=np.float64)
    rho_fit = np.full((n_voxels,), float(invalid_fill_value), dtype=np.float64)
    r_squared = np.zeros((n_voxels,), dtype=np.float64)
    sse = np.full((n_voxels,), np.nan, dtype=np.float64)
    threshold_failed = np.zeros((n_voxels,), dtype=bool)
    b1_valid = np.ones((n_voxels,), dtype=bool)
    effective_flip_angles: Optional[np.ndarray] = None

    if b1_scale_map is not None:
        flat_b1 = np.reshape(np.asarray(b1_scale_map, dtype=np.float64), (n_voxels,))
        b1_valid = np.isfinite(flat_b1) & (flat_b1 > 0.0)
        effective_flip_angles = flat_b1[:, None] * flip_angles_deg[None, :]
        b1_valid &= np.all(np.isfinite(effective_flip_angles), axis=1)

    fa_list = [float(v) for v in flip_angles_deg]
    for idx in np.flatnonzero(mask_flat & finite_signal & b1_valid):
        si = [float(v) for v in flat_signal[idx, :]]
        if effective_flip_angles is not None:
            fa_values = [float(v) for v in effective_flip_angles[idx, :]]
        else:
            fa_values = fa_list
        try:
            fit = fit_fn(fa_values, si, float(tr_ms))
        except Exception:
            continue

        if len(fit) < 3:
            continue
        t1_value = float(fit[0])
        rho_value = float(fit[1])
        r2_value = float(fit[2])
        sse_value = float(fit[5]) if len(fit) > 5 else float("nan")

        if not (np.isfinite(t1_value) and np.isfinite(rho_value) and np.isfinite(r2_value)):
            continue

        r_squared[idx] = r2_value
        sse[idx] = sse_value
        if r2_value < rsquared_threshold:
            threshold_failed[idx] = True
            continue

        t1_fit[idx] = t1_value
        rho_fit[idx] = rho_value

    t1_map = np.reshape(t1_fit, spatial_shape)
    rho_map = np.reshape(rho_fit, spatial_shape)
    r_squared_map = np.reshape(r_squared, spatial_shape)
    sse_map = np.reshape(sse, spatial_shape)

    valid_mask = (t1_fit != float(invalid_fill_value)) & mask_flat

    return {
        "t1_map": t1_map,
        "rho_map": rho_map,
        "r_squared_map": r_squared_map,
        "sse_map": sse_map,
        "metrics": {
            "total_voxels": int(n_voxels),
            "mask_voxels": int(mask_flat.sum()),
            "valid_fits": int(valid_mask.sum()),
            "threshold_failed": int((threshold_failed & mask_flat).sum()),
            "finite_signal_voxels": int((finite_signal & mask_flat).sum()),
            "b1_valid_voxels": int((b1_valid & mask_flat).sum()),
            "t1_mean_ms": float(np.mean(t1_fit[valid_mask])) if np.any(valid_mask) else float("nan"),
            "t1_median_ms": float(np.median(t1_fit[valid_mask])) if np.any(valid_mask) else float("nan"),
            "r2_mean": float(np.mean(r_squared[mask_flat])) if np.any(mask_flat) else float("nan"),
            "r2_median": float(np.median(r_squared[mask_flat])) if np.any(mask_flat) else float("nan"),
        },
    }


def _default_output_label(config: ParametricT1Config) -> str:
    first_name = config.vfa_files[0].name
    if first_name.endswith(".nii.gz"):
        return first_name[:-7]
    return config.vfa_files[0].stem


def run_parametric_t1_pipeline(
    config: ParametricT1Config,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run the parametric T1 workflow for VFA fitting."""
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    _emit_event(
        event_callback,
        "run_start",
        output_dir=str(config.output_dir),
        fit_type=config.fit_type,
        rsquared_threshold=float(config.rsquared_threshold),
    )

    vfa_data, affine, header = _load_vfa_data(config)
    n_flips = int(vfa_data.shape[-1])

    flip_angles_deg, tr_ms, sidecars, tr_source = _resolve_flip_angles_and_tr_ms(config, n_flips)
    _emit_event(
        event_callback,
        "inputs_resolved",
        n_flips=n_flips,
        tr_ms=float(tr_ms),
        tr_source=tr_source,
        flip_angles_deg=[float(v) for v in flip_angles_deg],
        sidecars=[str(path) for path in sidecars],
    )

    mask = None
    if config.mask_file is not None:
        mask_data, _, _ = _load_nifti(config.mask_file)
        mask = np.squeeze(mask_data)
        if mask.shape != vfa_data.shape[:-1]:
            raise ValueError(f"mask shape {mask.shape} does not match VFA spatial shape {vfa_data.shape[:-1]}")
    b1_scale_map, b1_map_path = _resolve_b1_scale_map(config, vfa_data.shape[:-1])
    b1_mode = "none"
    if b1_map_path is not None:
        b1_mode = "explicit" if config.b1_map_file is not None else "auto"
    _emit_event(
        event_callback,
        "b1_map_resolved",
        b1_mode=b1_mode,
        b1_map_path=str(b1_map_path) if b1_map_path else None,
    )

    fit_type = config.fit_type.strip().lower()
    if fit_type == "t1_fa_linear_fit":
        fit_result = _fit_t1_fa_linear_map(
            vfa_data=vfa_data,
            flip_angles_deg=flip_angles_deg,
            tr_ms=tr_ms,
            rsquared_threshold=float(config.rsquared_threshold),
            invalid_fill_value=float(config.invalid_fill_value),
            mask=mask,
            b1_scale_map=b1_scale_map,
        )
    elif fit_type == "t1_fa_fit":
        fit_result = _fit_t1_fa_model_map(
            vfa_data=vfa_data,
            flip_angles_deg=flip_angles_deg,
            tr_ms=tr_ms,
            rsquared_threshold=float(config.rsquared_threshold),
            invalid_fill_value=float(config.invalid_fill_value),
            mask=mask,
            b1_scale_map=b1_scale_map,
            fit_fn=t1_fa_nonlinear_fit,
        )
    else:
        fit_result = _fit_t1_fa_model_map(
            vfa_data=vfa_data,
            flip_angles_deg=flip_angles_deg,
            tr_ms=tr_ms,
            rsquared_threshold=float(config.rsquared_threshold),
            invalid_fill_value=float(config.invalid_fill_value),
            mask=mask,
            b1_scale_map=b1_scale_map,
            fit_fn=t1_fa_two_point_fit,
        )

    output_label = config.output_label.strip() or _default_output_label(config)
    t1_path = config.output_dir / f"{config.output_basename}_{config.fit_type}_{output_label}.nii.gz"
    _save_nifti(t1_path, fit_result["t1_map"], affine, header)
    _emit_event(event_callback, "artifact_written", artifact_type="t1_map", path=str(t1_path))

    rsquared_path: Optional[Path] = None
    if config.write_r_squared:
        rsquared_path = config.output_dir / f"Rsquared_{config.fit_type}_{output_label}.nii.gz"
        _save_nifti(rsquared_path, fit_result["r_squared_map"], affine, header)
        _emit_event(event_callback, "artifact_written", artifact_type="r_squared", path=str(rsquared_path))

    rho_path: Optional[Path] = None
    if config.write_rho_map:
        rho_path = config.output_dir / f"{config.output_basename}_rho_{config.fit_type}_{output_label}.nii.gz"
        _save_nifti(rho_path, fit_result["rho_map"], affine, header)
        _emit_event(event_callback, "artifact_written", artifact_type="rho_map", path=str(rho_path))

    finished_at = datetime.now(timezone.utc)
    summary = {
        "meta": {
            "pipeline": "parametric_t1_vfa",
            "status": "ok",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": float((finished_at - started_at).total_seconds()),
        },
        "config": config.to_dict(),
        "resolved_inputs": {
            "tr_ms": float(tr_ms),
            "tr_source": tr_source,
            "flip_angles_deg": [float(v) for v in flip_angles_deg],
            "n_flips": int(n_flips),
            "sidecars": [str(path) for path in sidecars],
            "b1_mode": b1_mode,
            "b1_map_path": str(b1_map_path) if b1_map_path else None,
        },
        "outputs": {
            "t1_map_path": str(t1_path),
            "rsquared_map_path": str(rsquared_path) if rsquared_path else None,
            "rho_map_path": str(rho_path) if rho_path else None,
        },
        "metrics": fit_result["metrics"],
    }

    summary_path = config.output_dir / "parametric_t1_run.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    summary["meta"]["summary_path"] = str(summary_path)
    _emit_event(event_callback, "run_done", summary_path=str(summary_path), status="ok")
    return summary
