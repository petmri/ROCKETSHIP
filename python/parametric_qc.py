"""QC figures for a parametric T1 run.

Three questions a person asks after a T1 fit, one figure each: are the values
physiological, how much of the volume did the R^2 threshold reject, and does the map look
like anatomy rather than noise. Everything here reads what the fit already produced -- no
refitting, no second pass over the data.

Figures are decorative. A run that produced numbers is a successful run even when
matplotlib is absent or one plot fails, so every entry point here degrades to "no figure"
rather than raising. That is the opposite of the policy for results, deliberately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Shared with the DCE figures so a study's plots look like one set.
_INK = "#1F2933"
_FILL = "#2A7F62"
_MARK = "#C73E1D"
_DPI = 150


def _plt():
    """matplotlib's pyplot on the Agg backend, or None if it is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _finite(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def write_qc_figures(
    output_dir: Path,
    t1_map: np.ndarray,
    r_squared_map: Optional[np.ndarray],
    rsquared_threshold: float,
    label: str,
    invalid_fill_value: float = float("nan"),
) -> Dict[str, str]:
    """Write the QC set and return `{name: path}` for the ones that were produced.

    An empty dict is a normal outcome, not a failure: no matplotlib, or a map with nothing
    finite in it, both land there.
    """
    plt = _plt()
    if plt is None:
        return {}

    figures: Dict[str, str] = {}
    # Voxels the fit filled rather than fitted are not data; excluding them is what makes
    # the histogram show the tissue rather than a spike at the fill value.
    t1_valid = _valid_t1(t1_map, invalid_fill_value)

    for name, fn in (
        ("t1_histogram", lambda: _t1_histogram(plt, output_dir, t1_valid, label)),
        ("r2_histogram", lambda: _r2_histogram(
            plt, output_dir, r_squared_map, rsquared_threshold, label)),
        ("t1_montage", lambda: _t1_montage(plt, output_dir, t1_map, invalid_fill_value, label)),
    ):
        try:
            path = fn()
        except Exception:
            path = None  # one bad plot must not cost the others, or the run
        if path is not None:
            figures[name] = str(path)
    return figures


def _valid_t1(t1_map: np.ndarray, invalid_fill_value: float) -> np.ndarray:
    values = _finite(t1_map)
    if np.isfinite(invalid_fill_value):
        values = values[values != invalid_fill_value]
    # A T1 of zero or less is not a measurement; it is a fit that failed into the floor.
    return values[values > 0.0]


def _t1_histogram(plt, output_dir: Path, t1_valid: np.ndarray, label: str) -> Optional[Path]:
    if t1_valid.size == 0:
        return None
    # Clipped at the 99th percentile: a handful of runaway voxels otherwise stretch the axis
    # far enough that the tissue peak collapses into the first bin.
    upper = float(np.percentile(t1_valid, 99.0))
    shown = t1_valid[t1_valid <= upper] if upper > 0 else t1_valid
    median = float(np.median(t1_valid))

    path = output_dir / f"qc_t1_histogram_{label}.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(shown, bins=60, color=_FILL, edgecolor="black", alpha=0.85)
    ax.axvline(median, color=_MARK, linestyle="--", linewidth=1.5, label=f"median {median:.0f} ms")
    ax.set_title("T1 distribution over fitted voxels")
    ax.set_xlabel("T1 (ms)")
    ax.set_ylabel("voxels")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="upper right")
    _stamp(ax, f"n = {t1_valid.size:,}  ·  99th pct shown")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path


def _r2_histogram(
    plt, output_dir: Path, r_squared_map: Optional[np.ndarray], threshold: float, label: str
) -> Optional[Path]:
    if r_squared_map is None:
        return None
    values = _finite(r_squared_map)
    # R^2 is only meaningful on voxels that were fitted; the background sits at exactly 0.
    values = values[values != 0.0]
    if values.size == 0:
        return None

    rejected = int(np.count_nonzero(values < threshold))
    share = 100.0 * rejected / values.size

    # R^2 goes arbitrarily negative -- it is negative whenever the fit is worse than a flat
    # line, and a failed voxel can reach -200. Plotting the true range crushes [0,1] into
    # one bar, but silently cropping to [0,1] is worse: it hides most of what the threshold
    # rejects, so the annotation and the bars disagree. Clip into a labelled underflow bin.
    underflow = int(np.count_nonzero(values < 0.0))
    shown = np.clip(values, 0.0, 1.0)

    path = output_dir / f"qc_r2_histogram_{label}.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(shown, bins=60, range=(0.0, 1.0), color=_FILL, edgecolor="black", alpha=0.85)
    ax.axvline(
        threshold,
        color=_MARK,
        linestyle="--",
        linewidth=1.5,
        label=f"threshold {threshold:g} — rejects {rejected:,} of {values.size:,} ({share:.1f}%)",
    )
    ax.set_title("Goodness of fit, and what the threshold discards")
    ax.set_xlabel(
        f"R²   ({underflow:,} voxels below 0 — fit worse than a flat line — clipped into the first bin)"
        if underflow
        else "R²"
    )
    ax.set_ylabel("voxels")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path


def _t1_montage(
    plt, output_dir: Path, t1_map: np.ndarray, invalid_fill_value: float, label: str
) -> Optional[Path]:
    volume = np.asarray(t1_map, dtype=np.float64)
    # A single-slice acquisition is a legitimate 2-D map; treat it as a one-slice volume
    # rather than declining to draw it.
    if volume.ndim == 2:
        volume = volume[:, :, np.newaxis]
    if volume.ndim != 3 or min(volume.shape) == 0:
        return None
    finite = _valid_t1(volume, invalid_fill_value)
    if finite.size == 0:
        return None
    # A shared window across every panel, so a slice reads as brighter because its T1 is
    # higher and not because it was scaled on its own.
    lo, hi = (float(np.percentile(finite, 2.0)), float(np.percentile(finite, 98.0)))
    if not hi > lo:
        return None

    n_slices = volume.shape[2]
    picks = _slice_picks(n_slices)
    cols = min(len(picks), 4)
    rows = int(np.ceil(len(picks) / cols))

    path = output_dir / f"qc_t1_montage_{label}.png"
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for ax, k in zip([a for row in axes for a in row], picks):
        panel = np.ma.masked_invalid(volume[:, :, k])
        image = ax.imshow(panel.T, origin="lower", cmap="magma", vmin=lo, vmax=hi)
        ax.set_title(f"slice {k}", fontsize=9, color=_INK)
        ax.set_xticks([])
        ax.set_yticks([])
    # Blank any panel the slice count did not fill, so an odd number does not leave axes
    # with ticks and a frame sitting in the grid.
    for ax in [a for row in axes for a in row][len(picks):]:
        ax.axis("off")
    fig.suptitle("T1 map (ms)", color=_INK)
    fig.colorbar(image, ax=axes, shrink=0.8, label="T1 (ms)")
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _slice_picks(n_slices: int, limit: int = 12) -> list:
    """Up to `limit` slices spread through the volume, skipping the empty end slices."""
    if n_slices <= limit:
        return list(range(n_slices))
    # Trimmed at both ends: the first and last slices of a volume are usually air, and
    # spending panels on them is what makes a montage look like it failed.
    return [int(round(v)) for v in np.linspace(n_slices * 0.1, n_slices * 0.9, limit)]


def _stamp(ax, text: str) -> None:
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=_INK,
        alpha=0.7,
    )
