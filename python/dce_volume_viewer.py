"""Slice-at-a-time viewer for the 3-D maps and 4-D dynamics a DCE run produces.

Deliberately 2-D: one slice at a time, with a slider per spatial and temporal axis. No
volume rendering -- the questions this answers ("is this Ktrans map plausible", "did the
bolus arrive when the AIF says it did") are answered by looking at slices.

Data is shown as stored: slices are taken along the array's third axis, and the plane is
turned a fixed quarter turn counter-clockwise so it sits upright instead of on its side.
The affine is not consulted to reorient or reslice anything, and the rotation is the same
for every volume, so what you see is the array the pipeline wrote. Voxel sizes are used for
display aspect only -- scaling, never resampling.

Volumes are held as lazy nibabel images and read one at a time through `dataobj`.
`get_fdata()` is avoided throughout: it upcasts to float64, which doubles the footprint of
a dynamic that is float32 on disk, and it materialises every timepoint whether or not the
viewer is showing them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Display numpy arrays as `arr[row, col]` rather than pyqtgraph's default column-major
# reading, so slices are not silently transposed.
pg.setConfigOption("imageAxisOrder", "row-major")

# Slices are taken along the stored third axis, always. No affine-driven choice: the point
# is to show what was written, and a fixture whose affine disagrees with its array layout
# (sub-02downsample is (128, 128, 1, 64) but reports axcodes ('L', 'S', 'P')) would otherwise
# be sliced into 128 strips one voxel wide.
SLICE_AXIS = 2

# Quarter turns counter-clockwise applied to every plane before it is drawn. NIfTI's first
# array axis is the fastest-varying in-plane direction, and reading row-major puts it on the
# display's vertical axis, which shows the image on its side. This is a fixed rotation applied
# identically to every volume -- not an affine-driven reorientation -- so volumes stay
# consistent with each other and with the arrays on disk.
DISPLAY_QUARTER_TURNS_CCW = 1

# Offered in the LUT combo. Greyscale first because it is the honest default for signal
# intensity; the perceptually-uniform maps are for parameter maps, where colour carries
# meaning. `turbo` and `hot` are included because readers ask for them, not because they
# are good choices for quantitative display.
COLORMAPS: Tuple[Tuple[str, str], ...] = (
    ("Greyscale", "grey"),
    ("Inferno", "inferno"),
    ("Viridis", "viridis"),
    ("Turbo", "turbo"),
    ("Hot", "hot"),
)

# Percentiles used for the initial window. Full min/max is unusable on DCE data, where a
# handful of vessel or artefact voxels sit orders of magnitude above tissue.
WINDOW_PERCENTILES = (2.0, 98.0)

# Timepoints sampled when picking the initial window for a 4-D volume. Levels taken from a
# single frame would clip badly once contrast arrives, and levels recomputed per frame make
# the image flicker while scrubbing, so sample across the series once and then hold.
LEVEL_SAMPLE_FRAMES = 5

NIFTI_SUFFIXES = ("*.nii", "*.nii.gz")


@dataclass
class Volume:
    """One loadable image, kept lazy until a slice is actually requested."""

    label: str
    path: Path
    _image: object = field(default=None, repr=False)
    _levels: Optional[Tuple[float, float]] = field(default=None, repr=False)

    @property
    def image(self):
        if self._image is None:
            import nibabel as nib

            self._image = nib.load(str(self.path))
        return self._image

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(n) for n in self.image.shape)

    @property
    def n_time(self) -> int:
        shape = self.shape
        return int(shape[3]) if len(shape) > 3 else 1

    @property
    def is_4d(self) -> bool:
        return self.n_time > 1

    def spatial_shape(self) -> Tuple[int, int, int]:
        shape = self.shape
        dims = list(shape[:3]) + [1] * max(0, 3 - len(shape))
        return int(dims[0]), int(dims[1]), int(dims[2])

    def zooms(self) -> Tuple[float, float, float]:
        raw = tuple(float(z) for z in self.image.header.get_zooms()[:3])
        padded = raw + (1.0,) * (3 - len(raw))
        # A zero or non-finite zoom would collapse the display aspect; treat it as isotropic.
        return tuple(z if np.isfinite(z) and z > 0 else 1.0 for z in padded[:3])  # type: ignore[return-value]

    def n_slices(self) -> int:
        return int(self.spatial_shape()[SLICE_AXIS])

    def frame(self, t: int) -> np.ndarray:
        """One 3-D volume, read straight from `dataobj` so nothing else is materialised."""
        data = self.image.dataobj
        if self.is_4d:
            arr = np.asarray(data[..., int(t)])
        else:
            arr = np.asarray(data)
        while arr.ndim < 3:
            arr = arr[..., None]
        return arr

    def slice_2d(self, t: int, k: int) -> np.ndarray:
        """Plane `k` of the stored array, turned upright for display.

        The only thing done to the samples is the quarter turn; no interpolation, no
        resampling, and no per-volume decision about which way is up.
        """
        arr = self.frame(t)
        k = int(np.clip(k, 0, arr.shape[SLICE_AXIS] - 1))
        plane = np.asarray(arr[:, :, k], dtype=np.float32)
        # rot90 returns a negative-strided view; make it contiguous before it reaches Qt.
        return np.ascontiguousarray(np.rot90(plane, DISPLAY_QUARTER_TURNS_CCW))

    def pixel_aspect(self) -> float:
        """Displayed row-to-column size ratio, so non-square voxels are not shown square.

        Scaling only -- no resampling. The quarter turn in `slice_2d` swaps which stored
        axis lands on which display axis, so the zooms are read in the same swapped order.
        """
        col_mm, row_mm, _ = self.zooms()
        return float(row_mm / col_mm) if col_mm > 0 else 1.0

    def levels(self) -> Tuple[float, float]:
        """Robust display window, computed once and then held across both sliders."""
        if self._levels is not None:
            return self._levels

        if self.is_4d:
            frames = np.unique(
                np.linspace(0, self.n_time - 1, num=min(LEVEL_SAMPLE_FRAMES, self.n_time)).astype(int)
            )
            samples = [self.frame(int(t)) for t in frames]
        else:
            samples = [self.frame(0)]

        finite = np.concatenate([s[np.isfinite(s)].ravel() for s in samples]) if samples else np.zeros(1)
        if finite.size == 0:
            self._levels = (0.0, 1.0)
            return self._levels

        lo, hi = (float(v) for v in np.percentile(finite, WINDOW_PERCENTILES))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
        self._levels = (lo, hi)
        return self._levels


def stored_index(row: int, col: int, plane_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Undo `Volume.slice_2d`'s quarter turn: display (row, col) back to stored (i, j).

    Paired with the `np.rot90` call there, which satisfies `rot90(m)[r, c] == m[c, N - 1 - r]`
    for `N = m.shape[1]`, the number of display rows. Change the turn and this changes with it.
    """
    return int(col), int(plane_shape[0] - 1 - row)


def discover_result_volumes(*roots: Path) -> List[Volume]:
    """Every NIfTI under the given directories, de-duplicated and sorted by name."""
    seen: Dict[Path, None] = {}
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        if not root.is_dir():
            continue
        for pattern in NIFTI_SUFFIXES:
            for path in sorted(root.rglob(pattern)):
                seen.setdefault(path.resolve(), None)
    return [Volume(label=p.name, path=p) for p in seen]


class VolumeViewer(QWidget):
    """Volume picker, slice and time sliders, and pyqtgraph's window/level histogram."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._volumes: List[Volume] = []
        self._current: Optional[Volume] = None
        self._suspend_refresh = False
        # The plane currently on screen, kept so the hover readout can look up a value
        # without re-reading the file on every mouse move.
        self._plane: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.volume_combo = QComboBox()
        self.volume_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.lut_combo = QComboBox()
        for label, name in COLORMAPS:
            self.lut_combo.addItem(label, name)
        self.add_button = QPushButton("Add Volumes...")
        top.addWidget(QLabel("Volume:"))
        top.addWidget(self.volume_combo, 1)
        top.addWidget(QLabel("LUT:"))
        top.addWidget(self.lut_combo)
        top.addWidget(self.add_button)
        layout.addLayout(top)

        # A 2-D image keeps ImageView's own timeline hidden -- the sliders below are ours, so
        # that 4-D data gets a genuine time axis rather than a flattened stack.
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.view.invertY(True)
        layout.addWidget(self.image_view, 1)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_label = QLabel("Slice -/-")
        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("Slice"))
        slice_row.addWidget(self.slice_slider, 1)
        slice_row.addWidget(self.slice_label)
        layout.addLayout(slice_row)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_label = QLabel("Frame -/-")
        self.time_row_widget = QWidget()
        time_row = QHBoxLayout(self.time_row_widget)
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.addWidget(QLabel("Time "))
        time_row.addWidget(self.time_slider, 1)
        time_row.addWidget(self.time_label)
        layout.addWidget(self.time_row_widget)

        self.status_label = QLabel("No volumes loaded.")
        self.status_label.setEnabled(False)
        # Voxel readout sits beside the status line rather than following the cursor: a
        # floating tooltip over a slice covers the pixels you are trying to read.
        self.hover_label = QLabel("")
        self.hover_label.setEnabled(False)
        self.hover_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.hover_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        status_row.addWidget(self.hover_label)
        layout.addLayout(status_row)

        self.image_view.scene.sigMouseMoved.connect(self._on_mouse_moved)
        self.volume_combo.currentIndexChanged.connect(self._on_volume_changed)
        self.lut_combo.currentIndexChanged.connect(self._apply_colormap)
        self.slice_slider.valueChanged.connect(self._refresh_image)
        self.time_slider.valueChanged.connect(self._refresh_image)
        self.add_button.clicked.connect(self._on_add_clicked)

        self._apply_colormap()
        self._set_controls_enabled(False)

    def set_volumes(self, volumes: Sequence[Volume], keep_selection: bool = True) -> None:
        """Replace the volume list, holding the current selection by name where possible."""
        previous = self._current.path if (keep_selection and self._current) else None
        self._volumes = list(volumes)

        self._suspend_refresh = True
        self.volume_combo.clear()
        for volume in self._volumes:
            self.volume_combo.addItem(volume.label, str(volume.path))
            self.volume_combo.setItemData(self.volume_combo.count() - 1, str(volume.path), Qt.ToolTipRole)
        self._suspend_refresh = False

        if not self._volumes:
            self._current = None
            self._plane = None
            self._set_controls_enabled(False)
            self.image_view.clear()
            self.status_label.setText("No volumes loaded.")
            self.hover_label.setText("")
            return

        index = 0
        if previous is not None:
            for i, volume in enumerate(self._volumes):
                if volume.path == previous:
                    index = i
                    break
        self.volume_combo.setCurrentIndex(index)
        self._on_volume_changed(index)

    def add_paths(self, paths: Sequence[Path]) -> None:
        known = {v.path for v in self._volumes}
        additions = [Volume(label=Path(p).name, path=Path(p).resolve()) for p in paths if Path(p).resolve() not in known]
        if additions:
            self.set_volumes(self._volumes + additions)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.volume_combo, self.lut_combo, self.slice_slider, self.time_slider):
            widget.setEnabled(enabled)

    def _on_add_clicked(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self, "Select NIfTI volumes", "", "NIfTI (*.nii *.nii.gz);;All Files (*)"
        )
        if selected:
            self.add_paths([Path(p) for p in selected])

    def _on_volume_changed(self, index: int) -> None:
        if self._suspend_refresh or not (0 <= index < len(self._volumes)):
            return
        volume = self._volumes[index]
        try:
            n_slices = volume.n_slices()
            n_time = volume.n_time
        except Exception as exc:  # unreadable or non-image file picked from the dialog
            self._current = None
            self._plane = None
            self._set_controls_enabled(False)
            self.image_view.clear()
            self.status_label.setText(f"Could not read {volume.label}: {exc}")
            self.hover_label.setText("")
            return

        self._current = volume
        self._set_controls_enabled(True)

        self._suspend_refresh = True
        self.slice_slider.setRange(0, max(0, n_slices - 1))
        self.slice_slider.setValue(n_slices // 2)
        self.slice_slider.setEnabled(n_slices > 1)
        self.time_slider.setRange(0, max(0, n_time - 1))
        self.time_slider.setValue(0)
        self.time_row_widget.setVisible(volume.is_4d)
        self._suspend_refresh = False

        # New volume, new data range: let the histogram re-window rather than carrying the
        # previous volume's levels onto a map with a completely different unit.
        self._refresh_image(reset_levels=True)

    def _refresh_image(self, *_args, reset_levels: bool = False) -> None:
        if self._suspend_refresh or self._current is None:
            return
        volume = self._current
        t = int(self.time_slider.value())
        k = int(self.slice_slider.value())
        try:
            plane = volume.slice_2d(t, k)
        except Exception as exc:
            self.status_label.setText(f"Could not read slice: {exc}")
            return

        levels = volume.levels() if reset_levels else self.image_view.getHistogramWidget().getLevels()
        # Stretch for voxel aspect first, then frame the view. Letting `setImage` auto-range
        # would fit the unstretched extent and leave the taller half of an anisotropic slice
        # off screen.
        self.image_view.setImage(plane, autoLevels=False, levels=levels, autoRange=False)
        self.image_view.getImageItem().setRect(0, 0, plane.shape[1], plane.shape[0] * volume.pixel_aspect())
        if reset_levels:
            self.image_view.autoRange()
        self._plane = plane

        self.slice_label.setText(f"{k + 1}/{volume.n_slices()}")
        self.time_label.setText(f"{t + 1}/{volume.n_time}")
        zx, zy, zz = volume.zooms()
        self.status_label.setText(
            f"{volume.label}  |  {'x'.join(str(n) for n in volume.shape)}  |  "
            f"{zx:.3g} x {zy:.3g} x {zz:.3g} mm"
        )

    def _on_mouse_moved(self, scene_pos) -> None:
        """Report the voxel under the cursor, in the coordinates the file stores it at."""
        plane, volume = self._plane, self._current
        if plane is None or volume is None:
            return
        # `setRect` only scales the item, so item-local coordinates are array pixels again --
        # x indexes display columns, y display rows, matching the row-major axis order.
        point = self.image_view.getImageItem().mapFromScene(scene_pos)
        row, col = int(np.floor(point.y())), int(np.floor(point.x()))
        if not (0 <= row < plane.shape[0] and 0 <= col < plane.shape[1]):
            self.hover_label.setText("")
            return

        i, j = stored_index(row, col, plane.shape)
        index = [str(i), str(j), str(int(self.slice_slider.value()))]
        if volume.is_4d:
            index.append(str(int(self.time_slider.value())))
        self.hover_label.setText(f"[{', '.join(index)}] = {float(plane[row, col]):.6g}")

    def _apply_colormap(self) -> None:
        name = self.lut_combo.currentData() or "grey"
        try:
            cmap = pg.colormap.get(name, source="matplotlib")
        except Exception:
            cmap = pg.colormap.get("grey")
        if cmap is not None:
            self.image_view.setColorMap(cmap)
