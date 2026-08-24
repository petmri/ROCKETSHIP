"""PySide6 GUI for running the Python parametric T1 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dce_volume_viewer import Volume, VolumeViewer, discover_result_volumes
from gui_common import (
    GuiCommonMixin,
    WINDOW_QSS,
    build_figures_panel,
    build_log_view,
    build_run_bar,
    form_label as _form_label,
    paths_to_text as _paths_to_text,
    resolve_path as _resolve_path,
    resolve_paths as _resolve_paths,
    text_to_paths as _text_to_paths,
)
import parametric_config
from parametric_pipeline import ALLOWED_BACKENDS
from run_reporting import Reporter
from version import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "parametric_run_example.json"
CLI_ENTRYPOINT = REPO_ROOT / "run_parametric_python_cli.py"
OPTIONS_DOC_PATH = REPO_ROOT / "docs" / "parametric_options.md"

# Main-window tab order, matching the DCE GUI so the two read alike: starting a run moves
# off Inputs to the log, and finishing moves on to the figures.
TAB_INPUTS = 0
TAB_LOG = 1
TAB_FIGURES = 2
TAB_RESULTS = 3

# Widest text the run bar's status label has to hold, so the progress bar's left edge does
# not move as a run progresses. Parametric fits one map, so there is no per-model line.
WIDEST_STAGE_TEXT = "Status: failed (exit=255)"

# The three fit types ParametricT1Config.validate accepts, and what each one is, since the
# names alone do not say. A free-text field here only ever produced a late ValueError.
FIT_TYPES = (
    ("t1_fa_fit", "Non-linear fit over all flip angles (default)"),
    ("t1_fa_linear_fit", "Linearised fit; faster, more noise-sensitive"),
    ("t1_fa_two_point_fit", "Closed form from two flip angles"),
)


def _float_list_to_text(values: List[float]) -> str:
    return ", ".join(f"{float(v):g}" for v in values)


def _text_to_float_list(raw: str) -> List[float]:
    text = raw.strip()
    if not text:
        return []
    parts = [token.strip() for token in text.replace("\n", ",").split(",") if token.strip()]
    return [float(token) for token in parts]


def _select(combo: QComboBox, value: str) -> None:
    """Select `value` if the combo offers it, leaving the current choice alone if not."""
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "on"}


def _same(a: Any, b: Any) -> bool:
    """Equal after the round trip through a text box, where 0.6 comes back as "0.6"."""
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


class ParametricGuiWindow(GuiCommonMixin, QMainWindow):
    """Main window for configuring and running parametric T1 CLI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"ROCKETSHIP Parametric T1 v{__version__} (Python GUI)")
        self.resize(1000, 700)

        self._stdout_buffer = ""
        self._event_paths: set[str] = set()
        self._log_reporter: Optional[Reporter] = None
        self._process: Optional[QProcess] = None
        self._config_path = DEFAULT_CONFIG_PATH
        self._loaded_payload: Dict[str, Any] = {}
        self._last_run_config_path: Optional[Path] = None

        self._build_ui()
        self._load_config(DEFAULT_CONFIG_PATH)

    # -- layout ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        self.setStyleSheet(WINDOW_QSS)
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)
        self.tabs.insertTab(TAB_INPUTS, self._build_inputs_tab(), "Inputs")
        self.tabs.insertTab(TAB_LOG, self._build_log_tab(), "CLI Output")
        self.tabs.insertTab(TAB_FIGURES, self._build_figures_tab(), "QC Figures")
        self.tabs.insertTab(TAB_RESULTS, self._build_results_tab(), "Results")

        # Below the tabs rather than inside Inputs: the tab you watch during a run is CLI
        # Output, and a Stop button you have to change tabs to reach is the wrong place.
        self._build_run_controls(root_layout)

        self.load_button.clicked.connect(self._on_load_config_clicked)
        self.save_button.clicked.connect(self._on_save_config_clicked)
        self.reset_button.clicked.connect(lambda: self._load_config(DEFAULT_CONFIG_PATH))
        self.options_button.clicked.connect(self._on_open_options_doc)
        self.run_button.clicked.connect(self._start_run)
        self.stop_button.clicked.connect(self._stop_run_hard)
        self.figure_list.currentItemChanged.connect(self._on_figure_selected)
        self.open_output_button.clicked.connect(self._open_output_dir)

    def _build_inputs_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        # A little air above the first group, so it does not sit flush against the tab bar.
        layout.addSpacing(6)

        config_group = QGroupBox("Run Config")
        config_row = QHBoxLayout(config_group)
        self.config_path_edit = QLineEdit(str(DEFAULT_CONFIG_PATH))
        self.config_path_edit.setReadOnly(True)
        # The styled border and padding eat into the field, clipping the path text at the
        # default height. Give back the pixels they take.
        self.config_path_edit.setMinimumHeight(self.config_path_edit.sizeHint().height() + 4)
        self.load_button = QPushButton("Load Config")
        self.save_button = QPushButton("Save Config As")
        self.reset_button = QPushButton("Reset Defaults")
        config_row.addWidget(self.config_path_edit, 1)
        config_row.addWidget(self.load_button)
        config_row.addWidget(self.save_button)
        config_row.addWidget(self.reset_button)
        layout.addWidget(config_group)

        self._build_core_settings(layout)
        self._build_input_settings(layout)
        self._build_resolved_settings(layout)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        # Always on, not as-needed: expanding the collapsible section is what pushes this
        # tab past the viewport, and a scrollbar appearing at that moment shifts the layout
        # underneath the click.
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        return scroll

    def _build_core_settings(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Core Settings")
        form = QFormLayout(group)

        self.output_dir_edit = QLineEdit()
        self.fit_type_combo = QComboBox()
        for value, hint in FIT_TYPES:
            self.fit_type_combo.addItem(value, value)
            self.fit_type_combo.setItemData(self.fit_type_combo.count() - 1, hint, Qt.ToolTipRole)
        self.backend_combo = QComboBox()
        for value in sorted(ALLOWED_BACKENDS):
            self.backend_combo.addItem(value, value)
        self.output_basename_edit = QLineEdit()
        self.output_label_edit = QLineEdit()
        self.rsq_threshold_edit = QLineEdit()
        self.invalid_fill_edit = QLineEdit()
        self.xy_smooth_sigma_edit = QLineEdit()
        self.mask_file_edit = QLineEdit()
        self.odd_echoes_check = QCheckBox()
        self.write_rsq_check = QCheckBox()
        self.write_rho_check = QCheckBox()
        self.write_figures_check = QCheckBox()

        form.addRow(
            _form_label("Output folder", "output_dir", "Where maps, summary and figures land."),
            self._line_edit_with_browse(
                self.output_dir_edit,
                lambda: self._choose_directory_for(self.output_dir_edit, "Select output folder"),
            ),
        )
        form.addRow(
            _form_label("Fit method", "fit_type", "How T1 is estimated from the flip angles."),
            self.fit_type_combo,
        )
        form.addRow(
            _form_label(
                "Backend", "backend",
                "auto tries GPU, then CPU acceleration, then the plain Python path.",
            ),
            self.backend_combo,
        )
        form.addRow(_form_label("Output name", "output_basename"), self.output_basename_edit)
        form.addRow(_form_label("Run label", "output_label"), self.output_label_edit)
        form.addRow(
            _form_label(
                "R-squared threshold", "rsquared_threshold",
                "Voxels fitting worse than this are replaced by the fill value.",
            ),
            self.rsq_threshold_edit,
        )
        form.addRow(
            _form_label("Fill value", "invalid_fill_value", "Written where a fit was rejected."),
            self.invalid_fill_edit,
        )
        form.addRow(
            _form_label(
                "In-plane smoothing", "xy_smooth_sigma", "Gaussian sigma in voxels; 0 disables.",
            ),
            self.xy_smooth_sigma_edit,
        )
        form.addRow(
            _form_label("Mask", "mask_file", "Restrict fitting to this mask. Empty fits everything."),
            self._line_edit_with_browse(
                self.mask_file_edit,
                lambda: self._choose_file_for(self.mask_file_edit, "Select mask"),
            ),
        )
        form.addRow(_form_label("Odd echoes only", "odd_echoes"), self.odd_echoes_check)
        form.addRow(_form_label("Write R-squared map", "write_r_squared"), self.write_rsq_check)
        form.addRow(_form_label("Write rho map", "write_rho_map"), self.write_rho_check)
        form.addRow(_form_label("Write QC figures", "write_qc_figures"), self.write_figures_check)
        parent_layout.addWidget(group)

    def _build_input_settings(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("VFA Inputs")
        form = QFormLayout(group)
        self.vfa_files_edit = QPlainTextEdit()
        self.vfa_files_edit.setFixedHeight(90)
        self.flip_angles_edit = QLineEdit()
        self.tr_ms_edit = QLineEdit()
        self.b1_map_file_edit = QLineEdit()

        form.addRow(
            _form_label("VFA images", "vfa_files", "One path per line, one per flip angle."),
            self._text_area_with_browse(
                self.vfa_files_edit,
                lambda: self._choose_files_for(self.vfa_files_edit, "Select VFA images"),
            ),
        )
        form.addRow(
            _form_label(
                "Flip angles (deg)", "flip_angles_deg",
                "Leave empty to read them from the JSON sidecars.",
            ),
            self.flip_angles_edit,
        )
        form.addRow(
            _form_label("TR (ms)", "tr_ms", "Leave empty to read it from the JSON sidecars."),
            self.tr_ms_edit,
        )
        form.addRow(
            _form_label(
                "B1 map", "b1_map_file",
                "Empty looks for B1_scaled_FAreg.nii beside the VFA images, and uses\n"
                "the nominal flip angles only if none is found.",
            ),
            self._line_edit_with_browse(
                self.b1_map_file_edit,
                lambda: self._choose_file_for(self.b1_map_file_edit, "Select B1 map"),
            ),
        )
        parent_layout.addWidget(group)

    def _build_resolved_settings(self, parent_layout: QVBoxLayout) -> None:
        """Every known setting, its resolved value, and where that value came from.

        The form above shows what *you* set; this shows what the run will actually *use*.
        The two differ wherever a field is left empty and parametric_defaults.json supplies
        the value, which is exactly the case that is otherwise invisible. It is the DCE
        override table's `source` column, for a config flat enough not to need the rest of
        that table.
        """
        group, header, body = self._collapsible_section(
            "Resolved Settings (what this run will use)",
            "Every key from python/parametric_defaults.json, with the value this run "
            "resolves to and whether it came from your config or the defaults file.",
        )
        header.addStretch(1)
        self.options_button = QPushButton("Open Options Doc")
        header.addWidget(self.options_button)

        self.resolved_table = QTableWidget(0, 3)
        self.resolved_table.setHorizontalHeaderLabels(["key", "value", "source"])
        head = self.resolved_table.horizontalHeader()
        head.setStretchLastSection(False)
        head.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.Stretch)
        head.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.resolved_table.setMinimumHeight(320)
        # Read-only: the form above is where a value is changed. A second editable view of
        # one value is how two controls end up disagreeing.
        self.resolved_table.setEditTriggers(QTableWidget.NoEditTriggers)
        body.addWidget(self.resolved_table)

        row = QHBoxLayout()
        self.refresh_resolved_button = QPushButton("Refresh")
        self.refresh_resolved_button.clicked.connect(self._refresh_resolved_table)
        row.addWidget(self.refresh_resolved_button)
        row.addStretch(1)
        body.addLayout(row)
        parent_layout.addWidget(group)

    def _build_log_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.log_view = build_log_view()
        self._reset_log_view()
        layout.addWidget(self.log_view)
        return panel

    def _build_figures_tab(self) -> QWidget:
        panel = build_figures_panel()
        self.figure_list = panel.list_widget
        self.figure_preview = panel.preview
        return panel.widget

    def _build_results_tab(self) -> QWidget:
        """Slice viewer for the fitted maps and the VFA images they came from."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.results_viewer = VolumeViewer()
        layout.addWidget(self.results_viewer, 1)
        row = QHBoxLayout()
        self.open_output_button = QPushButton("Open Output Directory")
        row.addWidget(self.open_output_button)
        row.addStretch(1)
        layout.addLayout(row)
        return container

    def _build_run_controls(self, parent_layout: QVBoxLayout) -> None:
        bar = build_run_bar(
            run_text="Run Parametric T1",
            widest_stage_text=WIDEST_STAGE_TEXT,
            widest_detail_text="",
        )
        self.run_button = bar.run_button
        self.stop_button = bar.stop_button
        self.stage_label = bar.stage_label
        self.progress = bar.progress
        parent_layout.addWidget(bar.group)

    def _text_area_with_browse(self, edit: QPlainTextEdit, on_browse: Any) -> QWidget:
        """Multi-line path list plus a Browse button, top-aligned against the tall box."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(on_browse)
        layout.addWidget(browse, 0, Qt.AlignTop)
        return row

    def _reset_log_view(self) -> None:
        """Idle text, so an untouched tab is not an unexplained black rectangle."""
        self.log_view.setPlainText(
            "[idle] No run yet.\n"
            "Pipeline output appears here once you press Run Parametric T1."
        )

    def _on_open_options_doc(self) -> None:
        if not OPTIONS_DOC_PATH.exists():
            QMessageBox.warning(self, "Missing docs", f"Options doc not found: {OPTIONS_DOC_PATH}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(OPTIONS_DOC_PATH))):
            QMessageBox.information(self, "Options Doc", f"Open this file:\n{OPTIONS_DOC_PATH}")

    # -- config ----------------------------------------------------------------- #

    def _load_config(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(self, "Missing config", f"Config file not found: {path}")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._loaded_payload = payload
        self._config_path = path
        self.config_path_edit.setText(str(path))
        defaults = parametric_config.load_defaults()

        def shown(key: str) -> str:
            """The config's value, or the defaults-file value it will fall back to.

            Showing the resolved value rather than a blank field is what makes the form
            honest: an empty box next to a key that has a default reads as "off".
            """
            value = parametric_config.resolve_optional(payload, key, None, defaults=defaults)
            return "" if value is None else str(value)

        self.output_dir_edit.setText(str(payload.get("output_dir", "")))
        _select(self.fit_type_combo, shown("fit_type"))
        _select(self.backend_combo, shown("backend"))
        self.output_basename_edit.setText(shown("output_basename"))
        self.output_label_edit.setText(shown("output_label"))
        self.rsq_threshold_edit.setText(shown("rsquared_threshold"))
        self.invalid_fill_edit.setText(shown("invalid_fill_value"))
        self.xy_smooth_sigma_edit.setText(shown("xy_smooth_sigma"))
        self.mask_file_edit.setText(shown("mask_file"))
        self.b1_map_file_edit.setText(shown("b1_map_file"))
        self.tr_ms_edit.setText(shown("tr_ms"))
        self.odd_echoes_check.setChecked(_as_bool(shown("odd_echoes")))
        self.write_rsq_check.setChecked(_as_bool(shown("write_r_squared")))
        self.write_rho_check.setChecked(_as_bool(shown("write_rho_map")))
        self.write_figures_check.setChecked(_as_bool(shown("write_qc_figures")))
        self.vfa_files_edit.setPlainText(
            _paths_to_text([str(v) for v in (payload.get("vfa_files") or [])])
        )
        self.flip_angles_edit.setText(
            _float_list_to_text([float(v) for v in (payload.get("flip_angles_deg") or [])])
        )
        self._refresh_resolved_table()

    def _collect_config_payload(self) -> Dict[str, Any]:
        base_dir = self._base_dir()
        output_dir_text = self.output_dir_edit.text().strip()
        output_dir = (
            str(REPO_ROOT / "out" / "parametric_gui")
            if output_dir_text in {"", "."}
            else _resolve_path(output_dir_text, base_dir)
        )
        payload: Dict[str, Any] = {
            "output_dir": output_dir,
            "fit_type": self.fit_type_combo.currentData(),
            "backend": self.backend_combo.currentData(),
            "output_basename": self.output_basename_edit.text().strip(),
            "output_label": self.output_label_edit.text().strip(),
            "rsquared_threshold": float(self.rsq_threshold_edit.text().strip() or "0"),
            "odd_echoes": bool(self.odd_echoes_check.isChecked()),
            "xy_smooth_sigma": float(self.xy_smooth_sigma_edit.text().strip() or "0"),
            "write_r_squared": bool(self.write_rsq_check.isChecked()),
            "write_rho_map": bool(self.write_rho_check.isChecked()),
            "write_qc_figures": bool(self.write_figures_check.isChecked()),
            "invalid_fill_value": float(self.invalid_fill_edit.text().strip() or "-1"),
            "vfa_files": _resolve_paths(_text_to_paths(self.vfa_files_edit.toPlainText()), base_dir),
            "flip_angles_deg": _text_to_float_list(self.flip_angles_edit.text()),
        }
        # Empty means "discover it from the sidecars", which is not the same as a value, so
        # these keys are left out of the payload rather than written as null.
        if self.tr_ms_edit.text().strip():
            payload["tr_ms"] = float(self.tr_ms_edit.text().strip())
        if self.mask_file_edit.text().strip():
            payload["mask_file"] = _resolve_path(self.mask_file_edit.text(), base_dir)
        if self.b1_map_file_edit.text().strip():
            payload["b1_map_file"] = _resolve_path(self.b1_map_file_edit.text(), base_dir)
        return payload

    def _refresh_resolved_table(self) -> None:
        """Fill the resolved view from the form as it currently stands.

        Provenance is measured against the *loaded config*, not the payload the form
        serializes -- the form fills every field, so by the time a payload exists every key
        looks like it was chosen deliberately. Reading the source from the file the user
        opened is what makes the column mean anything.
        """
        try:
            payload = self._collect_config_payload()
        except ValueError:
            return  # a half-typed number; this refreshes again on the next load or run
        defaults = parametric_config.load_defaults()
        loaded = self._loaded_payload
        keys = sorted({*defaults.defaults, *defaults.required, *defaults.optional})

        self.resolved_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            value = parametric_config.resolve_optional(payload, key, None, defaults=defaults)
            text = "" if value is None else str(value)
            source = self._source_for(key, value, loaded, defaults)
            key_item = QTableWidgetItem(key)
            unit = defaults.units.get(key)
            if unit:
                key_item.setToolTip(f"unit: {unit}")
            self.resolved_table.setItem(row, 0, key_item)
            self.resolved_table.setItem(row, 1, QTableWidgetItem(text))
            self.resolved_table.setItem(row, 2, QTableWidgetItem(source))

    def _source_for(self, key: str, value: Any, loaded: Dict[str, Any], defaults: Any) -> str:
        """Where the value in front of the user came from."""
        if value is None or value == "" or value == []:
            # Nothing to attribute. Which of the two this is decides whether the run stops.
            return "REQUIRED - must set" if key in defaults.required else "unset (optional)"
        if parametric_config.was_supplied(loaded, key):
            supplied = parametric_config.resolve_optional(loaded, key, None, defaults=defaults)
            # A path key is re-anchored on the way into the form, so the literal text in the
            # config and the absolute path on screen differ without anyone having edited it.
            # Compare what they both mean, not how they are written.
            if key in parametric_config.PATH_VALUED_KEYS:
                supplied = _resolve_path(str(supplied), self._base_dir())
            return "run config" if _same(supplied, value) else "edited here"
        if key in defaults.defaults:
            return "defaults file" if _same(defaults.defaults[key], value) else "edited here"
        return "edited here"

    def _prepare_run_config_path(self, payload: Dict[str, Any]) -> Path:
        # payload["output_dir"] is already absolute (resolved in _collect_config_payload).
        output_dir = Path(payload["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "parametric_gui_last_run_config.json"
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._last_run_config_path = config_path
        return config_path

    # -- running ---------------------------------------------------------------- #

    def _start_run(self) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return
        try:
            payload = self._collect_config_payload()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid config", str(exc))
            return

        config_path = self._prepare_run_config_path(payload)
        self.log_view.clear()
        self.figure_list.clear()
        self.figure_preview.setText("No figure selected")
        self._event_paths.clear()
        self._stdout_buffer = ""
        self._log_reporter = self._new_log_reporter()
        self.progress.setValue(0)
        self.stage_label.setText("Status: starting")

        self._start_process(CLI_ENTRYPOINT, config_path)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # Follow the run to the log, but only from Inputs: if the user has deliberately
        # opened another tab, leave them there.
        if self.tabs.currentIndex() == TAB_INPUTS:
            self.tabs.setCurrentIndex(TAB_LOG)

    def _on_process_finished(self, exit_code: int, _exit_status: Any) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if exit_code == 0:
            self.stage_label.setText("Status: done")
            self.progress.setValue(100)
        else:
            self.stage_label.setText(f"Status: failed (exit={exit_code})")
        self._append_log_line(f"Process finished with exit code {exit_code}")
        if exit_code != 0:
            return
        self._populate_results_viewer()
        # Move on to the figures once there is something to look at. A failed run leaves the
        # user on the log, which is where the reason is.
        if self.tabs.currentIndex() in (TAB_INPUTS, TAB_LOG):
            self.tabs.setCurrentIndex(TAB_FIGURES)
            if self.figure_list.currentItem() is None and self.figure_list.count() > 0:
                self.figure_list.setCurrentRow(0)

    def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "run_start":
            self.stage_label.setText("Status: reading inputs")
            self.progress.setValue(10)
        elif event_type == "inputs_resolved":
            self.stage_label.setText("Status: fitting")
            self.progress.setValue(35)
        elif event_type == "artifact_written":
            path = str(event.get("path", ""))
            if path.lower().endswith(".png"):
                self._add_figure(path)
            current = self.progress.value()
            self.progress.setValue(min(95, max(current, current + 10)))
        elif event_type == "run_error":
            self.stage_label.setText("Status: error")
        elif event_type == "run_done":
            self.stage_label.setText("Status: done")
            self.progress.setValue(100)

    def _populate_results_viewer(self) -> None:
        """List the run's maps alongside the images that produced them.

        Inputs come from the config the run was launched with, not from the form, so an edit
        made mid-run cannot misdescribe what was processed. Failures here are non-fatal -- a
        run that produced numbers is still a successful run even if one file will not open.
        """
        try:
            output_dir = Path(_resolve_path(self.output_dir_edit.text(), self._base_dir()))
            volumes = discover_result_volumes(output_dir)
            known = {v.path for v in volumes}
            for label, path in self._last_run_input_volumes():
                if path not in known:
                    volumes.append(Volume(label=label, path=path))
                    known.add(path)
            self.results_viewer.set_volumes(volumes)
        except Exception as exc:  # pragma: no cover - defensive, viewer is not load-bearing
            self._append_log_line(f"Could not populate the Results tab: {exc}")

    def _last_run_input_volumes(self) -> List[tuple]:
        """(label, path) for the image inputs of the most recent run, from its saved config."""
        if self._last_run_config_path is None or not self._last_run_config_path.exists():
            return []
        payload = json.loads(self._last_run_config_path.read_text(encoding="utf-8"))
        # Anchored to the run config the GUI wrote, not the one it was loaded from -- that
        # file is what these paths were serialized next to.
        base_dir = self._last_run_config_path.parent
        found: List[tuple] = []
        for key, suffix in (("vfa_files", "VFA"), ("mask_file", "mask"), ("b1_map_file", "B1")):
            value = payload.get(key)
            entries = value if isinstance(value, list) else ([value] if value else [])
            for text in entries:
                path = Path(_resolve_path(str(text), base_dir))
                if path.is_file():
                    found.append((f"{path.name}  ({suffix})", path.resolve()))
        return found

    def _open_output_dir(self) -> None:
        output_path = Path(self._collect_config_payload()["output_dir"])
        output_path.mkdir(parents=True, exist_ok=True)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path))):
            QMessageBox.information(self, "Output Directory", f"Open this folder:\n{output_path}")


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    app = QApplication(sys.argv)
    win = ParametricGuiWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
