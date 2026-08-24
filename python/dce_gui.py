"""PySide6 GUI for running the Python DCE CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTabWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


import dce_config
from cli_overrides import coerce_override_value
from dce_file_discovery import (
    DISCOVERABLE_FILE_LISTS,
    discover_dce_input_paths,
    missing_required_inputs,
    session_from_paths,
)
from dce_volume_viewer import Volume, VolumeViewer, discover_result_volumes
import gui_common
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
from run_reporting import Reporter
from version import __version__


REPO_ROOT = Path(__file__).resolve().parents[1]

# Main-window tab order. Named because the run follows the user through them: starting a run
# moves off Inputs to the log, and finishing moves on to the figures.
TAB_INPUTS = 0
TAB_LOG = 1
TAB_FIGURES = 2
TAB_RESULTS = 3


# Widest text each run-bar label has to hold, used to reserve width up front. Without this
# the labels resize as a run progresses and the progress bar's left edge jumps with them.
# Stage text comes from `_handle_event` / `_on_process_finished`; the model line is the
# longest model name in `_build_model_flags` with the largest plausible counter.
WIDEST_STAGE_TEXT = "Stage: failed (exit=255)"
WIDEST_MODEL_TEXT = "Model: tissue_uptake (9/9)"

# Baseline-end detectors, as (label, canonical stage_overrides value). The values are the
# ones `_normalize_steady_state_auto_method` accepts; `none` means "use steady_state_end".
BASELINE_END_METHODS = (
    ("TV denoising", "tv"),
    ("GLR edge", "glr"),
    ("Piecewise constant", "piecewise_constant"),
    ("Biexponential fit", "biexp_fit"),
    ("Legacy Sobel", "legacy_sobel"),
    ("None (use manual baseline end)", "none"),
)

# stage_overrides keys that have a dedicated Core Settings widget. They are kept out of the
# Advanced table so one key never has two controls that can disagree. The AIF mode is not
# listed: it is no longer a stage override at all, just the top-level `aif_mode` field.
PROMOTED_OVERRIDE_KEYS = {"steady_state_auto_method"}

# Per-item data on the value column, used to decide whether a row is an override.
DEFAULT_TEXT_ROLE = Qt.UserRole
HAS_DEFAULT_ROLE = Qt.UserRole + 1
REQUIRED_ROLE = Qt.UserRole + 2

# What a run config that says nothing about the detector actually gets. Run configs are
# minimal by design, so an absent key means "the defaults file decides" -- not "off".
DEFAULT_BASELINE_METHOD = str(dce_config.load_defaults().default_for("steady_state_auto_method"))
DEFAULT_AIF_MODE = "fitted"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "dce_run_example_bids.json"
CLI_ENTRYPOINT = REPO_ROOT / "run_dce_python_cli.py"
OPTIONS_DOC_PATH = REPO_ROOT / "docs" / "dce_options.md"


def _value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _text_to_value(text: str) -> Any:
    """Read an override-table cell exactly as `--set KEY=VALUE` reads its value."""
    return coerce_override_value(text)


class DceGuiWindow(GuiCommonMixin, QMainWindow):
    """Main window for configuring and running DCE CLI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"ROCKETSHIP DCE v{__version__} (Python GUI)")
        self.resize(1000, 700)

        self._stdout_buffer = ""
        self._event_paths: set[str] = set()
        self._log_reporter: Optional[Reporter] = None
        self._process: Optional[QProcess] = None
        self._config_path = DEFAULT_CONFIG_PATH
        self._last_run_config_path: Optional[Path] = None
        self._file_list_edits: List[QPlainTextEdit] = []
        self._file_list_browse_buttons: List[QPushButton] = []
        self._updating_overrides = False

        self._build_ui()
        # The checkbox is built pre-connection, so apply its starting state by hand.
        self._apply_auto_find_lock(self.auto_find_check.isChecked())
        self._load_config(DEFAULT_CONFIG_PATH)

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

        # Run/Stop and progress sit below the tabs rather than inside Inputs: the tab you
        # watch during a run is CLI Output, and a Stop button you have to change tabs to
        # reach is the wrong place for it.
        self._build_run_controls(root_layout)

        self.load_button.clicked.connect(self._on_load_config_clicked)
        self.save_button.clicked.connect(self._on_save_config_clicked)
        self.reset_button.clicked.connect(lambda: self._load_config(DEFAULT_CONFIG_PATH))
        self.options_button.clicked.connect(self._on_open_options_doc)
        self.run_button.clicked.connect(self._start_run)
        self.stop_button.clicked.connect(self._stop_run_hard)
        self.figure_list.currentItemChanged.connect(self._on_figure_selected)
        self.override_add_button.clicked.connect(self._add_override_row)
        self.override_remove_button.clicked.connect(self._remove_override_rows)
        self.override_table.itemChanged.connect(self._on_override_item_changed)
        self.aif_edit.textChanged.connect(self._refresh_baseline_override_notice)
        self.auto_find_check.toggled.connect(self._on_auto_find_toggled)
        self.subject_source_edit.textChanged.connect(self._schedule_auto_fill)
        self.subject_tp_edit.textChanged.connect(self._schedule_auto_fill)

    def _build_core_settings(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Core Settings")
        form = QFormLayout(group)

        self.subject_source_edit = QLineEdit()
        self.subject_tp_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.checkpoint_dir_edit = QLineEdit()
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["auto", "cpu", "gpufit"])
        self.aif_mode_combo = QComboBox()
        self.aif_mode_combo.addItems(["fitted", "raw", "imported"])
        self.baseline_method_combo = QComboBox()
        for label, value in BASELINE_END_METHODS:
            self.baseline_method_combo.addItem(label, value)
        self.write_xls_check = QCheckBox()

        form.addRow(
            _form_label(
                "BIDS Raw Data Folder (optional)",
                "subject_source_path",
                "This session's rawdata folder, e.g. <bids_root>/rawdata/sub-01/ses-01.\n"
                "Used as the fallback source for DCE metadata (<source>/dce/*DCE.json).\n"
                "Leave blank for data that is not in BIDS layout, and give TR, flip angle\n"
                "and temporal resolution in the settings table instead.",
            ),
            self._line_edit_with_browse(
                self.subject_source_edit,
                lambda: self._choose_directory_for(self.subject_source_edit, "Select BIDS Raw Data Folder"),
            ),
        )
        form.addRow(
            _form_label(
                "BIDS Derivatives Folder (optional)",
                "subject_tp_path",
                "This session's derivatives folder, e.g.\n"
                "<bids_root>/derivatives/<pipeline>/sub-01/ses-01.\n"
                "Recorded in the run summary; input files are given explicitly below.\n"
                "Leave blank for data that is not in BIDS layout.",
            ),
            self._line_edit_with_browse(
                self.subject_tp_edit,
                lambda: self._choose_directory_for(self.subject_tp_edit, "Select BIDS Derivatives Folder"),
            ),
        )
        form.addRow(
            _form_label("Output Folder", "output_dir"),
            self._line_edit_with_browse(
                self.output_dir_edit,
                lambda: self._choose_directory_for(self.output_dir_edit, "Select Output Folder"),
            ),
        )
        form.addRow(
            _form_label("Checkpoint Folder", "checkpoint_dir"),
            self._line_edit_with_browse(
                self.checkpoint_dir_edit,
                lambda: self._choose_directory_for(self.checkpoint_dir_edit, "Select Checkpoint Folder"),
            ),
        )
        form.addRow(_form_label("Fitting Backend", "backend"), self.backend_combo)
        form.addRow(_form_label("AIF Mode", "aif_mode"), self.aif_mode_combo)
        form.addRow(
            _form_label(
                "Auto End Baseline Algorithm",
                "stage_overrides.steady_state_auto_method",
                "How the end of the pre-contrast baseline is detected.\n"
                "Ignored when stage_overrides.steady_state_end is set explicitly.",
            ),
            self._baseline_method_row(),
        )
        form.addRow(_form_label("Write Excel Output", "write_xls"), self.write_xls_check)
        parent_layout.addWidget(group)

    def _collapsible_section(self, title: str, tooltip: str = "") -> tuple:
        """Group box whose body is hidden behind an expand/collapse caret.

        Returns (group, header_layout, body_layout); the header row stays visible when
        collapsed, so controls that belong there (e.g. the auto-find checkbox) can be
        added by the caller."""
        group = QGroupBox()
        # No title, so WINDOW_QSS should not reserve the strip a title would sit in.
        group.setProperty("titleless", True)
        layout = QVBoxLayout(group)

        toggle = QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(False)
        toggle.setArrowType(Qt.RightArrow)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        if tooltip:
            toggle.setToolTip(tooltip)

        header = QHBoxLayout()
        header.addWidget(toggle)
        layout.addLayout(header)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body.setVisible(False)
        layout.addWidget(body)

        toggle.toggled.connect(
            lambda expanded, b=body, t=toggle: self._set_section_expanded(b, t, expanded)
        )
        return group, header, body_layout

    @staticmethod
    def _set_section_expanded(body: QWidget, toggle: QToolButton, expanded: bool) -> None:
        body.setVisible(expanded)
        toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _build_file_lists(self, parent_layout: QVBoxLayout) -> None:
        group, header, body_layout = self._collapsible_section("Input File Lists")

        self.auto_find_check = QCheckBox("Auto find BIDS files")
        self.auto_find_check.setChecked(True)
        self.auto_find_check.setToolTip(
            "Fill the file lists from the BIDS Derivatives Folder using the dceprep naming\n"
            "convention, and keep them in sync when the BIDS folders change.\n"
            "The fields stay readable but locked. Uncheck to edit them by hand."
        )
        self.auto_find_status = QLabel("")
        header.addSpacing(12)
        header.addWidget(self.auto_find_check)
        header.addWidget(self.auto_find_status, 1)

        form = QFormLayout()
        body_layout.addLayout(form)

        self.dynamic_edit = QPlainTextEdit()
        self.aif_edit = QPlainTextEdit()
        self.roi_edit = QPlainTextEdit()
        self.t1map_edit = QPlainTextEdit()
        self.noise_edit = QPlainTextEdit()
        self.drift_edit = QPlainTextEdit()

        form.addRow(
            _form_label("Dynamic Files", "dynamic_files"),
            self._file_list_with_browse(
                self.dynamic_edit,
                lambda: self._choose_files_for(self.dynamic_edit, "Select Dynamic Files"),
            ),
        )
        form.addRow(
            _form_label("AIF Files", "aif_files"),
            self._file_list_with_browse(
                self.aif_edit,
                lambda: self._choose_files_for(self.aif_edit, "Select AIF Files"),
            ),
        )
        form.addRow(
            _form_label("ROI Files", "roi_files"),
            self._file_list_with_browse(
                self.roi_edit,
                lambda: self._choose_files_for(self.roi_edit, "Select ROI Files"),
            ),
        )
        form.addRow(
            _form_label("T1 Map Files", "t1map_files"),
            self._file_list_with_browse(
                self.t1map_edit,
                lambda: self._choose_files_for(self.t1map_edit, "Select T1 Map Files"),
            ),
        )
        form.addRow(
            _form_label("Noise Files", "noise_files"),
            self._file_list_with_browse(
                self.noise_edit,
                lambda: self._choose_files_for(self.noise_edit, "Select Noise Files"),
            ),
        )
        form.addRow(
            _form_label("Drift Files", "drift_files"),
            self._file_list_with_browse(
                self.drift_edit,
                lambda: self._choose_files_for(self.drift_edit, "Select Drift Files"),
            ),
        )

        # Discovery kind -> the field it fills. drift_files has no BIDS convention, so
        # auto-find leaves it empty rather than guessing.
        self._auto_fill_targets = {
            "dynamic": ("Dynamic", self.dynamic_edit),
            "aif_mask": ("AIF", self.aif_edit),
            "roi_mask": ("ROI", self.roi_edit),
            "t1_map": ("T1 Map", self.t1map_edit),
            "noise_mask": ("Noise", self.noise_edit),
        }
        self._auto_fill_timer = QTimer(self)
        self._auto_fill_timer.setSingleShot(True)
        self._auto_fill_timer.setInterval(250)
        self._auto_fill_timer.timeout.connect(self._auto_fill_input_files)

        parent_layout.addWidget(group)

    def _file_list_with_browse(self, edit: QPlainTextEdit, on_browse: Any) -> QWidget:
        """One-line path list plus an inline Browse button, matching the Core Settings rows.

        The widget stays a QPlainTextEdit because a list is still one path per line; it is
        just clamped to a single visible row, so extra paths scroll and the full list shows
        in the tooltip."""
        edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Match the one-line look of the Core Settings fields, plus the pixels the styled
        # border and padding take, so the path text is not clipped.
        edit.setFixedHeight(QLineEdit().sizeHint().height() + 4)
        edit.setPlaceholderText("One path per line")
        edit.textChanged.connect(lambda: self._refresh_file_list_tooltip(edit))
        self._refresh_file_list_tooltip(edit)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(on_browse)
        layout.addWidget(browse)

        self._file_list_edits.append(edit)
        self._file_list_browse_buttons.append(browse)
        return row

    def _refresh_file_list_tooltip(self, edit: QPlainTextEdit) -> None:
        paths = _text_to_paths(edit.toPlainText())
        if not paths:
            edit.setToolTip("One path per line")
            return
        header = f"{len(paths)} file{'s' if len(paths) != 1 else ''}:"
        edit.setToolTip(header + "\n" + "\n".join(paths))

    def _on_auto_find_toggled(self, checked: bool) -> None:
        self._apply_auto_find_lock(checked)
        if checked:
            self._auto_fill_input_files()
        else:
            self._set_auto_find_status("", missing=False)

    def _apply_auto_find_lock(self, locked: bool) -> None:
        """Lock the file lists while auto-find owns them, keeping them readable."""
        palette = QApplication.palette()
        role = QPalette.Window if locked else QPalette.Base
        background = palette.color(role).name()
        for edit in self._file_list_edits:
            edit.setReadOnly(locked)
            edit.setStyleSheet(f"QPlainTextEdit {{ background-color: {background}; }}")
        for button in self._file_list_browse_buttons:
            button.setEnabled(not locked)

    def _schedule_auto_fill(self) -> None:
        if self.auto_find_check.isChecked():
            self._auto_fill_timer.start()

    def _set_auto_find_status(self, text: str, missing: bool) -> None:
        self.auto_find_status.setText(text)
        self.auto_find_status.setStyleSheet("color: #b00020;" if missing else "")

    def _auto_fill_input_files(self) -> None:
        """Refill the file lists from the BIDS folders using the dceprep convention."""
        if not self.auto_find_check.isChecked():
            return

        derivatives_text = _resolve_path(self.subject_tp_edit.text(), self._base_dir())
        if not derivatives_text:
            self._set_auto_find_status("Set the BIDS Derivatives Folder", missing=True)
            return
        derivatives_path = Path(derivatives_text)
        if not derivatives_path.is_dir():
            self._set_auto_find_status(f"No such folder: {derivatives_path}", missing=True)
            return

        rawdata_text = _resolve_path(self.subject_source_edit.text(), self._base_dir())
        session = session_from_paths(
            derivatives_path, Path(rawdata_text) if rawdata_text else None
        )
        found = discover_dce_input_paths(session)

        for kind, (_label, edit) in self._auto_fill_targets.items():
            path = found.get(kind)
            edit.setPlainText(str(path) if path is not None else "")
        self.drift_edit.setPlainText("")

        missing_labels = [self._auto_fill_targets[kind][0] for kind in missing_required_inputs(found)]
        if missing_labels:
            self._set_auto_find_status("Not found: " + ", ".join(missing_labels), missing=True)
        else:
            optional_note = "" if found.get("noise_mask") else " (no noise mask)"
            self._set_auto_find_status(f"All required inputs found{optional_note}", missing=False)

    def _build_model_flags(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Model Flags")
        layout = QHBoxLayout(group)
        self.model_checks: Dict[str, QCheckBox] = {}
        for name in ["tofts", "ex_tofts", "patlak", "tissue_uptake", "two_cxm", "fxr", "auc", "nested", "FXL_rr"]:
            cb = QCheckBox(name)
            self.model_checks[name] = cb
            layout.addWidget(cb)
        layout.addStretch(1)
        parent_layout.addWidget(group)

    def _build_stage_overrides(self, parent_layout: QVBoxLayout) -> None:
        group, header, body_layout = self._collapsible_section(
            "Advanced (Stage Overrides)",
            "Every key from python/dce_defaults.json, showing the value this run will use.\n"
            "Only rows you change are written into the saved config.",
        )
        header.addStretch(1)
        # The options doc explains these keys, so it belongs with them rather than in the
        # config bar, where it read as another config-file action.
        self.options_button = QPushButton("Open Options Doc")
        header.addWidget(self.options_button)

        self.override_table = QTableWidget(0, 3)
        self.override_table.setHorizontalHeaderLabels(["key", "value", "source"])
        header_view = self.override_table.horizontalHeader()
        header_view.setStretchLastSection(False)
        # Keys are long (`voxel_upper_limit_ktrans_RR`); give them and `source` the width they
        # need and let the editable value column take the rest.
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.Stretch)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.override_table.setMinimumHeight(360)
        self.override_table.setSortingEnabled(True)
        body_layout.addWidget(self.override_table)

        controls = QHBoxLayout()
        self.override_add_button = QPushButton("Add Key")
        self.override_remove_button = QPushButton("Reset Selected")
        self.override_remove_button.setToolTip(
            "Restore the value from python/dce_defaults.json, so the key stops being an override."
        )
        controls.addWidget(self.override_add_button)
        controls.addWidget(self.override_remove_button)
        controls.addStretch(1)
        body_layout.addLayout(controls)

        parent_layout.addWidget(group)

    def _build_run_controls(self, parent_layout: QVBoxLayout) -> None:
        bar = build_run_bar(
            run_text="Run DCE",
            widest_stage_text=WIDEST_STAGE_TEXT,
            widest_detail_text=WIDEST_MODEL_TEXT,
            detail_idle_text="Model: -",
        )
        self.run_button = bar.run_button
        self.stop_button = bar.stop_button
        self.stage_label = bar.stage_label
        self.model_label = bar.detail_label
        self.progress = bar.progress
        parent_layout.addWidget(bar.group)

    def _build_inputs_tab(self) -> QWidget:
        """Config file selection plus every settings section, in one scrolling column."""
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
        self._build_model_flags(layout)
        self._build_file_lists(layout)
        self._build_stage_overrides(layout)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        # Always on, not as-needed: expanding a collapsible section is what pushes this tab
        # past the viewport, and a scrollbar that appears at that moment shifts the layout
        # underneath the click. Platforms drawing overlay scrollbars (macOS, depending on the
        # system setting) still fade theirs out; this is the policy, not a guarantee.
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        return scroll

    def _build_results_tab(self) -> QWidget:
        """Slice viewer for the fitted maps and the dynamic they came from."""
        self.results_viewer = VolumeViewer()
        return self.results_viewer

    def _populate_results_viewer(self) -> None:
        """List the run's maps alongside the inputs that produced them.

        Inputs come from the config the run was launched with, not from the form: auto-find
        is free to rewrite those fields while the run is in flight, so the widgets are not a
        reliable record of what was actually processed. Failures here are non-fatal -- a run
        that produced numbers is still a successful run even if one file will not open.
        """
        try:
            volumes = discover_result_volumes(
                Path(_resolve_path(self.output_dir_edit.text(), self._base_dir()))
            )
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
        payload = json.loads(self._last_run_config_path.read_text())
        # Anchored to the run config the GUI wrote, not the config it was loaded from --
        # that file is what these paths were serialized next to.
        base_dir = self._last_run_config_path.parent
        found: List[tuple] = []
        for key, suffix in (
            ("dynamic_files", "dynamic"),
            ("t1map_files", "T1 map"),
            ("aif_files", "AIF mask"),
            ("roi_files", "ROI mask"),
        ):
            for text in payload.get(key, []) or []:
                path = Path(_resolve_path(str(text), base_dir))
                if path.is_file():
                    found.append((f"{path.name}  ({suffix})", path.resolve()))
        return found

    def _build_log_tab(self) -> QWidget:
        # No group box around this: the tab already frames and titles it, so a second
        # border with the same caption inside it is just noise.
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.log_view = build_log_view()
        self._reset_log_view()
        layout.addWidget(self.log_view)
        return panel

    def _reset_log_view(self) -> None:
        """Idle text, so an untouched tab is not an unexplained black rectangle."""
        self.log_view.setPlainText(
            "[idle] No run yet.\n"
            "Pipeline output appears here once you press Run DCE."
        )

    def _build_figures_tab(self) -> QWidget:
        panel = build_figures_panel()
        self.figure_list = panel.list_widget
        self.figure_preview = panel.preview
        return panel.widget

    def _on_open_options_doc(self) -> None:
        if not OPTIONS_DOC_PATH.exists():
            QMessageBox.warning(self, "Missing docs", f"Options doc not found: {OPTIONS_DOC_PATH}")
            return
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(OPTIONS_DOC_PATH)))
        if not opened:
            QMessageBox.information(self, "Options Doc", f"Open this file:\n{OPTIONS_DOC_PATH}")

    def _set_overrides_from_dict(self, stage_overrides: Dict[str, Any]) -> None:
        """Show every known key with the value this run will actually use.

        Rows come from `dce_defaults.json`, not from the run config, so the table stays a
        complete picture of the knobs even though a run config only carries its overrides.
        The `source` column says where each value came from."""
        defaults = dce_config.load_defaults()
        default_lc = {k.lower(): k for k in defaults.defaults}
        known = set(defaults.defaults) | set(defaults.required) | set(defaults.optional)
        known_lc = {k.lower() for k in known}
        config_lc = {str(k).strip().lower(): v for k, v in stage_overrides.items()}
        extra = sorted(k for k in config_lc if k not in known_lc)
        rows = sorted(known, key=str.lower) + extra

        self._updating_overrides = True
        self.override_table.setSortingEnabled(False)
        self.override_table.setRowCount(0)
        for key in rows:
            lc = key.lower()
            if lc in PROMOTED_OVERRIDE_KEYS:
                continue
            has_default = lc in default_lc
            default_text = _value_to_text(defaults.default_for(key)) if has_default else ""
            value_text = _value_to_text(config_lc[lc]) if lc in config_lc else default_text

            row = self.override_table.rowCount()
            self.override_table.insertRow(row)
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            self.override_table.setItem(row, 0, key_item)
            value_item = QTableWidgetItem(value_text)
            value_item.setData(DEFAULT_TEXT_ROLE, default_text)
            value_item.setData(HAS_DEFAULT_ROLE, has_default)
            value_item.setData(REQUIRED_ROLE, lc in {k.lower() for k in defaults.required})
            self.override_table.setItem(row, 1, value_item)
            source_item = QTableWidgetItem("")
            source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
            self.override_table.setItem(row, 2, source_item)
            self._refresh_override_source(row)
        self.override_table.setSortingEnabled(True)
        self._updating_overrides = False

    @staticmethod
    def _override_is_set(value_item: QTableWidgetItem) -> bool:
        """Whether this row would be written into the saved config."""
        value_text = value_item.text().strip()
        if bool(value_item.data(HAS_DEFAULT_ROLE)):
            return value_text != str(value_item.data(DEFAULT_TEXT_ROLE) or "").strip()
        return value_text != ""

    def _refresh_override_source(self, row: int) -> None:
        value_item = self.override_table.item(row, 1)
        source_item = self.override_table.item(row, 2)
        if value_item is None or source_item is None:
            return
        if self._override_is_set(value_item):
            source_item.setText("override")
        elif bool(value_item.data(HAS_DEFAULT_ROLE)):
            source_item.setText("dce_defaults.json")
        elif bool(value_item.data(REQUIRED_ROLE)):
            source_item.setText("REQUIRED - must set")
        else:
            source_item.setText("unset (optional)")

    def _on_override_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_overrides or item.column() != 1:
            return
        self._updating_overrides = True
        self._refresh_override_source(item.row())
        self._updating_overrides = False

    def _baseline_method_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.baseline_method_combo, 1)
        self.baseline_override_label = QLabel("")
        layout.addWidget(self.baseline_override_label)
        return row

    def _refresh_baseline_override_notice(self) -> None:
        """Warn when the AIF sidecar will pin the baseline end and the detector won't run.

        `_resolve_baseline_window` gives `SteadyStateEndTimeIndex` in the AIF file's JSON
        sidecar priority over `steady_state_auto_method`, so without this the dropdown would
        claim a detector that the run silently ignores."""
        notice = ""
        aif_paths = _text_to_paths(self.aif_edit.toPlainText())
        if aif_paths:
            sidecar = Path(_resolve_path(aif_paths[0], self._base_dir()))
            for suffix in (".nii.gz", ".nii"):
                if sidecar.name.endswith(suffix):
                    sidecar = sidecar.with_name(sidecar.name[: -len(suffix)] + ".json")
                    break
            if sidecar.exists():
                try:
                    payload = json.loads(sidecar.read_text())
                except Exception:
                    payload = {}
                if "SteadyStateEndTimeIndex" in payload:
                    notice = f"overridden: AIF sidecar pins baseline end = {payload['SteadyStateEndTimeIndex']}"
        self.baseline_override_label.setText(notice)
        self.baseline_override_label.setStyleSheet("color: #b00020;" if notice else "")
        self.baseline_override_label.setToolTip(
            f"{sidecar}\nDelete SteadyStateEndTimeIndex from that sidecar to let the detector run."
            if notice
            else ""
        )

    def _set_aif_mode(self, payload: Dict[str, Any]) -> None:
        """Show the mode the run will use, tolerating the numeric MATLAB spellings."""
        text = str(payload.get("aif_mode", DEFAULT_AIF_MODE)).strip().lower()
        text = {"1": "fitted", "fit": "fitted", "2": "raw", "3": "imported", "import": "imported"}.get(text, text)
        if self.aif_mode_combo.findText(text) >= 0:
            self.aif_mode_combo.setCurrentText(text)

    def _set_baseline_method(self, value: Any) -> None:
        """Select the configured detector, tolerating the aliases the pipeline accepts."""
        text = "" if value is None else str(value).strip().lower()
        index = self.baseline_method_combo.findData(text)
        if index < 0 and text in {"", "off", "disabled", "manual", "false", "0"}:
            index = self.baseline_method_combo.findData("none")
        if index < 0:
            # An alias the pipeline understands but this list doesn't spell the same way
            # (e.g. "find_end_ss_tv"); keep the raw value rather than silently retargeting.
            self.baseline_method_combo.addItem(str(value), text)
            index = self.baseline_method_combo.count() - 1
        self.baseline_method_combo.setCurrentIndex(index)

    def _stage_overrides_to_dict(self) -> Dict[str, Any]:
        """Only the keys this run actually overrides.

        A row left at its `dce_defaults.json` value is omitted: restating a default in the
        run config forks it, which is what the defaults file exists to prevent."""
        out: Dict[str, Any] = {}
        for row in range(self.override_table.rowCount()):
            key_item = self.override_table.item(row, 0)
            value_item = self.override_table.item(row, 1)
            if key_item is None or value_item is None:
                continue
            key = key_item.text().strip()
            if key == "" or key.lower() in PROMOTED_OVERRIDE_KEYS:
                continue
            if not self._override_is_set(value_item):
                continue
            out[key] = _text_to_value(value_item.text())

        method = self.baseline_method_combo.currentData()
        if str(method) != DEFAULT_BASELINE_METHOD:
            out["steady_state_auto_method"] = method
        return out

    def _add_override_row(self) -> None:
        """Add a free-form key. Everything in `dce_defaults.json` is already listed, so this
        is only for keys the defaults file does not know about."""
        self._updating_overrides = True
        self.override_table.setSortingEnabled(False)
        row = self.override_table.rowCount()
        self.override_table.insertRow(row)
        self.override_table.setItem(row, 0, QTableWidgetItem("new_key"))
        value_item = QTableWidgetItem("")
        value_item.setData(DEFAULT_TEXT_ROLE, "")
        value_item.setData(HAS_DEFAULT_ROLE, False)
        value_item.setData(REQUIRED_ROLE, False)
        self.override_table.setItem(row, 1, value_item)
        source_item = QTableWidgetItem("")
        source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
        self.override_table.setItem(row, 2, source_item)
        self._refresh_override_source(row)
        self.override_table.setSortingEnabled(True)
        self._updating_overrides = False

    def _remove_override_rows(self) -> None:
        """Reset selected rows to the defaults-file value; drop rows that have no default."""
        self._updating_overrides = True
        rows = sorted({index.row() for index in self.override_table.selectedIndexes()}, reverse=True)
        for row in rows:
            value_item = self.override_table.item(row, 1)
            if value_item is None:
                continue
            key_item = self.override_table.item(row, 0)
            known = bool(value_item.data(HAS_DEFAULT_ROLE)) or bool(value_item.data(REQUIRED_ROLE))
            if not known and key_item is not None and not dce_config.load_defaults().knows(key_item.text().strip()):
                self.override_table.removeRow(row)
                continue
            value_item.setText(str(value_item.data(DEFAULT_TEXT_ROLE) or ""))
            self._refresh_override_source(row)
        self._updating_overrides = False

    def _load_config(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(self, "Missing config", f"Config file not found: {path}")
            return
        payload = json.loads(path.read_text())
        self._config_path = path
        self.config_path_edit.setText(str(path))

        self.subject_source_edit.setText(str(payload.get("subject_source_path", "")))
        self.subject_tp_edit.setText(str(payload.get("subject_tp_path", "")))
        self.output_dir_edit.setText(str(payload.get("output_dir", "")))
        self.checkpoint_dir_edit.setText(str(payload.get("checkpoint_dir") or ""))
        self.backend_combo.setCurrentText(str(payload.get("backend", "auto")))
        self._set_aif_mode(payload)
        self.write_xls_check.setChecked(bool(payload.get("write_xls", True)))

        self.dynamic_edit.setPlainText(_paths_to_text(list(payload.get("dynamic_files", []))))
        self.aif_edit.setPlainText(_paths_to_text(list(payload.get("aif_files", []))))
        self.roi_edit.setPlainText(_paths_to_text(list(payload.get("roi_files", []))))
        self.t1map_edit.setPlainText(_paths_to_text(list(payload.get("t1map_files", []))))
        self.noise_edit.setPlainText(_paths_to_text(list(payload.get("noise_files", []))))
        self.drift_edit.setPlainText(_paths_to_text(list(payload.get("drift_files", []))))

        model_flags = dict(payload.get("model_flags", {}))
        for name, cb in self.model_checks.items():
            cb.setChecked(int(model_flags.get(name, 0)) == 1)

        stage_overrides = dict(payload.get("stage_overrides", {}))
        configured_method = next(
            (v for k, v in stage_overrides.items() if k.strip().lower() == "steady_state_auto_method"),
            DEFAULT_BASELINE_METHOD,
        )
        self._set_baseline_method(configured_method)
        self._set_overrides_from_dict(stage_overrides)

        # Let the config decide whether auto-find owns the file lists. A config that names
        # its files is being explicit and must be honoured; one that names only the session
        # folders is asking for the convention, which is how the shipped BIDS example is
        # written. Non-BIDS configs land on "off", so their fields stay editable instead of
        # being locked behind a BIDS warning they can do nothing about.
        self.auto_find_check.setChecked(self._config_wants_auto_find(payload))

        # The BIDS folders may be unchanged by this load, so re-derive explicitly rather
        # than relying on their textChanged signal to schedule it.
        self._schedule_auto_fill()

    @staticmethod
    def _config_wants_auto_find(payload: Dict[str, Any]) -> bool:
        """True when the config names session folders but leaves the file lists to the
        convention -- the only case where discovering files is what the author asked for."""
        if not str(payload.get("subject_tp_path") or "").strip():
            return False
        return not any(
            payload.get(key) for key, _ in DISCOVERABLE_FILE_LISTS
        )

    def _collect_config_payload(self) -> Dict[str, Any]:
        model_flags = {name: (1 if cb.isChecked() else 0) for name, cb in self.model_checks.items()}
        base_dir = self._base_dir()
        output_dir_text = self.output_dir_edit.text().strip()
        if output_dir_text in {"", "."}:
            output_dir = str(REPO_ROOT / "out" / "dce_gui")
        else:
            output_dir = _resolve_path(output_dir_text, base_dir)
        payload = {
            "subject_source_path": _resolve_path(self.subject_source_edit.text(), base_dir),
            "subject_tp_path": _resolve_path(self.subject_tp_edit.text(), base_dir),
            "output_dir": output_dir,
            "checkpoint_dir": _resolve_path(self.checkpoint_dir_edit.text(), base_dir),
            "backend": self.backend_combo.currentText(),
            "write_xls": self.write_xls_check.isChecked(),
            "aif_mode": self.aif_mode_combo.currentText(),
            "dynamic_files": _resolve_paths(_text_to_paths(self.dynamic_edit.toPlainText()), base_dir),
            "aif_files": _resolve_paths(_text_to_paths(self.aif_edit.toPlainText()), base_dir),
            "roi_files": _resolve_paths(_text_to_paths(self.roi_edit.toPlainText()), base_dir),
            "t1map_files": _resolve_paths(_text_to_paths(self.t1map_edit.toPlainText()), base_dir),
            "noise_files": _resolve_paths(_text_to_paths(self.noise_edit.toPlainText()), base_dir),
            "drift_files": _resolve_paths(_text_to_paths(self.drift_edit.toPlainText()), base_dir),
            "model_flags": model_flags,
            "stage_overrides": dce_config.resolve_override_paths(
                self._stage_overrides_to_dict(), base_dir
            ),
        }
        return payload

    def _prepare_run_config_path(self, payload: Dict[str, Any]) -> Path:
        # payload["output_dir"] is already an absolute path (resolved in _collect_config_payload).
        output_dir = Path(payload["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "dce_gui_last_run_config.json"
        config_path.write_text(json.dumps(payload, indent=2) + "\n")
        self._last_run_config_path = config_path
        return config_path

    def _start_run(self) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return

        payload = self._collect_config_payload()
        config_path = self._prepare_run_config_path(payload)
        self.log_view.clear()
        self.figure_list.clear()
        self.figure_preview.setText("No figure selected")
        self._event_paths.clear()
        self._stdout_buffer = ""
        self._log_reporter = self._new_log_reporter()
        self.progress.setValue(0)
        self.stage_label.setText("Stage: starting")
        self.model_label.setText("Model: -")

        self._start_process(CLI_ENTRYPOINT, config_path)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # Follow the run to the log, but only from Inputs: if the user has deliberately
        # opened another tab, leave them there.
        if self.tabs.currentIndex() == TAB_INPUTS:
            self.tabs.setCurrentIndex(TAB_LOG)

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if exit_code == 0:
            self.stage_label.setText("Stage: done")
            self.progress.setValue(100)
        else:
            self.stage_label.setText(f"Stage: failed (exit={exit_code})")
        self._append_log_line(f"Process finished with exit code {exit_code}")
        if exit_code == 0:
            self._populate_results_viewer()
        # Move on to the figures once there is something to look at. A failed run leaves the
        # user on the log, which is where the reason is.
        if exit_code == 0 and self.tabs.currentIndex() in (TAB_INPUTS, TAB_LOG):
            self.tabs.setCurrentIndex(TAB_FIGURES)
            # Switching to an empty preview defeats the point of switching, so show the
            # first figure unless the user already picked one mid-run.
            if self.figure_list.currentItem() is None and self.figure_list.count() > 0:
                self.figure_list.setCurrentRow(0)

    def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "stage_start":
            stage = str(event.get("stage", "?"))
            self.stage_label.setText(f"Stage: {stage} (running)")
            if stage == "A":
                self.progress.setValue(5)
            elif stage == "B":
                self.progress.setValue(35)
            elif stage == "D":
                self.progress.setValue(65)
            return

        if event_type == "stage_done":
            stage = str(event.get("stage", "?"))
            self.stage_label.setText(f"Stage: {stage} ({event.get('status', '')})")
            if stage == "A":
                self.progress.setValue(33)
            elif stage == "B":
                self.progress.setValue(66)
            elif stage == "D":
                self.progress.setValue(95)
            return

        if event_type in {"model_start", "model_done"}:
            model = str(event.get("model", "?"))
            model_idx = int(event.get("model_index", 0) or 0)
            model_total = int(event.get("model_total", 0) or 0)
            self.model_label.setText(f"Model: {model} ({model_idx}/{model_total})")
            if model_total > 0:
                done = model_idx if event_type == "model_done" else max(model_idx - 1, 0)
                frac = float(done) / float(model_total)
                self.progress.setValue(65 + int(30 * frac))
            return

        if event_type == "artifact_written":
            path = str(event.get("path", ""))
            if path.lower().endswith(".png"):
                self._add_figure(path)
            return

        if event_type == "run_error":
            self.stage_label.setText(f"Error in stage {event.get('stage', '?')}")
            return

        if event_type == "run_done":
            self.progress.setValue(100)
            return

def main(argv: Optional[List[str]] = None) -> int:
    del argv
    app = QApplication(sys.argv)
    win = DceGuiWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
