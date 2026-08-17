"""PySide6 GUI for running the Python DCE CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFontDatabase, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


import dce_config
from bids_discovery import BidsSession
from dce_file_discovery import discover_dce_input_paths, missing_required_inputs


EVENT_PREFIX = "ROCKETSHIP_EVENT "
REPO_ROOT = Path(__file__).resolve().parents[1]

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
# Advanced table so one key never has two controls that can disagree. `aif_curve_mode` and
# its MATLAB alias `aif_type` are here because `_resolve_stage_b_aif_mode` treats them and
# the top-level `aif_mode` as three spellings of one setting, with these two winning -- so a
# table row would silently override the AIF Mode combo.
PROMOTED_OVERRIDE_KEYS = {"steady_state_auto_method", "aif_curve_mode"}

# Per-item data on the value column, used to decide whether a row is an override.
DEFAULT_TEXT_ROLE = Qt.UserRole
HAS_DEFAULT_ROLE = Qt.UserRole + 1
REQUIRED_ROLE = Qt.UserRole + 2

# What a run config that says nothing about the detector actually gets. Run configs are
# minimal by design, so an absent key means "the defaults file decides" -- not "off".
DEFAULT_BASELINE_METHOD = str(dce_config.load_defaults().default_for("steady_state_auto_method"))
DEFAULT_AIF_MODE = "fitted"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "dce_run_example.json"
CLI_ENTRYPOINT = REPO_ROOT / "run_dce_python_cli.py"
OPTIONS_DOC_PATH = REPO_ROOT / "docs" / "dce_options.md"


def _value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _text_to_value(text: str) -> Any:
    raw = text.strip()
    if raw == "":
        return ""
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _paths_to_text(values: List[str]) -> str:
    return "\n".join(values)


def _text_to_paths(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _resolve_repo_path(text: str) -> str:
    """Resolve a possibly-relative path against REPO_ROOT (matches the CLI's cwd,
    which the GUI always sets to REPO_ROOT for the subprocess it launches)."""
    stripped = text.strip()
    if not stripped:
        return ""
    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return str(candidate)


def _resolve_repo_paths(values: List[str]) -> List[str]:
    return [_resolve_repo_path(v) for v in values if str(v).strip()]


def _form_label(text: str, config_key: str, hint: str = "") -> QLabel:
    """Friendly field label that still exposes the underlying config key on hover."""
    label = QLabel(text)
    tooltip = f"config key: {config_key}"
    if hint:
        tooltip = f"{hint}\n\n{tooltip}"
    label.setToolTip(tooltip)
    return label


class DceGuiWindow(QMainWindow):
    """Main window for configuring and running DCE CLI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROCKETSHIP DCE (Python GUI v1)")
        self.resize(1500, 900)

        self._stdout_buffer = ""
        self._event_paths: set[str] = set()
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
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        top_controls = QHBoxLayout()
        self.config_path_edit = QLineEdit(str(DEFAULT_CONFIG_PATH))
        self.config_path_edit.setReadOnly(True)
        self.load_button = QPushButton("Load Config")
        self.save_button = QPushButton("Save Config As")
        self.reset_button = QPushButton("Reset Defaults")
        self.options_button = QPushButton("Open Options Doc")
        top_controls.addWidget(QLabel("Config:"))
        top_controls.addWidget(self.config_path_edit, 1)
        top_controls.addWidget(self.load_button)
        top_controls.addWidget(self.save_button)
        top_controls.addWidget(self.reset_button)
        top_controls.addWidget(self.options_button)
        root_layout.addLayout(top_controls)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 600])

        self._build_core_settings(left_layout)
        self._build_model_flags(left_layout)
        self._build_file_lists(left_layout)
        self._build_stage_overrides(left_layout)
        self._build_run_controls(left_layout)
        left_layout.addStretch(1)

        self._build_logs_and_figures(right_layout)

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
                "BIDS Raw Data Folder",
                "subject_source_path",
                "This session's rawdata folder, e.g. <bids_root>/rawdata/sub-01/ses-01.\n"
                "Used as the fallback source for DCE metadata (<source>/dce/*DCE.json).",
            ),
            self._line_edit_with_browse(
                self.subject_source_edit,
                lambda: self._choose_directory_for(self.subject_source_edit, "Select BIDS Raw Data Folder"),
            ),
        )
        form.addRow(
            _form_label(
                "BIDS Derivatives Folder",
                "subject_tp_path",
                "This session's derivatives folder, e.g.\n"
                "<bids_root>/derivatives/<pipeline>/sub-01/ses-01.\n"
                "Recorded in the run summary; input files are given explicitly below.",
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

    def _line_edit_with_browse(self, edit: QLineEdit, on_browse: Any) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(on_browse)
        layout.addWidget(browse)
        return row

    def _file_list_with_browse(self, edit: QPlainTextEdit, on_browse: Any) -> QWidget:
        """One-line path list plus an inline Browse button, matching the Core Settings rows.

        The widget stays a QPlainTextEdit because a list is still one path per line; it is
        just clamped to a single visible row, so extra paths scroll and the full list shows
        in the tooltip."""
        edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        edit.setFixedHeight(QLineEdit().sizeHint().height())
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

    def _session_for_discovery(self, derivatives_path: Path, rawdata_path: Path) -> BidsSession:
        is_session_dir = derivatives_path.name.startswith("ses-")
        return BidsSession(
            bids_root=derivatives_path,
            subject=derivatives_path.parent.name if is_session_dir else derivatives_path.name,
            session=derivatives_path.name if is_session_dir else None,
            rawdata_path=rawdata_path,
            derivatives_path=derivatives_path,
        )

    def _auto_fill_input_files(self) -> None:
        """Refill the file lists from the BIDS folders using the dceprep convention."""
        if not self.auto_find_check.isChecked():
            return

        derivatives_text = _resolve_repo_path(self.subject_tp_edit.text())
        if not derivatives_text:
            self._set_auto_find_status("Set the BIDS Derivatives Folder", missing=True)
            return
        derivatives_path = Path(derivatives_text)
        if not derivatives_path.is_dir():
            self._set_auto_find_status(f"No such folder: {derivatives_path}", missing=True)
            return

        rawdata_text = _resolve_repo_path(self.subject_source_edit.text())
        session = self._session_for_discovery(
            derivatives_path, Path(rawdata_text) if rawdata_text else derivatives_path
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

    def _dialog_start_dir(self, current_text: str) -> str:
        text = current_text.strip()
        if text:
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = (REPO_ROOT / candidate).resolve()
            if candidate.is_file():
                return str(candidate.parent)
            if candidate.exists():
                return str(candidate)
        return str(REPO_ROOT)

    def _choose_directory_for(self, edit: QLineEdit, title: str) -> None:
        start_dir = self._dialog_start_dir(edit.text())
        chosen = QFileDialog.getExistingDirectory(self, title, start_dir)
        if chosen:
            edit.setText(chosen)

    def _choose_files_for(self, edit: QPlainTextEdit, title: str) -> None:
        existing_paths = _text_to_paths(edit.toPlainText())
        start_seed = existing_paths[0] if existing_paths else ""
        start_dir = self._dialog_start_dir(start_seed)
        selected, _ = QFileDialog.getOpenFileNames(self, title, start_dir, "All Files (*)")
        if selected:
            edit.setPlainText(_paths_to_text(selected))

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
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run DCE")
        self.stop_button = QPushButton("Hard Stop")
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.stage_label = QLabel("Stage: idle")
        self.model_label = QLabel("Model: -")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.model_label)
        layout.addWidget(self.progress)
        parent_layout.addWidget(group)

    def _build_logs_and_figures(self, right_layout: QVBoxLayout) -> None:
        logs_group = QGroupBox("CLI Output / Progress")
        logs_layout = QVBoxLayout(logs_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log_view.setStyleSheet(
            "QPlainTextEdit {"
            " background-color: #000000;"
            " color: #ffffff;"
            " selection-background-color: #ffffff;"
            " selection-color: #000000;"
            " border: 1px solid #444444;"
            "}"
        )
        logs_layout.addWidget(self.log_view)
        right_layout.addWidget(logs_group, 2)

        fig_group = QGroupBox("QC Figures")
        fig_layout = QVBoxLayout(fig_group)
        self.figure_list = QListWidget()
        self.figure_preview = QLabel("No figure selected")
        self.figure_preview.setAlignment(Qt.AlignCenter)
        self.figure_preview.setMinimumHeight(300)
        self.figure_preview.setStyleSheet("border: 1px solid #888;")
        fig_layout.addWidget(self.figure_list, 1)
        fig_layout.addWidget(self.figure_preview, 3)
        right_layout.addWidget(fig_group, 3)

    def _on_open_options_doc(self) -> None:
        if not OPTIONS_DOC_PATH.exists():
            QMessageBox.warning(self, "Missing docs", f"Options doc not found: {OPTIONS_DOC_PATH}")
            return
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(OPTIONS_DOC_PATH)))
        if not opened:
            QMessageBox.information(self, "Options Doc", f"Open this file:\n{OPTIONS_DOC_PATH}")

    def _on_load_config_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load config JSON", str(REPO_ROOT), "JSON (*.json)")
        if not path_str:
            return
        self._load_config(Path(path_str))

    def _on_save_config_clicked(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(self, "Save config JSON", str(REPO_ROOT / "out"), "JSON (*.json)")
        if not path_str:
            return
        path = Path(path_str)
        payload = self._collect_config_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        self.config_path_edit.setText(str(path))
        self._config_path = path

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
            sidecar = Path(_resolve_repo_path(aif_paths[0]))
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
        """Show the mode the run will actually use.

        `stage_overrides.aif_curve_mode` beats the top-level `aif_mode` in
        `_resolve_stage_b_aif_mode`, so a config carrying both must display the winner.
        Collecting writes `aif_mode` only, which collapses the two spellings to one."""
        overrides = payload.get("stage_overrides", {})
        mode = next(
            (
                v
                for k, v in overrides.items()
                if k.strip().lower() == "aif_curve_mode" and str(v).strip()
            ),
            payload.get("aif_mode", DEFAULT_AIF_MODE),
        )
        text = str(mode).strip().lower()
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

        # The BIDS folders may be unchanged by this load, so re-derive explicitly rather
        # than relying on their textChanged signal to schedule it.
        self._schedule_auto_fill()

    def _collect_config_payload(self) -> Dict[str, Any]:
        model_flags = {name: (1 if cb.isChecked() else 0) for name, cb in self.model_checks.items()}
        output_dir_text = self.output_dir_edit.text().strip()
        if output_dir_text in {"", "."}:
            output_dir = str(REPO_ROOT / "out" / "dce_gui")
        else:
            output_dir = _resolve_repo_path(output_dir_text)
        payload = {
            "subject_source_path": _resolve_repo_path(self.subject_source_edit.text()),
            "subject_tp_path": _resolve_repo_path(self.subject_tp_edit.text()),
            "output_dir": output_dir,
            "checkpoint_dir": _resolve_repo_path(self.checkpoint_dir_edit.text()),
            "backend": self.backend_combo.currentText(),
            "write_xls": self.write_xls_check.isChecked(),
            "aif_mode": self.aif_mode_combo.currentText(),
            "dynamic_files": _resolve_repo_paths(_text_to_paths(self.dynamic_edit.toPlainText())),
            "aif_files": _resolve_repo_paths(_text_to_paths(self.aif_edit.toPlainText())),
            "roi_files": _resolve_repo_paths(_text_to_paths(self.roi_edit.toPlainText())),
            "t1map_files": _resolve_repo_paths(_text_to_paths(self.t1map_edit.toPlainText())),
            "noise_files": _resolve_repo_paths(_text_to_paths(self.noise_edit.toPlainText())),
            "drift_files": _resolve_repo_paths(_text_to_paths(self.drift_edit.toPlainText())),
            "model_flags": model_flags,
            "stage_overrides": self._stage_overrides_to_dict(),
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
        self.progress.setValue(0)
        self.stage_label.setText("Stage: starting")
        self.model_label.setText("Model: -")

        proc = QProcess(self)
        proc.setWorkingDirectory(str(REPO_ROOT))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        self._process = proc

        args = [str(CLI_ENTRYPOINT), "--config", str(config_path), "--events", "on"]
        proc.start(sys.executable, args)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _stop_run_hard(self) -> None:
        if self._process is None:
            return
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self.log_view.appendPlainText("Hard stop requested: process killed.")

    def _append_log_line(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += chunk
        lines = self._stdout_buffer.splitlines(keepends=False)
        if self._stdout_buffer and not self._stdout_buffer.endswith(("\n", "\r")):
            self._stdout_buffer = lines[-1]
            lines = lines[:-1]
        else:
            self._stdout_buffer = ""

        for line in lines:
            clean = line.rstrip()
            if clean == "":
                continue
            self._append_log_line(clean)
            if clean.startswith(EVENT_PREFIX):
                payload_text = clean[len(EVENT_PREFIX) :]
                try:
                    event = json.loads(payload_text)
                except Exception:
                    continue
                self._handle_event(event)

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if exit_code == 0:
            self.stage_label.setText("Stage: done")
            self.progress.setValue(100)
        else:
            self.stage_label.setText(f"Stage: failed (exit={exit_code})")
        self._append_log_line(f"Process finished with exit code {exit_code}")

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

    def _add_figure(self, path: str) -> None:
        if path in self._event_paths:
            return
        p = Path(path)
        if not p.exists():
            return
        self._event_paths.add(path)
        self.figure_list.addItem(QListWidgetItem(path))

    def _on_figure_selected(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            self.figure_preview.setText("No figure selected")
            return
        path = current.text()
        pix = QPixmap(path)
        if pix.isNull():
            self.figure_preview.setText(f"Unable to load image: {path}")
            return
        scaled = pix.scaled(self.figure_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.figure_preview.setPixmap(scaled)


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    app = QApplication(sys.argv)
    win = DceGuiWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
