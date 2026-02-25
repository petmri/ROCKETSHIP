"""PySide6 GUI for running the Python DCE CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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
    QVBoxLayout,
    QWidget,
)


EVENT_PREFIX = "ROCKETSHIP_EVENT "
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "dce_default.json"
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

        self._build_ui()
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
        self._build_file_lists(left_layout)
        self._build_model_flags(left_layout)
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
        self.aif_mode_combo.addItems(["auto", "fitted", "raw", "imported"])
        self.write_xls_check = QCheckBox()

        form.addRow(
            "subject_source_path",
            self._line_edit_with_browse(
                self.subject_source_edit,
                lambda: self._choose_directory_for(self.subject_source_edit, "Select subject_source_path"),
            ),
        )
        form.addRow(
            "subject_tp_path",
            self._line_edit_with_browse(
                self.subject_tp_edit,
                lambda: self._choose_directory_for(self.subject_tp_edit, "Select subject_tp_path"),
            ),
        )
        form.addRow(
            "output_dir",
            self._line_edit_with_browse(
                self.output_dir_edit,
                lambda: self._choose_directory_for(self.output_dir_edit, "Select output_dir"),
            ),
        )
        form.addRow(
            "checkpoint_dir",
            self._line_edit_with_browse(
                self.checkpoint_dir_edit,
                lambda: self._choose_directory_for(self.checkpoint_dir_edit, "Select checkpoint_dir"),
            ),
        )
        form.addRow("backend", self.backend_combo)
        form.addRow("aif_mode", self.aif_mode_combo)
        form.addRow("write_xls", self.write_xls_check)
        parent_layout.addWidget(group)

    def _build_file_lists(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Input File Lists (one path per line)")
        form = QFormLayout(group)
        self.dynamic_edit = QPlainTextEdit()
        self.aif_edit = QPlainTextEdit()
        self.roi_edit = QPlainTextEdit()
        self.t1map_edit = QPlainTextEdit()
        self.noise_edit = QPlainTextEdit()
        self.drift_edit = QPlainTextEdit()

        self.dynamic_edit.setFixedHeight(60)
        self.aif_edit.setFixedHeight(60)
        self.roi_edit.setFixedHeight(60)
        self.t1map_edit.setFixedHeight(60)
        self.noise_edit.setFixedHeight(50)
        self.drift_edit.setFixedHeight(50)

        form.addRow(
            "dynamic_files",
            self._text_edit_with_browse(
                self.dynamic_edit,
                lambda: self._choose_files_for(self.dynamic_edit, "Select dynamic_files"),
            ),
        )
        form.addRow(
            "aif_files",
            self._text_edit_with_browse(
                self.aif_edit,
                lambda: self._choose_files_for(self.aif_edit, "Select aif_files"),
            ),
        )
        form.addRow(
            "roi_files",
            self._text_edit_with_browse(
                self.roi_edit,
                lambda: self._choose_files_for(self.roi_edit, "Select roi_files"),
            ),
        )
        form.addRow(
            "t1map_files",
            self._text_edit_with_browse(
                self.t1map_edit,
                lambda: self._choose_files_for(self.t1map_edit, "Select t1map_files"),
            ),
        )
        form.addRow(
            "noise_files",
            self._text_edit_with_browse(
                self.noise_edit,
                lambda: self._choose_files_for(self.noise_edit, "Select noise_files"),
            ),
        )
        form.addRow(
            "drift_files",
            self._text_edit_with_browse(
                self.drift_edit,
                lambda: self._choose_files_for(self.drift_edit, "Select drift_files"),
            ),
        )
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

    def _text_edit_with_browse(self, edit: QPlainTextEdit, on_browse: Any) -> QWidget:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        controls = QHBoxLayout()
        browse = QPushButton("Browse...")
        clear = QPushButton("Clear")
        browse.clicked.connect(on_browse)
        clear.clicked.connect(lambda: edit.clear())
        controls.addWidget(browse)
        controls.addWidget(clear)
        controls.addStretch(1)
        layout.addLayout(controls)
        return row

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
        group = QGroupBox("Stage Overrides (all keys exposed)")
        layout = QVBoxLayout(group)
        self.override_table = QTableWidget(0, 2)
        self.override_table.setHorizontalHeaderLabels(["key", "value"])
        self.override_table.horizontalHeader().setStretchLastSection(True)
        self.override_table.setMinimumHeight(360)
        layout.addWidget(self.override_table)

        controls = QHBoxLayout()
        self.override_add_button = QPushButton("Add Override")
        self.override_remove_button = QPushButton("Remove Selected")
        controls.addWidget(self.override_add_button)
        controls.addWidget(self.override_remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)
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
        self.override_table.setRowCount(0)
        for key in sorted(stage_overrides.keys(), key=lambda x: x.lower()):
            row = self.override_table.rowCount()
            self.override_table.insertRow(row)
            self.override_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.override_table.setItem(row, 1, QTableWidgetItem(_value_to_text(stage_overrides[key])))

    def _stage_overrides_to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for row in range(self.override_table.rowCount()):
            key_item = self.override_table.item(row, 0)
            value_item = self.override_table.item(row, 1)
            if key_item is None:
                continue
            key = key_item.text().strip()
            if key == "":
                continue
            value_text = value_item.text() if value_item is not None else ""
            out[key] = _text_to_value(value_text)
        return out

    def _add_override_row(self) -> None:
        row = self.override_table.rowCount()
        self.override_table.insertRow(row)
        self.override_table.setItem(row, 0, QTableWidgetItem("new_key"))
        self.override_table.setItem(row, 1, QTableWidgetItem(""))

    def _remove_override_rows(self) -> None:
        rows = sorted({index.row() for index in self.override_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.override_table.removeRow(row)

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
        self.checkpoint_dir_edit.setText(str(payload.get("checkpoint_dir", "")))
        self.backend_combo.setCurrentText(str(payload.get("backend", "auto")))
        self.aif_mode_combo.setCurrentText(str(payload.get("aif_mode", "auto")))
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
        self._set_overrides_from_dict(stage_overrides)

    def _collect_config_payload(self) -> Dict[str, Any]:
        model_flags = {name: (1 if cb.isChecked() else 0) for name, cb in self.model_checks.items()}
        payload = {
            "subject_source_path": self.subject_source_edit.text().strip(),
            "subject_tp_path": self.subject_tp_edit.text().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
            "checkpoint_dir": self.checkpoint_dir_edit.text().strip(),
            "backend": self.backend_combo.currentText(),
            "write_xls": self.write_xls_check.isChecked(),
            "aif_mode": self.aif_mode_combo.currentText(),
            "dynamic_files": _text_to_paths(self.dynamic_edit.toPlainText()),
            "aif_files": _text_to_paths(self.aif_edit.toPlainText()),
            "roi_files": _text_to_paths(self.roi_edit.toPlainText()),
            "t1map_files": _text_to_paths(self.t1map_edit.toPlainText()),
            "noise_files": _text_to_paths(self.noise_edit.toPlainText()),
            "drift_files": _text_to_paths(self.drift_edit.toPlainText()),
            "model_flags": model_flags,
            "stage_overrides": self._stage_overrides_to_dict(),
        }
        return payload

    def _prepare_run_config_path(self, payload: Dict[str, Any]) -> Path:
        output_dir_raw = str(payload.get("output_dir", "")).strip()
        if output_dir_raw in {"", "."}:
            output_dir = REPO_ROOT / "out" / "dce_gui"
        else:
            output_dir = Path(output_dir_raw).expanduser()
            if not output_dir.is_absolute():
                output_dir = (REPO_ROOT / output_dir).resolve()
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
