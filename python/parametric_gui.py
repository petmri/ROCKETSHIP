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
    QVBoxLayout,
    QWidget,
)


EVENT_PREFIX = "ROCKETSHIP_EVENT "
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "parametric_default.json"
CLI_ENTRYPOINT = REPO_ROOT / "run_parametric_python_cli.py"


def _paths_to_text(values: List[str]) -> str:
    return "\n".join(values)


def _text_to_paths(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _float_list_to_text(values: List[float]) -> str:
    return ", ".join(f"{float(v):g}" for v in values)


def _text_to_float_list(raw: str) -> List[float]:
    text = raw.strip()
    if not text:
        return []
    parts = [token.strip() for token in text.replace("\n", ",").split(",") if token.strip()]
    return [float(token) for token in parts]


class ParametricGuiWindow(QMainWindow):
    """Main window for configuring and running parametric T1 CLI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROCKETSHIP Parametric T1 (Python GUI v1)")
        self.resize(1400, 850)

        self._stdout_buffer = ""
        self._artifacts_seen: set[str] = set()
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
        top_controls.addWidget(QLabel("Config:"))
        top_controls.addWidget(self.config_path_edit, 1)
        top_controls.addWidget(self.load_button)
        top_controls.addWidget(self.save_button)
        top_controls.addWidget(self.reset_button)
        root_layout.addLayout(top_controls)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        splitter.addWidget(right_panel)
        splitter.setSizes([850, 550])

        self._build_core_settings(left_layout)
        self._build_input_settings(left_layout)
        self._build_run_controls(left_layout)
        left_layout.addStretch(1)

        self._build_logs_and_summary(right_layout)

        self.load_button.clicked.connect(self._on_load_config_clicked)
        self.save_button.clicked.connect(self._on_save_config_clicked)
        self.reset_button.clicked.connect(lambda: self._load_config(DEFAULT_CONFIG_PATH))
        self.run_button.clicked.connect(self._start_run)
        self.stop_button.clicked.connect(self._stop_run_hard)
        self.open_output_button.clicked.connect(self._open_output_dir)

    def _build_core_settings(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Core Settings")
        form = QFormLayout(group)

        self.output_dir_edit = QLineEdit()
        self.fit_type_edit = QLineEdit("t1_fa_fit")
        self.output_basename_edit = QLineEdit("T1_map")
        self.output_label_edit = QLineEdit("")
        self.tr_ms_edit = QLineEdit("")
        self.rsq_threshold_edit = QLineEdit("0.6")
        self.invalid_fill_edit = QLineEdit("-1.0")
        self.xy_smooth_sigma_edit = QLineEdit("0.0")
        self.mask_file_edit = QLineEdit("")
        self.b1_map_file_edit = QLineEdit("")
        self.script_preferences_edit = QLineEdit("")
        self.odd_echoes_check = QCheckBox()
        self.write_rsq_check = QCheckBox()
        self.write_rho_check = QCheckBox()

        form.addRow(
            "output_dir",
            self._line_edit_with_browse(
                self.output_dir_edit,
                lambda: self._choose_directory_for(self.output_dir_edit, "Select output_dir"),
            ),
        )
        form.addRow("fit_type", self.fit_type_edit)
        form.addRow("output_basename", self.output_basename_edit)
        form.addRow("output_label", self.output_label_edit)
        form.addRow("tr_ms", self.tr_ms_edit)
        form.addRow("rsquared_threshold", self.rsq_threshold_edit)
        form.addRow("invalid_fill_value", self.invalid_fill_edit)
        form.addRow("xy_smooth_sigma", self.xy_smooth_sigma_edit)
        form.addRow("odd_echoes", self.odd_echoes_check)
        form.addRow(
            "mask_file",
            self._line_edit_with_browse(
                self.mask_file_edit,
                lambda: self._choose_file_for(self.mask_file_edit, "Select mask_file"),
            ),
        )
        form.addRow(
            "b1_map_file",
            self._line_edit_with_browse(
                self.b1_map_file_edit,
                lambda: self._choose_file_for(self.b1_map_file_edit, "Select b1_map_file"),
            ),
        )
        form.addRow(
            "script_preferences_path",
            self._line_edit_with_browse(
                self.script_preferences_edit,
                lambda: self._choose_file_for(self.script_preferences_edit, "Select script_preferences_path"),
            ),
        )
        form.addRow("write_r_squared", self.write_rsq_check)
        form.addRow("write_rho_map", self.write_rho_check)
        parent_layout.addWidget(group)

    def _build_input_settings(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("VFA Inputs")
        form = QFormLayout(group)
        self.vfa_files_edit = QPlainTextEdit()
        self.vfa_files_edit.setFixedHeight(90)
        self.flip_angles_edit = QLineEdit()

        form.addRow(
            "vfa_files",
            self._text_edit_with_browse(
                self.vfa_files_edit,
                lambda: self._choose_files_for(self.vfa_files_edit, "Select vfa_files"),
            ),
        )
        form.addRow("flip_angles_deg", self.flip_angles_edit)
        parent_layout.addWidget(group)

    def _build_run_controls(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run Parametric T1")
        self.stop_button = QPushButton("Hard Stop")
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.stage_label = QLabel("Status: idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress)
        parent_layout.addWidget(group)

    def _build_logs_and_summary(self, right_layout: QVBoxLayout) -> None:
        logs_group = QGroupBox("CLI Output / Events")
        logs_layout = QVBoxLayout(logs_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        logs_layout.addWidget(self.log_view)
        right_layout.addWidget(logs_group, 3)

        summary_group = QGroupBox("Run Summary / Artifacts")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_view = QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setPlaceholderText("Summary metrics will appear after run completion.")
        self.artifact_list = QListWidget()
        self.open_output_button = QPushButton("Open Output Directory")
        summary_layout.addWidget(self.summary_view, 2)
        summary_layout.addWidget(QLabel("Artifacts"))
        summary_layout.addWidget(self.artifact_list, 2)
        summary_layout.addWidget(self.open_output_button)
        right_layout.addWidget(summary_group, 2)

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

    def _choose_file_for(self, edit: QLineEdit, title: str) -> None:
        start_dir = self._dialog_start_dir(edit.text())
        path, _ = QFileDialog.getOpenFileName(self, title, start_dir, "NIfTI (*.nii *.nii.gz);;All Files (*)")
        if path:
            edit.setText(path)

    def _choose_files_for(self, edit: QPlainTextEdit, title: str) -> None:
        existing = _text_to_paths(edit.toPlainText())
        start_seed = existing[0] if existing else ""
        start_dir = self._dialog_start_dir(start_seed)
        selected, _ = QFileDialog.getOpenFileNames(self, title, start_dir, "NIfTI (*.nii *.nii.gz);;All Files (*)")
        if selected:
            edit.setPlainText(_paths_to_text(selected))

    def _append_log_line(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _on_load_config_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load config JSON", str(REPO_ROOT), "JSON (*.json)")
        if not path_str:
            return
        self._load_config(Path(path_str))

    def _on_save_config_clicked(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save config JSON", str(REPO_ROOT / "out"), "JSON (*.json)"
        )
        if not path_str:
            return
        path = Path(path_str)
        payload = self._collect_config_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.config_path_edit.setText(str(path))
        self._config_path = path

    def _load_config(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(self, "Missing config", f"Config file not found: {path}")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._config_path = path
        self.config_path_edit.setText(str(path))

        self.output_dir_edit.setText(str(payload.get("output_dir", "")))
        self.fit_type_edit.setText(str(payload.get("fit_type", "t1_fa_fit")))
        self.output_basename_edit.setText(str(payload.get("output_basename", "T1_map")))
        self.output_label_edit.setText(str(payload.get("output_label", "")))
        self.tr_ms_edit.setText("" if payload.get("tr_ms") is None else str(payload.get("tr_ms")))
        self.rsq_threshold_edit.setText(str(payload.get("rsquared_threshold", 0.6)))
        self.invalid_fill_edit.setText(str(payload.get("invalid_fill_value", -1.0)))
        self.xy_smooth_sigma_edit.setText(str(payload.get("xy_smooth_sigma", payload.get("xy_smooth_size", 0.0))))
        self.mask_file_edit.setText(str(payload.get("mask_file", "")))
        self.b1_map_file_edit.setText(str(payload.get("b1_map_file", "")))
        self.script_preferences_edit.setText(str(payload.get("script_preferences_path", "")))
        self.odd_echoes_check.setChecked(bool(payload.get("odd_echoes", False)))
        self.write_rsq_check.setChecked(bool(payload.get("write_r_squared", True)))
        self.write_rho_check.setChecked(bool(payload.get("write_rho_map", False)))
        self.vfa_files_edit.setPlainText(_paths_to_text(list(payload.get("vfa_files", []))))
        self.flip_angles_edit.setText(_float_list_to_text(list(payload.get("flip_angles_deg", []))))

    def _collect_config_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "output_dir": self.output_dir_edit.text().strip(),
            "fit_type": self.fit_type_edit.text().strip() or "t1_fa_fit",
            "output_basename": self.output_basename_edit.text().strip() or "T1_map",
            "output_label": self.output_label_edit.text().strip(),
            "rsquared_threshold": float(self.rsq_threshold_edit.text().strip() or "0.6"),
            "odd_echoes": bool(self.odd_echoes_check.isChecked()),
            "xy_smooth_sigma": float(self.xy_smooth_sigma_edit.text().strip() or "0.0"),
            "write_r_squared": bool(self.write_rsq_check.isChecked()),
            "write_rho_map": bool(self.write_rho_check.isChecked()),
            "invalid_fill_value": float(self.invalid_fill_edit.text().strip() or "-1.0"),
            "vfa_files": _text_to_paths(self.vfa_files_edit.toPlainText()),
            "flip_angles_deg": _text_to_float_list(self.flip_angles_edit.text()),
        }
        tr_text = self.tr_ms_edit.text().strip()
        if tr_text:
            payload["tr_ms"] = float(tr_text)
        mask_text = self.mask_file_edit.text().strip()
        if mask_text:
            payload["mask_file"] = mask_text
        b1_text = self.b1_map_file_edit.text().strip()
        if b1_text:
            payload["b1_map_file"] = b1_text
        script_prefs_text = self.script_preferences_edit.text().strip()
        if script_prefs_text:
            payload["script_preferences_path"] = script_prefs_text
        return payload

    def _prepare_run_config_path(self, payload: Dict[str, Any]) -> Path:
        output_dir_raw = str(payload.get("output_dir", "")).strip()
        if output_dir_raw in {"", "."}:
            output_dir = REPO_ROOT / "out" / "parametric_gui"
            payload["output_dir"] = str(output_dir)
        else:
            output_dir = Path(output_dir_raw).expanduser()
            if not output_dir.is_absolute():
                output_dir = (REPO_ROOT / output_dir).resolve()
                payload["output_dir"] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "parametric_gui_last_run_config.json"
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._last_run_config_path = config_path
        return config_path

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
        self.summary_view.clear()
        self.artifact_list.clear()
        self._artifacts_seen.clear()
        self._stdout_buffer = ""
        self.progress.setValue(0)
        self.stage_label.setText("Status: starting")

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
            self._append_log_line("Hard stop requested: process killed.")

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
            if not clean:
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
            self.stage_label.setText("Status: done")
            self.progress.setValue(100)
        else:
            self.stage_label.setText(f"Status: failed (exit={exit_code})")
        self._append_log_line(f"Process finished with exit code {exit_code}")

    def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "run_start":
            self.stage_label.setText("Status: run_start")
            self.progress.setValue(10)
            return
        if event_type == "inputs_resolved":
            self.stage_label.setText("Status: inputs_resolved")
            self.progress.setValue(35)
            return
        if event_type == "artifact_written":
            path = str(event.get("path", ""))
            self._add_artifact(path)
            current = self.progress.value()
            self.progress.setValue(min(95, max(current, current + 20)))
            return
        if event_type == "run_done":
            self.stage_label.setText("Status: run_done")
            self.progress.setValue(100)
            summary_path = str(event.get("summary_path", "")).strip()
            if summary_path:
                self._load_summary(Path(summary_path))
            return

    def _add_artifact(self, path: str) -> None:
        if not path or path in self._artifacts_seen:
            return
        p = Path(path)
        if not p.exists():
            return
        self._artifacts_seen.add(path)
        self.artifact_list.addItem(QListWidgetItem(path))

    def _load_summary(self, summary_path: Path) -> None:
        if not summary_path.exists():
            return
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        outputs = payload.get("outputs", {})
        lines = [
            f"summary_path: {summary_path}",
            f"valid_fits: {metrics.get('valid_fits', '-')}",
            f"threshold_failed: {metrics.get('threshold_failed', '-')}",
            f"t1_mean_ms: {metrics.get('t1_mean_ms', '-')}",
            f"t1_median_ms: {metrics.get('t1_median_ms', '-')}",
            f"r2_mean: {metrics.get('r2_mean', '-')}",
            f"r2_median: {metrics.get('r2_median', '-')}",
            "",
            "outputs:",
            f"  t1_map_path: {outputs.get('t1_map_path', '-')}",
            f"  rsquared_map_path: {outputs.get('rsquared_map_path', '-')}",
            f"  rho_map_path: {outputs.get('rho_map_path', '-')}",
        ]
        self.summary_view.setPlainText("\n".join(lines))
        for key in ("t1_map_path", "rsquared_map_path", "rho_map_path"):
            path = str(outputs.get(key, "") or "").strip()
            if path:
                self._add_artifact(path)

    def _open_output_dir(self) -> None:
        payload = self._collect_config_payload()
        output_dir = str(payload.get("output_dir", "")).strip()
        if not output_dir:
            output_path = REPO_ROOT / "out" / "parametric_gui"
        else:
            output_path = Path(output_dir).expanduser()
            if not output_path.is_absolute():
                output_path = (REPO_ROOT / output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path)))
        if not opened:
            QMessageBox.information(self, "Output Directory", f"Open this folder:\n{output_path}")


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    app = QApplication(sys.argv)
    win = ParametricGuiWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
