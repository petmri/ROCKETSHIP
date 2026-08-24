"""Front end shared by the DCE and parametric GUIs.

Both windows do the same things around the edges of their settings forms: style themselves,
resolve paths against the config that holds them, open browse dialogs, run a CLI as a
subprocess and render its event stream, and show the figures that come back. Those parts
live here so the two interfaces cannot drift, which is what happened while they were two
copies -- the same `--set` value parsed three ways is the previous example.

What stays in each GUI is what genuinely differs: which settings exist, how a config payload
is assembled, and what an event means for the progress bar.

`GuiCommonMixin` is mixed into a `QMainWindow` and expects the subclass to provide:

    _config_path      Path      the config currently loaded (anchors relative paths)
    _process          Optional[QProcess]
    _stdout_buffer    str
    _event_paths      set[str]  figure paths already listed
    _log_reporter     Optional[Reporter]
    log_view          QPlainTextEdit
    _handle_event(event: dict)  -> None
    _load_config(path: Path)    -> None
    _collect_config_payload()   -> dict

Widgets a window does not build (a GUI with no figure list, say) simply mean the matching
mixin methods are never called.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, List, Optional

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QFontDatabase, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import run_reporting
from run_reporting import CallbackStream, Reporter

REPO_ROOT = Path(__file__).resolve().parents[1]

# The prefix a CLI puts in front of each JSON event when `--events on` is passed. This is
# the protocol between a GUI and the subprocess it drives; changing it breaks both.
EVENT_PREFIX = "ROCKETSHIP_EVENT "

# Window-wide palette. The default group-box grey sits within a few percent of the white of
# the fields inside it, which on some displays made the edge of a text box invisible. These
# darken the panel and give every white field a border, so "editable" reads at a glance.
# Set once on the window and inherited by every child; widgets that carry their own
# stylesheet (the log view, the progress bar) keep it, since a widget's own sheet wins.
PANEL_BG = "#F0F0F0"
PANEL_BORDER = "#b4b4bb"
FIELD_BG = "#ffffff"
FIELD_BORDER = "#9a9aa2"
WINDOW_QSS = f"""
QGroupBox {{
    background-color: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    border-radius: 4px;
    /* Tall enough that the title clears the frame: with a shorter margin the top border is
       drawn through the middle of the text. */
    margin-top: 20px;
    padding: 3px 8px 5px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
}}
/* The collapsible sections are group boxes with no title, so they need no room above. */
QGroupBox[titleless="true"] {{
    margin-top: 0;
    padding: 4px 8px;
}}
QLineEdit, QPlainTextEdit, QListWidget, QTableWidget {{
    background-color: {FIELD_BG};
    border: 1px solid {FIELD_BORDER};
    border-radius: 3px;
}}
QLineEdit:disabled, QPlainTextEdit:disabled {{
    background-color: #f0f0f2;
    color: #6a6a70;
}}
"""


# --------------------------------------------------------------------------- #
# Text and path helpers
# --------------------------------------------------------------------------- #


def paths_to_text(values: List[str]) -> str:
    return "\n".join(values)


def text_to_paths(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_path(text: str, base_dir: Path) -> str:
    """Resolve a possibly-relative path against `base_dir`.

    Both CLIs resolve relative paths in a config against that config file's own directory,
    not the process cwd. `base_dir` must track wherever the currently-loaded config lives so
    relative paths keep meaning what they meant when authored, even after the GUI
    re-serializes the payload into a different directory (typically output_dir) to run it.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate)


def resolve_paths(values: List[str], base_dir: Path) -> List[str]:
    return [resolve_path(v, base_dir) for v in values if str(v).strip()]


def form_label(text: str, config_key: str, hint: str = "") -> QLabel:
    """Friendly field label that still exposes the underlying config key on hover."""
    label = QLabel(text)
    tooltip = f"config key: {config_key}"
    if hint:
        tooltip = f"{hint}\n\n{tooltip}"
    label.setToolTip(tooltip)
    return label


# --------------------------------------------------------------------------- #
# Widget builders
# --------------------------------------------------------------------------- #


def build_log_view() -> QPlainTextEdit:
    """The terminal-looking panel a run's rendered output goes into."""
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    view.setStyleSheet(
        "QPlainTextEdit {"
        " background-color: #000000;"
        " color: #ffffff;"
        " selection-background-color: #ffffff;"
        " selection-color: #000000;"
        " border: 1px solid #444444;"
        "}"
    )
    return view


@dataclass
class FiguresPanel:
    """The QC-figures tab: a list of what was written, and a preview of the selected one."""

    widget: QWidget
    list_widget: QListWidget
    preview: QLabel


def build_figures_panel() -> FiguresPanel:
    panel = QWidget()
    layout = QVBoxLayout(panel)
    figure_list = QListWidget()
    preview = QLabel("No figure selected")
    preview.setAlignment(Qt.AlignCenter)
    preview.setMinimumHeight(300)
    preview.setStyleSheet("border: 1px solid #888;")
    layout.addWidget(figure_list, 1)
    layout.addWidget(preview, 3)
    return FiguresPanel(widget=panel, list_widget=figure_list, preview=preview)


@dataclass
class RunBar:
    """Run/Stop, the status labels and the progress bar, as one row."""

    group: QGroupBox
    run_button: QPushButton
    stop_button: QPushButton
    stage_label: QLabel
    detail_label: QLabel
    progress: QProgressBar


def build_run_bar(
    run_text: str,
    widest_stage_text: str,
    widest_detail_text: str,
    detail_idle_text: str = "",
) -> RunBar:
    """Build the run bar.

    The two `widest_*` strings reserve label width up front. Without them the labels resize
    as a run progresses and the progress bar's left edge jumps with them; pass the longest
    message each label will actually show.
    """
    # This bar shows on every tab, so it is a single row: buttons, status text and the
    # progress bar side by side. Each child is explicitly AlignVCenter rather than left
    # to fill the row -- styles disagree about how much height a group box reserves for
    # its title, and stacked rows drifted to the top of the box on macOS.
    group = QGroupBox("Run")
    group.setSizePolicy(group.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)
    layout = QHBoxLayout(group)
    layout.setContentsMargins(8, 2, 8, 2)
    layout.setSpacing(8)

    run_button = QPushButton(run_text)
    stop_button = QPushButton("Hard Stop")
    stop_button.setEnabled(False)
    stage_label = QLabel("Stage: idle")
    detail_label = QLabel(detail_idle_text)
    # Minimum rather than fixed width: this pins the progress bar's left edge for every
    # message these labels actually show, without truncating an unexpectedly long one.
    metrics = stage_label.fontMetrics()
    stage_label.setMinimumWidth(metrics.horizontalAdvance(widest_stage_text) + 8)
    if widest_detail_text:
        detail_label.setMinimumWidth(metrics.horizontalAdvance(widest_detail_text) + 8)

    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    progress.setTextVisible(True)
    progress.setFixedHeight(18)
    progress.setMinimumWidth(200)
    # Styled rather than left native: the macOS style draws no text on a progress bar and
    # gives it a near-white groove, so on that platform an idle bar was invisible against
    # the group box. An explicit border and track render the same everywhere.
    progress.setStyleSheet(
        "QProgressBar {"
        " border: 1px solid #909090;"
        " border-radius: 3px;"
        " background-color: #ffffff;"
        " text-align: center;"
        " color: #000000;"
        "}"
        "QProgressBar::chunk {"
        " background-color: #3b82f6;"
        " border-radius: 2px;"
        "}"
    )

    for widget in (run_button, stop_button, stage_label, detail_label):
        layout.addWidget(widget, 0, Qt.AlignVCenter)
    # No alignment flag here: a fixed-height widget is centred by the layout anyway, and
    # an explicit flag would stop it stretching to fill the remaining width.
    layout.addWidget(progress, 1)
    return RunBar(
        group=group,
        run_button=run_button,
        stop_button=stop_button,
        stage_label=stage_label,
        detail_label=detail_label,
        progress=progress,
    )


# --------------------------------------------------------------------------- #
# Mixin
# --------------------------------------------------------------------------- #


class GuiCommonMixin:
    """Path resolution, browse dialogs, the subprocess lifecycle and figure preview."""

    # -- paths and dialogs --------------------------------------------------- #

    def _base_dir(self) -> Path:
        """Directory relative paths in the current config are anchored to.

        Mirrors each CLI's `base_dir=config_path.parent` semantics, so a relative path shown
        in the GUI (as loaded verbatim from JSON) keeps resolving the way the CLI would,
        regardless of where the GUI later writes the run config it launches.
        """
        config_path = getattr(self, "_config_path", None)
        return config_path.parent if config_path is not None else REPO_ROOT

    def _dialog_start_dir(self, current_text: str) -> str:
        text = current_text.strip()
        if text:
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = (self._base_dir() / candidate).resolve()
            if candidate.is_file():
                return str(candidate.parent)
            if candidate.exists():
                return str(candidate)
        return str(self._base_dir())

    def _choose_directory_for(self, edit: QLineEdit, title: str) -> None:
        chosen = QFileDialog.getExistingDirectory(self, title, self._dialog_start_dir(edit.text()))
        if chosen:
            edit.setText(chosen)

    def _choose_file_for(self, edit: QLineEdit, title: str) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, title, self._dialog_start_dir(edit.text()), "All Files (*)"
        )
        if chosen:
            edit.setText(chosen)

    def _choose_files_for(self, edit: QPlainTextEdit, title: str) -> None:
        existing = text_to_paths(edit.toPlainText())
        start_dir = self._dialog_start_dir(existing[0] if existing else "")
        selected, _ = QFileDialog.getOpenFileNames(self, title, start_dir, "All Files (*)")
        if selected:
            edit.setPlainText(paths_to_text(selected))

    def _line_edit_with_browse(self, edit: QLineEdit, on_browse: Any) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(on_browse)
        layout.addWidget(browse)
        return row

    # -- config file dialogs ------------------------------------------------- #

    def _on_load_config_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load config JSON", str(REPO_ROOT), "JSON (*.json)"
        )
        if path_str:
            self._load_config(Path(path_str))

    def _on_save_config_clicked(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save config JSON", str(REPO_ROOT / "out"), "JSON (*.json)"
        )
        if not path_str:
            return
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._collect_config_payload(), indent=2) + "\n", encoding="utf-8")
        self.config_path_edit.setText(str(path))
        self._config_path = path

    # -- log ------------------------------------------------------------------ #

    def _append_log_line(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _new_log_reporter(self) -> Reporter:
        """The renderer the log view shows a run through.

        The same one the CLIs use, so a terminal run and the GUI log describe a run
        identically rather than being two descriptions that drift.
        """
        return Reporter(
            stream=CallbackStream(self._append_log_line),
            verbosity=run_reporting.GUI_VERBOSITY,
            color=False,
            tty=False,
        )

    # -- subprocess ----------------------------------------------------------- #

    def _start_process(self, entrypoint: Path, config_path: Path) -> QProcess:
        """Launch a CLI with `--events on` and wire its output to this window."""
        proc = QProcess(self)
        proc.setWorkingDirectory(str(REPO_ROOT))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        self._process = proc
        proc.start(sys.executable, [str(entrypoint), "--config", str(config_path), "--events", "on"])
        return proc

    def _stop_run_hard(self) -> None:
        if self._process is None:
            return
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._append_log_line("Hard stop requested: process killed.")

    def _on_process_output(self) -> None:
        """Split the subprocess stream into lines and route each one.

        Event lines drive both the progress bar and the log; the log shows what an event
        *means* rather than its JSON, which is what made this panel a wall of text before.
        Anything that is not an event, or will not decode, is shown raw rather than dropped.
        """
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += chunk
        lines = self._stdout_buffer.split("\n")
        self._stdout_buffer = lines.pop()
        for line in lines:
            clean = line.rstrip("\r")
            if not clean:
                continue
            if not clean.startswith(EVENT_PREFIX):
                self._append_log_line(clean)
                continue
            try:
                event = json.loads(clean[len(EVENT_PREFIX) :])
            except Exception:
                # Not decodable: show it raw rather than swallowing it.
                self._append_log_line(clean)
                continue
            if self._log_reporter is not None:
                self._log_reporter.handle_event(event)
            self._handle_event(event)

    # -- figures --------------------------------------------------------------- #

    def _add_figure(self, path: str) -> None:
        if path in self._event_paths or not Path(path).exists():
            return
        self._event_paths.add(path)
        self.figure_list.addItem(QListWidgetItem(path))

    def _on_figure_selected(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem] = None
    ) -> None:
        if current is None:
            self.figure_preview.setText("No figure selected")
            return
        pix = QPixmap(current.text())
        if pix.isNull():
            self.figure_preview.setText(f"Unable to load image: {current.text()}")
            return
        self.figure_preview.setPixmap(
            pix.scaled(self.figure_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
