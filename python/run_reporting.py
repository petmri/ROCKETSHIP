"""Human-readable rendering of pipeline progress.

The pipelines already emit a structured event stream (`_emit_progress` in
`dce_pipeline.py`, `_emit_event` in `parametric_pipeline.py`). That stream is the
machine record and is unchanged here. This module is the other half: a single
renderer that turns those same events into text a person can read, shared by the
CLIs, the batch drivers and the GUI log view so all four agree on what a run looks
like.

Verbosity is a property of the *rendering*, not of the run: every level sees the
same events, and the JSONL event log always records all of them.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO


class Verbosity(IntEnum):
    """How much of the event stream to render.

    QUIET is for scripts (errors only), NORMAL for someone watching a run,
    DETAILED for someone diagnosing one, DEBUG for someone filing a bug report.
    """

    QUIET = 0
    NORMAL = 1
    DETAILED = 2
    DEBUG = 3


VERBOSITY_CHOICES = ("quiet", "normal", "detailed", "debug")

DEFAULT_VERBOSITY = Verbosity.NORMAL

# The GUI log is not a terminal, but it is watched live, so it gets the level a
# person diagnosing a run would want.
GUI_VERBOSITY = Verbosity.DETAILED


def parse_verbosity(name: str) -> Verbosity:
    try:
        return Verbosity[str(name).strip().upper()]
    except KeyError:
        raise ValueError(
            f"Unknown verbosity {name!r}; expected one of {', '.join(VERBOSITY_CHOICES)}"
        ) from None


def add_verbosity_argument(parser: Any) -> None:
    """Add the standard --verbosity/-v/-q trio to an argparse parser."""
    group = parser.add_argument_group("output")
    group.add_argument(
        "--verbosity",
        choices=VERBOSITY_CHOICES,
        default=None,
        help=(
            "How much to print: quiet (errors only), normal (progress, the default), "
            "detailed (adds inputs, settings and every file written), debug (adds raw events)"
        ),
    )
    group.add_argument(
        "-v",
        dest="verbose_count",
        action="count",
        default=0,
        help="Shorthand for --verbosity detailed; -vv for debug",
    )
    group.add_argument(
        "-q",
        dest="quiet",
        action="store_true",
        help="Shorthand for --verbosity quiet",
    )


def verbosity_from_args(args: Any) -> Verbosity:
    """Resolve --verbosity/-v/-q into one level, explicit flag winning."""
    if getattr(args, "verbosity", None):
        return parse_verbosity(args.verbosity)
    if getattr(args, "quiet", False):
        return Verbosity.QUIET
    count = int(getattr(args, "verbose_count", 0) or 0)
    if count >= 2:
        return Verbosity.DEBUG
    if count == 1:
        return Verbosity.DETAILED
    return DEFAULT_VERBOSITY


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def shorten_path(path: Any, base: Optional[Path] = None) -> str:
    """Render a path the shortest way that is still unambiguous to the reader."""
    if path is None:
        return "-"
    try:
        resolved = Path(str(path))
    except (TypeError, ValueError):
        return str(path)
    for anchor in (base, Path.cwd()):
        if anchor is None:
            continue
        try:
            return str(resolved.relative_to(anchor))
        except ValueError:
            continue
    try:
        return "~/" + str(resolved.relative_to(Path.home()))
    except ValueError:
        return str(resolved)


def plural(count: int, singular: str, suffix: str = "s") -> str:
    return singular if count == 1 else singular + suffix


class _Palette:
    """ANSI styling, or a no-op when the stream cannot use it."""

    _CODES = {
        "bold": "1",
        "dim": "2",
        "ok": "32",
        "bad": "31",
        "warn": "33",
        "head": "36",
    }

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def __call__(self, style: str, text: str) -> str:
        if not self.enabled or style not in self._CODES:
            return text
        return f"\033[{self._CODES[style]}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self("bold", text)

    def dim(self, text: str) -> str:
        return self("dim", text)

    def ok(self, text: str) -> str:
        return self("ok", text)

    def bad(self, text: str) -> str:
        return self("bad", text)

    def warn(self, text: str) -> str:
        return self("warn", text)

    def head(self, text: str) -> str:
        return self("head", text)


def _color_allowed(stream: Any) -> bool:
    # NO_COLOR is the de-facto opt-out; honour it before anything else.
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


_UNICODE_MARKS = {"ok": "✓", "bad": "✗", "run": "→", "sep": "·"}
_ASCII_MARKS = {"ok": "+", "bad": "x", "run": ">", "sep": "-"}


def _marks_for(stream: Any) -> Dict[str, str]:
    """Prefer the nicer glyphs, but not at the cost of a UnicodeEncodeError."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "".join(_UNICODE_MARKS.values()).encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return dict(_ASCII_MARKS)
    return dict(_UNICODE_MARKS)


class CallbackStream:
    """A minimal write()-able stream that forwards completed lines to a callback.

    Lets the GUI reuse the same Reporter as the CLIs: its log view takes whole
    lines, so partial writes are buffered until a newline arrives.
    """

    encoding = "utf-8"

    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._callback(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._callback(self._buffer)
            self._buffer = ""

    def isatty(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Reporter
# --------------------------------------------------------------------------- #

_STAGE_LABELS = {
    "A": "signal to concentration",
    "B": "AIF fitting",
    "D": "kinetic model fitting",
}

# Columns the description, the detail and the elapsed time occupy, so that nested
# lines still line their timings up with the stage lines above them.
_BODY_WIDTH = 48
_DETAIL_WIDTH = 14
_ELAPSED_WIDTH = 8
_LABEL_WIDTH = 9


class Reporter:
    """Renders pipeline events as human-readable progress.

    One instance per run. Feed it events with `handle_event`; it keeps only the
    state needed to render (stage timings and an artifact tally), never anything
    the caller relies on afterwards.
    """

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        verbosity: Verbosity = DEFAULT_VERBOSITY,
        color: Optional[bool] = None,
        tty: Optional[bool] = None,
        nested: bool = False,
    ) -> None:
        self.stream: Any = stream if stream is not None else sys.stdout
        self.verbosity = Verbosity(verbosity)
        self._tty = _color_allowed(self.stream) if tty is None else bool(tty)
        self.c = _Palette(self._tty if color is None else color)
        self.mark = _marks_for(self.stream)

        self._open_text: Optional[str] = None
        self._open_indent: str = "  "
        self._open_started: float = 0.0
        self._artifacts: List[Dict[str, str]] = []
        self._run_started: Optional[float] = None
        self._output_dir: Optional[str] = None
        self._event_log: Optional[str] = None
        self._voxels_fitted = 0
        self._models_done: List[str] = []
        self._header_shown = False
        self._error_shown = False
        # Events carry the time the pipeline reached them. Preferring that to the clock
        # here keeps timings honest for readers that are not the running process -- the
        # GUI reading down a pipe, or anyone replaying a saved event log.
        self._now_stamp: Optional[datetime] = None
        self._open_stamp: Optional[datetime] = None
        self._run_stamp: Optional[datetime] = None
        self._last_blank = True
        # A nested reporter renders one run inside another's progress (a batch session):
        # it shows the stages and skips the framing its parent has already printed.
        self.nested = bool(nested)
        self._base_indent = 2 if nested else 1

    # -- low-level writing --------------------------------------------------- #

    def _raw(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except (ValueError, OSError):
            # A closed or broken stream must not take the run down with it.
            pass

    def _clear_open(self) -> None:
        if self._tty and self._open_text is not None:
            self._raw("\r" + " " * (len(self._open_text) + len(self._open_indent) + 2) + "\r")

    def _restore_open(self) -> None:
        if self._tty and self._open_text is not None:
            self._raw(f"{self._open_indent}{self.mark['run']} {self._open_text}")

    def line(self, text: str = "") -> None:
        """Write one finished line, stepping around any in-place line in progress."""
        if not text.strip():
            if self._last_blank:
                return
            self._last_blank = True
        else:
            self._last_blank = False
        self._clear_open()
        self._raw(text + "\n")
        self._restore_open()

    def at(self, level: Verbosity, text: str = "") -> None:
        if self.verbosity >= level:
            self.line(text)

    def blank(self, level: Verbosity = Verbosity.NORMAL) -> None:
        self.at(level, "")

    # -- structured blocks --------------------------------------------------- #

    def header(self, title: str, rows: Sequence[tuple]) -> None:
        """A title plus aligned label/value rows, e.g. Config / Output / Backend."""
        if self.verbosity < Verbosity.NORMAL:
            return
        self.line()
        self.line(self.c.bold(title))
        for label, value in rows:
            if value is None:
                continue
            self.line(f"  {self.c.dim(str(label).ljust(_LABEL_WIDTH))}{value}")

    def listing(
        self, title: str, items: Sequence[tuple], level: Verbosity, indent: int = 1
    ) -> None:
        """A titled list of name/value pairs, e.g. the inputs discovery block."""
        if self.verbosity < level or not items:
            return
        pad = "  " * indent
        width = max(len(str(name)) for name, _ in items)
        self.line()
        self.line(f"{pad}{self.c.dim(title)}")
        for name, value in items:
            self.line(f"{pad}  {self.c.dim(str(name).ljust(width))}  {value}")
        self.line()

    def note(self, text: str, level: Verbosity = Verbosity.DETAILED) -> None:
        """A dim, indented aside. Multi-line notices keep their own indentation."""
        if self.verbosity < level:
            return
        lines = str(text).splitlines() or [""]
        pad = "  " * self._base_indent
        if len(lines) > 1:
            self.line()
        for line in lines:
            self.line(f"{pad}{self.c.dim(line)}")
        if len(lines) > 1:
            self.line()

    def warn(self, text: str) -> None:
        """A visible caution. Multi-line text keeps its shape under the first line."""
        if self.verbosity < Verbosity.NORMAL:
            return
        lines = str(text).splitlines() or [""]
        self.line(f"  {self.c.warn(self.mark['bad'] + ' ' + lines[0])}")
        for line in lines[1:]:
            self.line(f"    {self.c.warn(line)}")

    def error(self, text: str) -> None:
        # Errors survive --quiet: a silent failure is the one thing worse than noise.
        self._clear_open()
        self._open_text = None
        self._raw("\n" + self.c.bad(f"{self.mark['bad']} {text}") + "\n")

    # -- units of work ------------------------------------------------------- #

    def start(self, text: str, indent: Optional[int] = None) -> None:
        """Begin a unit of work whose completion will be reported by `settle`."""
        if self.verbosity < Verbosity.NORMAL:
            return
        indent = self._base_indent if indent is None else indent
        self._clear_open()
        self._open_text = text
        self._open_indent = "  " * indent
        self._open_started = time.perf_counter()
        self._open_stamp = self._now_stamp
        self._last_blank = False
        opener = f"{self._open_indent}{self.mark['run']} {text}"
        self._raw(opener if self._tty else opener + "\n")

    def settle(self, status: str = "done", ok: bool = True, detail: str = "") -> None:
        """Finish the open unit of work, reporting how long it took."""
        if self.verbosity < Verbosity.NORMAL:
            return
        elapsed = format_duration(
            self._elapsed_since(self._open_stamp, self._open_started)
        )
        glyph = self.mark["ok"] if ok else self.mark["bad"]
        glyph = self.c.ok(glyph) if ok else self.c.bad(glyph)
        tail = f"{detail:>{_DETAIL_WIDTH}}{elapsed:>{_ELAPSED_WIDTH}}"
        if self._tty and self._open_text is not None:
            body = self._open_text.ljust(_BODY_WIDTH - len(self._open_indent) - 2)
            self._clear_open()
            self._raw(f"{self._open_indent}{glyph} {body}{self.c.dim(tail)}\n")
        else:
            label = f"{status} in {elapsed}" if not detail else f"{status}  {detail}  {elapsed}"
            self._raw(f"{self._open_indent}  {glyph} {self.c.dim(label)}\n")
        self._open_text = None

    def _elapsed_since(self, stamp: Optional[datetime], fallback_start: float) -> float:
        """Seconds elapsed, measured by the pipeline's own clock where it reported one."""
        if stamp is not None and self._now_stamp is not None:
            return max(0.0, (self._now_stamp - stamp).total_seconds())
        return max(0.0, time.perf_counter() - fallback_start)

    def _settle_if_open(self, **kwargs: Any) -> None:
        if self._open_text is not None:
            self.settle(**kwargs)

    # -- event stream -------------------------------------------------------- #

    def handle_event(self, event: Dict[str, Any]) -> None:
        """Render one pipeline event. Unknown events are shown only at DEBUG."""
        if not isinstance(event, dict):
            return
        kind = str(event.get("type", ""))
        self._now_stamp = _parse_stamp(event.get("timestamp_utc"))
        if self.verbosity >= Verbosity.DEBUG:
            self.line(self.c.dim("    event " + json.dumps(event, default=str)))

        handler = getattr(self, f"_on_{kind}", None)
        if handler is None:
            return  # unknown event: shown above at DEBUG, ignored otherwise
        try:
            handler(event)
        except Exception:
            # Rendering is decorative. A bug here must never be the reason a finished run
            # reports failure -- but at DEBUG, someone is looking for exactly this.
            if self.verbosity >= Verbosity.DEBUG:
                raise

    def _on_cli_config(self, event: Dict[str, Any]) -> None:
        if self.nested:
            return
        payload = event.get("config") or {}
        self._output_dir = event.get("resolved_output_dir")
        self._event_log = event.get("event_log_path")
        self._header_shown = True
        models = _enabled_models(payload)
        rows = [
            ("Config", shorten_path(event.get("config_path"))),
            ("Output", shorten_path(self._output_dir)),
            ("Backend", payload.get("backend")),
            ("Models", ", ".join(models) if models else None),
            ("Fit", payload.get("fit_type")),
            ("Revision", self._revision_row(event)),
        ]
        self.header(_titled("ROCKETSHIP DCE" if models else "ROCKETSHIP", event), rows)
        if self.verbosity >= Verbosity.DETAILED:
            self._describe_settings(payload)

    def _revision_row(self, event: Dict[str, Any]) -> Optional[str]:
        """The commit a run came from -- detail, not identity, so DETAILED and above."""
        if self.verbosity < Verbosity.DETAILED:
            return None
        return event.get("git_revision")

    def _describe_settings(self, payload: Dict[str, Any]) -> None:
        overrides = payload.get("stage_overrides") or {}
        items = [(key, overrides[key]) for key in sorted(overrides) if not str(key).startswith("_")]
        self.listing("Settings from this config", items, Verbosity.DETAILED)

    def _on_run_start(self, event: Dict[str, Any]) -> None:
        self._run_started = time.perf_counter()
        self._run_stamp = self._now_stamp
        if self._output_dir is None:
            self._output_dir = event.get("output_dir")
        if self.nested:
            return
        if not self._header_shown:
            # Reached directly, without a CLI in front to describe the config.
            self.header(
                _titled("ROCKETSHIP", event),
                [
                    ("Output", shorten_path(self._output_dir)),
                    ("Backend", event.get("backend")),
                    ("Fit", event.get("fit_type")),
                    ("Revision", self._revision_row(event)),
                ],
            )
            self._header_shown = True
        if self.verbosity >= Verbosity.DETAILED:
            # Each pipeline names its own defaults file; the header calls both "defaults".
            defaults_path = event.get("dce_defaults_path") or event.get("defaults_path")
            self.note(f"defaults     {shorten_path(defaults_path)}")
            if event.get("checkpoint_dir"):
                self.note(f"checkpoints  {shorten_path(event.get('checkpoint_dir'))}")
        self.blank()

    def _on_stage_start(self, event: Dict[str, Any]) -> None:
        stage = str(event.get("stage", "?"))
        label = f"Stage {stage}  {_STAGE_LABELS.get(stage, 'working')}"
        if stage == "D":
            pad = "  " * self._base_indent
            self.at(Verbosity.NORMAL, f"{pad}{self.c.dim(self.mark['run'] + ' ' + label)}")
            return
        self.start(label)

    def _on_stage_done(self, event: Dict[str, Any]) -> None:
        status = str(event.get("status", "ok"))
        self._settle_if_open(ok=(status == "ok"), status="done" if status == "ok" else status)
        shapes = event.get("array_shapes")
        if isinstance(shapes, dict) and shapes:
            items = [(name, "x".join(str(n) for n in dims)) for name, dims in sorted(shapes.items())]
            self.listing(
                f"Stage {event.get('stage', '?')} arrays", items, Verbosity.DETAILED, indent=2
            )
        skipped = event.get("models_skipped")
        if skipped:
            self.warn(f"models skipped: {', '.join(str(m) for m in skipped)}")

    def _on_model_start(self, event: Dict[str, Any]) -> None:
        model = str(event.get("model", "?"))
        index = event.get("model_index")
        total = event.get("model_total")
        counter = f"  ({index}/{total})" if total and int(total) > 1 else ""
        self.start(f"fitting {model}{counter}", indent=self._base_indent + 1)

    def _on_model_done(self, event: Dict[str, Any]) -> None:
        voxels = event.get("voxel_count")
        self._models_done.append(str(event.get("model", "?")))
        if isinstance(voxels, int):
            self._voxels_fitted = max(self._voxels_fitted, voxels)
        detail = f"{format_count(voxels)} voxels" if voxels is not None else ""
        self._settle_if_open(detail=detail)

    def _on_checkpoint_written(self, event: Dict[str, Any]) -> None:
        path = shorten_path(event.get("path"), _as_path(self._output_dir))
        self.note(f"checkpoint {event.get('stage', '')}  {path}")

    def _on_artifact_written(self, event: Dict[str, Any]) -> None:
        record = {
            "type": str(event.get("artifact_type", "file")),
            "name": str(event.get("name", "")),
            "path": str(event.get("path", "")),
        }
        self._artifacts.append(record)
        if self.verbosity >= Verbosity.DETAILED:
            path = shorten_path(record["path"], _as_path(self._output_dir))
            self.note(f"{record['type']:<9}{record['name']:<16}  {path}")

    def _on_notice(self, event: Dict[str, Any]) -> None:
        level = event.get("level", int(Verbosity.DETAILED))
        self.note(str(event.get("text", "")), Verbosity(int(level)))

    def _on_inputs_resolved(self, event: Dict[str, Any]) -> None:
        angles = event.get("flip_angles_deg") or []
        items = [
            ("flip angles", ", ".join(f"{float(a):g}" for a in angles) + " deg" if angles else None),
            ("TR", f"{event.get('tr_ms')} ms (from {event.get('tr_source')})"),
        ]
        self.listing("Inputs", [(k, v) for k, v in items if v], Verbosity.DETAILED)
        self.start("Fitting T1 map")

    def _on_b1_map_resolved(self, event: Dict[str, Any]) -> None:
        if event.get("b1_mode") not in (None, "none"):
            self.note(f"B1 map ({event.get('b1_mode')})  {shorten_path(event.get('b1_map_path'))}")

    def _on_backend_resolved(self, event: Dict[str, Any]) -> None:
        self.note(
            f"backend requested={event.get('requested')} selected={event.get('selected')} "
            f"({event.get('reason')})"
        )

    def _on_run_error(self, event: Dict[str, Any]) -> None:
        # The pipeline reports the stage that failed and the entry point reports the run
        # that failed; when both fire for the same failure, the first one said it.
        if self._error_shown:
            return
        self._error_shown = True
        self._settle_if_open(status="failed", ok=False)
        self.error(
            f"Stage {event.get('stage', '?')} failed: "
            f"{event.get('error_type', 'Error')}: {event.get('error', '')}"
        )

    def _on_run_done(self, event: Dict[str, Any]) -> None:
        self._settle_if_open()
        if self.nested or self.verbosity < Verbosity.NORMAL:
            return
        elapsed = (
            format_duration(self._elapsed_since(self._run_stamp, self._run_started))
            if self._run_started
            else "-"
        )
        self.line()
        self.line(self.c.ok(self.c.bold(f"{self.mark['ok']} Finished in {elapsed}")))
        rows: List[tuple] = []
        if self._voxels_fitted:
            models = ", ".join(self._models_done) or "-"
            rows.append(("Fitted", f"{format_count(self._voxels_fitted)} voxels  ({models})"))
        if self._artifacts:
            rows.append(("Wrote", f"{len(self._artifacts)} {plural(len(self._artifacts), 'file')}"))
        rows.append(("Results", shorten_path(self._output_dir)))
        summary_path = event.get("summary_path")
        if summary_path:
            rows.append(("Summary", shorten_path(summary_path)))
        if self._event_log and self.verbosity >= Verbosity.DETAILED:
            rows.append(("Events", shorten_path(self._event_log)))
        for label, value in rows:
            self.line(f"  {self.c.dim(str(label).ljust(_LABEL_WIDTH))}{value}")
        self.line()


# Beyond this many entries the queue listing stops being a summary and starts being
# the wall of text it exists to replace.
QUEUE_LISTING_LIMIT = 12


def report_queue(
    reporter: Reporter,
    entries: Sequence[tuple],
    *,
    checked: bool = True,
    noun: str = "session",
) -> None:
    """Say what was found and what is queued, before any of it runs.

    `entries` is (name, problem) pairs, where problem is None for a ready item. Shared by
    both batch drivers; each discovers its own inputs and describes the failures its own way.
    """
    total = len(entries)
    blocked = [(name, problem) for name, problem in entries if problem]
    headline = f"Found {total} {plural(total, noun)}"
    if checked:
        headline += f", {total - len(blocked)} with complete inputs"
    reporter.at(Verbosity.NORMAL, "")
    reporter.at(Verbosity.NORMAL, reporter.c.bold(headline))

    # Everything is listed while the list is still readable; past that, only the entries
    # with a problem, since those are the ones the user can act on.
    shown = entries if total <= QUEUE_LISTING_LIMIT else blocked
    width = max((len(str(name)) for name, _ in shown), default=0) + 2
    for name, problem in shown:
        note = reporter.c.warn(str(problem)) if problem else reporter.c.dim("ready")
        reporter.at(Verbosity.NORMAL, f"  {str(name):<{width}}{note}")
    if total > QUEUE_LISTING_LIMIT:
        reporter.at(
            Verbosity.NORMAL,
            reporter.c.dim(f"  ({total} {plural(total, noun)}; listing only those with problems)"),
        )
    reporter.at(Verbosity.NORMAL, "")


def report_batch_finish(
    reporter: Reporter,
    elapsed_sec: float,
    tally: Sequence[tuple],
    rows: Sequence[tuple] = (),
) -> None:
    """Close a batch with its timing, its tally and where the record went."""
    if reporter.verbosity < Verbosity.NORMAL:
        return
    failed = any(count for label, count in tally if label == "failed")
    glyph = reporter.mark["bad"] if failed else reporter.mark["ok"]
    headline = f"{glyph} Finished in {format_duration(elapsed_sec)}"
    reporter.line()
    reporter.line(
        reporter.c.bold(reporter.c.bad(headline) if failed else reporter.c.ok(headline))
    )
    counted = ", ".join(f"{count} {label}" for label, count in tally if count)
    reporter.line(f"  {reporter.c.dim('Sessions'.ljust(_LABEL_WIDTH))}{counted or 'none'}")
    for label, value in rows:
        reporter.line(f"  {reporter.c.dim(str(label).ljust(_LABEL_WIDTH))}{value}")
    reporter.line()


def _parse_stamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _as_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    try:
        return Path(str(value))
    except (TypeError, ValueError):
        return None


def _titled(base: str, event: Dict[str, Any]) -> str:
    """Header title with the running version appended, when the run reported one.

    Read from the event rather than imported, so a GUI rendering a subprocess -- or anyone
    replaying an old event log -- reports the version that produced the run, not the one
    doing the reading.
    """
    running = event.get("version")
    return f"{base} v{running}" if running else base


def _enabled_models(payload: Dict[str, Any]) -> List[str]:
    flags = payload.get("model_flags")
    if not isinstance(flags, dict):
        return []
    return [name for name, on in flags.items() if on]


# --------------------------------------------------------------------------- #
# Notices
# --------------------------------------------------------------------------- #
#
# A few messages come from deep inside the pipeline, where no event callback is in
# scope (config-time BIDS discovery, per-scan value provenance, backend choice).
# They post here instead of calling print, so the entry point decides where they go
# and at what verbosity. With no sink installed they print, which keeps the pipeline
# usable as a library and matches how it behaved before.

_notice_sink: Optional[Callable[[str, Verbosity], None]] = None


def set_notice_sink(sink: Optional[Callable[[str, Verbosity], None]]) -> None:
    global _notice_sink
    _notice_sink = sink


def notice(text: str, level: Verbosity = Verbosity.DETAILED) -> None:
    """Report a message that has no event of its own."""
    if _notice_sink is not None:
        _notice_sink(text, level)
        return
    print(f"[DCE] {text}", flush=True)


def reporter_notice_sink(reporter: Reporter) -> Callable[[str, Verbosity], None]:
    """Route notices into a Reporter, so they respect its verbosity."""

    def _sink(text: str, level: Verbosity) -> None:
        reporter.note(text, level)

    return _sink


def event_notice_sink(emit: Callable[[Dict[str, Any]], None]) -> Callable[[str, Verbosity], None]:
    """Route notices into the machine event stream, for consumers like the GUI."""

    def _sink(text: str, level: Verbosity) -> None:
        emit({"type": "notice", "text": text, "level": int(level)})

    return _sink
