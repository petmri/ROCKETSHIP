"""Rendering of pipeline progress for humans.

The pipelines emit a structured event stream; `run_reporting` is the one place that turns
it into text. These tests pin the parts that are easy to break without noticing: the
default the CLIs pick, the machine protocol the GUI depends on, and the rule that a
reporter never lets a rendering problem take a run down with it.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import run_reporting  # noqa: E402
import version  # noqa: E402
from run_reporting import CallbackStream, Reporter, Verbosity  # noqa: E402


TINY_CONFIG = REPO_ROOT / "tests/python/dce_run_tiny.json"


def _render(events, verbosity=Verbosity.NORMAL, **kwargs) -> str:
    buffer = io.StringIO()
    reporter = Reporter(stream=buffer, verbosity=verbosity, color=False, tty=False, **kwargs)
    for event in events:
        reporter.handle_event(event)
    return buffer.getvalue()


RUN = [
    {"type": "cli_config", "config_path": "/x/run.json", "resolved_output_dir": "/x/out",
     "event_log_path": "/x/out/events.jsonl",
     "config": {"backend": "cpu", "model_flags": {"tofts": 1, "patlak": 0}}},
    {"type": "run_start", "output_dir": "/x/out", "backend": "cpu"},
    {"type": "stage_start", "stage": "A"},
    {"type": "stage_done", "stage": "A", "status": "ok", "array_shapes": {"Ct": [18, 16]}},
    {"type": "artifact_written", "stage": "A", "artifact_type": "figure",
     "name": "timecurves_png", "path": "/x/out/curves.png"},
    {"type": "stage_start", "stage": "D"},
    {"type": "model_start", "stage": "D", "model": "tofts", "model_index": 1, "model_total": 1},
    {"type": "model_done", "stage": "D", "model": "tofts", "model_index": 1,
     "model_total": 1, "voxel_count": 5014},
    {"type": "run_done", "summary_path": "/x/out/run.json", "status": "ok"},
]


# --------------------------------------------------------------------------- #
# Verbosity resolution
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], Verbosity.NORMAL),
        (["-v"], Verbosity.DETAILED),
        (["-vv"], Verbosity.DEBUG),
        (["-q"], Verbosity.QUIET),
        (["--verbosity", "debug"], Verbosity.DEBUG),
        # An explicit level is a statement of intent and outranks the shorthands.
        (["-vv", "--verbosity", "quiet"], Verbosity.QUIET),
    ],
)
def test_verbosity_flags_resolve_to_one_level(argv, expected) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    run_reporting.add_verbosity_argument(parser)
    assert run_reporting.verbosity_from_args(parser.parse_args(argv)) is expected


@pytest.mark.unit
def test_parse_verbosity_rejects_an_unknown_level() -> None:
    with pytest.raises(ValueError, match="quietish"):
        run_reporting.parse_verbosity("quietish")


# --------------------------------------------------------------------------- #
# What each level shows
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_normal_shows_progress_and_a_summary_but_no_raw_json() -> None:
    out = _render(RUN)
    assert "Stage A" in out
    assert "fitting tofts" in out
    assert "5,014 voxels" in out
    assert "Finished in" in out
    # The wall of text this replaced: raw event JSON must not be back at the default.
    assert '"type":' not in out


@pytest.mark.unit
def test_quiet_says_nothing_on_a_successful_run() -> None:
    assert _render(RUN, Verbosity.QUIET).strip() == ""


@pytest.mark.unit
def test_quiet_still_reports_a_failure() -> None:
    # A silent failure is the one thing worse than noise.
    out = _render(
        [{"type": "run_error", "stage": "B", "error_type": "ValueError", "error": "bad AIF"}],
        Verbosity.QUIET,
    )
    assert "bad AIF" in out and "Stage B" in out


@pytest.mark.unit
def test_detailed_adds_the_files_written_and_the_array_shapes() -> None:
    normal = _render(RUN, Verbosity.NORMAL)
    detailed = _render(RUN, Verbosity.DETAILED)
    assert "curves.png" not in normal
    assert "curves.png" in detailed
    assert "18x16" in detailed


@pytest.mark.unit
def test_debug_includes_the_raw_events() -> None:
    out = _render(RUN, Verbosity.DEBUG)
    assert '"type": "stage_done"' in out
    assert "Stage A" in out  # debug is additive, not a different renderer


@pytest.mark.unit
def test_an_unknown_event_is_ignored_rather_than_raising() -> None:
    # The renderer must never be the reason a finished run reports failure.
    assert _render([{"type": "something_new_entirely", "detail": 1}]) == ""
    assert _render(["not a dict", None, {}]) == ""


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_elapsed_comes_from_the_event_timestamps_not_the_wall_clock() -> None:
    # The GUI and any log replay read events long after they happened, so the pipeline's
    # own clock is the only one that gives the right answer.
    out = _render(
        [
            {"type": "stage_start", "stage": "A", "timestamp_utc": "2026-08-24T03:00:00+00:00"},
            {"type": "stage_done", "stage": "A", "status": "ok",
             "timestamp_utc": "2026-08-24T03:00:42+00:00"},
        ]
    )
    assert "42s" in out


@pytest.mark.unit
@pytest.mark.parametrize(
    "seconds, expected",
    [(0.62, "0.6s"), (12.7, "13s"), (95, "1m 35s"), (3720, "1h 02m"), (None, "-")],
)
def test_durations_read_the_way_a_person_would_say_them(seconds, expected) -> None:
    assert run_reporting.format_duration(seconds) == expected


# --------------------------------------------------------------------------- #
# Paths and streams
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_paths_are_shortened_against_the_directory_they_belong_to() -> None:
    assert run_reporting.shorten_path("/x/out/maps/k.nii", Path("/x/out")) == "maps/k.nii"
    # Nothing to shorten against still has to come back usable.
    assert run_reporting.shorten_path("/elsewhere/k.nii", Path("/x/out")).endswith("k.nii")
    assert run_reporting.shorten_path(None) == "-"


@pytest.mark.unit
def test_callback_stream_hands_over_whole_lines_only() -> None:
    seen: list[str] = []
    stream = CallbackStream(seen.append)
    stream.write("one\ntw")
    assert seen == ["one"]  # the partial line waits for its newline
    stream.write("o\n")
    assert seen == ["one", "two"]


@pytest.mark.unit
def test_a_broken_stream_does_not_take_the_run_down() -> None:
    buffer = io.StringIO()
    reporter = Reporter(stream=buffer, verbosity=Verbosity.NORMAL, color=False, tty=False)
    buffer.close()
    for event in RUN:
        reporter.handle_event(event)  # must not raise


# --------------------------------------------------------------------------- #
# Nested rendering (a batch session inside a batch)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_nested_reporter_shows_stages_but_not_the_framing_its_parent_printed() -> None:
    nested = _render(RUN, Verbosity.DETAILED, nested=True)
    assert "Stage A" in nested
    assert "ROCKETSHIP DCE" not in nested
    assert "Finished in" not in nested


@pytest.mark.unit
def test_a_nested_reporter_still_shortens_paths_against_the_output_dir() -> None:
    # It skips the header that usually carries output_dir, so it has to take it from run_start.
    assert "curves.png" in _render(RUN, Verbosity.DETAILED, nested=True)
    assert "/x/out/curves.png" not in _render(RUN, Verbosity.DETAILED, nested=True)


# --------------------------------------------------------------------------- #
# Notices
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_notice_prints_when_no_entry_point_has_claimed_it(capsys) -> None:
    # Keeps the pipeline usable as a library, where nobody installed a sink.
    run_reporting.set_notice_sink(None)
    run_reporting.notice("something worth saying")
    assert "something worth saying" in capsys.readouterr().out


@pytest.mark.unit
def test_a_notice_routed_to_a_reporter_obeys_its_verbosity() -> None:
    buffer = io.StringIO()
    reporter = Reporter(stream=buffer, verbosity=Verbosity.NORMAL, color=False, tty=False)
    run_reporting.set_notice_sink(run_reporting.reporter_notice_sink(reporter))
    try:
        run_reporting.notice("detail only", Verbosity.DETAILED)
        run_reporting.notice("worth saying at normal", Verbosity.NORMAL)
    finally:
        run_reporting.set_notice_sink(None)
    out = buffer.getvalue()
    assert "detail only" not in out
    assert "worth saying at normal" in out


@pytest.mark.unit
def test_a_notice_routed_to_the_event_stream_becomes_an_event() -> None:
    events: list[dict] = []
    run_reporting.set_notice_sink(run_reporting.event_notice_sink(events.append))
    try:
        run_reporting.notice("into the stream", Verbosity.NORMAL)
    finally:
        run_reporting.set_notice_sink(None)
    assert events == [
        {"type": "notice", "text": "into the stream", "level": int(Verbosity.NORMAL)}
    ]
    # And it renders back out the far side, which is how the GUI sees it.
    assert "into the stream" in _render(events)


# --------------------------------------------------------------------------- #
# The queue summary shared by both batch drivers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_queue_summary_names_what_is_ready_and_what_is_blocked() -> None:
    buffer = io.StringIO()
    reporter = Reporter(stream=buffer, verbosity=Verbosity.NORMAL, color=False, tty=False)
    run_reporting.report_queue(
        reporter, [("sub-01_ses-01", None), ("sub-02_ses-01", "missing t1_map")]
    )
    out = buffer.getvalue()
    assert "Found 2 sessions, 1 with complete inputs" in out
    assert "missing t1_map" in out


@pytest.mark.unit
def test_a_long_queue_lists_only_the_sessions_the_user_can_act_on() -> None:
    buffer = io.StringIO()
    reporter = Reporter(stream=buffer, verbosity=Verbosity.NORMAL, color=False, tty=False)
    entries = [(f"sub-{i:02d}_ses-01", None) for i in range(40)]
    entries.append(("sub-99_ses-01", "missing aif_mask"))
    run_reporting.report_queue(reporter, entries)
    out = buffer.getvalue()
    assert "Found 41 sessions" in out
    assert "sub-99_ses-01" in out
    assert "sub-05_ses-01" not in out  # a healthy session in a long queue is not news


# --------------------------------------------------------------------------- #
# Build identity
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_header_reports_the_version_the_run_carried_not_the_one_reading_it() -> None:
    """A GUI renders a subprocess and a log replay renders the past. Either can be a
    different build from the one that produced the events, so the version travels in the
    event rather than being imported by whoever is drawing the header."""
    events = [dict(RUN[0], version="1.9.9", git_revision="deadbee")] + RUN[1:]
    assert "ROCKETSHIP DCE v1.9.9" in _render(events)


@pytest.mark.unit
def test_a_run_that_reports_no_version_still_gets_a_header() -> None:
    # Events predating this field, and library callers that never set it.
    assert "ROCKETSHIP DCE" in _render(RUN)
    assert " v" not in _render(RUN).splitlines()[1]


@pytest.mark.unit
def test_the_revision_is_detail_and_appears_only_when_detail_was_asked_for() -> None:
    events = [dict(RUN[0], version="2.0.rc", git_revision="8109e50-dirty")] + RUN[1:]
    assert "8109e50-dirty" not in _render(events, Verbosity.NORMAL)
    assert "8109e50-dirty" in _render(events, Verbosity.DETAILED)


@pytest.mark.unit
def test_build_identity_is_one_shape_for_every_caller() -> None:
    """Summaries and headers read the same dict, so they cannot disagree."""
    identity = version.build_identity()
    assert identity["version"] == version.__version__
    assert set(identity) == {"version", "git_revision"}


@pytest.mark.unit
def test_a_source_tree_that_is_not_a_checkout_reports_no_revision(tmp_path) -> None:
    """A release download has no .git. That is not an error -- the version is still known."""
    assert version._git(tmp_path, "rev-parse", "--short", "HEAD") is None


# --------------------------------------------------------------------------- #
# End-to-end contracts the other interfaces rely on
# --------------------------------------------------------------------------- #


def _run_cli(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "python/dce_cli.py"), "--config", str(TINY_CONFIG), *extra],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )


@pytest.mark.integration
def test_the_cli_default_is_readable_progress_not_the_event_stream() -> None:
    result = _run_cli()
    assert result.returncode == 0, result.stderr
    assert "ROCKETSHIP_EVENT" not in result.stdout
    assert "Stage A" in result.stdout and "Finished in" in result.stdout


@pytest.mark.integration
def test_events_on_still_speaks_the_protocol_the_gui_parses() -> None:
    result = _run_cli("--events", "on")
    assert result.returncode == 0, result.stderr
    prefixed = [ln for ln in result.stdout.splitlines() if ln.startswith("ROCKETSHIP_EVENT ")]
    assert prefixed, "the GUI drives its progress bar off these lines"
    types = {json.loads(ln[len("ROCKETSHIP_EVENT ") :])["type"] for ln in prefixed}
    assert {"run_start", "stage_start", "stage_done", "run_done"} <= types
    # Human progress must not be interleaved into the machine stream.
    assert "Finished in" not in result.stdout


@pytest.mark.integration
def test_the_event_log_records_everything_whatever_the_console_showed(tmp_path) -> None:
    log = tmp_path / "events.jsonl"
    assert _run_cli("-q", "--event-log", str(log)).returncode == 0
    types = [json.loads(line)["type"] for line in log.read_text().splitlines()]
    assert {"run_start", "stage_done", "run_done"} <= set(types)


@pytest.mark.integration
def test_the_summary_records_which_rocketship_produced_it(tmp_path) -> None:
    """Result files outlive the terminal that made them, so the build is written down
    whatever the console showed -- here at the quietest level, which prints nothing."""
    out = tmp_path / "run"
    assert _run_cli("-q", "--output-dir", str(out)).returncode == 0
    provenance = json.loads((out / "dce_pipeline_run.json").read_text())["provenance"]
    assert provenance["version"] == version.__version__
    assert "git_revision" in provenance


@pytest.mark.integration
def test_the_gui_log_renders_events_instead_of_dumping_them() -> None:
    """The GUI's 'CLI output' used to be the raw event stream, which is what made it a
    wall of text. It reads the same events now, but shows what they mean."""
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    import dce_gui

    class _Chunk:
        def __init__(self, text: str) -> None:
            self._data = text.encode("utf-8")

        def __bytes__(self) -> bytes:
            return self._data

    payload = "".join(
        f"{dce_gui.EVENT_PREFIX}{json.dumps(event)}\n" for event in RUN
    )
    shown: list[str] = []
    handled: list[dict] = []
    window = SimpleNamespace(
        _process=SimpleNamespace(readAllStandardOutput=lambda: _Chunk(payload)),
        _stdout_buffer="",
        _log_reporter=Reporter(
            stream=CallbackStream(shown.append),
            verbosity=run_reporting.GUI_VERBOSITY,
            color=False,
            tty=False,
        ),
        _append_log_line=shown.append,
        _handle_event=handled.append,
    )
    dce_gui.DceGuiWindow._on_process_output(window)

    log = "\n".join(shown)
    assert "ROCKETSHIP_EVENT" not in log
    assert '"type":' not in log
    assert "Stage A" in log and "5,014 voxels" in log
    # The progress bar still needs every event, unrendered.
    assert [event["type"] for event in handled] == [event["type"] for event in RUN]


@pytest.mark.unit
def test_a_bug_in_the_renderer_does_not_fail_a_finished_run() -> None:
    """Rendering is decorative; a run that computed its maps must not report failure
    because the progress line could not be drawn."""
    buffer = io.StringIO()
    reporter = Reporter(stream=buffer, verbosity=Verbosity.NORMAL, color=False, tty=False)
    reporter._on_stage_done = lambda event: 1 / 0  # type: ignore[assignment]
    for event in RUN:
        reporter.handle_event(event)
    assert "Finished in" in buffer.getvalue()


@pytest.mark.unit
def test_debug_re_raises_so_a_renderer_bug_is_still_findable() -> None:
    reporter = Reporter(
        stream=io.StringIO(), verbosity=Verbosity.DEBUG, color=False, tty=False
    )
    reporter._on_stage_done = lambda event: 1 / 0  # type: ignore[assignment]
    with pytest.raises(ZeroDivisionError):
        reporter.handle_event({"type": "stage_done", "stage": "A", "status": "ok"})
