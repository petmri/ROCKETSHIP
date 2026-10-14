"""The front end both GUIs share, and that each window still builds.

GUI *behaviour* is out of scope for this suite (see tests/README.md). What is in scope is
that the windows construct at all and that the shared module keeps its contract: a
cross-file refactor with no coverage is how a code path dies quietly, which is exactly what
happened to the batch script.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

pytest.importorskip("PySide6")

import gui_common  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    """One offscreen QApplication for the module; Qt allows only a single instance."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_relative_path_anchors_to_the_config_that_holds_it(tmp_path) -> None:
    """Both CLIs resolve a config's relative paths against that config's own directory."""
    assert gui_common.resolve_path("sub/x.nii", tmp_path) == str(tmp_path / "sub/x.nii")


@pytest.mark.unit
def test_an_absolute_path_is_left_alone(tmp_path) -> None:
    assert gui_common.resolve_path("/abs/x.nii", tmp_path) == "/abs/x.nii"


@pytest.mark.unit
def test_blank_paths_stay_blank_rather_than_becoming_the_base_dir(tmp_path) -> None:
    # "" resolving to the base directory would silently turn an empty field into a path.
    assert gui_common.resolve_path("   ", tmp_path) == ""
    assert gui_common.resolve_paths(["", "  ", "a.nii"], tmp_path) == [str(tmp_path / "a.nii")]


@pytest.mark.unit
def test_a_path_list_survives_the_round_trip_through_its_text_box() -> None:
    paths = ["/a/one.nii", "/b/two.nii"]
    assert gui_common.text_to_paths(gui_common.paths_to_text(paths)) == paths
    # Blank lines and stray whitespace are what a hand-edited box actually contains.
    assert gui_common.text_to_paths("  /a.nii \n\n\n /b.nii\n") == ["/a.nii", "/b.nii"]


# --------------------------------------------------------------------------- #
# Shared widgets
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_run_bar_reserves_label_width_so_the_progress_bar_stops_jumping(qt_app) -> None:
    bar = gui_common.build_run_bar("Run", "Stage: failed (exit=255)", "Model: tissue_uptake (9/9)")
    assert bar.stage_label.minimumWidth() > 0
    assert bar.detail_label.minimumWidth() > bar.stage_label.minimumWidth() // 4
    assert not bar.stop_button.isEnabled(), "Stop is meaningless before a run starts"
    assert bar.progress.value() == 0


@pytest.mark.unit
def test_a_run_bar_with_no_detail_label_still_builds(qt_app) -> None:
    """Parametric has no per-model line, so it passes no widest-detail text."""
    bar = gui_common.build_run_bar("Run", "Stage: failed (exit=255)", "")
    assert bar.detail_label.minimumWidth() == 0


@pytest.mark.unit
def test_the_log_view_is_read_only_and_monospaced(qt_app) -> None:
    view = gui_common.build_log_view()
    assert view.isReadOnly()
    assert view.font().fixedPitch() or "mono" in view.font().family().lower()


@pytest.mark.unit
def test_every_theme_colour_is_defined_before_the_stylesheet_uses_it() -> None:
    # An f-string that lost a token renders "None" into the sheet rather than failing.
    assert "None" not in gui_common.WINDOW_QSS
    for token in (gui_common.PANEL_BG, gui_common.FIELD_BG, gui_common.FIELD_BORDER):
        assert token in gui_common.WINDOW_QSS


# --------------------------------------------------------------------------- #
# The windows themselves
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_the_dce_window_builds_with_its_tabs_and_settings(qt_app) -> None:
    import dce_config
    import dce_gui

    window = dce_gui.DceGuiWindow()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "Inputs",
        "CLI Output",
        "QC Figures",
        "Results",
    ]
    # The table is a view of the defaults file, not of the run config, so it must show every
    # known key regardless of what the loaded config sets -- except the handful promoted to
    # their own control, which would otherwise be two widgets editing one value.
    known = dce_config.load_defaults()
    expected = {
        key.lower()
        for key in (*known.defaults, *known.required, *known.optional)
        if key.lower() not in dce_gui.PROMOTED_OVERRIDE_KEYS
    }
    shown = {
        window.override_table.item(row, 0).text().lower()
        for row in range(window.override_table.rowCount())
    }
    assert shown == expected
    assert window.styleSheet(), "the window palette must be applied"
    assert window.log_view.toPlainText().startswith("[idle]")



@pytest.mark.integration
def test_the_parametric_window_matches_the_dce_shape(qt_app) -> None:
    """The point of the rebuild: the two windows are the same interface over two pipelines."""
    import dce_gui
    import parametric_gui

    window = parametric_gui.ParametricGuiWindow()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        window.tabs.tabText(i) for i in range(dce_gui.DceGuiWindow().tabs.count())
    ]
    assert window.styleSheet(), "the window palette must be applied"
    assert window.log_view.toPlainText().startswith("[idle]")


@pytest.mark.integration
def test_the_parametric_backend_is_reachable_from_the_gui(qt_app) -> None:
    """backend was a validated config field with no control anywhere -- JSON only."""
    import parametric_gui
    from parametric_pipeline import ALLOWED_BACKENDS

    window = parametric_gui.ParametricGuiWindow()
    offered = {window.backend_combo.itemData(i) for i in range(window.backend_combo.count())}
    assert offered == set(ALLOWED_BACKENDS)
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("cpu"))
    assert window._collect_config_payload()["backend"] == "cpu"


@pytest.mark.integration
def test_the_fit_type_control_offers_only_what_the_pipeline_accepts(qt_app) -> None:
    """It was free text, so a typo became a ValueError several seconds into a run."""
    import parametric_gui
    from parametric_pipeline import ParametricT1Config

    window = parametric_gui.ParametricGuiWindow()
    for index in range(window.fit_type_combo.count()):
        window.fit_type_combo.setCurrentIndex(index)
        payload = window._collect_config_payload()
        payload["vfa_files"] = ["/nonexistent.nii"]
        config = ParametricT1Config.from_dict(payload)
        # validate() rejects an unknown fit_type before it looks at any file.
        with pytest.raises(FileNotFoundError):
            config.validate()


@pytest.mark.integration
def test_the_resolved_view_says_where_each_value_came_from(qt_app) -> None:
    """The form fills every field, so without this the source of a value is invisible."""
    import parametric_gui

    window = parametric_gui.ParametricGuiWindow()
    rows = {
        window.resolved_table.item(r, 0).text(): window.resolved_table.item(r, 2).text()
        for r in range(window.resolved_table.rowCount())
    }
    # The shipped example names output_label and leaves the preferences to the file.
    assert rows["output_label"] == "run config"
    assert rows["rsquared_threshold"] == "defaults file"
    # Optional keys with nothing set are not errors, and must not be labelled as required.
    assert rows["tr_ms"] == "unset (optional)"
    assert rows["mask_file"] == "unset (optional)"

    window.rsq_threshold_edit.setText("0.9")
    window._refresh_resolved_table()
    changed = {
        window.resolved_table.item(r, 0).text(): window.resolved_table.item(r, 2).text()
        for r in range(window.resolved_table.rowCount())
    }
    assert changed["rsquared_threshold"] == "edited here"
    # A path re-anchored on load is not an edit, and must not be reported as one.
    assert changed["output_dir"] == "run config"


@pytest.mark.integration
def test_both_windows_take_their_shared_behaviour_from_the_mixin(qt_app) -> None:
    import dce_gui
    import parametric_gui

    shared = (
        "_base_dir",
        "_dialog_start_dir",
        "_choose_directory_for",
        "_choose_files_for",
        "_line_edit_with_browse",
        "_collapsible_section",
        "_append_log_line",
        "_on_process_output",
        "_stop_run_hard",
        "_add_figure",
        "_on_figure_selected",
        "_on_load_config_clicked",
        "_on_save_config_clicked",
    )
    for window_class in (dce_gui.DceGuiWindow, parametric_gui.ParametricGuiWindow):
        for name in shared:
            assert name not in vars(window_class), f"{window_class.__name__}.{name} is a copy"
            assert hasattr(window_class, name), f"{window_class.__name__}.{name} is unreachable"
