"""QC figures for a parametric T1 run.

These pin two things that are easy to get wrong and hard to notice: that a figure never
takes a run down with it, and that what a figure *says* matches what it *draws* -- the R^2
plot annotates a rejected count, and R^2 goes arbitrarily negative, so cropping the axis to
[0, 1] once made the caption disagree with the bars.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import parametric_qc  # noqa: E402


def _t1_volume(shape=(8, 8, 5), value: float = 1400.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return value + rng.normal(0.0, 40.0, size=shape)


@pytest.mark.unit
def test_a_normal_run_produces_the_whole_set(tmp_path) -> None:
    figures = parametric_qc.write_qc_figures(
        output_dir=tmp_path,
        t1_map=_t1_volume(),
        r_squared_map=np.linspace(0.0, 1.0, 8 * 8 * 5).reshape(8, 8, 5),
        rsquared_threshold=0.6,
        label="probe",
    )
    assert set(figures) == {"t1_histogram", "r2_histogram", "t1_montage"}
    for path in figures.values():
        assert Path(path).stat().st_size > 0


@pytest.mark.unit
def test_a_single_slice_map_still_gets_a_montage(tmp_path) -> None:
    """A 2-D map is a single-slice acquisition, not a malformed volume."""
    figures = parametric_qc.write_qc_figures(
        output_dir=tmp_path,
        t1_map=_t1_volume((16, 16)),
        r_squared_map=None,
        rsquared_threshold=0.6,
        label="flat",
    )
    assert "t1_montage" in figures


@pytest.mark.unit
def test_the_r_squared_plot_counts_the_voxels_it_also_draws(tmp_path) -> None:
    """R^2 is negative wherever the fit is worse than a flat line, and can reach -200.

    Cropping those out of the axis while still counting them in the caption is the defect
    this guards: the plot then claims a rejection the bars do not show.
    """
    values = np.concatenate([
        np.full(40, -250.0),          # catastrophic fits
        np.full(10, 0.3),             # poor but on-scale
        np.full(50, 0.95),            # good
    ]).reshape(-1, 1, 1)
    figures = parametric_qc.write_qc_figures(
        output_dir=tmp_path,
        t1_map=_t1_volume((10, 10, 3)),
        r_squared_map=values,
        rsquared_threshold=0.6,
        label="neg",
    )
    assert "r2_histogram" in figures
    # The clipping is what keeps the drawn bars consistent with the counted total.
    shown = np.clip(values.reshape(-1), 0.0, 1.0)
    assert int((values.reshape(-1) < 0.6).sum()) == 50
    assert int((shown < 0.6).sum()) == 50, "clipped values must stay on the rejected side"


@pytest.mark.unit
def test_filled_voxels_are_excluded_rather_than_plotted_as_data(tmp_path) -> None:
    volume = _t1_volume((10, 10, 3))
    volume[:, :, 0] = -1.0  # the invalid_fill_value
    kept = parametric_qc._valid_t1(volume, -1.0)
    assert kept.size == 200 and float(kept.min()) > 0.0


@pytest.mark.unit
def test_a_map_with_nothing_fitted_yields_no_figures_and_no_error(tmp_path) -> None:
    figures = parametric_qc.write_qc_figures(
        output_dir=tmp_path,
        t1_map=np.full((4, 4, 2), np.nan),
        r_squared_map=np.zeros((4, 4, 2)),
        rsquared_threshold=0.6,
        label="empty",
    )
    assert figures == {}


@pytest.mark.unit
def test_a_broken_plot_costs_only_itself(tmp_path, monkeypatch) -> None:
    """Figures are decorative: a run that produced numbers must still succeed."""
    monkeypatch.setattr(
        parametric_qc, "_t1_histogram", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    figures = parametric_qc.write_qc_figures(
        output_dir=tmp_path,
        t1_map=_t1_volume(),
        r_squared_map=np.full((8, 8, 5), 0.9),
        rsquared_threshold=0.6,
        label="partial",
    )
    assert "t1_histogram" not in figures
    assert "r2_histogram" in figures and "t1_montage" in figures


@pytest.mark.unit
def test_no_matplotlib_is_not_a_failed_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(parametric_qc, "_plt", lambda: None)
    assert parametric_qc.write_qc_figures(
        output_dir=tmp_path,
        t1_map=_t1_volume(),
        r_squared_map=None,
        rsquared_threshold=0.6,
        label="nolib",
    ) == {}


@pytest.mark.unit
def test_a_thick_volume_samples_slices_instead_of_drawing_all_of_them() -> None:
    picks = parametric_qc._slice_picks(200)
    assert len(picks) == 12
    assert picks == sorted(picks)
    # Trimmed at both ends, where a volume is usually air.
    assert picks[0] > 0 and picks[-1] < 199
    assert parametric_qc._slice_picks(5) == [0, 1, 2, 3, 4]


@pytest.mark.integration
def test_the_pipeline_emits_its_figures_as_figure_artifacts(tmp_path) -> None:
    """The GUI figure tab keys off artifact_type=figure, so parametric must speak it too."""
    from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline

    example = REPO_ROOT / "python/parametric_run_example.json"
    import json

    payload = json.loads(example.read_text())
    payload["output_dir"] = str(tmp_path / "out")
    config = ParametricT1Config.from_dict(payload, base_dir=example.parent)
    if not all(p.exists() for p in config.vfa_files):
        pytest.skip("VFA fixture assets missing")

    events = []
    result = run_parametric_t1_pipeline(config, event_callback=events.append)
    figures = [e for e in events if e.get("artifact_type") == "figure"]
    assert figures, "no figure artifacts emitted"
    assert {e["name"] for e in figures} <= {"t1_histogram", "r2_histogram", "t1_montage"}
    for event in figures:
        assert Path(event["path"]).exists()
    assert result["outputs"]["qc_figures"], "figures must be recorded in the summary too"


@pytest.mark.integration
def test_figures_can_be_turned_off(tmp_path) -> None:
    from parametric_pipeline import ParametricT1Config, run_parametric_t1_pipeline

    example = REPO_ROOT / "python/parametric_run_example.json"
    import json

    payload = json.loads(example.read_text())
    payload["output_dir"] = str(tmp_path / "out")
    payload["write_qc_figures"] = False
    config = ParametricT1Config.from_dict(payload, base_dir=example.parent)
    if not all(p.exists() for p in config.vfa_files):
        pytest.skip("VFA fixture assets missing")

    events = []
    run_parametric_t1_pipeline(config, event_callback=events.append)
    assert not [e for e in events if e.get("artifact_type") == "figure"]
    assert not list((tmp_path / "out").glob("qc_*.png"))
