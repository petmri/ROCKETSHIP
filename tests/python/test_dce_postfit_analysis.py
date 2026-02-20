"""Unit tests for DCE Part E post-fit analysis helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dce_postfit_analysis import (  # noqa: E402
    DceFitStats,
    compute_aic,
    compute_ftest,
    get_sse_and_fp,
    run_aic_analysis,
    run_ftest_analysis,
    voxel_values_to_volume,
    write_aic_roi_csv,
    write_ftest_roi_csv,
)


def _build_result_matrix(sse_values: list[float], sse_col_1based: int) -> np.ndarray:
    n_rows = len(sse_values)
    n_cols = max(sse_col_1based, 4)
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    out[:, sse_col_1based - 1] = np.asarray(sse_values, dtype=np.float64)
    return out


@pytest.mark.unit
def test_get_sse_and_fp_for_tofts_voxels() -> None:
    stats = DceFitStats(
        model_name="tofts",
        timer=np.linspace(0.0, 2.5, 6),
        fitting_results=_build_result_matrix([1.0, 2.0, 3.0], sse_col_1based=3),
    )
    sse, fp, n = get_sse_and_fp(stats, "voxel")
    assert fp == 2
    assert n == 6
    np.testing.assert_allclose(sse, [1.0, 2.0, 3.0], rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_compute_ftest_returns_low_p_for_improved_higher_model() -> None:
    lower = DceFitStats(
        model_name="tofts",
        timer=np.linspace(0.0, 4.0, 15),
        fitting_results=_build_result_matrix([10.0, 11.0], sse_col_1based=3),
    )
    higher = DceFitStats(
        model_name="ex_tofts",
        timer=np.linspace(0.0, 4.0, 15),
        fitting_results=_build_result_matrix([7.0, 8.0], sse_col_1based=4),
    )

    out = compute_ftest(lower, higher, "voxel")
    p_values = np.asarray(out["p_values"], dtype=np.float64)
    assert np.all(np.isfinite(p_values))
    assert float(np.nanmean(p_values)) < 0.1


@pytest.mark.unit
def test_compute_aic_ranks_best_model_per_roi_and_writes_csv(tmp_path: Path) -> None:
    timer = np.linspace(0.0, 3.0, 10)
    tofts = DceFitStats(
        model_name="tofts",
        timer=timer,
        roi_results=_build_result_matrix([5.0, 3.0], sse_col_1based=3),
        roi_names=["roi_a", "roi_b"],
    )
    ex_tofts = DceFitStats(
        model_name="ex_tofts",
        timer=timer,
        roi_results=_build_result_matrix([4.0, 4.0], sse_col_1based=4),
        roi_names=["roi_a", "roi_b"],
    )
    patlak = DceFitStats(
        model_name="patlak",
        timer=timer,
        roi_results=_build_result_matrix([6.0, 2.0], sse_col_1based=3),
        roi_names=["roi_a", "roi_b"],
    )

    out = compute_aic([tofts, ex_tofts, patlak], "roi")
    best = np.asarray(out["best_model_names"]).tolist()
    assert best == ["ex_tofts", "patlak"]

    csv_path = write_aic_roi_csv(tmp_path / "aic_summary.csv", ["roi_a", "roi_b"], out)
    text = csv_path.read_text(encoding="utf-8")
    assert "best_model" in text
    assert "AIC_tofts" in text
    assert "AIC_ex_tofts" in text
    assert "AIC_patlak" in text


@pytest.mark.unit
def test_write_ftest_roi_csv_writes_expected_columns(tmp_path: Path) -> None:
    csv_path = write_ftest_roi_csv(
        tmp_path / "ftest_summary.csv",
        ["roi_1", "roi_2"],
        np.asarray([0.01, 0.2], dtype=np.float64),
        np.asarray([7.0, 8.0], dtype=np.float64),
        np.asarray([10.0, 11.0], dtype=np.float64),
        higher_model_name="ex_tofts",
        lower_model_name="tofts",
    )
    text = csv_path.read_text(encoding="utf-8")
    assert "Residual_ex_tofts" in text
    assert "Residual_tofts" in text
    assert "roi_1" in text


@pytest.mark.unit
def test_voxel_values_to_volume_maps_1_based_indices() -> None:
    values = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    tumind = np.asarray([1, 5, 8], dtype=np.int64)
    out = voxel_values_to_volume(values, tumind, shape=(2, 2, 2), fill_value=-1.0)
    assert out.shape == (2, 2, 2)
    flat = out.reshape(-1)
    np.testing.assert_allclose(flat[[0, 4, 7]], values, rtol=0.0, atol=0.0)
    assert flat[1] == pytest.approx(-1.0)


@pytest.mark.unit
def test_run_ftest_analysis_writes_roi_artifacts(tmp_path: Path) -> None:
    lower = DceFitStats(
        model_name="tofts",
        timer=np.linspace(0.0, 4.0, 15),
        roi_results=_build_result_matrix([10.0, 11.0], sse_col_1based=3),
        roi_names=["roi_1", "roi_2"],
    )
    higher = DceFitStats(
        model_name="ex_tofts",
        timer=np.linspace(0.0, 4.0, 15),
        roi_results=_build_result_matrix([7.0, 8.0], sse_col_1based=4),
        roi_names=["roi_1", "roi_2"],
    )

    out = run_ftest_analysis(lower, higher, region="roi", output_dir=tmp_path / "ftest")
    assert Path(out["summary_json_path"]).exists()
    assert Path(out["p_values_vector_path"]).exists()
    assert Path(out["roi_csv_path"]).exists()


@pytest.mark.unit
def test_run_aic_analysis_writes_voxel_artifacts_with_volume_maps(tmp_path: Path) -> None:
    timer = np.linspace(0.0, 3.0, 10)
    tumind = np.asarray([1, 8], dtype=np.int64)
    dims = (2, 2, 2)
    tofts = DceFitStats(
        model_name="tofts",
        timer=timer,
        fitting_results=_build_result_matrix([5.0, 3.0], sse_col_1based=3),
        tumind_1based=tumind,
        dimensions=dims,
    )
    ex_tofts = DceFitStats(
        model_name="ex_tofts",
        timer=timer,
        fitting_results=_build_result_matrix([4.0, 4.0], sse_col_1based=4),
        tumind_1based=tumind,
        dimensions=dims,
    )

    out = run_aic_analysis([tofts, ex_tofts], region="voxel", output_dir=tmp_path / "aic")
    assert Path(out["summary_json_path"]).exists()
    assert Path(out["aic_matrix_path"]).exists()
    assert Path(out["best_model_index_path"]).exists()
    assert Path(out["best_model_index_map_path"]).exists()
    assert Path(out["best_vs_second_map_path"]).exists()
