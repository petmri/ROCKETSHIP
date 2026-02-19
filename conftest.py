from __future__ import annotations

from pathlib import Path

import pytest


@pytest.hookimpl
def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rocketship parity")
    group.addoption(
        "--run-parity",
        "--parity",
        action="store_true",
        default=False,
        help="Enable dataset-backed parity tests. Alias: --parity",
    )
    group.addoption(
        "--run-full-parity",
        "--full-parity",
        action="store_true",
        default=False,
        help="Enable full-volume parity tests (slow). Alias: --full-parity",
    )
    group.addoption(
        "--run-multi-model-backend-parity",
        "--mm-parity",
        action="store_true",
        default=False,
        help="Enable multi-model CPU-vs-auto backend parity checks. Alias: --mm-parity",
    )
    group.addoption(
        "--dataset-root",
        "--ds-root",
        action="store",
        default="",
        help="Override downsample parity dataset root. Alias: --ds-root",
    )
    group.addoption(
        "--full-root",
        "--fr-root",
        action="store",
        default="",
        help="Override full-volume parity dataset root. Alias: --fr-root",
    )
    group.addoption(
        "--roi-stride",
        "--stride",
        action="store",
        type=int,
        default=12,
        help="ROI stride for multi-model parity sparse masks. Alias: --stride",
    )
    group.addoption(
        "--parity-summary-dir",
        action="store",
        default=".pytest_cache/parity_summaries",
        help="Directory to write parity summary JSON reports.",
    )
    group.addoption("--parity-ve-ktrans-min", action="store", type=float, default=1e-6)
    group.addoption("--parity-downsample-ktrans-corr-min", action="store", type=float, default=0.99)
    group.addoption("--parity-downsample-ktrans-mse-max", action="store", type=float, default=0.001)
    group.addoption("--parity-downsample-ve-corr-min", action="store", type=float, default=0.97)
    group.addoption("--parity-downsample-ve-mse-max", action="store", type=float, default=0.002)
    group.addoption("--parity-full-ktrans-corr-min", action="store", type=float, default=0.99)
    group.addoption("--parity-full-ktrans-mse-max", action="store", type=float, default=0.001)
    group.addoption("--parity-full-ve-corr-min", action="store", type=float, default=0.97)
    group.addoption("--parity-full-ve-mse-max", action="store", type=float, default=0.002)
    group.addoption("--parity-model-ktrans-corr-min", action="store", type=float, default=0.95)
    group.addoption("--parity-model-ktrans-mse-max", action="store", type=float, default=0.01)
    group.addoption("--parity-model-param-corr-min", action="store", type=float, default=0.90)
    group.addoption("--parity-model-param-mse-max", action="store", type=float, default=0.02)
    group.addoption("--parity-cpu-auto-ktrans-corr-min", action="store", type=float, default=0.98)
    group.addoption("--parity-cpu-auto-ktrans-mse-max", action="store", type=float, default=0.002)
    group.addoption("--parity-cpu-auto-param-corr-min", action="store", type=float, default=0.95)
    group.addoption("--parity-cpu-auto-param-mse-max", action="store", type=float, default=0.01)
    group.addoption("--parity-ex-tofts-ktrans-corr-min", action="store", type=float, default=0.85)
    group.addoption("--parity-ktrans-upper-exclude", action="store", type=float, default=1.9)
    group.addoption("--parity-required-models", "--req-models", action="store", default="tofts,ex_tofts,patlak")
    group.addoption("--parity-cpu-optional-models", "--cpu-opt-models", action="store", default="patlak")
    group.addoption("--parity-require-all-models", "--all-models", action="store_true", default=False)
    group.addoption(
        "--run-osipi-slow",
        "--osipi-slow",
        action="store_true",
        default=False,
        help="Enable long-running OSIPI reliability fits. Alias: --osipi-slow",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent


@pytest.hookimpl
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Treat non-parity tests as portability-targeted checks by default.
    for item in items:
        if "parity" not in item.keywords:
            item.add_marker(pytest.mark.portability)


@pytest.fixture(scope="session")
def run_parity(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-parity"))


@pytest.fixture(scope="session")
def run_full_parity(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-full-parity"))


@pytest.fixture(scope="session")
def run_multi_model_backend_parity(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-multi-model-backend-parity"))


@pytest.fixture(scope="session")
def run_osipi_slow(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-osipi-slow"))


@pytest.fixture(scope="session")
def parity_dataset_root(request: pytest.FixtureRequest) -> str:
    option_value = str(request.config.getoption("--dataset-root") or "").strip()
    return option_value


@pytest.fixture(scope="session")
def parity_full_root(request: pytest.FixtureRequest) -> str:
    option_value = str(request.config.getoption("--full-root") or "").strip()
    return option_value


@pytest.fixture(scope="session")
def parity_roi_stride(request: pytest.FixtureRequest) -> int:
    return max(1, int(request.config.getoption("--roi-stride")))


@pytest.fixture(scope="session")
def parity_summary_dir(request: pytest.FixtureRequest, repo_root: Path) -> Path | None:
    option_value = str(request.config.getoption("--parity-summary-dir") or "").strip()
    if not option_value:
        return None
    path = Path(option_value).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def parity_thresholds(request: pytest.FixtureRequest) -> dict:
    cfg = request.config
    return {
        "ve_ktrans_min": float(cfg.getoption("--parity-ve-ktrans-min")),
        "downsample_ktrans_corr_min": float(cfg.getoption("--parity-downsample-ktrans-corr-min")),
        "downsample_ktrans_mse_max": float(cfg.getoption("--parity-downsample-ktrans-mse-max")),
        "downsample_ve_corr_min": float(cfg.getoption("--parity-downsample-ve-corr-min")),
        "downsample_ve_mse_max": float(cfg.getoption("--parity-downsample-ve-mse-max")),
        "full_ktrans_corr_min": float(cfg.getoption("--parity-full-ktrans-corr-min")),
        "full_ktrans_mse_max": float(cfg.getoption("--parity-full-ktrans-mse-max")),
        "full_ve_corr_min": float(cfg.getoption("--parity-full-ve-corr-min")),
        "full_ve_mse_max": float(cfg.getoption("--parity-full-ve-mse-max")),
        "model_ktrans_corr_min": float(cfg.getoption("--parity-model-ktrans-corr-min")),
        "model_ktrans_mse_max": float(cfg.getoption("--parity-model-ktrans-mse-max")),
        "model_param_corr_min": float(cfg.getoption("--parity-model-param-corr-min")),
        "model_param_mse_max": float(cfg.getoption("--parity-model-param-mse-max")),
        "cpu_auto_ktrans_corr_min": float(cfg.getoption("--parity-cpu-auto-ktrans-corr-min")),
        "cpu_auto_ktrans_mse_max": float(cfg.getoption("--parity-cpu-auto-ktrans-mse-max")),
        "cpu_auto_param_corr_min": float(cfg.getoption("--parity-cpu-auto-param-corr-min")),
        "cpu_auto_param_mse_max": float(cfg.getoption("--parity-cpu-auto-param-mse-max")),
        "ex_tofts_ktrans_corr_min": float(cfg.getoption("--parity-ex-tofts-ktrans-corr-min")),
        "ktrans_upper_exclude": float(cfg.getoption("--parity-ktrans-upper-exclude")),
        "required_models_raw": str(cfg.getoption("--parity-required-models") or "").strip(),
        "cpu_optional_models_raw": str(cfg.getoption("--parity-cpu-optional-models") or "").strip(),
        "require_all_models": bool(cfg.getoption("--parity-require-all-models")),
    }
