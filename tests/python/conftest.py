from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.hookimpl
def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rocketship parity")
    group.addoption(
        "--run-parity",
        action="store_true",
        default=False,
        help="Enable dataset-backed parity tests.",
    )
    group.addoption(
        "--run-full-parity",
        action="store_true",
        default=False,
        help="Enable full-volume parity tests (slow).",
    )
    group.addoption(
        "--run-multi-model-backend-parity",
        action="store_true",
        default=False,
        help="Enable multi-model CPU-vs-auto backend parity checks.",
    )
    group.addoption(
        "--dataset-root",
        action="store",
        default="",
        help="Override downsample parity dataset root.",
    )
    group.addoption(
        "--full-root",
        action="store",
        default="",
        help="Override full-volume parity dataset root.",
    )
    group.addoption(
        "--roi-stride",
        action="store",
        type=int,
        default=12,
        help="ROI stride for multi-model parity sparse masks.",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def run_parity(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-parity")) or os.environ.get(
        "ROCKETSHIP_RUN_PIPELINE_PARITY", "0"
    ) == "1"


@pytest.fixture(scope="session")
def run_full_parity(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-full-parity")) or os.environ.get(
        "ROCKETSHIP_RUN_FULL_VOLUME_PARITY", "0"
    ) == "1"


@pytest.fixture(scope="session")
def run_multi_model_backend_parity(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-multi-model-backend-parity")) or os.environ.get(
        "ROCKETSHIP_RUN_MULTI_MODEL_BACKEND_PARITY", "0"
    ) == "1"


@pytest.fixture(scope="session")
def parity_dataset_root(request: pytest.FixtureRequest) -> str:
    option_value = str(request.config.getoption("--dataset-root") or "").strip()
    if option_value:
        return option_value
    return os.environ.get("ROCKETSHIP_BBB_DOWNSAMPLED_ROOT", "").strip()


@pytest.fixture(scope="session")
def parity_full_root(request: pytest.FixtureRequest) -> str:
    option_value = str(request.config.getoption("--full-root") or "").strip()
    if option_value:
        return option_value
    return os.environ.get("ROCKETSHIP_BBB_FULL_ROOT", "").strip()


@pytest.fixture(scope="session")
def parity_roi_stride(request: pytest.FixtureRequest) -> int:
    if request.config.getoption("--roi-stride"):
        return max(1, int(request.config.getoption("--roi-stride")))
    env_value = os.environ.get("ROCKETSHIP_PARITY_MULTI_MODEL_ROI_STRIDE", "12").strip()
    try:
        return max(1, int(env_value))
    except ValueError:
        return 12
