from __future__ import annotations

from pathlib import Path

import pytest


@pytest.hookimpl
def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rocketship parity")
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
    # The DCE parity gate has exactly two thresholds, applied to every gated model/parameter/
    # region alike (see tests/python/test_dce_pipeline_parity_metrics.py "Gate policy").
    # Scatter is normalized by the reference RMS so one bound is honest across parameters whose
    # scales differ 50x.
    #
    # nrmse tightened 0.25 -> 0.10 on 2026-08-12, after regenerating the MATLAB reference under the
    # shipped `voxel_MaxFunEvals = 200` took the measured worst case from 0.1423 to 0.0255. At 0.25
    # the gate could not see a uniform sub-25% bias: an injected 20% ex_tofts Ktrans error passed
    # (nrmse ~0.20) while injected 100% Ktrans, 5% ve and halved-SSE errors all failed. 0.10 keeps
    # ~4x margin on the worst observed check and catches that 20% case.
    group.addoption("--parity-gate-corr-min", action="store", type=float, default=0.95)
    group.addoption("--parity-gate-nrmse-max", action="store", type=float, default=0.10)
    group.addoption("--parity-required-models", "--req-models", action="store", default="tofts,ex_tofts,patlak")
    group.addoption("--parity-cpu-optional-models", "--cpu-opt-models", action="store", default="patlak")
    group.addoption("--parity-require-all-models", "--all-models", action="store_true", default=False)
    group.addoption(
        "--parity-suite",
        action="store",
        default="standard",
        help=(
            "Comma-set of parity suites to run: standard (default, gated tofts/patlak Ktrans), "
            "allmodels (adds reported-only ex_tofts/tissue_uptake/2cxm), or all."
        ),
    )
    group.addoption(
        "--parity-thresholds",
        action="store",
        default="",
        help=(
            "Optional path to a JSON file overriding parity gate thresholds "
            "(keys as in tests/python/parity_thresholds_default.json). "
            "Preferred over the individual --parity-*-corr-min/-mse-max flags."
        ),
    )
    group.addoption(
        "--run-runtime-parity",
        action="store_true",
        default=False,
        help="Enable Python-vs-MATLAB runtime parity tests for DCE and T1 workflows.",
    )
    group.addoption(
        "--runtime-parity-matlab-cmd",
        action="store",
        default="matlab",
        help="MATLAB command used by runtime parity tests (default: matlab).",
    )
    group.addoption(
        "--runtime-parity-max-python-over-matlab-ratio",
        action="store",
        type=float,
        default=1.5,
        help="Maximum allowed Python/Matlab wall-clock runtime ratio for runtime parity tests.",
    )
    group.addoption(
        "--runtime-parity-dce-root",
        action="store",
        default="",
        help="Override DCE BIDS fixture root used by runtime parity tests.",
    )
    group.addoption(
        "--runtime-parity-t1-root",
        action="store",
        default="",
        help="Override T1 BIDS fixture root used by runtime parity tests.",
    )
    group.addoption(
        "--run-phantom-gt",
        action="store_true",
        default=False,
        help="Enable synthetic-phantom ground-truth reliability checks.",
    )
    group.addoption(
        "--phantom-gt-root",
        action="store",
        default="",
        help="Override BIDS root used by phantom ground-truth reliability checks.",
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
def run_multi_model_backend_parity(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-multi-model-backend-parity"))


@pytest.fixture(scope="session")
def parity_suite(request: pytest.FixtureRequest) -> set[str]:
    raw = str(request.config.getoption("--parity-suite") or "standard")
    tokens = {t.strip().lower() for t in raw.split(",") if t.strip()}
    if "all" in tokens:
        tokens |= {"standard", "allmodels"}
    # Back-compat: the deprecated multi-model flag implies the allmodels suite.
    if request.config.getoption("--run-multi-model-backend-parity"):
        tokens |= {"allmodels"}
    tokens.add("standard")  # the gated standard suite is always part of a parity run
    return tokens


@pytest.fixture(scope="session")
def run_runtime_parity(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-runtime-parity"))


@pytest.fixture(scope="session")
def runtime_parity_matlab_cmd(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--runtime-parity-matlab-cmd") or "matlab").strip()


@pytest.fixture(scope="session")
def runtime_parity_max_python_over_matlab_ratio(request: pytest.FixtureRequest) -> float:
    return float(request.config.getoption("--runtime-parity-max-python-over-matlab-ratio"))


@pytest.fixture(scope="session")
def runtime_parity_dce_root(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--runtime-parity-dce-root") or "").strip()


@pytest.fixture(scope="session")
def runtime_parity_t1_root(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--runtime-parity-t1-root") or "").strip()


@pytest.fixture(scope="session")
def run_phantom_gt(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-phantom-gt"))


@pytest.fixture(scope="session")
def phantom_gt_root(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--phantom-gt-root") or "").strip()


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
    values = {
        "gate_corr_min": float(cfg.getoption("--parity-gate-corr-min")),
        "gate_nrmse_max": float(cfg.getoption("--parity-gate-nrmse-max")),
        "required_models_raw": str(cfg.getoption("--parity-required-models") or "").strip(),
        "cpu_optional_models_raw": str(cfg.getoption("--parity-cpu-optional-models") or "").strip(),
        "require_all_models": bool(cfg.getoption("--parity-require-all-models")),
    }
    # Optional JSON override (preferred single-file surface). Overlays only the keys present.
    thresholds_path = str(cfg.getoption("--parity-thresholds") or "").strip()
    if thresholds_path:
        import json

        path = Path(thresholds_path).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent / path).resolve()
        overrides = json.loads(path.read_text(encoding="utf-8"))
        for key, val in overrides.items():
            if key.startswith("_"):
                continue
            if key in values and isinstance(values[key], float):
                values[key] = float(val)
            else:
                values[key] = val
    return values
