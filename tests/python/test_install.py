"""Unit tests for installer-side CUDA/asset selection logic."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import subprocess
import sys
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import install as installer  # noqa: E402


@pytest.mark.unit
def test_rank_prefers_highest_compatible_cuda() -> None:
    asset_ids = [
        "windows-x64-cuda12.8",
        "windows-x64-cuda12.4",
        "windows-x64-cpu",
    ]
    ranked = installer._rank_host_asset_ids(asset_ids, (12, 4))
    assert ranked == ["windows-x64-cuda12.4", "windows-x64-cuda12.8", "windows-x64-cpu"]


@pytest.mark.unit
def test_rank_prefers_cpu_when_cuda_assets_are_all_too_new() -> None:
    asset_ids = [
        "windows-x64-cuda12.8",
        "windows-x64-cuda12.4",
        "windows-x64-cpu",
    ]
    ranked = installer._rank_host_asset_ids(asset_ids, (12, 2))
    assert ranked == ["windows-x64-cpu", "windows-x64-cuda12.4", "windows-x64-cuda12.8"]


@pytest.mark.unit
def test_rank_prefers_latest_cuda_when_local_version_unknown() -> None:
    asset_ids = [
        "linux-x64-cuda11.8",
        "linux-x64-cuda12.8",
        "linux-x64-cpu",
    ]
    ranked = installer._rank_host_asset_ids(asset_ids, None)
    assert ranked == ["linux-x64-cpu", "linux-x64-cuda12.8", "linux-x64-cuda11.8"]


@pytest.mark.unit
def test_discover_host_asset_ids_from_release_assets() -> None:
    release_payload = {
        "assets": [
            {"name": "gpufit-dev-latest-windows-x64-cuda12.8.tar.gz"},
            {"name": "gpufit-dev-latest-windows-x64-cuda12.4.zip"},
            {"name": "gpufit-dev-latest-windows-x64-cpu.tar.gz"},
            {"name": "gpufit-dev-latest-linux-x64-cuda12.8.tar.gz"},
            {"name": "README.txt"},
        ]
    }
    asset_ids = installer._discover_host_asset_ids(release_payload, "windows-x64")
    assert asset_ids == ["windows-x64-cuda12.8", "windows-x64-cuda12.4", "windows-x64-cpu"]


@pytest.mark.unit
def test_detect_local_cuda_version_from_environment() -> None:
    with patch.dict(os.environ, {"CUDA_VERSION": "12.4"}, clear=False):
        with patch("install._detect_driver_cuda_capability", return_value=(None, "nvidia-smi")):
            with patch(
                "install._detect_toolkit_cuda_version",
                return_value=((12, 4), "CUDA_VERSION"),
            ):
                version, source = installer._detect_local_cuda_version()
    assert version == (12, 4)
    assert "CUDA_VERSION" in source


@pytest.mark.unit
def test_detect_local_cuda_version_from_nvcc_output() -> None:
    fake_nvcc = subprocess.CompletedProcess(
        args=["nvcc", "--version"],
        returncode=0,
        stdout="Cuda compilation tools, release 12.4, V12.4.131",
        stderr="",
    )
    fake_smi = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=1,
        stdout="",
        stderr="",
    )

    with patch.dict(os.environ, {"CUDA_VERSION": ""}, clear=False):
        with patch("install.subprocess.run", side_effect=[fake_smi, fake_nvcc]):
            version, source = installer._detect_local_cuda_version()
    assert version == (12, 4)
    assert "nvcc --version" in source


@pytest.mark.unit
def test_detect_local_cuda_version_caps_toolkit_to_driver() -> None:
    fake_smi = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="CUDA Version: 12.4",
        stderr="",
    )
    fake_nvcc = subprocess.CompletedProcess(
        args=["nvcc", "--version"],
        returncode=0,
        stdout="Cuda compilation tools, release 12.8, V12.8.0",
        stderr="",
    )
    with patch.dict(os.environ, {"CUDA_VERSION": ""}, clear=False):
        with patch("install.subprocess.run", side_effect=[fake_smi, fake_nvcc]):
            version, source = installer._detect_local_cuda_version()
    assert version == (12, 4)
    assert "capped by driver" in source


@pytest.mark.unit
def test_find_release_defaults_to_latest_stable_tag() -> None:
    releases_payload = [
        {"tag_name": "nightly-20260216", "prerelease": True, "draft": False},
        {"tag_name": "v1.4.1", "prerelease": False, "draft": False},
        {"tag_name": "dev-20260216-abcdef12", "prerelease": True, "draft": False},
        {"tag_name": "v1.4.0", "prerelease": False, "draft": False},
    ]
    with patch("install._github_json", return_value=releases_payload):
        release = installer._find_release(
            repo="ironictoo/Gpufit",
            release_tag=None,
            prefer_prerelease=False,
            token=None,
        )
    assert release["tag_name"] == "v1.4.1"


@pytest.mark.unit
def test_parse_matlab_symbol_probe_output() -> None:
    sample = "\n".join(
        [
            "noise",
            "ROCKETSHIP_MATLAB_SYMBOL:GpufitCudaAvailableMex=1",
            "ROCKETSHIP_MATLAB_SYMBOL:gpufit_constrained=0",
            "ROCKETSHIP_MATLAB_SYMBOL:cpufit=1",
        ]
    )
    parsed = installer._parse_matlab_symbol_probe_output(sample)
    assert parsed["GpufitCudaAvailableMex"]
    assert not parsed["gpufit_constrained"]
    assert parsed["cpufit"]


@pytest.mark.unit
def test_collect_matlab_bundle_files() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        matlab_dir = root / "matlab"
        matlab_dir.mkdir(parents=True, exist_ok=True)
        (matlab_dir / "CpufitMex.mexa64").write_bytes(b"x")
        (matlab_dir / "cpufit.m").write_text("function cpufit\n", encoding="utf-8")
        (root / "python" / "ignore.txt").parent.mkdir(parents=True, exist_ok=True)
        (root / "python" / "ignore.txt").write_text("y", encoding="utf-8")

        files = installer._collect_matlab_bundle_files(root)
        names = sorted(p.name for p in files)
        assert names == ["CpufitMex.mexa64", "cpufit.m"]


@pytest.mark.unit
def test_check_matlab_install_warns_and_continues_when_matlab_is_absent(capsys) -> None:
    with patch("install.shutil.which", return_value=None):
        symbol_status, runtime_status = installer._check_matlab_install(
            repo_root=REPO_ROOT,
            matlab_cmd="matlab",
            host_key="linux-x64",
            skip=False,
        )
    assert symbol_status["status"] == "SKIP"
    assert runtime_status["status"] == "SKIP"
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.unit
def test_check_matlab_install_still_fails_when_matlab_lacks_symbols() -> None:
    with patch("install.shutil.which", return_value="/usr/bin/matlab"):
        with patch("install._probe_matlab_symbols", return_value={"gpufit": True}):
            with pytest.raises(RuntimeError, match="GpufitCudaAvailableMex"):
                installer._check_matlab_install(
                    repo_root=REPO_ROOT,
                    matlab_cmd="matlab",
                    host_key="linux-x64",
                    skip=False,
                )


@pytest.mark.unit
def test_check_matlab_install_tolerates_missing_symbols_on_macos() -> None:
    with patch("install.shutil.which", return_value="/usr/bin/matlab"):
        with patch("install._probe_matlab_symbols", return_value={}):
            with patch("install._probe_matlab_runtime_gpufit", return_value={"status": "SKIP", "detail": ""}):
                symbol_status, _ = installer._check_matlab_install(
                    repo_root=REPO_ROOT,
                    matlab_cmd="matlab",
                    host_key="macos-arm64",
                    skip=False,
                )
    assert symbol_status["status"] == "SKIP"


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher only")
def test_gui_launcher_points_at_venv_relative_to_repo_root() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        launcher = installer._write_gui_launcher(repo_root, repo_root / ".venv" / "bin" / "python")

        assert launcher == repo_root / "rocketship.sh"
        assert os.access(launcher, os.X_OK)
        body = launcher.read_text(encoding="utf-8")
        assert 'VENV_DIR="$REPO_ROOT"/.venv' in body
        assert 'source "$ACTIVATE"' in body


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher only")
def test_gui_launcher_uses_absolute_path_for_venv_outside_repo() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td) / "repo"
        repo_root.mkdir()
        outside_venv = Path(td) / "envs" / "rocketship"
        launcher = installer._write_gui_launcher(repo_root, outside_venv / "bin" / "python")

        assert f"VENV_DIR={outside_venv}" in launcher.read_text(encoding="utf-8")


@pytest.mark.integration
@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher only")
def test_gui_launcher_activates_venv_and_dispatches_to_the_requested_gui() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        venv_bin = repo_root / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(sys.executable)
        (venv_bin / "activate").write_text(
            'PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd):$PATH"\nexport PATH\n',
            encoding="utf-8",
        )
        for name in ("run_dce_python_gui.py", "run_parametric_python_gui.py"):
            (repo_root / name).write_text(
                f"import sys\nprint({name!r}, sys.argv[1:])\n", encoding="utf-8"
            )

        launcher = installer._write_gui_launcher(repo_root, venv_bin / "python")

        def run(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [str(launcher), *args], check=True, capture_output=True, text=True
            )

        assert "run_dce_python_gui.py []" in run().stdout
        assert "run_dce_python_gui.py ['--x']" in run("dce", "--x").stdout
        assert "run_parametric_python_gui.py ['--x']" in run("parametric", "--x").stdout
