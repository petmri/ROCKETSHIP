"""Unit tests for installer-side CUDA/asset selection logic."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import install_python_acceleration as installer  # noqa: E402


class TestInstallPythonAcceleration(unittest.TestCase):
    def test_rank_prefers_highest_compatible_cuda(self) -> None:
        asset_ids = [
            "windows-x64-cuda12.8",
            "windows-x64-cuda12.4",
            "windows-x64-cpu",
        ]
        ranked = installer._rank_host_asset_ids(asset_ids, (12, 4))
        self.assertEqual(ranked, ["windows-x64-cuda12.4", "windows-x64-cuda12.8", "windows-x64-cpu"])

    def test_rank_prefers_cpu_when_cuda_assets_are_all_too_new(self) -> None:
        asset_ids = [
            "windows-x64-cuda12.8",
            "windows-x64-cuda12.4",
            "windows-x64-cpu",
        ]
        ranked = installer._rank_host_asset_ids(asset_ids, (12, 2))
        self.assertEqual(ranked, ["windows-x64-cpu", "windows-x64-cuda12.4", "windows-x64-cuda12.8"])

    def test_rank_prefers_latest_cuda_when_local_version_unknown(self) -> None:
        asset_ids = [
            "linux-x64-cuda11.8",
            "linux-x64-cuda12.8",
            "linux-x64-cpu",
        ]
        ranked = installer._rank_host_asset_ids(asset_ids, None)
        self.assertEqual(ranked, ["linux-x64-cuda12.8", "linux-x64-cuda11.8", "linux-x64-cpu"])

    def test_discover_host_asset_ids_from_release_assets(self) -> None:
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
        self.assertEqual(asset_ids, ["windows-x64-cuda12.8", "windows-x64-cuda12.4", "windows-x64-cpu"])

    def test_detect_local_cuda_version_from_environment(self) -> None:
        with patch.dict(os.environ, {"CUDA_VERSION": "12.4"}, clear=False):
            version, source = installer._detect_local_cuda_version()
        self.assertEqual(version, (12, 4))
        self.assertEqual(source, "CUDA_VERSION")

    def test_detect_local_cuda_version_from_nvcc_output(self) -> None:
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
            with patch("install_python_acceleration.subprocess.run", side_effect=[fake_nvcc, fake_smi]):
                version, source = installer._detect_local_cuda_version()
        self.assertEqual(version, (12, 4))
        self.assertEqual(source, "nvcc --version")

    def test_find_release_prefers_dev_latest_prerelease_tag(self) -> None:
        releases_payload = [
            {"tag_name": "nightly-20260216", "prerelease": True, "draft": False},
            {"tag_name": "dev-20260216-abcdef12", "prerelease": True, "draft": False},
            {"tag_name": "dev-latest", "prerelease": True, "draft": False},
        ]
        with patch("install_python_acceleration._github_json", return_value=releases_payload):
            release = installer._find_release(
                repo="ironictoo/Gpufit",
                release_tag=None,
                prefer_prerelease=True,
                token=None,
            )
        self.assertEqual(release["tag_name"], "dev-latest")


if __name__ == "__main__":
    unittest.main()
