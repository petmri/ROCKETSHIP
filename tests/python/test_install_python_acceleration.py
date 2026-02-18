"""Unit tests for installer-side CUDA/asset selection logic."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
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
        self.assertEqual(ranked, ["linux-x64-cpu", "linux-x64-cuda12.8", "linux-x64-cuda11.8"])

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
            with patch("install_python_acceleration._detect_driver_cuda_capability", return_value=(None, "nvidia-smi")):
                with patch(
                    "install_python_acceleration._detect_toolkit_cuda_version",
                    return_value=((12, 4), "CUDA_VERSION"),
                ):
                    version, source = installer._detect_local_cuda_version()
        self.assertEqual(version, (12, 4))
        self.assertIn("CUDA_VERSION", source)

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
            with patch("install_python_acceleration.subprocess.run", side_effect=[fake_smi, fake_nvcc]):
                version, source = installer._detect_local_cuda_version()
        self.assertEqual(version, (12, 4))
        self.assertIn("nvcc --version", source)

    def test_detect_local_cuda_version_caps_toolkit_to_driver(self) -> None:
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
            with patch("install_python_acceleration.subprocess.run", side_effect=[fake_smi, fake_nvcc]):
                version, source = installer._detect_local_cuda_version()
        self.assertEqual(version, (12, 4))
        self.assertIn("capped by driver", source)

    def test_find_release_defaults_to_latest_stable_tag(self) -> None:
        releases_payload = [
            {"tag_name": "nightly-20260216", "prerelease": True, "draft": False},
            {"tag_name": "v1.4.1", "prerelease": False, "draft": False},
            {"tag_name": "dev-20260216-abcdef12", "prerelease": True, "draft": False},
            {"tag_name": "v1.4.0", "prerelease": False, "draft": False},
        ]
        with patch("install_python_acceleration._github_json", return_value=releases_payload):
            release = installer._find_release(
                repo="ironictoo/Gpufit",
                release_tag=None,
                prefer_prerelease=False,
                token=None,
            )
        self.assertEqual(release["tag_name"], "v1.4.1")

    def test_parse_matlab_symbol_probe_output(self) -> None:
        sample = "\n".join(
            [
                "noise",
                "ROCKETSHIP_MATLAB_SYMBOL:GpufitCudaAvailableMex=1",
                "ROCKETSHIP_MATLAB_SYMBOL:gpufit_constrained=0",
                "ROCKETSHIP_MATLAB_SYMBOL:cpufit=1",
            ]
        )
        parsed = installer._parse_matlab_symbol_probe_output(sample)
        self.assertTrue(parsed["GpufitCudaAvailableMex"])
        self.assertFalse(parsed["gpufit_constrained"])
        self.assertTrue(parsed["cpufit"])

    def test_collect_matlab_bundle_files(self) -> None:
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
            self.assertEqual(names, ["CpufitMex.mexa64", "cpufit.m"])


if __name__ == "__main__":
    unittest.main()
