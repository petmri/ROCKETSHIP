#!/usr/bin/env python3
"""Set up ROCKETSHIP Python environment and install Gpufit/Cpufit acceleration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


GITHUB_API_BASE = "https://api.github.com"
DEFAULT_GPUFIT_REPO = "ironictoo/Gpufit"


def _log(message: str) -> None:
    print(f"[install] {message}", flush=True)


def _run(cmd: Sequence[str]) -> None:
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_root_from_python(python_path: Path) -> Path:
    parent = python_path.parent
    if parent.name in {"bin", "Scripts"}:
        return parent.parent
    return parent


def _github_json(url: str, token: Optional[str]) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "rocketship-install-python-acceleration",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp:  # nosec B310 - expected HTTPS URL
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _download(url: str, destination: Path, token: Optional[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "rocketship-install-python-acceleration",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp, destination.open("wb") as out_f:  # nosec B310 - expected HTTPS URL
        shutil.copyfileobj(resp, out_f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest().lower()


def _parse_sha256sums(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest = parts[0].strip().lower()
        filename = parts[-1].lstrip("*").strip()
        if filename and digest:
            out[filename] = digest
    return out


def _safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            target = output_dir / member.name
            resolved = target.resolve()
            if not str(resolved).startswith(str(output_dir.resolve())):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(output_dir)


def _safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in zf.namelist():
            target = output_dir / name
            resolved = target.resolve()
            if not str(resolved).startswith(str(output_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip archive: {name}")
        zf.extractall(output_dir)


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    name_lc = archive_path.name.lower()
    if name_lc.endswith(".tar.gz") or name_lc.endswith(".tgz") or name_lc.endswith(".tar"):
        _safe_extract_tar(archive_path, output_dir)
        return
    if name_lc.endswith(".zip"):
        _safe_extract_zip(archive_path, output_dir)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


def _find_release(
    repo: str,
    release_tag: Optional[str],
    prefer_prerelease: bool,
    token: Optional[str],
) -> Dict[str, object]:
    if release_tag:
        encoded = urllib.parse.quote(release_tag, safe="")
        url = f"{GITHUB_API_BASE}/repos/{repo}/releases/tags/{encoded}"
        try:
            payload = _github_json(url, token)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(f"Release tag not found: {release_tag}") from exc
            raise
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected GitHub release payload type for tag lookup")
        return payload

    url = f"{GITHUB_API_BASE}/repos/{repo}/releases?per_page=100"
    payload = _github_json(url, token)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected GitHub releases payload type")

    releases = [r for r in payload if isinstance(r, dict) and not bool(r.get("draft", False))]
    if not releases:
        raise RuntimeError(f"No releases found in {repo}")

    if prefer_prerelease:
        for rel in releases:
            if bool(rel.get("prerelease", False)):
                return rel
        raise RuntimeError(
            "No prerelease release found. Use --release-tag to target a specific release."
        )

    return releases[0]


def _asset_id_candidates_for_host() -> Tuple[str, List[str]]:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows" and machine in {"amd64", "x86_64"}:
        return "windows-x64", ["windows-x64-cuda12.8", "windows-x64-cuda12.4"]
    if system == "linux" and machine in {"amd64", "x86_64"}:
        return "linux-x64", ["linux-x64-cuda12.8", "linux-x64-cuda11.8"]
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        return "macos-x64", ["macos-x64-cpu"]
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64", ["macos-arm64-cpu"]

    raise RuntimeError(
        f"Unsupported host platform for prebuilt assets: system={system} machine={machine}"
    )


def _as_asset_list(release_payload: Dict[str, object]) -> List[Dict[str, object]]:
    assets = release_payload.get("assets", [])
    if not isinstance(assets, list):
        return []
    typed_assets: List[Dict[str, object]] = []
    for a in assets:
        if isinstance(a, dict):
            typed_assets.append(a)
    return typed_assets


def _pick_release_archive_asset(
    release_payload: Dict[str, object],
    preferred_asset_ids: Iterable[str],
) -> Dict[str, object]:
    assets = _as_asset_list(release_payload)
    archive_assets: List[Dict[str, object]] = []
    for a in assets:
        name = str(a.get("name", ""))
        if name.endswith(".tar.gz") or name.endswith(".zip"):
            archive_assets.append(a)

    if not archive_assets:
        raise RuntimeError("No archive assets found in selected release")

    for asset_id in preferred_asset_ids:
        for a in archive_assets:
            name = str(a.get("name", ""))
            if name.endswith(f"-{asset_id}.tar.gz") or name.endswith(f"-{asset_id}.zip"):
                return a

    available = ", ".join(sorted(str(a.get("name", "")) for a in archive_assets))
    raise RuntimeError(
        "No archive asset matched this host.\n"
        f"Tried asset IDs: {list(preferred_asset_ids)}\n"
        f"Available archive assets: {available}"
    )


def _pick_sha256_asset(release_payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    for a in _as_asset_list(release_payload):
        if str(a.get("name", "")) == "SHA256SUMS.txt":
            return a
    return None


def _bundle_root(extract_dir: Path) -> Path:
    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    if (extract_dir / "python").is_dir():
        return extract_dir
    raise RuntimeError(f"Unable to locate extracted package root in {extract_dir}")


def _latest_file(globbed: List[Path]) -> Optional[Path]:
    if not globbed:
        return None
    return sorted(globbed, key=lambda p: p.name)[-1]


def _install_acceleration_packages(venv_python: Path, package_root: Path) -> None:
    wheels_dir = package_root / "wheels"
    cpufit_src = package_root / "python" / "src" / "Cpufit"
    gpufit_src = package_root / "python" / "src" / "Gpufit"

    cpufit_target: Optional[Path] = None
    gpufit_target: Optional[Path] = None
    if wheels_dir.is_dir():
        cpufit_target = _latest_file(list(wheels_dir.glob("pyCpufit-*.whl")))
        gpufit_target = _latest_file(list(wheels_dir.glob("pyGpufit-*.whl")))

    if cpufit_target is None:
        if not cpufit_src.is_dir():
            raise RuntimeError("Could not find pyCpufit wheel or source package in release bundle")
        cpufit_target = cpufit_src
    if gpufit_target is None:
        if not gpufit_src.is_dir():
            raise RuntimeError("Could not find pyGpufit wheel or source package in release bundle")
        gpufit_target = gpufit_src

    _log(f"Installing pyCpufit from {cpufit_target}")
    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "--force-reinstall", str(cpufit_target)])
    _log(f"Installing pyGpufit from {gpufit_target}")
    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "--force-reinstall", str(gpufit_target)])


def _verify_install(venv_python: Path) -> None:
    code = (
        "import pycpufit.cpufit as cf; "
        "import pygpufit.gpufit as gf; "
        "print('pycpufit:', cf.__file__); "
        "print('pygpufit:', gf.__file__); "
        "print('cuda_available:', bool(gf.cuda_available()))"
    )
    _run([str(venv_python), "-c", code])


def _ensure_venv(repo_root: Path, venv_path_arg: str, recreate: bool) -> Path:
    venv_dir = Path(venv_path_arg).expanduser()
    if not venv_dir.is_absolute():
        venv_dir = (repo_root / venv_dir).resolve()

    if recreate and venv_dir.exists():
        _log(f"Removing existing virtual environment: {venv_dir}")
        shutil.rmtree(venv_dir)

    python_path = _venv_python(venv_dir)
    if not python_path.exists():
        _log(f"Creating virtual environment at: {venv_dir}")
        _run([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        _log(f"Using existing virtual environment: {venv_dir}")
    return python_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set up ROCKETSHIP Python environment and install pyCpufit/pyGpufit "
            "from ironictoo/Gpufit release assets."
        )
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_GPUFIT_REPO,
        help="GitHub repo in owner/name form (default: ironictoo/Gpufit)",
    )
    parser.add_argument(
        "--release-tag",
        default=None,
        help="Install from a specific GitHub release tag (default: latest prerelease).",
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help=(
            "Explicit release asset id suffix (for example macos-arm64-cpu or "
            "linux-x64-cuda12.8). By default this is auto-detected."
        ),
    )
    parser.add_argument(
        "--venv-path",
        default=".venv",
        help="Virtual environment path (default: .venv in repo root).",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Delete and recreate virtual environment before installing.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip requirements_gui.txt (GUI deps are installed by default).",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip SHA256 verification even if SHA256SUMS.txt is present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    try:
        host_key, auto_asset_candidates = _asset_id_candidates_for_host()
        if args.asset_id:
            asset_candidates = [args.asset_id]
        else:
            asset_candidates = auto_asset_candidates
        _log(f"Detected host: {host_key}")
        _log(f"Asset candidates: {asset_candidates}")

        release = _find_release(
            repo=args.repo,
            release_tag=args.release_tag,
            prefer_prerelease=(args.release_tag is None),
            token=token,
        )
        release_tag = str(release.get("tag_name", "unknown"))
        release_name = str(release.get("name", ""))
        _log(f"Selected release: tag={release_tag} name={release_name}")

        asset = _pick_release_archive_asset(release, asset_candidates)
        asset_name = str(asset.get("name", ""))
        download_url = str(asset.get("browser_download_url", ""))
        if not download_url:
            raise RuntimeError(f"Selected asset has no download URL: {asset_name}")
        _log(f"Selected asset: {asset_name}")

        python_in_venv = _ensure_venv(
            repo_root=repo_root,
            venv_path_arg=args.venv_path,
            recreate=args.recreate_venv,
        )

        _run([str(python_in_venv), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        _run([str(python_in_venv), "-m", "pip", "install", "-r", str(repo_root / "requirements.txt")])
        if not args.no_gui:
            _run([str(python_in_venv), "-m", "pip", "install", "-r", str(repo_root / "requirements_gui.txt")])

        with tempfile.TemporaryDirectory(prefix="rocketship_gpufit_install_") as td:
            work_dir = Path(td)
            archive_path = work_dir / asset_name
            _log(f"Downloading asset to: {archive_path}")
            _download(download_url, archive_path, token=token)

            if not args.skip_sha256:
                sha_asset = _pick_sha256_asset(release)
                if sha_asset is not None:
                    sha_name = str(sha_asset.get("name", "SHA256SUMS.txt"))
                    sha_url = str(sha_asset.get("browser_download_url", ""))
                    if sha_url:
                        sha_path = work_dir / sha_name
                        _log("Downloading SHA256SUMS.txt for verification")
                        _download(sha_url, sha_path, token=token)
                        checksums = _parse_sha256sums(sha_path.read_text(encoding="utf-8"))
                        expected = checksums.get(asset_name)
                        if expected:
                            actual = _sha256(archive_path)
                            if actual != expected.lower():
                                raise RuntimeError(
                                    "SHA256 mismatch for downloaded asset:\n"
                                    f"  expected: {expected}\n"
                                    f"  actual:   {actual}"
                                )
                            _log("SHA256 verification passed")
                        else:
                            _log("SHA256SUMS.txt does not contain this asset name; skipping checksum check")
                else:
                    _log("No SHA256SUMS.txt asset found; skipping checksum verification")

            extract_dir = work_dir / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            _log(f"Extracting {asset_name}")
            _extract_archive(archive_path, extract_dir)
            package_root = _bundle_root(extract_dir)
            _log(f"Using extracted package root: {package_root}")

            _install_acceleration_packages(python_in_venv, package_root)

        _verify_install(python_in_venv)
        _log("Installation complete.")
        _log(f"Virtual environment: {_venv_root_from_python(python_in_venv)}")
        return 0

    except Exception as exc:
        _log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
