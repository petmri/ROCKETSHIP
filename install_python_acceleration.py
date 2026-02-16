#!/usr/bin/env python3
"""Set up ROCKETSHIP Python environment and install Gpufit/Cpufit acceleration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
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
        prereleases = [r for r in releases if bool(r.get("prerelease", False))]
        if prereleases:
            # Manual dev builds may reuse the persistent "dev-latest" tag.
            # Prefer that channel when present so users get the latest refreshed dev asset.
            for preferred_tag in ("dev-latest", "latest"):
                for rel in prereleases:
                    if str(rel.get("tag_name", "")) == preferred_tag:
                        return rel
            return prereleases[0]
        raise RuntimeError(
            "No prerelease release found. Use --release-tag to target a specific release."
        )

    return releases[0]


def _asset_id_candidates_for_host() -> Tuple[str, List[str]]:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows" and machine in {"amd64", "x86_64"}:
        return "windows-x64", ["windows-x64-cuda12.8", "windows-x64-cuda12.4", "windows-x64-cpu"]
    if system == "linux" and machine in {"amd64", "x86_64"}:
        return "linux-x64", ["linux-x64-cuda12.8", "linux-x64-cuda11.8", "linux-x64-cpu"]
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        return "macos-x64", ["macos-x64-cpu"]
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64", ["macos-arm64-cpu"]

    raise RuntimeError(
        f"Unsupported host platform for prebuilt assets: system={system} machine={machine}"
    )


def _parse_cuda_version_pair(text: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"(\d+)(?:\.(\d+))?", text)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or "0")
    return (major, minor)


def _format_cuda_version(version_pair: Tuple[int, int]) -> str:
    return f"{version_pair[0]}.{version_pair[1]}"


def _detect_local_cuda_version() -> Tuple[Optional[Tuple[int, int]], str]:
    env_cuda = os.environ.get("CUDA_VERSION", "").strip()
    if env_cuda:
        parsed = _parse_cuda_version_pair(env_cuda)
        if parsed is not None:
            return parsed, "CUDA_VERSION"

    probes: List[Tuple[List[str], str, str]] = [
        (["nvcc", "--version"], r"release\s+(\d+)\.(\d+)", "nvcc --version"),
        (["nvidia-smi"], r"CUDA Version:\s*(\d+)\.(\d+)", "nvidia-smi"),
    ]
    for cmd, pattern, source in probes:
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode != 0:
            continue
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            return (int(match.group(1)), int(match.group(2))), source

    return None, "not detected"


def _extract_host_asset_id(archive_name: str, host_key: str) -> Optional[str]:
    lower = archive_name.lower()
    if lower.endswith(".tar.gz"):
        suffix_len = len(".tar.gz")
    elif lower.endswith(".zip"):
        suffix_len = len(".zip")
    else:
        return None

    base = archive_name[:-suffix_len]
    marker = f"-{host_key}"
    idx = base.rfind(marker)
    if idx < 0:
        return None
    candidate = base[idx + 1 :]
    if not candidate.startswith(host_key):
        return None
    return candidate


def _discover_host_asset_ids(release_payload: Dict[str, object], host_key: str) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for asset in _as_asset_list(release_payload):
        name = str(asset.get("name", ""))
        asset_id = _extract_host_asset_id(name, host_key)
        if not asset_id:
            continue
        if asset_id in seen:
            continue
        seen.add(asset_id)
        out.append(asset_id)
    return out


def _cuda_version_from_asset_id(asset_id: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"-cuda(\d+)(?:\.(\d+))?$", asset_id, flags=re.IGNORECASE)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2) or "0"))


def _rank_host_asset_ids(asset_ids: List[str], local_cuda: Optional[Tuple[int, int]]) -> List[str]:
    seen: set[str] = set()
    ordered_unique: List[str] = []
    for aid in asset_ids:
        if aid in seen:
            continue
        seen.add(aid)
        ordered_unique.append(aid)

    cuda_assets: List[Tuple[str, Tuple[int, int]]] = []
    cpu_assets: List[str] = []
    other_assets: List[str] = []
    for aid in ordered_unique:
        if aid.endswith("-cpu"):
            cpu_assets.append(aid)
            continue
        cuda_version = _cuda_version_from_asset_id(aid)
        if cuda_version is not None:
            cuda_assets.append((aid, cuda_version))
            continue
        other_assets.append(aid)

    ordered_cuda: List[str] = []
    if cuda_assets:
        if local_cuda is None:
            ordered_cuda = [aid for aid, _ in sorted(cuda_assets, key=lambda item: item[1], reverse=True)]
        else:
            lower_or_equal = [item for item in cuda_assets if item[1] <= local_cuda]
            higher = [item for item in cuda_assets if item[1] > local_cuda]

            if not lower_or_equal and cpu_assets:
                ordered_cuda = [aid for aid, _ in sorted(higher, key=lambda item: item[1])]
                return cpu_assets + ordered_cuda + other_assets

            ordered_cuda.extend(aid for aid, _ in sorted(lower_or_equal, key=lambda item: item[1], reverse=True))
            ordered_cuda.extend(aid for aid, _ in sorted(higher, key=lambda item: item[1]))

    return ordered_cuda + other_assets + cpu_assets


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

    # Dependencies are installed from requirements.txt / requirements_gui.txt first.
    # Keep those versions stable to avoid pip backtracking NumPy into incompatible versions.
    install_base = [str(venv_python), "-m", "pip", "install", "--upgrade", "--force-reinstall", "--no-deps"]

    _log(f"Installing pyCpufit from {cpufit_target}")
    _run([*install_base, str(cpufit_target)])
    _log(f"Installing pyGpufit from {gpufit_target}")
    _run([*install_base, str(gpufit_target)])


def _verify_install(venv_python: Path) -> None:
    code = (
        "import pycpufit.cpufit as cf; "
        "import pygpufit.gpufit as gf; "
        "print('pycpufit:', cf.__file__); "
        "print('pygpufit:', gf.__file__); "
        "print('cuda_available:', bool(gf.cuda_available()))"
    )
    cmd = [str(venv_python), "-c", code]
    _log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode == 0:
        return

    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    details = "Acceleration package import check failed."
    combined = f"{result.stdout}\n{result.stderr}"
    if "cpufit_constrained" in combined:
        details += (
            " Installed Cpufit binary is missing symbol 'cpufit_constrained' "
            "(likely release packaging/export issue in the selected Gpufit asset)."
        )
    raise RuntimeError(details)


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
        _log(f"Detected host: {host_key}")

        release = _find_release(
            repo=args.repo,
            release_tag=args.release_tag,
            prefer_prerelease=(args.release_tag is None),
            token=token,
        )
        release_tag = str(release.get("tag_name", "unknown"))
        release_name = str(release.get("name", ""))
        _log(f"Selected release: tag={release_tag} name={release_name}")

        if args.asset_id:
            asset_candidates = [args.asset_id]
            _log(f"Asset id pinned by user: {args.asset_id}")
        else:
            local_cuda, cuda_source = _detect_local_cuda_version()
            if local_cuda is not None:
                _log(f"Detected local CUDA version: {_format_cuda_version(local_cuda)} (source: {cuda_source})")
            else:
                _log("Detected local CUDA version: not found")

            release_host_assets = _discover_host_asset_ids(release, host_key)
            if release_host_assets:
                _log(f"Host-matching assets in release: {release_host_assets}")
                asset_candidates = _rank_host_asset_ids(release_host_assets, local_cuda)
            else:
                _log("No host-matching assets parsed from release names; using built-in fallback list.")
                asset_candidates = auto_asset_candidates

        _log(f"Asset candidates: {asset_candidates}")

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
