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
MATLAB_PROBE_PREFIX = "ROCKETSHIP_MATLAB_SYMBOL:"
MATLAB_GPUFIT_SYMBOLS = ("GpufitCudaAvailableMex", "GpufitConstrainedMex", "GpufitMex", "gpufit_constrained", "gpufit")
MATLAB_CPUFIT_SYMBOLS = ("CpufitMex", "cpufit")


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


def _require_matlab_gpufit_by_default(host_key: str) -> bool:
    # macOS release assets currently ship Python acceleration only.
    return not host_key.startswith("macos-")


def _print_post_install_next_steps(repo_root: Path, venv_python: Path) -> None:
    venv_root = _venv_root_from_python(venv_python)
    if os.name == "nt":
        activate_cmd = f"{venv_root}\\Scripts\\activate"
    else:
        activate_cmd = f"source {venv_root}/bin/activate"
    print("\n\nInstallation was successful!")
    print("Next steps:")
    print(f"  1) Activate env: {activate_cmd}")
    print("  2) Run Benchmarks: python tests/python/run_dce_benchmark.py")
    print("  3) Run Python CLI: python run_dce_python_cli.py --help")
    print("  4) Run MATLAB CLI: matlab -batch \"cd(pwd); run_dce_cli('<subject_source_path>','<subject_tp_path>');\"")


def _probe_python_install_health(venv_python: Path) -> Dict[str, object]:
    code = (
        "import json\n"
        "out = {}\n"
        "try:\n"
        "    import pycpufit.cpufit as cf\n"
        "    out['pycpufit'] = True\n"
        "    out['pycpufit_path'] = cf.__file__\n"
        "except Exception as e:\n"
        "    out['pycpufit'] = False\n"
        "    out['pycpufit_error'] = str(e)\n"
        "try:\n"
        "    import pygpufit.gpufit as gf\n"
        "    out['pygpufit'] = True\n"
        "    out['pygpufit_path'] = gf.__file__\n"
        "    out['cuda_available'] = bool(gf.cuda_available())\n"
        "except Exception as e:\n"
        "    out['pygpufit'] = False\n"
        "    out['pygpufit_error'] = str(e)\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run([str(venv_python), "-c", code], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "status": "FAIL",
            "detail": (result.stderr or result.stdout or "python health probe failed").strip(),
        }
    try:
        payload = json.loads((result.stdout or "").strip())
    except Exception:
        return {"status": "FAIL", "detail": "python health probe output parse failed"}

    pycpufit_ok = bool(payload.get("pycpufit", False))
    pygpufit_ok = bool(payload.get("pygpufit", False))
    if not pycpufit_ok:
        return {"status": "FAIL", "detail": f"pycpufit import error: {payload.get('pycpufit_error', 'unknown')}"}
    if not pygpufit_ok:
        return {"status": "FAIL", "detail": f"pygpufit import error: {payload.get('pygpufit_error', 'unknown')}"}

    return {
        "status": "PASS",
        "detail": f"pycpufit+pygpufit import ok (cuda_available={bool(payload.get('cuda_available', False))})",
    }


def _probe_matlab_runtime_gpufit(repo_root: Path, matlab_cmd: str) -> Dict[str, str]:
    if shutil.which(matlab_cmd) is None:
        return {"status": "SKIP", "detail": f"'{matlab_cmd}' not found in PATH"}

    repo_escaped = str(repo_root).replace("'", "''")
    marker = "ROCKETSHIP_MATLAB_RUNTIME="
    batch_expr = (
        f"cd('{repo_escaped}'); "
        "addpath(fullfile(pwd,'external_programs')); "
        "if ~exist('GpufitCudaAvailableMex','file'), "
        f"fprintf('{marker}MISS\\n'); "
        "else, "
        "try, avail=GpufitCudaAvailableMex; "
        f"fprintf('{marker}OK:%d\\n', double(avail)); "
        "catch ME, msg=ME.message; msg=strrep(msg, char(10), ' | '); "
        f"fprintf('{marker}ERR:%s\\n', msg); "
        "end; "
        "end;"
    )
    result = subprocess.run(
        [matlab_cmd, "-noFigureWindows", "-batch", batch_expr],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    line_value: Optional[str] = None
    for line in combined.splitlines():
        if line.startswith(marker):
            line_value = line[len(marker) :].strip()
            break

    if line_value is None:
        tail = " | ".join([ln.strip() for ln in combined.splitlines() if ln.strip()][-3:])
        return {"status": "FAIL", "detail": f"runtime probe marker missing ({tail or 'no MATLAB output'})"}
    if line_value == "MISS":
        return {"status": "SKIP", "detail": "GpufitCudaAvailableMex not present on MATLAB path"}
    if line_value.startswith("OK:"):
        avail = line_value.split(":", 1)[1].strip()
        return {"status": "PASS", "detail": f"GpufitCudaAvailableMex loaded (cuda_available={avail})"}
    if line_value.startswith("ERR:"):
        return {"status": "FAIL", "detail": line_value.split(":", 1)[1].strip()}
    return {"status": "FAIL", "detail": f"unexpected runtime probe output: {line_value}"}


def _print_post_install_health_check(
    python_health: Dict[str, object],
    matlab_symbol_status: Dict[str, str],
    matlab_runtime_status: Dict[str, str],
) -> None:
    rows = [
        ("Python acceleration imports", str(python_health.get("status", "FAIL")), str(python_health.get("detail", ""))),
        ("MATLAB symbol presence", str(matlab_symbol_status.get("status", "SKIP")), str(matlab_symbol_status.get("detail", ""))),
        ("MATLAB MEX runtime load", str(matlab_runtime_status.get("status", "SKIP")), str(matlab_runtime_status.get("detail", ""))),
    ]

    header_left = "Check"
    header_mid = "Status"
    left_w = max(len(header_left), *(len(r[0]) for r in rows))
    mid_w = max(len(header_mid), *(len(r[1]) for r in rows))
    print("\nPost-install health check")
    print(f"  {header_left:<{left_w}}  {header_mid:<{mid_w}}  Details")
    print(f"  {'-' * left_w}  {'-' * mid_w}  {'-' * 40}")
    for check, status, detail in rows:
        print(f"  {check:<{left_w}}  {status:<{mid_w}}  {detail}")


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

    stable_releases = [r for r in releases if not bool(r.get("prerelease", False))]
    if stable_releases:
        return stable_releases[0]
    raise RuntimeError(
        "No stable (non-prerelease) release found. Use --release-tag to target a specific release."
    )


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


def _run_version_probe(
    cmd: List[str],
    pattern: str,
    source: str,
) -> Tuple[Optional[Tuple[int, int]], str]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, source
    if result.returncode != 0:
        return None, source
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(pattern, combined, flags=re.IGNORECASE)
    if not match:
        return None, source
    return (int(match.group(1)), int(match.group(2))), source


def _detect_driver_cuda_capability() -> Tuple[Optional[Tuple[int, int]], str]:
    # NVIDIA driver capability from `nvidia-smi` is the compatibility gate for prebuilt CUDA binaries.
    return _run_version_probe(
        ["nvidia-smi"],
        r"CUDA Version:\s*(\d+)\.(\d+)",
        "nvidia-smi",
    )


def _detect_toolkit_cuda_version() -> Tuple[Optional[Tuple[int, int]], str]:
    env_cuda = os.environ.get("CUDA_VERSION", "").strip()
    if env_cuda:
        parsed = _parse_cuda_version_pair(env_cuda)
        if parsed is not None:
            return parsed, "CUDA_VERSION"

    probes: List[Tuple[List[str], str, str]] = [
        (["nvcc", "--version"], r"release\s+(\d+)\.(\d+)", "nvcc --version"),
    ]
    for cmd, pattern, source in probes:
        parsed, parsed_source = _run_version_probe(cmd, pattern, source)
        if parsed is not None:
            return parsed, parsed_source

    return None, "not detected"


def _detect_local_cuda_version() -> Tuple[Optional[Tuple[int, int]], str]:
    driver_cuda, driver_source = _detect_driver_cuda_capability()
    toolkit_cuda, toolkit_source = _detect_toolkit_cuda_version()

    if driver_cuda is not None:
        # Driver capability is authoritative for runtime compatibility.
        if toolkit_cuda is not None and toolkit_cuda > driver_cuda:
            return driver_cuda, f"{driver_source} (capped by driver; toolkit={_format_cuda_version(toolkit_cuda)})"
        return driver_cuda, f"{driver_source} (driver capability)"

    if toolkit_cuda is not None:
        return toolkit_cuda, f"{toolkit_source} (toolkit hint; driver capability unavailable)"

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
            if cpu_assets:
                # When compatibility cannot be detected, prefer CPU build over blind CUDA pick.
                return cpu_assets + ordered_cuda + other_assets
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


def _collect_matlab_bundle_files(package_root: Path) -> List[Path]:
    matlab_dir = package_root / "matlab"
    if not matlab_dir.is_dir():
        return []
    matlab_files = sorted([p for p in matlab_dir.rglob("*") if p.is_file()])
    # Remove README
    matlab_files = [p for p in matlab_files if p.name.lower() != "readme.txt"]
    return matlab_files


def _install_matlab_bundle_files(package_root: Path, repo_root: Path) -> List[Path]:
    bundle_files = _collect_matlab_bundle_files(package_root)
    if not bundle_files:
        _log("No MATLAB files found in bundle; skipping MATLAB MEX install")
        return []

    target_dir = repo_root / "external_programs"
    target_dir.mkdir(parents=True, exist_ok=True)

    installed: List[Path] = []
    for src in bundle_files:
        dst = target_dir / src.name
        if dst.exists() and dst.is_file():
            try:
                if _sha256(dst) == _sha256(src):
                    continue
            except Exception:
                pass
        shutil.copy2(src, dst)
        installed.append(dst)

    _log(f"Installed {len(installed)} MATLAB file(s) into {target_dir}")
    return installed


def _parse_matlab_symbol_probe_output(output: str) -> Dict[str, bool]:
    symbols: Dict[str, bool] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith(MATLAB_PROBE_PREFIX):
            continue
        payload = line[len(MATLAB_PROBE_PREFIX) :]
        if "=" not in payload:
            continue
        name, value = payload.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        symbols[name] = value in {"1", "true", "True"}
    return symbols


def _probe_matlab_symbols(repo_root: Path, matlab_cmd: str) -> Dict[str, bool]:
    if shutil.which(matlab_cmd) is None:
        return {}

    all_symbols = list(MATLAB_GPUFIT_SYMBOLS) + list(MATLAB_CPUFIT_SYMBOLS)
    symbol_list = " ".join([f"'{s}'" for s in all_symbols])
    repo_escaped = str(repo_root).replace("'", "''")
    batch_expr = (
        f"cd('{repo_escaped}'); "
        "addpath(fullfile(pwd,'external_programs')); "
        f"syms={{ {symbol_list} }}; "
        "for k=1:numel(syms), fprintf('"
        + MATLAB_PROBE_PREFIX
        + "%s=%d\\n', syms{k}, double(exist(syms{k},'file')>0)); end;"
    )

    result = subprocess.run(
        [matlab_cmd, "-batch", batch_expr],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    symbols = _parse_matlab_symbol_probe_output(combined)
    return symbols


def _log_matlab_symbol_status(symbols: Dict[str, bool]) -> None:
    if not symbols:
        _log("MATLAB symbol probe unavailable")
        return
    for name in list(MATLAB_GPUFIT_SYMBOLS) + list(MATLAB_CPUFIT_SYMBOLS):
        status = bool(symbols.get(name, False))
        _log(f"MATLAB symbol {name}: {'found' if status else 'missing'}")


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
            "from ironictoo/Gpufit release assets, plus optional MATLAB MEX files."
        )
    )
    parser.add_argument(
        "-R",
        "--repo",
        default=DEFAULT_GPUFIT_REPO,
        help="GitHub repo in owner/name form (default: ironictoo/Gpufit)",
    )
    parser.add_argument(
        "-t",
        "--release-tag",
        default=None,
        help=(
            "Release tag to install. (default: latest stable tag) \n"
            "Use '-t dev-latest' for the dev channel. "
            "For other releases, syntax is the exact tag string, for example '-t v1.4.1'"
        ),
    )
    parser.add_argument(
        "-a",
        "--asset-id",
        default=None,
        help=(
            "Explicit release asset id suffix (for example macos-arm64-cpu or "
            "linux-x64-cuda12.8). By default this is auto-detected."
        ),
    )
    parser.add_argument(
        "-e",
        "--venv-path",
        default=".venv",
        help="Virtual environment path (default: .venv in repo root).",
    )
    parser.add_argument(
        "-x",
        "--recreate-venv",
        action="store_true",
        help="Delete and recreate virtual environment before installing.",
    )
    parser.add_argument(
        "-G",
        "--no-gui",
        action="store_true",
        help="Skip requirements_gui.txt (GUI deps are installed by default).",
    )
    parser.add_argument(
        "-k",
        "--no-sha256",
        action="store_true",
        help="Disable SHA256 verification even if SHA256SUMS.txt is present.",
    )
    parser.add_argument(
        "-M",
        "--no-matlab",
        action="store_true",
        help="Skip installation/probe of MATLAB files from the release bundle.",
    )
    parser.add_argument(
        "-m",
        "--matlab-cmd",
        default="matlab",
        help="MATLAB command used for post-install symbol probing (default: matlab).",
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
            prefer_prerelease=False,
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

            if not args.no_sha256:
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
            if not args.no_matlab:
                _install_matlab_bundle_files(package_root, repo_root)

        _verify_install(python_in_venv)

        matlab_symbols: Dict[str, bool] = {}
        if args.no_matlab:
            _log("MATLAB MEX install skipped by user")
        else:
            matlab_symbols = _probe_matlab_symbols(repo_root, args.matlab_cmd)
            _log_matlab_symbol_status(matlab_symbols)
            require_matlab_gpufit = _require_matlab_gpufit_by_default(host_key)
            if not require_matlab_gpufit:
                _log("MATLAB Gpufit symbol requirement disabled for this install")
            if require_matlab_gpufit:
                missing = [name for name in MATLAB_GPUFIT_SYMBOLS if not bool(matlab_symbols.get(name, False))]
                if missing:
                    raise RuntimeError(
                        "Required MATLAB Gpufit symbols were not detected in external_programs: "
                        + ", ".join(missing)
                        + ". Update the release bundle to include MATLAB GPUfit MEX wrappers/binaries, "
                        "or rerun with --no-matlab to skip MATLAB install/probe."
                    )

        python_health = _probe_python_install_health(python_in_venv)

        if args.no_matlab:
            matlab_symbol_status = {"status": "SKIP", "detail": "MATLAB MEX install skipped"}
            matlab_runtime_status = {"status": "SKIP", "detail": "MATLAB MEX install skipped"}
        elif shutil.which(args.matlab_cmd) is None:
            matlab_symbol_status = {"status": "SKIP", "detail": f"'{args.matlab_cmd}' not found in PATH"}
            matlab_runtime_status = {"status": "SKIP", "detail": f"'{args.matlab_cmd}' not found in PATH"}
        else:
            missing_symbols = [name for name in MATLAB_GPUFIT_SYMBOLS if not bool(matlab_symbols.get(name, False))]
            if missing_symbols and _require_matlab_gpufit_by_default(host_key):
                matlab_symbol_status = {"status": "FAIL", "detail": "missing: " + ", ".join(missing_symbols)}
            elif missing_symbols:
                matlab_symbol_status = {
                    "status": "SKIP",
                    "detail": "missing on macOS (expected for current release assets): " + ", ".join(missing_symbols),
                }
            else:
                matlab_symbol_status = {"status": "PASS", "detail": "all required MATLAB Gpufit symbols found"}
            matlab_runtime_status = _probe_matlab_runtime_gpufit(repo_root, args.matlab_cmd)

        _log("Installation complete.")
        _log(f"Virtual environment: {_venv_root_from_python(python_in_venv)}")
        _print_post_install_health_check(python_health, matlab_symbol_status, matlab_runtime_status)
        _print_post_install_next_steps(repo_root, python_in_venv)
        return 0

    except Exception as exc:
        _log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
