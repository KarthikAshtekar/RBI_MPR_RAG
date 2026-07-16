from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from .artifact_io import file_sha, now_iso, rel_path, write_json


POPPLER_RETRY_OUT = Path("reports/v2_unstructured_cohere/poppler_retry")
POPPLER_HELPER_JSON = "poppler_path_helper.json"
POPPLER_HELPER_PS1 = "poppler_path_helper.ps1"
POPPLER_COMMANDS = ("pdfinfo", "pdftoppm", "pdftotext")
POPPLER_VERIFY_COMMANDS = (
    ("pdfinfo", "-v"),
    ("pdftoppm", "-h"),
)
POPPLER_STATUS_COMMANDS = (
    ("pdfinfo", "-v"),
    ("pdftoppm", "-h"),
    ("pdftotext", "-v"),
)
POPPLER_CHECKSUM_TARGETS = [
    Path("reports/v2_unstructured_cohere"),
    Path("reports/v2_generation"),
    Path("reports/v2_sufficiency"),
    Path("reports/final_comparison"),
    Path("configs/v2_selected_retrieval.yaml"),
    Path("data/evaluation"),
    Path("data/raw"),
]


def safe_text(value: str | None, *, limit: int = 4000) -> str | None:
    if value is None:
        return None
    text = value.replace("\x00", "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def poppler_retry_dir(root: Path) -> Path:
    return root / POPPLER_RETRY_OUT


def common_poppler_search_roots(root: Path) -> list[Path]:
    user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
    local_app_data = Path(os.environ.get("LOCALAPPDATA", str(user_profile / "AppData" / "Local")))
    return [
        Path(r"C:\ProgramData\chocolatey\lib\poppler"),
        Path(r"C:\ProgramData\chocolatey\bin"),
        Path(r"C:\Program Files\poppler"),
        Path(r"C:\Program Files (x86)\poppler"),
        user_profile / "scoop" / "apps" / "poppler",
        local_app_data / "Microsoft" / "WinGet" / "Packages",
        local_app_data / "Microsoft" / "WinGet" / "Links",
        local_app_data / "Microsoft" / "WindowsApps",
        Path(r"C:\tools\poppler"),
        root / "tools" / "poppler",
    ]


def path_entries(env: dict[str, str] | None = None) -> list[str]:
    env = env or os.environ
    return [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]


def poppler_relevant_path_entries(env: dict[str, str] | None = None) -> list[str]:
    relevant: list[str] = []
    for entry in path_entries(env):
        lowered = entry.lower()
        entry_path = Path(entry)
        if "poppler" in lowered or (entry_path / "pdfinfo.exe").exists():
            relevant.append(entry)
    return relevant


def command_available(name: str, env: dict[str, str] | None = None) -> str | None:
    return shutil.which(name, path=(env or os.environ).get("PATH"))


def run_command_status(
    command: tuple[str, ...],
    *,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 30,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    runner = runner or subprocess.run
    executable = command_available(command[0], env)
    payload: dict[str, Any] = {
        "command": " ".join(command),
        "executable_path": executable,
        "available": executable is not None,
        "returncode": None,
        "stdout": None,
        "stderr": None,
        "error_type": None,
        "error_message": None,
    }
    if executable is None:
        payload["error_type"] = "ExecutableNotFound"
        payload["error_message"] = f"{command[0]} was not found on PATH"
        return payload
    try:
        completed = runner(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        payload.update(
            {
                "returncode": completed.returncode,
                "stdout": safe_text(completed.stdout),
                "stderr": safe_text(completed.stderr),
                "success": completed.returncode == 0,
            }
        )
    except Exception as exc:  # pragma: no cover - exact subprocess exceptions vary by platform
        payload.update(
            {
                "available": False,
                "error_type": type(exc).__name__,
                "error_message": safe_text(str(exc)),
                "success": False,
            }
        )
    return payload


def poppler_status(
    root: Path,
    *,
    commands: Iterable[tuple[str, ...]] = POPPLER_STATUS_COMMANDS,
    env: dict[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    env = dict(env or os.environ)
    command_checks = [run_command_status(command, env=env, runner=runner) for command in commands]
    python_which = {name: command_available(name, env) for name in POPPLER_COMMANDS}
    required = ("pdfinfo", "pdftoppm")
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "platform": "Windows",
        "command_checks": command_checks,
        "python_which": python_which,
        "path_relevant_entries": poppler_relevant_path_entries(env),
        "active_python_can_see_poppler": all(python_which.get(name) for name in required),
        "poppler_ready": all(check.get("available") and check.get("returncode") == 0 for check in command_checks if check["command"].split()[0] in required),
    }


def write_poppler_status(root: Path, name: str, payload: dict[str, Any]) -> None:
    out = poppler_retry_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{name}.json", payload)
    lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        f"Poppler ready: {payload.get('poppler_ready')}",
        f"Active Python can see Poppler: {payload.get('active_python_can_see_poppler')}",
        "",
        "## Python executable discovery",
        "",
    ]
    for command, detected in sorted((payload.get("python_which") or {}).items()):
        lines.append(f"- {command}: `{detected}`")
    lines += ["", "## Command checks", ""]
    for check in payload.get("command_checks", []):
        lines.append(
            f"- `{check['command']}`: available={check.get('available')}, "
            f"returncode={check.get('returncode')}, executable=`{check.get('executable_path')}`"
        )
        version_text = (check.get("stdout") or check.get("stderr") or "").strip().splitlines()
        if version_text:
            lines.append(f"  - first output line: `{version_text[0]}`")
        if check.get("error_message"):
            lines.append(f"  - error: {check.get('error_message')}")
    lines += ["", "## PATH entries relevant to Poppler", ""]
    entries = payload.get("path_relevant_entries") or []
    if entries:
        lines.extend(f"- `{entry}`" for entry in entries)
    else:
        lines.append("- None detected.")
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_poppler_bin(search_roots: Iterable[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if (root / "pdfinfo.exe").exists():
            candidates.append(root)
        if root.is_dir():
            try:
                for pdfinfo in root.rglob("pdfinfo.exe"):
                    candidates.append(pdfinfo.parent)
            except OSError:
                continue
    for candidate in candidates:
        if (candidate / "pdfinfo.exe").exists() and (candidate / "pdftoppm.exe").exists():
            return candidate
    return candidates[0] if candidates else None


def prepend_path(env: dict[str, str], bin_path: Path) -> dict[str, str]:
    updated = dict(env)
    current = path_entries(updated)
    bin_str = str(bin_path)
    without_duplicate = [entry for entry in current if entry.lower() != bin_str.lower()]
    updated["PATH"] = os.pathsep.join([bin_str, *without_duplicate])
    return updated


def write_poppler_helper(root: Path, bin_path: Path) -> None:
    out = poppler_retry_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "poppler_bin_path": str(bin_path),
        "instructions": "Source poppler_path_helper.ps1 or rerun repository scripts; the V2 runner reads this helper automatically.",
    }
    write_json(out / POPPLER_HELPER_JSON, payload)
    escaped = str(bin_path).replace("'", "''")
    ps1 = (
        "# Project-local Poppler PATH helper generated by setup_poppler_for_unstructured.py\n"
        f"$env:PATH = '{escaped}' + [IO.Path]::PathSeparator + $env:PATH\n"
        "pdfinfo -v\n"
    )
    (out / POPPLER_HELPER_PS1).write_text(ps1, encoding="utf-8")


def apply_poppler_path_from_helper(root: Path, env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    helper = poppler_retry_dir(root) / POPPLER_HELPER_JSON
    if not helper.exists():
        return False
    try:
        payload = json.loads(helper.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    bin_path = Path(payload.get("poppler_bin_path", ""))
    if not (bin_path / "pdfinfo.exe").exists():
        return False
    updated = prepend_path(dict(env), bin_path)
    env["PATH"] = updated["PATH"]
    return True


def package_manager_status(env: dict[str, str] | None = None) -> dict[str, str | None]:
    env = env or os.environ
    return {name: command_available(name, env) for name in ("choco", "winget", "scoop", "conda")}


def install_commands_for_available_managers(managers: dict[str, str | None]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    if managers.get("choco"):
        commands.append({"manager": "chocolatey", "command": ["choco", "install", "poppler", "-y"], "requires_conda": False})
    if managers.get("winget"):
        commands.append(
            {
                "manager": "winget",
                "command": [
                    "winget",
                    "install",
                    "oschwartz10612.Poppler",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
                "requires_conda": False,
            }
        )
    if managers.get("scoop"):
        commands.append({"manager": "scoop", "command": ["scoop", "install", "poppler"], "requires_conda": False})
    if managers.get("conda") and os.environ.get("CONDA_PREFIX"):
        commands.append({"manager": "conda", "command": ["conda", "install", "-c", "conda-forge", "poppler", "-y"], "requires_conda": True})
    return commands


def manual_installation_instructions() -> list[str]:
    return [
        "Install Poppler for Windows, recommended: oschwartz10612 Poppler package via winget.",
        "Command: winget install oschwartz10612.Poppler --accept-package-agreements --accept-source-agreements",
        "Alternative: install a Poppler Windows binary and locate its Library\\bin or bin folder.",
        "Add the folder containing pdfinfo.exe and pdftoppm.exe to PATH.",
        "Open a new PowerShell and verify: pdfinfo -v; pdftoppm -h.",
        "Then rerun: python scripts\\setup_poppler_for_unstructured.py",
    ]


def write_pre_poppler_retry_checksums(root: Path) -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    missing_targets: list[str] = []
    excluded = str(POPPLER_RETRY_OUT).replace("/", "\\")
    retry_abs = (root / POPPLER_RETRY_OUT).resolve()
    for target in POPPLER_CHECKSUM_TARGETS:
        path = root / target
        if path.is_file():
            entries.append({"path": rel_path(path, root), "sha256": file_sha(path)})
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file():
                    continue
                try:
                    child.relative_to(retry_abs)
                    continue
                except ValueError:
                    pass
                entries.append({"path": rel_path(child, root), "sha256": file_sha(child)})
        else:
            missing_targets.append(str(target).replace("/", "\\"))
    payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "targets": [str(target).replace("/", "\\") for target in POPPLER_CHECKSUM_TARGETS],
        "excluded_paths": [excluded],
        "entry_count": len(entries),
        "missing_targets": missing_targets,
        "entries": entries,
    }
    out = poppler_retry_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "pre_poppler_retry_checksums.json", payload)
    lines = [
        "# Pre-Poppler Retry Checksums",
        "",
        f"Entry count: {payload['entry_count']}",
        f"Missing targets: {len(missing_targets)}",
        "",
        "The Poppler retry output directory is excluded to avoid self-referential checksums.",
    ]
    if missing_targets:
        lines += ["", "## Missing targets", ""]
        lines.extend(f"- `{target}`" for target in missing_targets)
    (out / "pre_poppler_retry_checksums.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def setup_poppler(root: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> dict[str, Any]:
    root = root.resolve()
    out = poppler_retry_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    checksum_manifest = write_pre_poppler_retry_checksums(root)

    before = poppler_status(root, runner=runner)
    write_poppler_status(root, "poppler_status_before", before)

    attempts: list[dict[str, Any]] = []
    env = dict(os.environ)
    method = "already_available" if before["poppler_ready"] else None
    configured_bin: str | None = None
    if before["poppler_ready"]:
        detected = before.get("python_which", {}).get("pdfinfo")
        if detected:
            configured_bin = str(Path(detected).parent)
            write_poppler_helper(root, Path(configured_bin))

    if not before["poppler_ready"]:
        found = find_poppler_bin(common_poppler_search_roots(root))
        attempts.append(
            {
                "method": "search_existing_install",
                "searched_roots": [str(path) for path in common_poppler_search_roots(root)],
                "found_bin": str(found) if found else None,
                "success": found is not None,
            }
        )
        if found:
            env = prepend_path(env, found)
            os.environ["PATH"] = env["PATH"]
            write_poppler_helper(root, found)
            configured_bin = str(found)
            method = "configured_existing_install"

    if method is None:
        managers = package_manager_status(env)
        commands = install_commands_for_available_managers(managers)
        if not commands:
            attempts.append({"method": "package_manager_detection", "available": managers, "success": False})
        for spec in commands:
            command = spec["command"]
            attempt = {
                "method": spec["manager"],
                "command": " ".join(command),
                "success": False,
                "returncode": None,
                "stdout": None,
                "stderr": None,
                "error_type": None,
                "error_message": None,
            }
            try:
                completed = (runner or subprocess.run)(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env,
                )
                attempt.update(
                    {
                        "returncode": completed.returncode,
                        "stdout": safe_text(completed.stdout),
                        "stderr": safe_text(completed.stderr),
                        "success": completed.returncode == 0,
                    }
                )
            except Exception as exc:  # pragma: no cover - depends on local package manager
                attempt.update({"error_type": type(exc).__name__, "error_message": safe_text(str(exc))})
            attempts.append(attempt)
            found = find_poppler_bin(common_poppler_search_roots(root))
            if found:
                env = prepend_path(env, found)
                os.environ["PATH"] = env["PATH"]
                write_poppler_helper(root, found)
                configured_bin = str(found)
                method = f"installed_or_configured_via_{spec['manager']}"
                break
            fresh = poppler_status(root, commands=POPPLER_VERIFY_COMMANDS, env=env, runner=runner)
            if fresh["poppler_ready"]:
                method = f"installed_via_{spec['manager']}"
                detected = fresh.get("python_which", {}).get("pdfinfo")
                configured_bin = str(Path(detected).parent) if detected else None
                if configured_bin:
                    write_poppler_helper(root, Path(configured_bin))
                break

    after = poppler_status(root, commands=POPPLER_VERIFY_COMMANDS, env=env, runner=runner)
    write_poppler_status(root, "poppler_status_after", after)

    blocked_reason = None
    if not after["poppler_ready"]:
        blocked_reason = "Poppler verification failed; pdfinfo and pdftoppm are not both callable from the active Python subprocess environment."

    install_payload = {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "status": "ready" if after["poppler_ready"] else "blocked",
        "method": method or "not_configured",
        "configured_bin": configured_bin,
        "attempts": attempts,
        "manual_installation_instructions": manual_installation_instructions() if blocked_reason else [],
        "blocked_reason": blocked_reason,
        "checksum_entry_count": checksum_manifest["entry_count"],
    }
    write_json(out / "poppler_install_attempts.json", install_payload)
    lines = [
        "# Poppler Installation / Configuration Attempts",
        "",
        f"Status: {install_payload['status']}",
        f"Method: {install_payload['method']}",
        f"Configured bin: `{configured_bin}`",
        "",
        "## Attempts",
        "",
    ]
    if attempts:
        for attempt in attempts:
            lines.append(f"- {attempt.get('method')}: success={attempt.get('success')}, returncode={attempt.get('returncode')}")
            if attempt.get("error_message"):
                lines.append(f"  - error: {attempt['error_message']}")
    else:
        lines.append("- No install attempt required.")
    if blocked_reason:
        lines += ["", "## Blocked reason", "", blocked_reason, "", "## Manual installation instructions", ""]
        lines.extend(f"- {item}" for item in manual_installation_instructions())
    (out / "poppler_install_attempts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": install_payload["status"],
        "method": install_payload["method"],
        "poppler_ready_before": before["poppler_ready"],
        "poppler_ready_after": after["poppler_ready"],
        "configured_bin": configured_bin,
        "blocked_reason": blocked_reason,
        "outputs": {
            "pre_checksums": str(out / "pre_poppler_retry_checksums.json"),
            "status_before": str(out / "poppler_status_before.json"),
            "install_attempts": str(out / "poppler_install_attempts.json"),
            "status_after": str(out / "poppler_status_after.json"),
        },
    }
