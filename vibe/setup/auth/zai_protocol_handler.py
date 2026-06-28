from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Literal

from vibe.core.logger import logger
from vibe.core.utils.io import read_safe

ZaiProtocolHandlerStatus = Literal[
    "installed",
    "already_configured",
    "existing_handler",
    "missing_chaton",
    "missing_xdg_mime",
    "unsupported",
    "failed",
]

_DESKTOP_FILE_NAME = "chaton-zcode-handler.desktop"
_MIME_TYPE = "x-scheme-handler/zcode"


@dataclass(frozen=True, slots=True)
class ZaiProtocolHandlerInstallResult:
    status: ZaiProtocolHandlerStatus
    handler: str | None = None
    path: Path | None = None
    error: str | None = None


def install_zai_protocol_handler() -> ZaiProtocolHandlerInstallResult:
    if sys.platform != "linux":
        return ZaiProtocolHandlerInstallResult(status="unsupported")
    xdg_mime = shutil.which("xdg-mime")
    if xdg_mime is None:
        return ZaiProtocolHandlerInstallResult(status="missing_xdg_mime")
    chaton = _resolve_chaton_binary()
    if chaton is None:
        return ZaiProtocolHandlerInstallResult(status="missing_chaton")

    current = _query_current_handler(xdg_mime)
    if current and current != _DESKTOP_FILE_NAME:
        return ZaiProtocolHandlerInstallResult(
            status="existing_handler", handler=current
        )

    desktop_path = _write_desktop_file(chaton)
    return _register_handler(xdg_mime, current, desktop_path)


def _register_handler(
    xdg_mime: str, current: str | None, desktop_path: Path
) -> ZaiProtocolHandlerInstallResult:
    try:
        subprocess.run(
            [xdg_mime, "default", _DESKTOP_FILE_NAME, _MIME_TYPE],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_desktop_env(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.debug("Failed to register Z.ai protocol handler", exc_info=True)
        return ZaiProtocolHandlerInstallResult(
            status="failed", handler=current, path=desktop_path, error=str(exc)
        )
    try:
        _ensure_mimeapps_entries()
    except OSError as exc:
        logger.debug("Failed to write Z.ai MIME app association", exc_info=True)
        return ZaiProtocolHandlerInstallResult(
            status="failed", handler=current, path=desktop_path, error=str(exc)
        )
    return ZaiProtocolHandlerInstallResult(
        status="already_configured" if current == _DESKTOP_FILE_NAME else "installed",
        handler=_DESKTOP_FILE_NAME,
        path=desktop_path,
    )


def _resolve_chaton_binary() -> str | None:
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.exists() and argv0.name == "chaton":
        return str(argv0.resolve())
    if chaton := shutil.which("chaton"):
        return chaton
    return None


def _query_current_handler(xdg_mime: str) -> str | None:
    try:
        result = subprocess.run(
            [xdg_mime, "query", "default", _MIME_TYPE],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=_desktop_env(),
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    handler = result.stdout.strip()
    return handler or None


def _write_desktop_file(chaton: str) -> Path:
    applications_dir = _applications_dir()
    applications_dir.mkdir(parents=True, exist_ok=True)
    desktop_path = applications_dir / _DESKTOP_FILE_NAME
    desktop_path.write_text(_desktop_entry(chaton), encoding="utf-8")
    return desktop_path


def _applications_dir() -> Path:
    return _data_home() / "applications"


def _desktop_entry(chaton: str) -> str:
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Name=Chaton Z.ai Callback",
        f"Exec={_desktop_exec_arg(chaton)} --zai-callback %u",
        f"MimeType={_MIME_TYPE};",
        "NoDisplay=true",
        "Terminal=false",
        "",
    ])


def _desktop_exec_arg(value: str) -> str:
    if value and not any(ch.isspace() or ch in '"\\`$' for ch in value):
        return value
    escaped = (
        value
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
    )
    return f'"{escaped}"'


def _ensure_mimeapps_entries() -> None:
    path = _mimeapps_path()
    try:
        text = read_safe(path).text
    except OSError:
        text = ""
    text = _upsert_mimeapps_entry(
        text, "Default Applications", _MIME_TYPE, _DESKTOP_FILE_NAME, list_value=False
    )
    text = _upsert_mimeapps_entry(
        text, "Added Associations", _MIME_TYPE, _DESKTOP_FILE_NAME, list_value=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mimeapps_path() -> Path:
    return _config_home() / "mimeapps.list"


def _desktop_env() -> dict[str, str]:
    env = os.environ.copy()
    env["XDG_DATA_HOME"] = str(_data_home())
    env["XDG_CONFIG_HOME"] = str(_config_home())
    return env


def _data_home() -> Path:
    return _host_xdg_path("XDG_DATA_HOME", Path.home() / ".local" / "share")


def _config_home() -> Path:
    return _host_xdg_path("XDG_CONFIG_HOME", Path.home() / ".config")


def _host_xdg_path(env_var: str, fallback: Path) -> Path:
    if value := os.getenv(env_var):
        path = Path(value).expanduser()
        if not _is_snap_scoped_path(path):
            return path
    return fallback


def _is_snap_scoped_path(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(Path.home() / "snap")
    except ValueError:
        return False
    return True


def _upsert_mimeapps_entry(
    text: str, section: str, key: str, value: str, *, list_value: bool
) -> str:
    lines = text.splitlines()
    section_line = f"[{section}]"
    try:
        section_index = lines.index(section_line)
    except ValueError:
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend([section_line, _mimeapps_line(key, value, list_value=list_value)])
        return "\n".join(lines) + "\n"

    end_index = len(lines)
    for index in range(section_index + 1, len(lines)):
        if lines[index].startswith("[") and lines[index].endswith("]"):
            end_index = index
            break

    prefix = f"{key}="
    for index in range(section_index + 1, end_index):
        if not lines[index].startswith(prefix):
            continue
        lines[index] = _merged_mimeapps_line(
            key, lines[index].removeprefix(prefix), value, list_value=list_value
        )
        return "\n".join(lines) + "\n"

    lines.insert(section_index + 1, _mimeapps_line(key, value, list_value=list_value))
    return "\n".join(lines) + "\n"


def _mimeapps_line(key: str, value: str, *, list_value: bool) -> str:
    return f"{key}={value};" if list_value else f"{key}={value}"


def _merged_mimeapps_line(
    key: str, current: str, value: str, *, list_value: bool
) -> str:
    if not list_value:
        return _mimeapps_line(key, value, list_value=False)
    values = [part for part in current.split(";") if part]
    if value in values:
        values.remove(value)
    values.insert(0, value)
    return f"{key}={';'.join(values)};"
