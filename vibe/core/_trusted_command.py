from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path, PurePath
import shutil
import stat
import sys

TRUSTED_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_EMPTY_TRUSTED_SYSTEM_PATH = "/__vibe_no_trusted_executables__"
TRUSTED_GIT_CONFIG_ARGS = (
    "-c",
    "color.ui=false",
    "-c",
    "commit.gpgSign=false",
    "-c",
    "core.attributesFile=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.pager=cat",
    "-c",
    "diff.external=",
    "-c",
    "log.showSignature=false",
    "-c",
    "tag.gpgSign=false",
)

_SHELL_EXECUTABLES = frozenset({
    "bash",
    "cmd",
    "cmd.exe",
    "dash",
    "fish",
    "ksh",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "sh",
    "zsh",
})
_COMMAND_SPLITTING_WRAPPERS = frozenset({"env", "env.exe"})


class TrustedCommandError(ValueError):
    pass


def _current_user_can_write(path: Path) -> bool:
    return os.geteuid() != 0 and os.access(path, os.W_OK, effective_ids=True)


def _validate_linux_directory(path: Path) -> None:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise TrustedCommandError(
            f"trusted system path component could not be inspected: {path}"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or _current_user_can_write(path)
    ):
        raise TrustedCommandError(
            f"trusted system path component is not root-controlled: {path}"
        )


def _validate_linux_directory_ancestry(directory: Path) -> None:
    current = Path(directory.anchor)
    _validate_linux_directory(current)
    for part in directory.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise TrustedCommandError(
                f"trusted system path component could not be inspected: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            continue
        _validate_linux_directory(current)


def _linux_executable_is_untrusted(path: Path, metadata: os.stat_result) -> bool:
    unsafe_mode = stat.S_IWGRP | stat.S_IWOTH | stat.S_ISUID | stat.S_ISGID
    return (
        metadata.st_uid != 0
        or bool(metadata.st_mode & unsafe_mode)
        or _current_user_can_write(path)
    )


def validate_trusted_system_executable(path: str | Path) -> Path:
    lexical = Path(path)
    if not lexical.is_absolute():
        raise TrustedCommandError(
            f"trusted system executable must be an absolute path: {path!r}"
        )
    if sys.platform.startswith("linux"):
        _validate_linux_directory_ancestry(lexical.parent)
    try:
        resolved = lexical.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise TrustedCommandError(
            f"trusted system executable could not be resolved: {path}"
        ) from exc
    if sys.platform.startswith("linux"):
        _validate_linux_directory_ancestry(resolved.parent)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or not os.access(resolved, os.X_OK)
        or sys.platform.startswith("linux")
        and _linux_executable_is_untrusted(resolved, metadata)
    ):
        raise TrustedCommandError(
            f"trusted system executable is not a root-controlled executable: {resolved}"
        )
    return resolved


def trusted_system_path() -> str:
    if not sys.platform.startswith("linux"):
        return TRUSTED_SYSTEM_PATH
    directories: list[str] = []
    for candidate in TRUSTED_SYSTEM_PATH.split(os.pathsep):
        try:
            lexical = Path(candidate)
            _validate_linux_directory_ancestry(lexical)
            resolved = lexical.resolve(strict=True)
            _validate_linux_directory_ancestry(resolved)
        except (OSError, TrustedCommandError):
            continue
        rendered = str(resolved)
        if rendered not in directories:
            directories.append(rendered)
    return os.pathsep.join(directories) if directories else _EMPTY_TRUSTED_SYSTEM_PATH


def resolve_trusted_system_executable(name: str) -> Path:
    separators = tuple(separator for separator in (os.sep, os.altsep) if separator)
    if not name or "\0" in name or any(item in name for item in separators):
        raise TrustedCommandError(
            f"trusted system executable must be a bare name: {name!r}"
        )
    executable = shutil.which(name, path=trusted_system_path())
    if executable is None:
        raise TrustedCommandError(
            "trusted system executable is unavailable on the sanitized system "
            f"PATH: {name}"
        )
    return validate_trusted_system_executable(executable)


def minimal_trusted_git_environment(home: Path) -> dict[str, str]:
    if not home.is_absolute():
        raise TrustedCommandError("trusted Git HOME must be an absolute path")
    return {
        "PATH": trusted_system_path(),
        "HOME": str(home.resolve()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def validate_trusted_command_argv(argv: Sequence[str]) -> None:
    executable = PurePath(argv[0]).name.casefold()
    if executable in _SHELL_EXECUTABLES:
        raise ValueError(
            f"trusted verification checks cannot invoke a shell: {argv[0]!r}"
        )
    if executable in _COMMAND_SPLITTING_WRAPPERS:
        raise ValueError(
            "trusted verification checks cannot invoke a command-splitting "
            f"wrapper: {argv[0]!r}"
        )
    if executable != "uv" or "run" not in argv[1:]:
        return

    run_index = argv.index("run", 1)
    wrapper = next(
        (
            argument
            for argument in argv[run_index + 1 :]
            if argument.casefold() in _COMMAND_SPLITTING_WRAPPERS
            or (
                PurePath(argument).is_absolute()
                and PurePath(argument).name.casefold() in _COMMAND_SPLITTING_WRAPPERS
            )
        ),
        None,
    )
    if wrapper is not None:
        raise ValueError(
            "trusted verification checks cannot invoke a command-splitting "
            f"wrapper: {wrapper!r}"
        )
    shell = next(
        (
            argument
            for argument in argv[run_index + 1 :]
            if argument.casefold() in _SHELL_EXECUTABLES
            or (
                PurePath(argument).is_absolute()
                and PurePath(argument).name.casefold() in _SHELL_EXECUTABLES
            )
        ),
        None,
    )
    if shell is not None:
        raise ValueError(
            f"trusted verification checks cannot invoke a shell: {shell!r}"
        )


__all__ = [
    "TRUSTED_GIT_CONFIG_ARGS",
    "TRUSTED_SYSTEM_PATH",
    "TrustedCommandError",
    "minimal_trusted_git_environment",
    "resolve_trusted_system_executable",
    "trusted_system_path",
    "validate_trusted_command_argv",
    "validate_trusted_system_executable",
]
