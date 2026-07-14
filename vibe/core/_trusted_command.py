from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path, PurePath
import shutil

TRUSTED_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
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


def resolve_trusted_system_executable(name: str) -> Path:
    requested = Path(name)
    if not name or "\0" in name or requested.parent != Path("."):
        raise TrustedCommandError(
            f"trusted system executable must be a bare name: {name!r}"
        )
    executable = shutil.which(name, path=TRUSTED_SYSTEM_PATH)
    if executable is None:
        raise TrustedCommandError(
            "trusted system executable is unavailable on the sanitized system "
            f"PATH: {name}"
        )
    try:
        resolved = Path(executable).resolve(strict=True)
    except OSError as exc:
        raise TrustedCommandError(
            f"trusted system executable could not be resolved: {name}"
        ) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise TrustedCommandError(
            f"trusted system executable is not an executable file: {resolved}"
        )
    return resolved


def minimal_trusted_git_environment(home: Path) -> dict[str, str]:
    if not home.is_absolute():
        raise TrustedCommandError("trusted Git HOME must be an absolute path")
    return {
        "PATH": TRUSTED_SYSTEM_PATH,
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
    "validate_trusted_command_argv",
]
