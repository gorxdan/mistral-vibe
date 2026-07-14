from __future__ import annotations

from functools import lru_cache
import os

from vibe.core._trusted_command import (
    TrustedCommandError,
    resolve_trusted_system_executable,
    trusted_system_path,
)
from vibe.core.utils import is_windows

_UNSAFE_SHELL_ENV_NAMES = frozenset({
    "BASHOPTS",
    "BASH_COMPAT",
    "BASH_ENV",
    "BASH_XTRACEFD",
    "DYLD_INSERT_LIBRARIES",
    "ENV",
    "LD_AUDIT",
    "LD_PRELOAD",
    "LESSKEY",
    "LESSKEYIN",
    "LESSKEYIN_SYSTEM",
    "LESSKEY_CONTENT",
    "LESSKEY_SYSTEM",
    "LESSCLOSE",
    "LESSOPEN",
    "LESSSECURE_ALLOW",
    "POSIXLY_CORRECT",
    "POSIX_PEDANTIC",
    "PS4",
    "SHELLOPTS",
})


@lru_cache(maxsize=1)
def get_bash_executable() -> str | None:
    if is_windows():
        return None
    try:
        return str(resolve_trusted_system_executable("bash"))
    except TrustedCommandError:
        return None


def get_trusted_system_path() -> str:
    return os.defpath if is_windows() else trusted_system_path()


def get_autoapproved_shell_env(
    *, home: str = "/nonexistent", tmpdir: str = "/tmp"
) -> dict[str, str]:
    environment = {
        "PATH": get_trusted_system_path(),
        "HOME": home,
        "TMPDIR": tmpdir,
        "CI": "true",
        "NONINTERACTIVE": "1",
        "NO_TTY": "1",
    }
    if is_windows():
        environment.update({"GIT_PAGER": "more", "PAGER": "more"})
        return environment
    if shell_executable := get_bash_executable():
        environment["SHELL"] = shell_executable
    environment.update({
        "TERM": "dumb",
        "DEBIAN_FRONTEND": "noninteractive",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "PAGER": "cat",
        "LESS": "-FX",
        "LESSSECURE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    })
    return environment


def get_base_shell_env() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _UNSAFE_SHELL_ENV_NAMES and not name.startswith("BASH_FUNC_")
    }
    environment.update({"CI": "true", "NONINTERACTIVE": "1", "NO_TTY": "1"})

    if is_windows():
        environment["GIT_PAGER"] = "more"
        environment["PAGER"] = "more"
        return environment

    if shell_executable := get_bash_executable():
        environment["SHELL"] = shell_executable
    environment.update({
        "TERM": "dumb",
        "DEBIAN_FRONTEND": "noninteractive",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "LESS": "-FX",
        "LESSSECURE": "1",
        "LC_ALL": "en_US.UTF-8",
    })
    return environment
