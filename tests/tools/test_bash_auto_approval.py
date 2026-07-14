from __future__ import annotations

import pytest

from vibe.core.tools.base import BaseToolState, ToolPermission
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.permissions import PermissionContext


def _bash() -> Bash:
    config = BashToolConfig(allowlist=["cat", "echo", "grep", "printf", "pwd"])
    return Bash(config_getter=lambda: config, state=BaseToolState())


@pytest.mark.parametrize(
    "command",
    [
        "> out",
        "2> errors",
        "<> data",
        "echo hi > out",
        "echo hi >> out",
        "echo hi >| out",
        "echo hi &> out",
        "echo hi 2>&1",
        "echo hi <> data",
        "echo hi && pwd",
        "echo hi || pwd",
        "echo hi ; pwd",
        "echo hi | cat",
        "echo hi &",
        "! echo hi",
        "(echo hi)",
        "echo $(whoami)",
        "echo hi\npwd",
        "echo hi &&",
    ],
)
def test_shell_composition_and_output_redirects_are_not_auto_approved(
    command: str,
) -> None:
    permission = _bash().resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK


@pytest.mark.parametrize(
    "command",
    [
        "echo 'a|b'",
        "echo '$(literal)'",
        "echo a\\|b",
        "printf '%s' 'x;y'",
        "printf '%s' 'x>y'",
        "grep -F '$(literal)' README.md",
    ],
)
def test_quoted_and_escaped_operator_data_remains_auto_approved(command: str) -> None:
    permission = _bash().resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "sort -o pyproject.toml README.md",
        "sort --output=pyproject.toml README.md",
        "tree -o pyproject.toml .",
        "less -O pyproject.toml README.md",
        "less --LOG-FILE=pyproject.toml README.md",
        "date --set=2020-01-01",
        "date -s2030-01-01",
        "diff --output=report.patch before after",
        "uniq input.txt output.txt",
    ],
)
def test_effectful_read_command_modes_are_not_auto_approved(command: str) -> None:
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK


@pytest.mark.parametrize(
    "command",
    [
        "cat < /etc/passwd",
        "head < /home/dan/.ssh/config",
        "grep needle < /home/dan/.ssh/config",
        "sort < /home/dan/.ssh/config",
    ],
)
def test_outside_input_redirects_require_directory_permission(
    command: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert any(
        required.scope.value == "outside_directory"
        for required in permission.required_permissions
    )


@pytest.mark.parametrize(
    "command",
    [
        'cat "/etc/passwd"',
        'head "/home/dan/.ssh/config"',
        'grep needle "/home/dan/.ssh/config"',
        'diff /dev/null "/home/dan/.ssh/config"',
        'tree "/home/dan/.ssh"',
        'od -c "/home/dan/.ssh/config"',
        "grep -f/home/dan/.ssh/config README.md",
        "file -m/home/dan/.ssh/config README.md",
        "sort -T/tmp README.md",
        "sort -T.. README.md",
        "date --file=/etc/passwd",
        "date -f /etc/passwd",
        "date --reference=/etc/passwd",
        "cat ~-",
        "cat ~-/passwd",
        "cat ~nosuchvibeuser/file",
    ],
)
def test_static_outside_paths_require_directory_permission(
    command: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert any(
        required.scope.value == "outside_directory"
        for required in permission.required_permissions
    )


@pytest.mark.parametrize(
    "command",
    [
        "cat < $'\\x2fetc\\x2fpasswd'",
        "cat < ${HOME:0:1}etc${HOME:0:1}passwd",
        "date -f$HOME",
        "file -m$HOME",
        "grep -f$HOME needle",
        "date -f${HOME:0:1}etc${HOME:0:1}passwd",
        "file -m${HOME:0:1}etc${HOME:0:1}passwd",
        "cat $'\\x2fetc\\x2fpasswd'",
        "head $'\\057home\\057dan\\057.ssh\\057config'",
        "grep needle $'\\u002fetc\\u002fpasswd'",
        "less $'\\U0000002fetc\\U0000002fpasswd'",
        "cat {..,foo}/secret",
        "cat {/,./}etc/passwd",
        "cat {,/etc}/passwd",
        "head {,/home/dan/.ssh}/config",
        "head p$'\\057..\\057..\\057..\\057.ssh\\057config'",
        "cat pre$'\\u002f..\\u002f..\\u002fetc\\u002fpasswd'",
        "cat $HOME",
        "cat ${HOME}",
        "less $_",
        "cat $0",
        "cat ${HOME:0:1}etc${HOME:0:1}passwd",
        "cat *",
        "cat ./*",
        "less ?",
        "grep needle **/*",
        "cat [a-z]*",
    ],
)
def test_dynamic_outside_paths_require_explicit_user_approval(
    command: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True
    assert permission.required_permissions == []


@pytest.mark.parametrize("command", ["cat ~+", "cat ~+/README.md"])
def test_current_directory_tilde_does_not_crash_or_escape(
    command: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert not (
        isinstance(permission, PermissionContext)
        and any(
            required.scope.value == "outside_directory"
            for required in permission.required_permissions
        )
    )


def test_expanded_home_path_requires_explicit_user_approval(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "host-home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(
        BashArgs(command='cat "$HOME/.ssh/config"')
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True
    assert permission.required_permissions == []
