from __future__ import annotations

import shlex

import pytest

from tests.mock.utils import collect_result
from vibe.core._trusted_command import TrustedCommandError
from vibe.core.tools.base import BaseToolState, ToolError, ToolPermission
from vibe.core.tools.builtins._bash_command_policy import (
    command_uses_unmanaged_background,
    harden_automated_command,
)
from vibe.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    _analyze_command_policy,
)
from vibe.core.tools.permissions import PermissionContext


def _always_bash() -> Bash:
    return Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )


@pytest.mark.asyncio
async def test_command_policy_analyzed_once_per_authorization_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    analyze = _analyze_command_policy

    def count(command: str):
        calls.append(command)
        return analyze(command)

    monkeypatch.setattr("vibe.core.tools.builtins.bash._analyze_command_policy", count)
    bash_tool = _always_bash()
    args = BashArgs(command="echo hello")

    bash_tool.resolve_permission(args)
    assert calls == [args.command]

    await bash_tool._validate_execution_authorization(
        args, None, require_local_shell=False
    )
    assert calls == [args.command, args.command]


@pytest.mark.parametrize(
    "command",
    [
        "dotnet test tests/Fcc.Core.Tests 2>&1 | tail -20",
        "dotnet restore FccSandbox.sln 2>&1 | tail -12; echo done",
        "uv run pytest -q tests/tools/test_bash.py | head -20",
        "pyright vibe/core; echo checked",
        "python -m pytest -q | tail -20",
        "pytest & wait",
        "! pytest",
        "pytest || echo ignored",
        "bash -c 'pytest | tail -20'",
        "uv run --offline pytest | head -20",
        "python -I -m pytest | head -20",
        "npm --silent test | head -20",
        "ruff --config pyproject.toml check . | head -20",
        "make -j test | head -20",
        "npx --no-install pytest | head -20",
        "uvx ruff check . | head -20",
        "npm exec -- pytest | head -20",
        "pnpm dlx vitest | head -20",
        "find . -exec pytest \\;",
        "xargs --max-procs 1 pytest",
        "trap 'pytest' EXIT",
        "coproc pytest",
        "bash --norc -c 'pytest | head -20'",
        'cmd.exe /c "pytest | head -20"',
        'cmd.exe /c"pytest | head -20"',
        '%COMSPEC% /c"pytest | head -20"',
        'powershell.exe -Command "pytest | head -20"',
        "bash <<'EOF'\npytest | head -20\nEOF",
        "./bash <<0\npytest | head -20\n0",
        "/bin/bash <<'123'\npytest | head -20\n123",
        "bash <<< 'pytest | head -20'",
        "%COMSPEC% /c 'pytest | head -20'",
        "%SHELL% /c 'pytest | head -20'",
        "!SHELL! /c 'pytest | head -20'",
        'powershell.exe -CommandWithArgs "pytest | head -20"',
        'powershell.exe -NoProfile -Command "pytest | head -20"',
        "npx --call 'pytest | head -20'",
        "npx -c 'pytest | head -20'",
        'printf "pytest | head -20" | powershell.exe -Command -',
        'printf "pytest | head -20" | pwsh -File -',
        "dotnet exec app.dll | head -20",
        "dotnet app.dll | head -20",
        "dotnet msbuild app.csproj | head -20",
        "dotnet vstest tests.dll | head -20",
        "cat <(pytest)",
    ],
)
def test_verification_runner_composition_is_hard_denied(command: str) -> None:
    config = BashToolConfig()
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "directly" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "dotnet test tests/Fcc.Core.Tests",
        'dotnet test --filter "A|B"',
        "pytest && echo verified",
    ],
)
def test_direct_verification_runner_is_not_hard_denied(command: str) -> None:
    config = BashToolConfig()
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is not ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "python -m pip --disable-pip-version-check install requests",
        "npm --silent install",
        "uv --project . sync",
        "bash -c 'npm install'",
        "npx prettier .",
        "pnpm dlx eslint .",
        "yarn dlx eslint .",
        "uvx ruff check .",
        "cargo add serde",
        "cargo remove serde",
        "env -u TOKEN npm install",
        "nice -n 10 npm install",
        "sudo -u root npm install",
        "command -- npm install",
        "find . -exec npm install \\;",
        "bash <<'EOF'\nnpm install\nEOF",
        "npm remove left-pad",
        "npm rm left-pad",
        "npm -w app install",
        "python -m pip download requests",
        "python -m pip wheel requests",
        "env -S 'npm install'",
        "env -a fake npm install",
        "bash --norc -c 'npm install'",
        "xargs --max-procs 1 npm install",
        "uv run --with-requirements requirements.txt pytest",
        "python.exe -m pip --log /tmp/pip.log install requests",
        "npm.cmd install",
        "dotnet workload install maui",
        "apt-get -o A=B install curl",
        "uv tool install ruff",
    ],
)
def test_dependency_syntax_variants_require_explicit_authority(command: str) -> None:
    reason = Bash.model_approval_deferral_reason(BashArgs(command=command))

    assert reason is not None
    assert "explicit user approval" in reason


@pytest.mark.parametrize(
    "command",
    [
        "bash -c \"echo 'or npm install'\"",
        "bash -c \"printf '%s' 'and apt install curl'\"",
        "bash -c \"printf '%s' 'not pytest | head'\"",
        'printf "not pytest" | bash -c "echo fish"',
    ],
)
def test_static_shell_payload_data_is_not_hard_denied(command: str) -> None:
    config = BashToolConfig()
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert not (
        isinstance(permission, PermissionContext)
        and permission.permission is ToolPermission.NEVER
    )


@pytest.mark.parametrize("command", ["apt list", "dnf info bash", "pacman -Q"])
def test_system_package_inspection_does_not_require_change_authority(
    command: str,
) -> None:
    reason = Bash.model_approval_deferral_reason(BashArgs(command=command))

    assert reason is None or "trusted system path" in reason


@pytest.mark.parametrize("command", ["apt install curl", "dnf upgrade", "pacman -Syu"])
def test_system_package_mutations_require_explicit_authority(command: str) -> None:
    reason = Bash.model_approval_deferral_reason(BashArgs(command=command))

    assert reason is not None
    assert "System package changes" in reason


@pytest.mark.parametrize(
    "command", ["npm install", "dotnet test tests/Fcc.Core.Tests --no-restore"]
)
def test_explicit_authority_cannot_be_replaced_by_global_always_permission(
    command: str,
) -> None:
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.required_permissions == []
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "timeout 30 pytest",
        "dotnet publish --no-restore",
        "dotnet run --no-restore",
        "cargo +nightly test",
        "cargo --color always test",
        "dotnet --diagnostics test --no-restore",
        "make -j test",
        "make -j 4 test",
        "npm run custom-check",
        'cmd.exe /c "pytest"',
        'powershell.exe -Command "pytest"',
        "dotnet exec app.dll",
        "dotnet app.dll",
        "dotnet msbuild app.csproj",
        "dotnet vstest tests.dll",
        "dotnet custom-extension",
    ],
)
def test_wrapped_and_dotnet_project_execution_requires_explicit_authority(
    command: str,
) -> None:
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.required_permissions == []
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "setsid npm install",
        "taskset -c 0 pytest",
        "strace -o trace.log dotnet test",
        "chrt -r 1 npm install",
        "numactl --physcpubind=0 pytest",
        "prlimit --nproc=10:10 pytest",
        "unshare --fork pytest",
        "setpriv --no-new-privs pytest",
        "valgrind pytest",
        "perf stat pytest",
        "watch -n 1 pytest",
        "watch -n 1 'npm install'",
        "busybox env npm install",
        "mystery-wrapper bash -c 'npm install'",
        "systemd-run --wait pytest",
    ],
)
def test_unproven_sensitive_child_requires_explicit_authority(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.required_permissions == []
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "trap 'npm install' EXIT",
        "command trap 'npm install' EXIT",
        "builtin trap 'npm install' EXIT",
        "source script.sh",
        ". script.sh",
        "builtin source script.sh",
        "command source script.sh",
        "enable -f ./evil.so name",
        "builtin enable -f ./evil.so name",
        "BASH_ENV=setup.sh bash -c true",
        "env BASH_ENV=setup.sh bash -c true",
        "ENV=setup.sh bash --posix -c true",
        "env SHELLOPTS=xtrace PS4='$(fish)' bash -c true",
        "env 'BASH_FUNC_shadow%%=() { fish; }' bash -c true",
        "LD_PRELOAD=./fixture.so true",
        "env LD_PRELOAD=./fixture.so true",
        "env DYLD_INSERT_LIBRARIES=./fixture.dylib true",
        "env -uNAME BASH_ENV=fixture bash -c true",
        "env -C/tmp BASH_ENV=fixture bash -c true",
        "env -afixture BASH_ENV=fixture bash -c true",
        "git -c alias.run='!fish -c true' run",
        "git --config-env=alias.run:ALIAS run",
        "git --exec-path=/tmp/helpers status",
    ],
)
def test_opaque_shell_carrier_requires_explicit_authority(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.required_permissions == []
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "git diff --ext-diff",
        "git diff --ext-di",
        "git diff --output=verification.log",
        "git show --textconv",
        "git show --textco",
        "git log --ext-diff",
        "git log --show-signature -1",
        "git log --format=%G? -1",
        "git grep --textconv needle",
        "git cat-file --textconv HEAD:README.md",
        "git cat-file --filters HEAD:README.md",
        "git bisect run echo safe",
        "git submodule foreach --recursive echo safe",
    ],
)
def test_git_execution_carriers_require_explicit_authority(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "git grep -c fish",
        "git show -c fish",
        "git log -c fish",
        "git show --no-textconv",
    ],
)
def test_git_child_options_and_terminated_operands_remain_data(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "xargs -a commands.txt -I{} {} -c true",
        "printf fish | xargs -I{} {} -c true",
        "find . -exec {} -c true \\;",
        "xargs -I{} sh -c {}",
        "find . -exec sh -c '{}' \\;",
    ],
)
def test_placeholder_executable_requires_explicit_authority(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "xargs -I{} sudo -u {} echo ok",
        "xargs -I{} env NAME={} echo ok",
        "xargs -I{} bash --rcfile {} -c true",
        "xargs -I{} sh -c 'echo {}'",
        "xargs -I{} eval 'echo {}'",
        "xargs -I{} pipx run --spec {} echo ok",
        "xargs -I{} uvx --from {} echo ok",
        "xargs -I{} rg --pre={} pattern .",
        "xargs -I{} sed -e '{}' README.md",
        "xargs -I{} sed -e '1e echo {}' README.md",
        "xargs -I{} sed 's/x/echo {}/e' README.md",
        "xargs -I{} split --filter={} README.md out-",
        "xargs -I{} split --filter='echo {}' README.md out-",
        "xargs -I{} sort {} README.md",
        "xargs -I{} less '+!{}' README.md",
        "xargs -I{} less '+{}' README.md",
        "xargs -I{} less '+!echo {}' README.md",
        "xargs -I{} git -c core.pager={} log -1",
        "xargs -I{} git grep -O{} needle",
        "xargs -I{} git grep -O'echo {}' needle",
        "xargs -I{} git bisect run {}",
        "xargs -I{} git submodule foreach {}",
        "xargs -I{} git show {}",
        "xargs -I{} rg fish {}",
        "xargs -I{} find . {} fish \\;",
        "xargs -I@ find . -exec echo @ \\; -exec echo @ \\;",
    ],
)
def test_placeholder_in_execution_structure_requires_explicit_authority(
    command: str,
) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize("command", ["xargs -I{} echo {}"])
def test_placeholder_carrier_still_requires_explicit_authority(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True


def test_placeholder_carrier_budget_fails_closed_without_recursion(monkeypatch) -> None:
    from vibe.core.tools.builtins import (
        _bash_command_policy as policy,
        bash as bash_module,
    )

    monkeypatch.setattr(policy, "_MAX_DEPTH", 4)
    nested = " ".join(["xargs -I{}"] * 10 + ["echo {}"])
    command = f"find . -exec {nested} \\;"
    analysis = policy.analyze_command_policy(
        command,
        bash_module._extract_commands(command),
        extract_commands=bash_module._extract_commands,
    )

    assert "static-analysis limit" in (analysis.denial or "")


def test_command_analysis_depth_limit_is_independent_of_top_level_order(
    monkeypatch,
) -> None:
    from vibe.core.tools.builtins import (
        _bash_command_policy as policy,
        bash as bash_module,
    )

    monkeypatch.setattr(policy, "_MAX_DEPTH", 2)
    shallow = "bash -c 'echo ok'"
    deep = shallow
    for _ in range(3):
        deep = f"bash -c {shlex.quote(deep)}"

    for command in (f"{deep}; {shallow}", f"{shallow}; {deep}"):
        analysis = policy.analyze_command_policy(
            command,
            bash_module._extract_commands(command),
            extract_commands=bash_module._extract_commands,
        )
        assert "static-analysis limit" in (analysis.denial or "")


def test_execution_carrier_registries_are_disjoint_from_data_commands() -> None:
    from vibe.core.tools.builtins import _bash_command_policy as policy

    carriers = set(policy._CARRIER_HANDLERS) | set(policy._EXECUTING_VALUE_OPTIONS)

    assert carriers.isdisjoint(policy._ARGUMENTS_ARE_DATA)


@pytest.mark.parametrize(
    "command",
    [
        "trap -p",
        "trap -l",
        "enable -p",
        "command -v fish",
        "command -V fish",
        "builtin fish",
        "xargs -I{} echo {}",
        "find . -exec echo {} \\;",
        "find . -exec sh -c 'echo \"$1\"' sh {} \\;",
    ],
)
def test_nonexecuting_metadata_and_placeholder_data_are_not_hard_denied(
    command: str,
) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert not (
        isinstance(permission, PermissionContext)
        and permission.permission is ToolPermission.NEVER
    )


def test_trap_masked_verification_is_hard_denied() -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(
        BashArgs(command="trap 'pytest | head -1' EXIT")
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "directly" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "sed -e '1e pytest' input",
        "sed -e '/x/!e pytest' input",
        "sed -e '{e pytest;}' input",
        "sed 's/x/pytest/2e' input",
        "sed '/x/Is/a/pytest/e' input",
        "sed '\\%x%s/a/pytest/e' input",
        "sort --compress-program=pytest input",
        "sort --compress-prog=pytest input",
        "split --filter=pytest input",
        "split --filt=pytest input",
        "mapfile -C pytest -c 1",
        "readarray -Cpytest -c 1",
        "less +!pytest file",
        "rg --pre=pytest pattern .",
        "git grep -Opytest needle",
        "git grep -nOpytest needle",
        "git grep --open-files-in-page=pytest needle",
        "git bisect run pytest",
        "git submodule foreach --recursive pytest",
    ],
)
def test_indirect_verification_status_is_hard_denied(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "directly" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        'eval "$cmd"',
        'bash -c "$cmd"',
        'trap "$handler" EXIT',
        'rg --pre="$tool" pattern .',
        'git grep --open-files-in-pager="$tool" needle',
        'less "+!$tool" file',
    ],
)
def test_direct_dynamic_carrier_requires_approval_without_masking_denial(
    command: str,
) -> None:
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "bash < script.sh",
        "powershell.exe -File .\\verify.ps1",
        "powershell.exe -EncodedCommand ZQBjAGgAbwAgAG8AawA=",
        "powershell.exe -ec ZQBjAGgAbwAgAG8AawA=",
        "dotnet --unknown=value test",
        "powershell.exe -File .\\verify.ps1 | more",
        "powershell.exe -e ZQBjAGgAbwAgAG8AawA= | more",
        "bash < script.sh | head -20",
        'runner=pytest; "$runner" | head -20',
        "%COMSPEC% /c pytest",
        "%SHELL% /c pytest",
        "!SHELL! /c pytest",
    ],
)
def test_ambiguous_execution_carriers_require_explicit_authority(command: str) -> None:
    config = BashToolConfig(permission=ToolPermission.ALWAYS)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize("command", ["echo hi &&", "if true; then", "echo 'open"])
def test_malformed_shell_syntax_fails_closed_without_exception(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert "explicit user approval" in (permission.reason or "")


def test_nested_command_depth_fails_closed() -> None:
    package_command = "npm install"
    verification_command = "pytest | head -20"
    for _ in range(6):
        package_command = f"bash -c {shlex.quote(package_command)}"
        verification_command = f"bash -c {shlex.quote(verification_command)}"

    package_reason = Bash.model_approval_deferral_reason(
        BashArgs(command=package_command)
    )
    bash_tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    verification_permission = bash_tool.resolve_permission(
        BashArgs(command=verification_command)
    )

    assert package_reason is not None
    assert "explicit user approval" in package_reason
    assert isinstance(verification_permission, PermissionContext)
    assert verification_permission.permission is ToolPermission.NEVER


def test_windows_keeps_platform_independent_authority_gates(monkeypatch) -> None:
    monkeypatch.setattr("vibe.core.tools.builtins.bash.is_windows", lambda: True)
    bash_tool = _always_bash()

    package_permission = bash_tool.resolve_permission(BashArgs(command="npm install"))
    masked_permission = bash_tool.resolve_permission(
        BashArgs(command="pytest | head -20")
    )

    assert isinstance(package_permission, PermissionContext)
    assert package_permission.permission is ToolPermission.ASK
    assert isinstance(masked_permission, PermissionContext)
    assert masked_permission.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "fish -c true",
        "/usr/bin/fish -c true",
        "./fish -c true",
        "fish.exe -c true",
        "env fish -c true",
        "env -S 'fish -c true'",
        "env --split-string='fish -c true'",
        "uv run fish -c true",
        "uv run --offline fish -c true",
        "npx --no-install fish -c true",
        "npm exec -- fish -c true",
        "npm --silent exec -- fish -c true",
        "pnpm dlx fish -c true",
        "pnpm --silent dlx fish -c true",
        "yarn dlx fish -c true",
        "yarn --silent dlx fish -c true",
        "bun x fish -c true",
        "uvx fish -c true",
        "poetry run fish -c true",
        "pipx run fish -c true",
        "pnpm exec fish -c true",
        "yarn exec fish -c true",
        "composer exec fish -c true",
        "git submodule foreach 'fish -c true'",
        "command fish -c true",
        "command -- fish -c true",
        "exec fish -c true",
        "exec -- fish -c true",
        "exec -cl fish -c true",
        "nice fish -c true",
        "nice -10 fish -c true",
        "nohup fish -c true",
        "nohup -- fish -c true",
        "timeout 5 fish -c true",
        "sudo -u root fish -c true",
        "sudo -- fish -c true",
        "sudo -nE fish -c true",
        "stdbuf -oL fish -c true",
        "ionice -c 3 fish -c true",
        "time fish -c true",
        "bash -c 'fish -c true'",
        "bash -c 'exec fish -c true'",
        "sh -c 'command fish -c true'",
        "eval 'fish -c true'",
        "xargs fish -c true",
        "xargs -n 1 fish -c true",
        "find . -exec fish -c true \\;",
        "find . -execdir /usr/bin/fish -c true +",
        "printf 'echo hello' | fish",
        "fish <<< 'echo hello'",
        "fish <<'EOF'\necho hello\nEOF",
        "coproc fish -c true",
        "builtin command fish -c true",
        "builtin command -- fish -c true",
        "builtin -- command fish -c true",
        "builtin exec fish -c true",
        "builtin -- exec fish -c true",
        "builtin eval 'fish -c true'",
        "builtin -- eval 'fish -c true'",
        "builtin builtin command fish -c true",
        "builtin -- builtin -- exec fish -c true",
        "trap 'fish -c true' EXIT",
        "command trap 'fish -c true' EXIT",
        "builtin trap 'fish -c true' EXIT",
        "rg --pre=fish pattern .",
        "git grep --open-files-in-pager=fish needle",
        "git grep --open-files-in-page=fish needle",
        "git grep -nOfish needle",
        "git grep -Ofish needle",
        "git grep -O/usr/bin/fish needle",
        "git submodule foreach --recursive fish -c true",
        "git bisect run fish -c true",
        "sed -e '1e fish -c true' input",
        "sed -e '/x/e fish -c true' input",
        "sed -e '/x/!e fish -c true' input",
        "sed -e '0,/x/e fish -c true' input",
        "sed -e '{e fish -c true;}' input",
        "sed 's/x/fish -c true/2e' input",
        "sed '\\%x%e fish -v' input",
        "sed '/x/Ie fish -v' input",
        "sort --compress-program=fish input",
        "sort --compress-program fish input",
        "sort --compress-prog=fish input",
        "sort --compress-p=fish input",
        "split --filter='fish -c true' input",
        "split --filt='fish -c true' input",
        "mapfile -C 'fish -c true' -c 1",
        "readarray -Cfish -c 1",
        "builtin mapfile -C fish -c 1",
        "less '+!fish -c true' file",
        "env -C/tmp fish -c true",
        "env -uPATH fish -c true",
        "sudo --preserve-env=PATH fish -c true",
        "xargs -0 -- fish -c true",
        "xargs --null -- fish -c true",
        "xargs --replace fish -c true",
        "xargs --replace={} fish -c true",
        "xargs -i fish -c true",
        "xargs -i{} fish -c true",
        "xargs --eof fish -c true",
        "xargs -e fish -c true",
        "xargs --max-lines fish -c true",
        "xargs -l fish -c true",
        "xargs --show-limits fish -c true",
        "xargs --exit fish -c true",
        "pipx run --spec package fish -c true",
        "uvx --from package fish -c true",
        "fish --version -c true",
        "fish --help",
        "fish --print-debug-categories",
        "fish -v > version.txt",
        "fish -v &",
        "./fish -v",
        "/usr/bin/fish -v",
        "/tmp/fish -v",
        "fish.exe -v",
        "fish -v < version.txt",
        "fish -v <redirect>",
        "env fish --version",
        "fish --version && echo done",
        "fish -Dnot -c 'npm install'",
        "fish --help -c 'npm install'",
        "fish -n -i -c 'npm install'",
    ],
)
def test_statically_reachable_fish_execution_is_hard_denied(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "Fish" in (permission.reason or "")


def test_deep_static_fish_wrapper_chain_is_hard_denied() -> None:
    command = " ".join(["env"] * 32 + ["fish", "-c", "true"])
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "Fish" in (permission.reason or "")


def test_static_analysis_node_limit_fails_closed() -> None:
    command = " ".join(["env"] * 300 + ["echo", "hello"])
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "static-analysis limit" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "setsid -f sleep 60",
        "setsid --fork sleep 60",
        "setsid -fw sleep 60",
        "env setsid --fork sleep 60",
        "systemd-run sleep 60",
        "systemd-run --no-block sleep 60",
        "systemd-run --no-block --wait sleep 60",
        "systemd-run --unit=worker sleep 60",
        "systemd-run printf --wait",
        "command systemd-run --no-block sleep 60",
        "start-stop-daemon --start --background --exec /bin/sleep -- 60",
        "start-stop-daemon -bSx /bin/sleep -- 60",
        "ssh -f localhost true",
        "ssh -fN localhost",
    ],
)
def test_unmanaged_background_detector_rejects_detaching_launchers(
    command: str,
) -> None:
    assert command_uses_unmanaged_background(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "setsid sleep 60",
        "setsid --wait sleep 60",
        "setsid sleep --fork",
        "systemd-run --wait sleep 60",
        "systemd-run --scope sleep 60",
        "systemd-run --wait --unit=worker sleep 60",
        "systemd-run -u worker --wait sleep 60",
        "systemd-run --property Description=worker --wait sleep 60",
        "systemd-run --version",
        "start-stop-daemon --start --exec /bin/sleep -- 60",
        "start-stop-daemon --start --name --background --exec /bin/sleep",
        "ssh localhost true",
        "ssh localhost printf -f",
        "ssh -o BatchMode=yes localhost true",
        "ssh -o -f localhost true",
    ],
)
def test_unmanaged_background_detector_preserves_foreground_launchers(
    command: str,
) -> None:
    assert command_uses_unmanaged_background(command) is False


def test_dynamic_shell_carrier_remains_approval_gated() -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command="$shell -c true"))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "setsid fish -c true",
        "taskset -c 0 fish -c true",
        "strace -o trace.log fish -c true",
        "chrt -r 1 fish -c true",
        "numactl --physcpubind=0 fish -c true",
        "prlimit --nproc=10:10 fish -c true",
        "unshare --fork fish -c true",
        "setpriv --no-new-privs fish -c true",
        "valgrind fish -c true",
        "perf stat fish -c true",
        "watch -n 1 fish -c true",
        "watch -n 1 'fish -c true'",
        "mystery-wrapper bash -c 'fish -c true'",
        "mystery-wrapper 'fish -c true'",
        "systemd-run --wait fish -c true",
    ],
)
def test_unproven_fish_command_position_requires_explicit_approval(
    command: str,
) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.required_permissions == []
    assert "explicit user approval" in (permission.reason or "")


def test_multicall_fish_position_is_hard_denied() -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(
        BashArgs(command="busybox env fish -c true")
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER


@pytest.mark.parametrize("command", ["fish -v", "fish --version"])
def test_direct_fish_version_query_is_not_hard_denied(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is not ToolPermission.NEVER


@pytest.mark.parametrize("command", ["fish -v", "fish --version"])
def test_direct_fish_version_query_is_not_auto_approved(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK


@pytest.mark.parametrize(
    "command",
    [
        "echo fish",
        "grep fish README.md",
        "printf '%s' 'fish -c true'",
        "command -v fish",
        "command -V fish",
        "command -p -v fish",
        "command -pv fish",
        "builtin command -v fish",
        "builtin -- command -v fish",
        "bash -c 'echo fish'",
        "env NAME=fish echo hello",
        "env --argv0 fish echo hello",
        "exec -a fish echo hello",
        "sudo -u fish echo hello",
        "timeout --signal fish 1 echo hello",
        "stdbuf -o fish echo hello",
        "nice -n fish echo hello",
        "nohup --help fish",
        "nohup --version fish",
        "sudo --version fish",
        "make SHELL=fish",
        "cat fish",
        "rg fish",
        "git grep fish",
        "test -f fish",
        "ls fish",
        "stat fish",
        "wc fish",
        "cp fish output",
        "cd fish",
        "source fish",
        "python fish",
        "git show fish",
        "which fish",
        "type fish",
        "builtin fish",
        "cat README.md | grep fish",
        "echo 'npm install' | grep fish",
    ],
)
def test_benign_fish_arguments_are_not_hard_denied(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert not (
        isinstance(permission, PermissionContext)
        and permission.permission is ToolPermission.NEVER
    )


@pytest.mark.parametrize(
    "command",
    [
        "bash -lc true",
        "bash -ic true",
        "bash --rcfile /tmp/fixture -c true",
        "sh -c true",
        "dash -c true",
        "ksh -c true",
        "zsh -c true",
        'python -c \'__import__("os").system("fish -c true")\'',
        'node -e \'require("child_process").execSync("fish -c true")\'',
        "awk 'BEGIN { system(\"fish -c true\") }'",
        "/usr/bin/fis? -c true",
        "/usr/bin/fis[h] -c true",
        "/usr/bin/f*sh -c true",
        "env /usr/bin/{fish,bash} -c true",
    ],
)
def test_opaque_interpreter_or_executable_requires_explicit_authority(
    command: str,
) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert "explicit user approval" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    ["bash -o posix -c 'fish -c true'", "printf 'fish -c true' | bash -s ignored"],
)
def test_static_bash_carriers_preserve_fish_denial(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "Fish" in (permission.reason or "")


def test_static_stdin_bash_carrier_preserves_masked_verification_denial() -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(
        BashArgs(command="printf 'pytest | head -1' | bash -s ignored")
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "directly" in (permission.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "printf 'fish -c true' | ash",
        "printf 'fish -c true' | env ash",
        "printf 'pytest | head -1' | ash",
    ],
)
def test_static_stdin_ash_preserves_nested_denials(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER


def test_non_stdin_heredoc_is_not_treated_as_shell_input() -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(
        BashArgs(command="bash 3<<'EOF'\nfish -c true\nEOF")
    )

    assert not (
        isinstance(permission, PermissionContext)
        and permission.permission is ToolPermission.NEVER
    )


@pytest.mark.parametrize(
    "command",
    [
        "head fish",
        "tail pytest",
        "git log pytest",
        "git status fish",
        "git show fish",
        "du -sh fish",
        "echo BASH_ENV=fixture",
        "file pytest",
        "sed -n 1p fish",
        "sed 's/fish/chips/' input",
        "sort pytest",
        "split pytest",
        "less +G pytest",
        "less fish",
    ],
)
def test_unmodeled_command_operands_are_not_invented_as_child_commands(
    command: str,
) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    expected = (
        ToolPermission.ASK
        if command.startswith(("less +", "git status"))
        else ToolPermission.ALWAYS
    )
    assert permission.permission is expected


def test_default_read_only_operands_are_never_reparsed_as_child_commands() -> None:
    from vibe.core.tools.builtins.bash import _get_default_allowlist

    bash_tool = _always_bash()

    for command in _get_default_allowlist():
        permission = bash_tool.resolve_permission(BashArgs(command=f"{command} fish"))

        assert isinstance(permission, PermissionContext), command
        assert permission.permission is ToolPermission.ALWAYS, command


def test_masked_verification_depth_exhaustion_is_not_a_false_hard_denial() -> None:
    command = "echo ok"
    for _ in range(6):
        command = f"bash -c {shlex.quote(command)}"
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert not (
        isinstance(permission, PermissionContext)
        and permission.permission is ToolPermission.NEVER
    )


@pytest.mark.asyncio
async def test_run_rechecks_fish_denial_before_spawn(monkeypatch) -> None:
    async def fail_if_spawned(*_args, **_kwargs):
        raise AssertionError("spawn must not be reached")

    monkeypatch.setattr(Bash, "_start_foreground", fail_if_spawned)
    bash_tool = _always_bash()

    with pytest.raises(ToolError, match="Fish"):
        await collect_result(bash_tool.run(BashArgs(command="fish -c true")))


def test_benign_inspection_pipeline_is_not_hard_denied() -> None:
    config = BashToolConfig()
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command="cat README.md | head"))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is not ToolPermission.NEVER


@pytest.mark.parametrize(
    "command", ['printf "Write-Output hello" | powershell.exe -Command -']
)
def test_benign_static_powershell_stdin_is_not_hard_denied(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is not ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        'less "+|.cat >/tmp/proof" README.md',
        'less --cmd="!npm install" README.md',
        "less --cm='!printf pwn' README.md",
        "less --lesskey-content='x shell printf pwn\\n' +x README.md",
        "less --lesskey-src=project.lesskey +x README.md",
        "less -k project.lesskey +x README.md",
        "less '+/needle\n!printf pwn' README.md",
        "less '+:e /etc/passwd' README.md",
        "less '+t secret' README.md",
        "less -t secret",
        "less --tag=secret",
        "less '+!cp /etc/passwd copied-secret' README.md",
        "less '+!git clean -fdx' README.md",
        "sort --compress-program='git clean -fdx' README.md",
        "uv run ./tool.py",
        "uv tool run --from hostile-package hostile-command",
        "pipx run hostile-package",
        "pypy3 -c 'print(1)'",
        "nodejs -e 'console.log(1)'",
        "lua -e 'print(1)'",
        "ash -c 'echo ok'",
        "cmake --build .",
        "bazel test //...",
        "uv build",
        "uv venv --seed .venv",
        "pipenv install requests",
        "conda install requests",
        "mamba install requests",
        "apk add ripgrep",
        "zypper install ripgrep",
        "aptitude install ripgrep",
        "snap install ripgrep",
        "flatpak install app.example.Foo",
        "npm audit fix",
        "npm audit --fix",
        "uv cache clean",
        "uv cache prune",
        "uv python install 3.12",
        "uv python uninstall 3.12",
        "uv python pin 3.12",
        "uv version --bump patch",
        "uv version 1.2.3",
        "go env -w GOPROXY=off",
        "go env -u GOPROXY",
        "conda config --set auto_activate_base false",
        "conda config --add channels conda-forge",
        "bun pm cache rm",
        "cargo metadata",
        "npm audit --fix=true",
        "go env -w=true GOPROXY=off",
        "go env -u=true GOPROXY",
        "PATH+=:/tmp/toolchain malicious-command",
        "LD_PRELOAD+=/tmp/payload.so /bin/true",
        "env --argv0=sh busybox -c 'echo hi'",
        "exec -a sh busybox -c 'echo hi'",
        "busybox rm -rf build",
        "git log --show-signature -1",
        "git rebase --exec='printf pwn' HEAD~1",
        "git fetch --upload-pack='printf pwn' origin",
        "rg --hostname-bin=printf needle .",
        "file -S -z archive.Z",
        "file --no-sandbox --uncompress archive.Z",
        "file -C -m magic",
        "file --compile -m magic",
        "file --compi -m magic",
        "find -L .",
        "grep -R needle .",
        "grep --dereference-recursive needle .",
        "rg --follow needle .",
        "du -L .",
        "tree -l .",
        "ls -RL .",
        "ls --recursive --dereference .",
        "diff -r left right",
        "diff --rec left right",
        "poetry run cat README.md",
        "pnpm exec cat README.md",
        "yarn exec cat README.md",
        "composer exec cat README.md",
        "find -files0-from=roots.bin -name '*.py'",
        "sort --files0-from=inputs.bin",
        "wc --files0-from=inputs.bin",
        "du --files0-from=inputs.bin",
        "file --files-from=inputs.txt",
        "file -f inputs.txt",
        "md5sum --check checksums.txt",
        "sha1sum -c checksums.txt",
        "sha256sum --check checksums.txt",
        "sha512sum -c checksums.txt",
        "shasum --check checksums.txt",
        "cksum --check checksums.txt",
        "/usr/bin/sudo -n /bin/true",
        "env sudo -n /bin/true",
        "find . $'-\\x65xec' true ';'",
        "file $'-\\x7a' archive.Z",
        "file $'-\\572' archive.Z",
        "file $'-\\u007a' archive.Z",
        "file $'-\\u7a' archive.Z",
        "file $'-\\103' -m magic",
        "grep $'-\\x52' needle .",
        "diff $'-\\x72' left right",
        "sort $'--compress-program=printf' README.md",
        "less $'--cmd=+:e /etc/passwd' README.md",
        "less $'+:e \\x2fetc\\x2fpasswd' README.md",
        "date $'--set=tomorrow'",
        "go env $'-\\x77' GOPROXY=off",
        "rg $'--hostname-bin=printf' needle .",
        "date $((121401012026))",
        'date "$((121401012026))"',
        "date $[121401012026]",
        "find . $OPT true ';'",
        "find . -{ex,foo}ec true ';'",
        "find . -* true ';'",
        'printf %s $"message"',
        "command -p cat README.md",
        "touch candidate.txt",
        "rm -rf build",
        "tail -F README.md",
        "tail --follow=name README.md",
        "tail --retry README.md",
    ],
)
def test_linux_execution_profiles_require_explicit_user_authority(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True


@pytest.mark.parametrize(
    "command",
    [
        "printf '%s' \"\\$'-\\x65xec'\"",
        r"cat \*",
        r"cat foo\*bar",
        'cat foo"*"bar',
        "cat foo'*'bar",
        'cat "*"',
        "tail -f README.md",
        "tail --follow README.md",
        "tail --follow=descriptor README.md",
    ],
)
def test_quoted_or_descriptor_literals_remain_autoapprovable(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "printf $'\\0'",
        "printf $'\\x00'",
        "printf $'\\u0000'",
        "printf $'\\c@'",
        "printf $'\\uD800'",
        "printf $'\\U00110000'",
    ],
)
def test_invalid_ansi_c_command_text_fails_closed(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "UTF-8" in (permission.reason or "")


def test_autoapproval_requires_trusted_executable_resolution(monkeypatch) -> None:
    def missing(_name: str):
        raise TrustedCommandError("missing")

    monkeypatch.setattr(
        "vibe.core.tools.builtins._bash_command_policy.resolve_trusted_system_executable",
        missing,
    )
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command="cat pyproject.toml"))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True
    assert "trusted system path" in (permission.reason or "")


def test_trusted_system_executable_can_remain_autoapproved(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.tools.builtins._bash_command_policy.resolve_trusted_system_executable",
        lambda name: f"/usr/bin/{name}",
    )
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command="cat README.md"))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    ("command", "executable"),
    [
        ("/tmp/echo hello", "/tmp/echo"),
        ("./printf '%s' hello", "./printf"),
        ("env /workspace/true", "/workspace/true"),
        ("command /tmp/echo hello", "/tmp/echo"),
    ],
)
def test_path_qualified_builtin_requires_executable_attestation(
    monkeypatch, command: str, executable: str
) -> None:
    inspected: list[str] = []

    def reject(path):
        inspected.append(str(path))
        raise TrustedCommandError("untrusted")

    monkeypatch.setattr(
        "vibe.core.tools.builtins._bash_command_policy.validate_trusted_system_executable",
        reject,
    )
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True
    assert executable in inspected


@pytest.mark.parametrize("command", ["ECHO hello", "echo.exe hello", "true.cmd"])
def test_builtin_exemption_requires_exact_bash_spelling(
    monkeypatch, command: str
) -> None:
    inspected: list[str] = []

    def reject(name: str):
        inspected.append(name)
        raise TrustedCommandError("untrusted")

    monkeypatch.setattr(
        "vibe.core.tools.builtins._bash_command_policy.resolve_trusted_system_executable",
        reject,
    )
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True
    assert inspected


@pytest.mark.parametrize(
    ("command", "child"),
    [
        ("env echo hello", "echo"),
        ("timeout 1 true", "true"),
        ("nice printf '%s' hello", "printf"),
        ("stdbuf -o0 echo hello", "echo"),
    ],
)
def test_external_wrapper_child_does_not_inherit_shell_builtin_exemption(
    monkeypatch, command: str, child: str
) -> None:
    inspected: list[str] = []

    def resolve(name: str):
        inspected.append(name)
        if name == child:
            raise TrustedCommandError("untrusted")
        return f"/usr/bin/{name}"

    monkeypatch.setattr(
        "vibe.core.tools.builtins._bash_command_policy.resolve_trusted_system_executable",
        resolve,
    )
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True
    assert child in inspected


@pytest.mark.parametrize(
    "command",
    [
        "git log --oneline -20",
        "git show HEAD",
        "git blame vibe/core/agent_loop.py",
        "git grep authorization_fingerprint",
    ],
)
def test_git_inspection_can_remain_autoapproved(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "git diff",
        "git diff --cached",
        "git diff --no-ext-diff",
        "git diff -- --ext-diff",
        "git status --short",
    ],
)
def test_git_worktree_inspection_requires_explicit_user(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True
    assert permission.reason is not None
    assert "repository-configured filters" in permission.reason


@pytest.mark.parametrize(
    "command",
    [
        "/usr/bin/git log -1",
        "env git log -1",
        "command git log -1",
        "git log -1 && true",
        "git log -1 > history.txt",
        "git log --help",
        "git show -h",
    ],
)
def test_unhardenable_git_inspection_requires_explicit_user(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True


@pytest.mark.parametrize(
    "command",
    [
        "hash -p ./pwn ls; ls",
        "builtin hash -p ./pwn ls; ls",
        "command hash -p ./pwn ls; ls",
        "printf -v PATH %s .; ls",
        "builtin printf -v PATH %s .; ls",
        "read PATH <<< .; ls",
        "mapfile PATH < input; ls",
        "alias ls=./pwn; ls",
        "export PATH=.; ls",
        "enable -f ./evil.so replacement; replacement",
        "for PATH in .; do ls; done",
        "for ((PATH=0; PATH<1; PATH++)); do ls; done",
        "PATH=.; ls",
        "((PATH=0)); ls",
        "let PATH=0; ls",
        "redirect() { PATH=.; ls; }; redirect",
        "builtin cd /etc; cat shadow",
        "builtin command cd /etc; cat shadow",
        "builtin builtin cd /etc; cat shadow",
        "pushd /etc; cat shadow",
        "popd; cat shadow",
    ],
)
def test_shell_resolution_state_mutation_requires_explicit_user(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ASK
    assert permission.requires_explicit_user_approval is True
    assert "command resolution or execution state" in (permission.reason or "")


@pytest.mark.parametrize(
    ("command", "expected_flags", "excluded_flags"),
    [
        ("git log -p -1", {"--no-ext-diff", "--no-textconv"}, set()),
        ("git show HEAD", {"--no-ext-diff", "--no-textconv"}, set()),
        ("git blame README.md", {"--no-textconv"}, {"--no-ext-diff"}),
        ("git grep needle", {"--no-textconv"}, {"--no-ext-diff"}),
    ],
)
def test_automated_git_inspection_disables_external_renderers(
    command: str, expected_flags: set[str], excluded_flags: set[str]
) -> None:
    hardened = shlex.split(harden_automated_command(command))

    assert "core.hooksPath=/dev/null" in hardened
    assert "core.fsmonitor=false" in hardened
    assert "diff.external=" in hardened
    assert expected_flags.issubset(hardened)
    assert excluded_flags.isdisjoint(hardened)


@pytest.mark.parametrize(
    "command",
    ["/usr/bin/git log -1", "env git log -1", "git log -1 && true", "git log --help"],
)
def test_automated_hardening_does_not_rewrite_non_direct_git(command: str) -> None:
    assert harden_automated_command(command) == command


def test_executable_provenance_is_not_cached(monkeypatch) -> None:
    available = True

    def resolve(name: str):
        if not available:
            raise TrustedCommandError("changed")
        return f"/usr/bin/{name}"

    monkeypatch.setattr(
        "vibe.core.tools.builtins._bash_command_policy.resolve_trusted_system_executable",
        resolve,
    )
    bash_tool = _always_bash()

    first = bash_tool.resolve_permission(BashArgs(command="cat README.md"))
    available = False
    second = bash_tool.resolve_permission(BashArgs(command="cat README.md"))

    assert isinstance(first, PermissionContext)
    assert first.permission is ToolPermission.ALWAYS
    assert isinstance(second, PermissionContext)
    assert second.permission is ToolPermission.ASK
    assert second.requires_explicit_user_approval is True


def test_recursive_diff_without_dereference_can_remain_autoapproved() -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(
        BashArgs(command="diff -r --no-dereference left right")
    )

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "env python",
        "command python",
        "nice python",
        "timeout 5 python",
        "bash -c 'python'",
        "env bash",
        "command bash",
        "env nohup",
        "env su",
    ],
)
def test_wrapped_standalone_denylist_commands_remain_denied(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('sed "e echo safe; npm install" README.md', ToolPermission.ASK),
        ('sed "e true; fish -c echo" README.md', ToolPermission.NEVER),
        ('sed "e true; pytest" README.md', ToolPermission.NEVER),
    ],
)
def test_sed_direct_execution_consumes_the_rest_of_the_line(
    command: str, expected: ToolPermission
) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is expected


@pytest.mark.parametrize(
    "command",
    [
        "/usr/bin/passwd root",
        "env /usr/bin/passwd root",
        "env --argv0=echo /usr/bin/passwd root",
        "env -a echo /usr/bin/passwd root",
        "exec -a echo /usr/bin/passwd root",
        "busybox passwd root",
        "toybox passwd root",
        "sudo --us root passwd root",
        "timeout --sig KILL 1 passwd root",
        "env --arg=echo passwd root",
        "/usr/bin/gdb app",
        "./pdb app",
        "/usr/bin/vim file",
        "/usr/bin/bash -i",
    ],
)
def test_denylist_matches_normalized_executables(command: str) -> None:
    bash_tool = _always_bash()

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER


def test_policy_rejects_oversized_token_graph_without_suffix_scanning() -> None:
    command = "unknown " + " ".join(f"arg{index}" for index in range(1_100))
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "size exceeds" in (permission.reason or "")


def test_deep_shell_ast_does_not_recurse_in_permission_resolution() -> None:
    command = "(" * 1_200 + "true" + ")" * 1_200
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert permission is None or (
        isinstance(permission, PermissionContext)
        and permission.permission in {ToolPermission.ASK, ToolPermission.NEVER}
    )


@pytest.mark.parametrize("command", ["cat \ud800", "cat \udfff", "echo \ud800"])
def test_invalid_unicode_command_text_fails_closed(command: str) -> None:
    bash_tool = Bash(config_getter=BashToolConfig, state=BaseToolState())

    permission = bash_tool.resolve_permission(BashArgs(command=command))

    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert "UTF-8" in (permission.reason or "")
