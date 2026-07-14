"""Static Bash authority analysis; ambiguity must defer to explicit approval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import re
import shlex
import sys
from typing import NamedTuple

from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

from vibe.core._trusted_command import (
    TRUSTED_GIT_CONFIG_ARGS,
    TrustedCommandError,
    resolve_trusted_system_executable,
    validate_trusted_system_executable,
)
from vibe.core.tools._command_tokens import (
    UV_OPTION_SPEC,
    OptionSpec,
    ParsedLeadingCommand,
    command_name,
    normalize_bash_ansi_c,
    parse_leading_command,
    parse_python_module,
    split_bash_tokens,
    token_is_assignment,
    token_is_dynamic,
    unwrap_command_once,
    unwrap_command_tokens,
)
from vibe.core.utils import is_windows
from vibe.core.verification_contract import is_verification_command

type ExtractCommands = Callable[[str], list[str]]

_MAX_DEPTH = 32
_SHORT_OPTION_LENGTH = 2
_FAILURE_MASKING_OPERATORS = frozenset({"|", "|&", "||", ";", "&", "!"})
_AUTO_APPROVAL_OPERATOR_NODES = {
    "command_substitution": "$(...)",
    "process_substitution": "<(...) or >(...)",
    "subshell": "(...)",
}
_DYNAMIC_ARGUMENT_NODES = frozenset({
    "arithmetic_expansion",
    "command_substitution",
    "expansion",
    "process_substitution",
    "simple_expansion",
})
_AUTO_APPROVAL_UNNAMED_OPERATORS = frozenset({"!", "&", "&&", ";", "|", "|&", "||"})
_OUTPUT_REDIRECT_RE = re.compile(r"^\s*\d*(?:<>|&>>?|>>?|>\|)")
_FIND_EXECUTION_PREDICATES = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
_PACKAGE_LAUNCHERS = frozenset({"bunx", "npx", "uvx"})
_STDIN_SHELLS = frozenset({
    "ash",
    "bash",
    "cmd",
    "dash",
    "fish",
    "ksh",
    "powershell",
    "pwsh",
    "sh",
    "zsh",
})
_OPAQUE_WRAPPERS = frozenset({
    "chrt",
    "numactl",
    "perf",
    "prlimit",
    "setpriv",
    "setsid",
    "strace",
    "taskset",
    "unshare",
    "valgrind",
    "watch",
})
_SSH_VALUE_OPTIONS = frozenset("BbcDEeFIiJLlmOoPpQRSWw")
_START_STOP_DAEMON_VALUE_OPTIONS = frozenset("acdgIikNnPpRrsux")
_START_STOP_DAEMON_VALUE_LONG_OPTIONS = frozenset({
    "--chdir",
    "--chroot",
    "--chuid",
    "--exec",
    "--group",
    "--iosched",
    "--name",
    "--nicelevel",
    "--notify-timeout",
    "--pid",
    "--pidfile",
    "--ppid",
    "--procsched",
    "--retry",
    "--signal",
    "--startas",
    "--umask",
    "--user",
})
_SYSTEMD_RUN_FOREGROUND_OPTIONS = frozenset({"--scope", "--wait"})
_SYSTEMD_RUN_NEUTRAL_LONG_OPTIONS = frozenset({
    "--collect",
    "--ignore-failure",
    "--no-ask-password",
    "--on-clock-change",
    "--on-timezone-change",
    "--pipe",
    "--pty",
    "--quiet",
    "--remain-after-exit",
    "--same-dir",
    "--send-sighup",
    "--shell",
    "--slice-inherit",
    "--system",
    "--user",
})
_SYSTEMD_RUN_NEUTRAL_SHORT_OPTIONS = frozenset("dGPrqSt")
_SYSTEMD_RUN_VALUE_LONG_OPTIONS = frozenset({
    "--background",
    "--description",
    "--expand-environment",
    "--gid",
    "--host",
    "--json",
    "--machine",
    "--nice",
    "--on-active",
    "--on-boot",
    "--on-calendar",
    "--on-startup",
    "--on-unit-active",
    "--on-unit-inactive",
    "--path-property",
    "--property",
    "--service-type",
    "--setenv",
    "--slice",
    "--socket-property",
    "--timer-property",
    "--uid",
    "--unit",
    "--working-directory",
})
_SYSTEMD_RUN_VALUE_SHORT_OPTIONS = frozenset("EHMpu")
_OPAQUE_CODE_INTERPRETERS = frozenset({
    "awk",
    "deno",
    "gawk",
    "jruby",
    "lua",
    "luajit",
    "mawk",
    "nodejs",
    "nawk",
    "node",
    "perl",
    "php",
    "python",
    "python3",
    "rscript",
    "ruby",
    "tclsh",
})
_PACKAGE_MANAGERS = frozenset({
    "brew",
    "bun",
    "cargo",
    "conda",
    "composer",
    "gem",
    "go",
    "mamba",
    "micromamba",
    "npm",
    "pip",
    "pip3",
    "pipenv",
    "pipx",
    "pnpm",
    "poetry",
    "uv",
    "yarn",
})
_SYSTEM_PACKAGE_MANAGERS = frozenset({
    "apk",
    "apt",
    "apt-get",
    "aptitude",
    "dnf",
    "dnf5",
    "flatpak",
    "pacman",
    "snap",
    "yum",
    "zypper",
})
_PACKAGE_INSPECTION_SUBCOMMANDS = {
    "brew": frozenset({"info", "leaves", "list", "outdated", "search"}),
    "bun": frozenset({"pm"}),
    "cargo": frozenset({"metadata", "search", "tree", "verify-project"}),
    "conda": frozenset({"compare", "config", "info", "list", "search"}),
    "composer": frozenset({"audit", "check-platform-reqs", "outdated", "show"}),
    "gem": frozenset({"contents", "dependency", "environment", "list", "search"}),
    "go": frozenset({"env", "list", "version"}),
    "mamba": frozenset({"info", "list", "repoquery", "search"}),
    "micromamba": frozenset({"info", "list", "repoquery", "search"}),
    "npm": frozenset({"audit", "explain", "list", "outdated", "view"}),
    "pip": frozenset({"check", "freeze", "index", "inspect", "list", "show"}),
    "pip3": frozenset({"check", "freeze", "index", "inspect", "list", "show"}),
    "pipenv": frozenset({"check", "graph", "requirements", "verify"}),
    "pipx": frozenset({"environment", "list"}),
    "pnpm": frozenset({"audit", "list", "outdated", "view", "why"}),
    "poetry": frozenset({"check", "show"}),
    "uv": frozenset({"cache", "python", "version"}),
    "yarn": frozenset({"info", "list", "outdated", "why"}),
}
_SYSTEM_PACKAGE_INSPECTION_SUBCOMMANDS = {
    "apk": frozenset({"audit", "info", "list", "policy", "search", "stats"}),
    "apt": frozenset({"list", "policy", "search", "show"}),
    "apt-get": frozenset({"check", "indextargets"}),
    "dnf": frozenset({"check", "info", "list", "repoquery", "search"}),
    "dnf5": frozenset({"check", "info", "list", "repoquery", "search"}),
    "flatpak": frozenset({"info", "list", "remotes", "search"}),
    "snap": frozenset({"connections", "info", "list", "services"}),
    "yum": frozenset({"check", "info", "list", "search"}),
    "zypper": frozenset({"info", "packages", "repos", "search"}),
}
_GIT_TERMINAL_SUBCOMMANDS = frozenset({
    "add",
    "apply",
    "branch",
    "cat-file",
    "checkout",
    "cherry-pick",
    "clean",
    "clone",
    "commit",
    "describe",
    "diff",
    "fetch",
    "format-patch",
    "init",
    "log",
    "ls-files",
    "ls-remote",
    "merge",
    "merge-base",
    "mv",
    "notes",
    "pull",
    "push",
    "rebase",
    "reflog",
    "remote",
    "reset",
    "restore",
    "rev-list",
    "rev-parse",
    "rm",
    "show",
    "show-ref",
    "sparse-checkout",
    "stash",
    "status",
    "switch",
    "tag",
    "worktree",
})
_GIT_GLOBAL_OPTIONS = OptionSpec(
    flags=frozenset({"--bare", "--literal-pathspecs", "--no-pager", "--paginate"}),
    values=frozenset({"--git-dir", "--namespace", "--work-tree", "-C"}),
)
_AUTOMATED_GIT_INSPECTIONS = frozenset({"blame", "grep", "log", "show"})
_GIT_WORKTREE_INSPECTIONS = frozenset({"diff", "status"})
_PROJECT_RUNNERS = frozenset({
    "bandit",
    "bazel",
    "cargo",
    "cmake",
    "ctest",
    "dotnet",
    "eslint",
    "flake8",
    "go",
    "gradle",
    "gradlew",
    "jest",
    "make",
    "meson",
    "mvn",
    "mvnw",
    "ninja",
    "nox",
    "pre-commit",
    "py.test",
    "pyright",
    "pytest",
    "ruff",
    "tox",
    "tsc",
    "vitest",
})
_PROJECT_SUBCOMMANDS = {
    "cargo": frozenset({
        "bench",
        "build",
        "check",
        "clippy",
        "doc",
        "fix",
        "run",
        "test",
    }),
    "dotnet": frozenset({
        "build",
        "clean",
        "exec",
        "format",
        "msbuild",
        "pack",
        "publish",
        "restore",
        "run",
        "test",
        "vstest",
    }),
    "go": frozenset({"build", "generate", "run", "test", "vet"}),
    "pre-commit": frozenset({"run"}),
    "ruff": frozenset({"check", "format"}),
}
_SCRIPT_MANAGERS = frozenset({"bun", "npm", "pnpm", "yarn"})
_PYTHON_PROJECT_MODULES = frozenset({"mypy", "pyright", "pytest", "ruff", "unittest"})
_PACKAGE_MANAGER_MUTATIONS = {
    "brew": frozenset({"install", "reinstall", "uninstall", "update", "upgrade"}),
    "bun": frozenset({"add", "install", "remove", "update", "x"}),
    "cargo": frozenset({"add", "fetch", "install", "remove", "uninstall", "update"}),
    "composer": frozenset({"install", "remove", "require", "update"}),
    "gem": frozenset({"install", "uninstall", "update"}),
    "npm": frozenset({
        "ci",
        "exec",
        "i",
        "install",
        "rebuild",
        "remove",
        "rm",
        "uninstall",
        "update",
        "x",
    }),
    "pip": frozenset({"download", "install", "uninstall", "wheel"}),
    "pip3": frozenset({"download", "install", "uninstall", "wheel"}),
    "pipenv": frozenset({"install", "lock", "sync", "uninstall", "update"}),
    "pipx": frozenset({
        "inject",
        "install",
        "reinstall",
        "run",
        "uninstall",
        "upgrade",
    }),
    "pnpm": frozenset({
        "add",
        "dlx",
        "i",
        "import",
        "install",
        "remove",
        "rm",
        "uninstall",
        "up",
        "update",
    }),
    "poetry": frozenset({"add", "install", "lock", "remove", "update"}),
    "yarn": frozenset({"add", "dlx", "install", "remove", "up", "upgrade"}),
}
_SYSTEM_PACKAGE_MUTATIONS = {
    "apt": frozenset({
        "autoremove",
        "dist-upgrade",
        "full-upgrade",
        "install",
        "purge",
        "remove",
        "update",
        "upgrade",
    }),
    "apt-get": frozenset({
        "autoremove",
        "dist-upgrade",
        "install",
        "purge",
        "remove",
        "update",
        "upgrade",
    }),
    "dnf": frozenset({
        "autoremove",
        "downgrade",
        "install",
        "reinstall",
        "remove",
        "update",
        "upgrade",
    }),
    "yum": frozenset({
        "autoremove",
        "downgrade",
        "install",
        "reinstall",
        "remove",
        "update",
        "upgrade",
    }),
}
_PACMAN_MUTATION_PREFIXES = ("-R", "-S", "-U")
_PACMAN_MUTATION_OPTIONS = frozenset({"--remove", "--sync", "--upgrade"})
_UV_DEPENDENCY_MUTATIONS = frozenset({"add", "lock", "remove", "sync"})
_UV_PIP_MUTATIONS = frozenset({"install", "sync", "uninstall"})
_DOTNET_IMPLICIT_RESTORE_COMMANDS = frozenset({
    "build",
    "pack",
    "publish",
    "run",
    "test",
})
_PACKAGE_CHANGE_DEFERRAL = (
    "Package acquisition and dependency graph changes require explicit user "
    "approval; the safety judge cannot authorize them."
)
_SYSTEM_PACKAGE_CHANGE_DEFERRAL = (
    "System package changes require explicit user approval; the safety judge "
    "cannot authorize them."
)
_DOTNET_RESTORE_DEFERRAL = (
    "This dotnet command can restore dependencies; package acquisition requires "
    "explicit user approval. Use --no-restore only after a successful "
    "user-approved restore."
)
_PROJECT_EXECUTION_DEFERRAL = (
    "Verification and build commands execute project-controlled code and require "
    "explicit user approval; the safety judge cannot authorize them."
)
_PRIVILEGED_EXECUTION_DEFERRAL = (
    "Privilege-changing commands require explicit user approval; stored rules, "
    "auto-approve, and the safety judge cannot authorize them."
)
_PRIVILEGED_EXECUTABLES = frozenset({"doas", "pkexec", "su", "sudo"})
_AMBIGUOUS_AUTHORITY_DEFERRAL = (
    "Static analysis cannot prove this command's complete executable graph. It "
    "requires explicit user approval; the safety judge cannot authorize it."
)
_UNTRUSTED_EXECUTABLE_DEFERRAL = (
    "The executable is not pinned to the trusted system path and may be shadowed "
    "by project or user files. It requires explicit user approval."
)
_GIT_INSPECTION_DEFERRAL = (
    "Automated Git inspection must be an exact standalone `git` command so the "
    "host can disable executable renderers. This form requires explicit user "
    "approval."
)
_GIT_WORKTREE_INSPECTION_DEFERRAL = (
    "Git status and diff can invoke repository-configured filters while reading "
    "the worktree and require explicit user approval."
)
_SHELL_STATE_DEFERRAL = (
    "This shell builtin can change command resolution or execution state and "
    "requires explicit user approval."
)
_STATEFUL_SHELL_BUILTINS = frozenset({
    ".",
    "alias",
    "cd",
    "declare",
    "enable",
    "export",
    "hash",
    "let",
    "local",
    "mapfile",
    "popd",
    "pushd",
    "read",
    "readarray",
    "readonly",
    "set",
    "shopt",
    "source",
    "trap",
    "typeset",
    "unalias",
    "unset",
})
_ARITHMETIC_MUTATION_OPERATORS = frozenset({
    "%=",
    "&=",
    "*=",
    "++",
    "+=",
    "--",
    "-=",
    "/=",
    "<<=",
    "=",
    ">>=",
    "^=",
    "|=",
})
_MASKED_STATUS_DENIAL = (
    "Run verification commands directly, without pipelines or trailing shell "
    "commands. Shell composition can hide the verification exit status; Bash "
    "already bounds displayed output and persists truncated output."
)
_FISH_EXECUTION_DENIAL = (
    "Fish command execution is unsupported because Bash command policy cannot "
    "validate Fish syntax. Use Bash syntax or run an exact standalone "
    "`fish -v` or `fish --version` query."
)
_FISH_VERSION_QUERY_DEFERRAL = (
    "Fish version queries execute an unsupported interpreter and require "
    "explicit user approval; the safety judge cannot authorize them."
)
_COMMAND_ANALYSIS_LIMIT_DENIAL = (
    "Command nesting exceeds the safe static-analysis limit; split it into "
    "simpler Bash calls."
)
_COMMAND_ANALYSIS_SIZE_LIMIT_DENIAL = (
    "Command size exceeds the safe static-analysis limit; split it into smaller "
    "Bash calls."
)
_COMMAND_ENCODING_DENIAL = (
    "Command text is not valid UTF-8 and cannot be analyzed safely."
)
_COMMAND_ANALYSIS_NODE_LIMIT = 256
_COMMAND_ANALYSIS_BYTE_LIMIT = 131_072
_COMMAND_ANALYSIS_TOKEN_LIMIT = 1_024
_REPLACEMENT_LITERAL_ARGUMENT_COMMANDS = frozenset({"echo", "false", "printf", "true"})
_SHELL_BUILTINS = frozenset({
    ".",
    ":",
    "[",
    "alias",
    "bg",
    "bind",
    "break",
    "builtin",
    "caller",
    "cd",
    "command",
    "compgen",
    "complete",
    "compopt",
    "continue",
    "coproc",
    "declare",
    "dirs",
    "disown",
    "echo",
    "enable",
    "eval",
    "exec",
    "exit",
    "export",
    "false",
    "fc",
    "fg",
    "getopts",
    "hash",
    "help",
    "history",
    "jobs",
    "kill",
    "let",
    "local",
    "logout",
    "mapfile",
    "popd",
    "printf",
    "pushd",
    "pwd",
    "read",
    "readarray",
    "readonly",
    "return",
    "set",
    "shift",
    "shopt",
    "source",
    "suspend",
    "test",
    "times",
    "trap",
    "true",
    "type",
    "typeset",
    "ulimit",
    "umask",
    "unalias",
    "unset",
    "wait",
})
_CHECKSUM_MANIFEST_COMMANDS = frozenset({
    "cksum",
    "md5sum",
    "sha1sum",
    "sha256sum",
    "sha512sum",
    "shasum",
})
_ARGUMENTS_ARE_DATA = frozenset({
    "[",
    "basename",
    "cat",
    "cksum",
    "cmp",
    "comm",
    "cd",
    "cp",
    "cut",
    "date",
    "dd",
    "diff",
    "dirname",
    "du",
    "echo",
    "expand",
    "expr",
    "false",
    "file",
    "findstr",
    "fold",
    "fmt",
    "grep",
    "head",
    "id",
    "join",
    "ln",
    "ls",
    "md5sum",
    "more",
    "nl",
    "od",
    "paste",
    "pathchk",
    "printf",
    "pwd",
    "readlink",
    "realpath",
    "sha1sum",
    "sha256sum",
    "sha512sum",
    "shasum",
    "stat",
    "strings",
    "sum",
    "tac",
    "tail",
    "test",
    "tr",
    "tree",
    "true",
    "tsort",
    "type",
    "uname",
    "uniq",
    "wc",
    "where",
    "which",
    "whoami",
})
_NON_EXECUTING_WRAPPER_OPTIONS = {
    "nohup": frozenset({"--help", "--version"}),
    "sudo": frozenset({"--help", "--version", "-V"}),
}
_EXECUTING_VALUE_OPTIONS = {
    "sort": frozenset({"--compress-program"}),
    "split": frozenset({"--filter"}),
}
_SED_FLAGS = frozenset({
    "--debug",
    "--follow-symlinks",
    "--posix",
    "--quiet",
    "--sandbox",
    "--silent",
    "-E",
    "-n",
    "-r",
    "-s",
    "-u",
    "-z",
})
_SED_REGEX_ADDRESS = r"/(?:\\.|[^/])*/[IM]*"
_SED_SINGLE_ADDRESS = rf"(?:\d+(?:~\d+)?|\$|{_SED_REGEX_ADDRESS})"
_SED_ADDRESS = (
    rf"(?:(?:{_SED_SINGLE_ADDRESS})"
    rf"(?:\s*,\s*(?:{_SED_SINGLE_ADDRESS}|[+~]\d+))?\s*)?(?:!\s*)?"
)
_SED_DIRECT_EXECUTION = re.compile(
    rf"(?:^|[;\n{{])\s*{_SED_ADDRESS}e(?P<payload>[^\n]*)(?=\n|$)"
)
_SED_SUBSTITUTION = re.compile(
    rf"(?:^|[;\n{{])\s*{_SED_ADDRESS}s(?P<delimiter>[^\\\w\s])"
)
_SED_ALT_DIRECT_EXECUTION = re.compile(
    r"(?:^|[;\n{])\s*\\(?P<address_delimiter>[^\\\w\s])"
    r"(?:\\.|(?!(?P=address_delimiter)).)*(?P=address_delimiter)[IM]*"
    r"\s*(?:!\s*)?e(?P<payload>[^\n]*)(?=\n|$)"
)
_SED_ALT_SUBSTITUTION = re.compile(
    r"(?:^|[;\n{])\s*\\(?P<address_delimiter>[^\\\w\s])"
    r"(?:\\.|(?!(?P=address_delimiter)).)*(?P=address_delimiter)[IM]*"
    r"\s*(?:!\s*)?s(?P<delimiter>[^\\\w\s])"
)

_CARGO_OPTIONS = OptionSpec(
    flags=frozenset({
        "--frozen",
        "--locked",
        "--offline",
        "--quiet",
        "--verbose",
        "-q",
        "-v",
    }),
    values=frozenset({"--color", "--config", "--manifest-path", "--target-dir", "-Z"}),
    cargo_toolchain=True,
)
_DOTNET_OPTIONS = OptionSpec(
    flags=frozenset({
        "--diagnostics",
        "--help",
        "--info",
        "--list-runtimes",
        "--list-sdks",
        "--no-logo",
        "--version",
        "-h",
    }),
    values=frozenset({
        "--additionalprobingpath",
        "--architecture",
        "--fx-version",
        "--os",
        "--roll-forward",
        "--runtime",
        "--verbosity",
        "-a",
        "-r",
        "-v",
    }),
)
_GO_OPTIONS = OptionSpec(values=frozenset({"-C"}))
_MAKE_OPTIONS = OptionSpec(
    flags=frozenset({
        "--always-make",
        "--dry-run",
        "--keep-going",
        "--no-builtin-rules",
        "--no-builtin-variables",
        "--question",
        "--silent",
        "-B",
        "-k",
        "-n",
        "-q",
        "-s",
    }),
    values=frozenset({"--directory", "--eval", "--file", "-C", "-f"}),
    optional_numeric_values=frozenset({"--jobs", "-j"}),
)
_NPM_OPTIONS = OptionSpec(
    flags=frozenset({"--no-progress", "--silent", "--verbose", "-s"}),
    values=frozenset({
        "--cache",
        "--loglevel",
        "--prefix",
        "--registry",
        "--userconfig",
        "--workspace",
        "-w",
    }),
)
_PNPM_OPTIONS = OptionSpec(
    flags=frozenset({"--silent", "--workspace-root", "-s", "-w"}),
    values=frozenset({"--config-dir", "--dir", "--filter", "--store-dir", "-C"}),
)
_YARN_OPTIONS = OptionSpec(
    flags=frozenset({"--silent", "--verbose", "-s"}),
    values=frozenset({"--cache-folder", "--cwd", "--modules-folder", "--mutex"}),
)
_PIP_OPTIONS = OptionSpec(
    flags=frozenset({
        "--disable-pip-version-check",
        "--isolated",
        "--no-cache-dir",
        "--no-color",
        "--no-input",
        "--require-virtualenv",
        "--verbose",
        "-q",
        "-v",
    }),
    values=frozenset({
        "--cache-dir",
        "--cert",
        "--client-cert",
        "--exists-action",
        "--log",
        "--proxy",
        "--python",
        "--retries",
        "--timeout",
        "--trusted-host",
        "--use-deprecated",
        "--use-feature",
    }),
)
_APT_OPTIONS = OptionSpec(
    flags=frozenset({"--assume-no", "--assume-yes", "--download-only", "-d", "-y"}),
    values=frozenset({
        "--config-file",
        "--host-architecture",
        "--option",
        "--target-release",
        "-c",
        "-o",
        "-t",
    }),
)
_DNF_OPTIONS = OptionSpec(
    flags=frozenset({"--assumeno", "--assumeyes", "-y"}),
    values=frozenset({"--config", "--installroot", "--releasever", "--setopt", "-c"}),
)
_RUFF_OPTIONS = OptionSpec(
    flags=frozenset({
        "--isolated",
        "--quiet",
        "--silent",
        "--verbose",
        "-q",
        "-s",
        "-v",
    }),
    values=frozenset({"--config", "--output-format"}),
)
_PRE_COMMIT_OPTIONS = OptionSpec(values=frozenset({"--config", "-c"}))
_GENERIC_SCRIPT_OPTIONS = OptionSpec(flags=frozenset({"--silent", "--verbose", "-s"}))
_PACKAGE_LAUNCHER_OPTIONS = OptionSpec(
    flags=frozenset({"--no-install", "--yes", "-y"}),
    values=frozenset({
        "--cache",
        "--cache-dir",
        "--from",
        "--package",
        "--registry",
        "--spec",
        "--userconfig",
        "-p",
    }),
)
_OPTION_SPECS = {
    "apt": _APT_OPTIONS,
    "apt-get": _APT_OPTIONS,
    "cargo": _CARGO_OPTIONS,
    "dnf": _DNF_OPTIONS,
    "dotnet": _DOTNET_OPTIONS,
    "go": _GO_OPTIONS,
    "make": _MAKE_OPTIONS,
    "npm": _NPM_OPTIONS,
    "pip": _PIP_OPTIONS,
    "pip3": _PIP_OPTIONS,
    "pnpm": _PNPM_OPTIONS,
    "pre-commit": _PRE_COMMIT_OPTIONS,
    "ruff": _RUFF_OPTIONS,
    "uv": UV_OPTION_SPEC,
    "yarn": _YARN_OPTIONS,
    "yum": _DNF_OPTIONS,
}


class _ReplacementPayload(NamedTuple):
    body: str
    marker: str


class _CarrierPayloads(NamedTuple):
    payloads: tuple[str, ...] = ()
    recognized: bool = False
    ambiguous: bool = False
    dynamic: bool = False
    replacements: tuple[_ReplacementPayload, ...] = ()
    masks_status: bool = False


class _ShellPayload(NamedTuple):
    body: str


class _ShellInput(NamedTuple):
    payloads: tuple[_ShellPayload, ...] = ()
    ambiguous: bool = False


class _PendingCommand(NamedTuple):
    tokens: list[str]
    depth: int = 0
    replacement_markers: frozenset[str] = frozenset()
    allow_shell_builtin: bool = True


@dataclass(frozen=True, slots=True)
class CommandPolicyAnalysis:
    denial: str | None = None
    deferral: str | None = None


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


def shell_input_payloads(command: str) -> tuple[str, ...]:
    return tuple(payload.body for payload in _shell_stdin_payloads(command).payloads)


def command_analysis_preflight_denial(command: str) -> str | None:
    try:
        encoded = command.encode("utf-8")
        normalized = normalize_bash_ansi_c(command)
        normalized.encode("utf-8")
    except (UnicodeEncodeError, ValueError):
        return _COMMAND_ENCODING_DENIAL
    if "\x00" in normalized:
        return _COMMAND_ENCODING_DENIAL
    if len(encoded) > _COMMAND_ANALYSIS_BYTE_LIMIT:
        return _COMMAND_ANALYSIS_SIZE_LIMIT_DENIAL
    return None


def _unquoted_word_expands(value: str) -> bool:
    unquoted: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(value):
        character = value[index]
        if quote is not None:
            if character == "\\" and quote == '"' and index + 1 < len(value):
                index += 2
                continue
            if character == quote:
                quote = None
            index += 1
            continue
        if character == "\\" and index + 1 < len(value):
            index += 2
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
            continue
        unquoted.append(character)
        index += 1
    rendered = "".join(unquoted)
    glob = "*" in rendered or "?" in rendered or bool(re.search(r"\[[^]]+\]", rendered))
    brace = (
        "{" in rendered and "}" in rendered and ("," in rendered or ".." in rendered)
    )
    return glob or brace


def _has_dynamic_shell_arguments(command: str) -> bool:
    source = command.encode("utf-8")
    pending = [_get_parser().parse(source).root_node]
    while pending:
        node = pending.pop()
        if node.type in _DYNAMIC_ARGUMENT_NODES:
            return True
        if node.type == "command":
            if any(
                left.type == "$" and right.type == "string"
                for left, right in zip(node.children, node.children[1:], strict=False)
            ):
                return True
            for child in node.children:
                if child.type in {"command_name", "raw_string", "variable_assignment"}:
                    continue
                rendered = _node_text(child, source) or ""
                if rendered.startswith('$"'):
                    return True
                if child.type != "string" and _unquoted_word_expands(rendered):
                    return True
        pending.extend(reversed(node.children))
    return False


def _has_shell_state_mutation(command: str) -> bool:
    source = command.encode("utf-8")
    pending = [_get_parser().parse(source).root_node]
    while pending:
        node = pending.pop()
        if node.type in {
            "c_style_for_statement",
            "for_statement",
            "function_definition",
            "select_statement",
            "variable_assignment",
        }:
            return True
        if node.type == "compound_statement" and any(
            child.type == "((" for child in node.children
        ):
            arithmetic = [node]
            while arithmetic:
                child = arithmetic.pop()
                if not child.is_named and child.type in _ARITHMETIC_MUTATION_OPERATORS:
                    return True
                arithmetic.extend(reversed(child.children))
        pending.extend(reversed(node.children))
    return False


def _short_option_before_command(
    arguments: list[str], flag: str, *, value_options: frozenset[str] = frozenset()
) -> bool:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--", "-"} or not argument.startswith("-"):
            return False
        if argument.startswith("--"):
            index += 1
            continue
        consume_next = False
        for position, option in enumerate(argument[1:]):
            if option == flag:
                return True
            if option in value_options:
                consume_next = position == len(argument) - 2
                break
        index += 2 if consume_next else 1
    return False


def _setsid_detaches(arguments: list[str]) -> bool:
    for argument in arguments:
        if argument in {"--", "-"} or not argument.startswith("-"):
            break
        if argument == "--fork":
            return True
    return _short_option_before_command(arguments, "f")


def _systemd_run_detaches(arguments: list[str]) -> bool:
    if arguments and all(
        argument in {"--help", "--version", "-h"} for argument in arguments
    ):
        return False
    foreground = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--", "-"} or not argument.startswith("-"):
            break
        if argument == "--no-block":
            return True
        if argument in _SYSTEMD_RUN_FOREGROUND_OPTIONS:
            foreground = True
            index += 1
            continue
        if argument.startswith("--"):
            option, separator, _ = argument.partition("=")
            if option in _SYSTEMD_RUN_VALUE_LONG_OPTIONS:
                index += 1 if separator else 2
                continue
            if argument in _SYSTEMD_RUN_NEUTRAL_LONG_OPTIONS:
                index += 1
                continue
            return True
        consume_next = False
        for position, option in enumerate(argument[1:]):
            if option in _SYSTEMD_RUN_VALUE_SHORT_OPTIONS:
                consume_next = position == len(argument) - 2
                break
            if option not in _SYSTEMD_RUN_NEUTRAL_SHORT_OPTIONS:
                return True
        index += 2 if consume_next else 1
    return not foreground


def _start_stop_daemon_detaches(arguments: list[str]) -> bool:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            return False
        if argument == "--background":
            return True
        if argument.startswith("--"):
            option, separator, _ = argument.partition("=")
            index += (
                2
                if option in _START_STOP_DAEMON_VALUE_LONG_OPTIONS and not separator
                else 1
            )
            continue
        if not argument.startswith("-") or argument == "-":
            index += 1
            continue
        consume_next = False
        for position, option in enumerate(argument[1:]):
            if option == "b":
                return True
            if option in _START_STOP_DAEMON_VALUE_OPTIONS:
                consume_next = position == len(argument) - 2
                break
        index += 2 if consume_next else 1
    return False


def _launcher_detaches(tokens: list[str]) -> bool:
    name = command_name(tokens[0])
    arguments = tokens[1:]
    if name == "setsid":
        return _setsid_detaches(arguments)
    if name == "systemd-run":
        return _systemd_run_detaches(arguments)
    if name == "start-stop-daemon":
        return _start_stop_daemon_detaches(arguments)
    if name == "ssh":
        return _short_option_before_command(
            arguments, "f", value_options=_SSH_VALUE_OPTIONS
        )
    return False


def command_uses_unmanaged_background(command: str) -> bool:
    pending = [command]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if len(seen) > _COMMAND_ANALYSIS_NODE_LIMIT:
            return True
        source = current.encode("utf-8")
        tree = _get_parser().parse(source)
        nodes = [tree.root_node]
        while nodes:
            node = nodes.pop()
            if not node.is_named and node.type == "&":
                return True
            if node.type == "command":
                rendered = _node_text(node, source) or ""
                tokens = _tokenize(rendered)
                if tokens:
                    name = command_name(tokens[0])
                    if name == "coproc" or _launcher_detaches(tokens):
                        return True
                    carrier = _carrier_payloads(tokens)
                    if name in _STDIN_SHELLS | {"eval"} and (
                        carrier.dynamic or (carrier.ambiguous and not carrier.payloads)
                    ):
                        return True
                    pending.extend(carrier.payloads)
            nodes.extend(reversed(node.children))
        shell_input = _shell_stdin_payloads(current)
        if shell_input.ambiguous:
            return True
        pending.extend(payload.body for payload in shell_input.payloads)
    return False


def _shell_stdin_payloads(command: str) -> _ShellInput:
    source = command.encode("utf-8")
    tree = _get_parser().parse(source)
    payloads: list[_ShellPayload] = []
    ambiguous = False

    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.type == "pipeline":
            pipeline_input = _pipeline_shell_input(node, source)
            payloads.extend(pipeline_input.payloads)
            ambiguous = ambiguous or pipeline_input.ambiguous
        if node.type in {"heredoc_redirect", "herestring_redirect", "file_redirect"}:
            if _redirect_targets_stdin(node, source):
                receiver = _redirect_receiver(node)
                name_node = (
                    receiver.child_by_field_name("name")
                    if receiver is not None
                    else None
                )
                receiver_name = _node_text(name_node, source)
                if receiver_name is not None and (
                    command_name(receiver_name) in _STDIN_SHELLS
                    or token_is_dynamic(receiver_name)
                ):
                    payload = _redirect_payload(node, source)
                    if payload is None:
                        ambiguous = True
                    else:
                        payloads.append(_ShellPayload(payload))
        pending.extend(reversed(node.children))
    return _ShellInput(tuple(dict.fromkeys(payloads)), ambiguous)


def _redirect_targets_stdin(node: Node, source: bytes) -> bool:
    rendered = _node_text(node, source) or ""
    match = re.match(r"^\s*(\d*)<", rendered)
    return match is None or match.group(1) in {"", "0"}


def _redirect_receiver(node: Node) -> Node | None:
    parent = node.parent
    if parent is None:
        return None
    if parent.type == "command":
        return parent
    if parent.type != "redirected_statement":
        return None
    body = parent.child_by_field_name("body")
    while body is not None and body.type == "redirected_statement":
        body = body.child_by_field_name("body")
    return body if body is not None and body.type == "command" else None


def _node_text(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _redirect_payload(node: Node, source: bytes) -> str | None:
    if node.type == "heredoc_redirect":
        body = next(
            (child for child in node.named_children if child.type == "heredoc_body"),
            None,
        )
        return _node_text(body, source)
    if node.type == "file_redirect":
        text = _node_text(node, source) or ""
        return None if re.match(r"^\s*(?:0)?<", text) else ""
    value = next(iter(node.named_children), None)
    rendered = _node_text(value, source)
    if rendered is None:
        return None
    try:
        values = split_bash_tokens(rendered)
    except ValueError:
        return None
    return values[0] if len(values) == 1 else None


def _pipeline_shell_input(node: Node, source: bytes) -> _ShellInput:
    payloads: list[_ShellPayload] = []
    ambiguous = False
    stages = list(node.named_children)
    for producer, receiver in zip(stages, stages[1:], strict=False):
        receiver_command = _pipeline_command(receiver)
        if receiver_command is None:
            continue
        rendered = _node_text(receiver_command, source)
        tokens = _tokenize(rendered or "")
        if tokens is None or not tokens or not _reads_commands_from_stdin(tokens):
            continue
        payload = _static_pipeline_output(producer, source)
        if payload is None:
            ambiguous = True
        else:
            payloads.append(_ShellPayload(payload))
    return _ShellInput(tuple(payloads), ambiguous)


def _pipeline_command(node: Node) -> Node | None:
    current = node
    while current.type == "redirected_statement":
        body = current.child_by_field_name("body")
        if body is None:
            return None
        current = body
    return current if current.type == "command" else None


def _reads_commands_from_stdin(tokens: list[str], *, depth: int = 0) -> bool:
    if depth > _MAX_DEPTH:
        return True
    if _executable_token_is_dynamic(tokens[0]):
        return True
    name = command_name(tokens[0])
    arguments = tokens[1:]
    if name == "cmd":
        result = unwrap_command_tokens(tokens).shell_payload is None
    elif name in {"powershell", "pwsh"}:
        result = _powershell_reads_commands_from_stdin(arguments)
    elif name in {"ash", "bash", "dash", "fish", "ksh", "sh", "zsh"}:
        result = _unix_shell_reads_commands_from_stdin(name, arguments)
    else:
        return _wrapped_command_reads_stdin(tokens, depth=depth)
    if result:
        return True
    payload = unwrap_command_tokens(tokens).shell_payload
    return payload is not None and _payload_reads_commands_from_stdin(
        payload, depth=depth + 1
    )


def _wrapped_command_reads_stdin(tokens: list[str], *, depth: int) -> bool:
    unwrapped = unwrap_command_tokens(tokens)
    if unwrapped.shell_payload is not None:
        return _payload_reads_commands_from_stdin(
            unwrapped.shell_payload, depth=depth + 1
        )
    if unwrapped.ambiguous:
        return any(
            command_name(token) in _STDIN_SHELLS or token_is_dynamic(token)
            for token in tokens
        )
    if unwrapped.changed and unwrapped.tokens:
        return _reads_commands_from_stdin(list(unwrapped.tokens), depth=depth + 1)
    return False


def _payload_reads_commands_from_stdin(payload: str, *, depth: int) -> bool:
    if depth > _MAX_DEPTH:
        return True
    source = payload.encode("utf-8")
    tree = _get_parser().parse(source)
    if tree.root_node.has_error:
        return True

    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.type == "command":
            rendered = _node_text(node, source)
            tokens = _tokenize(rendered or "")
            if tokens is None or (
                tokens and _reads_commands_from_stdin(tokens, depth=depth)
            ):
                return True
        pending.extend(reversed(node.named_children))
    return False


def _powershell_reads_commands_from_stdin(arguments: list[str]) -> bool:
    for index, argument in enumerate(arguments):
        option = argument.casefold()
        if len(option) > 1 and "-encodedcommand".startswith(option):
            return False
        command_option = option in {"-command", "-c"} or (
            option not in {"-", "--"} and "-commandwithargs".startswith(option)
        )
        file_option = len(option) > 1 and "-file".startswith(option)
        if command_option or file_option:
            return index + 1 >= len(arguments) or arguments[index + 1] == "-"
    return True


def _unix_shell_reads_commands_from_stdin(name: str, arguments: list[str]) -> bool:
    index = 0
    forced_stdin = False
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            return index + 1 >= len(arguments)
        option = argument.partition("=")[0]
        if argument == "-c" or (
            argument.startswith("-")
            and not argument.startswith("--")
            and "c" in argument[1:]
        ):
            return False
        if (
            argument.startswith("-")
            and not argument.startswith("--")
            and "s" in argument[1:]
        ):
            forced_stdin = True
        if option in {"--init-file", "--rcfile", "+O", "+o", "-O", "-o"}:
            index += (
                1
                if "=" in argument or len(argument) > _SHORT_OPTION_LENGTH
                else _SHORT_OPTION_LENGTH
            )
            continue
        if argument.startswith(("-", "+")) and argument not in {"-", "+"}:
            index += 1
            continue
        return forced_stdin
    return True


def _static_pipeline_output(node: Node, source: bytes) -> str | None:
    command = _pipeline_command(node)
    rendered = _node_text(command, source)
    tokens = _tokenize(rendered or "")
    if tokens is None or not tokens or any(token_is_dynamic(token) for token in tokens):
        return None
    name = command_name(tokens[0])
    arguments = tokens[1:]
    if name == "printf":
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        return arguments[0] if len(arguments) == 1 else None
    if name == "echo":
        while arguments and arguments[0] in {"-E", "-e", "-n"}:
            arguments = arguments[1:]
        return " ".join(arguments)
    return None


def _tokenize(command: str) -> list[str] | None:
    try:
        return split_bash_tokens(command.replace(" <redirect>", ""))
    except ValueError:
        return None


def _is_direct_fish_version_query(command: str) -> bool:
    try:
        tokens = split_bash_tokens(command)
    except ValueError:
        return False
    return bool(
        tokens
        and len(tokens) == _SHORT_OPTION_LENGTH
        and tokens[0] == "fish"
        and tokens[1] in {"--version", "-v"}
    )


def _is_command_lookup(tokens: list[str]) -> bool:
    if command_name(tokens[0]) != "command":
        return False
    for argument in tokens[1:]:
        if argument == "--" or not argument.startswith("-"):
            return False
        if "v" in argument[1:] or "V" in argument[1:]:
            return True
    return False


def _parsed_subcommand(name: str, arguments: list[str]) -> ParsedLeadingCommand:
    return parse_leading_command(arguments, _OPTION_SPECS.get(name, OptionSpec()))


def _pip_invocation(tokens: list[str]) -> tuple[list[str] | None, bool]:
    name = command_name(tokens[0])
    if re.fullmatch(r"python(?:3(?:\.\d+)*)?", name) is None:
        return None, False
    parsed = parse_python_module(tokens[1:])
    if parsed.ambiguous:
        return None, True
    return (list(parsed.arguments), False) if parsed.module == "pip" else (None, False)


def _system_package_reason(name: str, arguments: list[str]) -> str | None:
    if name == "pacman":
        changed = any(
            argument in _PACMAN_MUTATION_OPTIONS
            or any(argument.startswith(prefix) for prefix in _PACMAN_MUTATION_PREFIXES)
            for argument in arguments
        )
        return _SYSTEM_PACKAGE_CHANGE_DEFERRAL if changed else None
    parsed = _parsed_subcommand(name, arguments)
    if parsed.ambiguous:
        return _AMBIGUOUS_AUTHORITY_DEFERRAL
    mutations = _SYSTEM_PACKAGE_MUTATIONS.get(name, frozenset())
    return (
        _SYSTEM_PACKAGE_CHANGE_DEFERRAL
        if command_name(parsed.token or "") in mutations
        else None
    )


def _dotnet_dependency_reason(arguments: list[str]) -> str | None:
    parsed = _parsed_subcommand("dotnet", arguments)
    if parsed.ambiguous:
        return _AMBIGUOUS_AUTHORITY_DEFERRAL
    subcommand = command_name(parsed.token or "")
    all_arguments = [*arguments]
    if subcommand == "restore" or (
        subcommand in _DOTNET_IMPLICIT_RESTORE_COMMANDS
        and "--no-restore" not in all_arguments
    ):
        return _DOTNET_RESTORE_DEFERRAL
    if subcommand in {"add", "remove"} and "package" in parsed.arguments:
        return _PACKAGE_CHANGE_DEFERRAL
    if subcommand == "package" and any(
        action in parsed.arguments for action in {"add", "remove"}
    ):
        return _PACKAGE_CHANGE_DEFERRAL
    if subcommand in {"tool", "workload"} and any(
        action in parsed.arguments
        for action in {"install", "restore", "uninstall", "update"}
    ):
        return _PACKAGE_CHANGE_DEFERRAL
    return None


def _dependency_reason(tokens: list[str]) -> str | None:
    name = command_name(tokens[0])
    arguments = tokens[1:]
    pip_arguments, python_ambiguous = _pip_invocation(tokens)
    if python_ambiguous:
        reason = _AMBIGUOUS_AUTHORITY_DEFERRAL
    else:
        if pip_arguments is not None:
            name = "pip"
            arguments = pip_arguments
        if name in _SYSTEM_PACKAGE_MANAGERS:
            reason = _system_package_reason(name, arguments)
        elif name in _PACKAGE_LAUNCHERS:
            benign = not arguments or all(
                argument in {"--help", "--version", "-h", "-v"}
                for argument in arguments
            )
            reason = None if benign else _PACKAGE_CHANGE_DEFERRAL
        elif name == "dotnet":
            reason = _dotnet_dependency_reason(arguments)
        elif name not in _PACKAGE_MANAGERS:
            reason = None
        else:
            reason = _package_manager_reason(name, arguments)
    return reason


def _package_manager_reason(name: str, arguments: list[str]) -> str | None:
    parsed = _parsed_subcommand(name, arguments)
    if parsed.ambiguous:
        return _AMBIGUOUS_AUTHORITY_DEFERRAL
    subcommand = command_name(parsed.token or "")
    if subcommand in _PACKAGE_INSPECTION_SUBCOMMANDS.get(
        name, frozenset()
    ) and not _package_inspection_is_observational(name, subcommand, parsed.arguments):
        return _PACKAGE_CHANGE_DEFERRAL
    if name == "uv" and _uv_changes_dependencies(
        subcommand, parsed.arguments, arguments
    ):
        return _PACKAGE_CHANGE_DEFERRAL
    if name == "go":
        go_change = subcommand in {"get", "install"} or (
            subcommand in {"mod", "work"}
            and parsed.arguments[:1] in {("download",), ("sync",), ("tidy",)}
        )
        return _PACKAGE_CHANGE_DEFERRAL if go_change else None
    mutations = _PACKAGE_MANAGER_MUTATIONS.get(name, frozenset())
    default_yarn_install = name == "yarn" and parsed.token is None
    return (
        _PACKAGE_CHANGE_DEFERRAL
        if subcommand in mutations or default_yarn_install
        else None
    )


def _package_inspection_is_observational(
    name: str, subcommand: str, arguments: tuple[str, ...]
) -> bool:
    options = {argument.partition("=")[0] for argument in arguments}
    match name, subcommand:
        case "npm", "audit":
            observational = "fix" not in arguments and "--fix" not in options
        case "uv", "cache":
            observational = not arguments or arguments[0] in {"--help", "-h", "dir"}
        case "uv", "python":
            observational = not arguments or arguments[0] in {
                "--help",
                "-h",
                "dir",
                "find",
                "list",
            }
        case "uv", "version":
            observational = not arguments or all(
                argument in {"--help", "-h"} for argument in arguments
            )
        case "go", "env":
            observational = not options.intersection({"-u", "-w"})
        case ("conda", "config") | ("bun", "pm"):
            observational = False
        case "cargo", "metadata":
            observational = "--frozen" in arguments or {
                "--locked",
                "--offline",
            }.issubset(arguments)
        case _:
            observational = True
    return observational


def _uv_changes_dependencies(
    subcommand: str, subcommand_arguments: tuple[str, ...], arguments: list[str]
) -> bool:
    with_packages = any(
        argument.partition("=")[0]
        in {"--with", "--with-editable", "--with-requirements"}
        for argument in arguments
    )
    uv_pip_change = (
        subcommand == "pip"
        and bool(subcommand_arguments)
        and (command_name(subcommand_arguments[0]) in _UV_PIP_MUTATIONS)
    )
    uv_tool_change = subcommand == "tool" and any(
        action in subcommand_arguments
        for action in {"install", "run", "uninstall", "upgrade"}
    )
    uv_seeded_venv = subcommand == "venv" and "--seed" in subcommand_arguments
    return (
        subcommand in _UV_DEPENDENCY_MUTATIONS | {"build"}
        or with_packages
        or uv_pip_change
        or uv_tool_change
        or uv_seeded_venv
    )


def _project_execution(tokens: list[str]) -> tuple[bool, bool]:
    if not tokens:
        return False, False
    rendered = shlex.join(tokens)
    if is_verification_command(rendered):
        return True, False
    name = command_name(tokens[0])
    arguments = tokens[1:]
    if tokens[0].casefold() in {"%comspec%", "!comspec!"}:
        result = (False, False)
    elif token_is_dynamic(tokens[0]):
        return False, True
    elif re.fullmatch(r"python(?:3(?:\.\d+)*)?", name):
        result = _python_project_execution(arguments)
    elif name in _SCRIPT_MANAGERS:
        result = _script_project_execution(name, arguments)
    elif name == "make":
        parsed = _parsed_subcommand(name, arguments)
        result = (
            (False, True)
            if parsed.ambiguous
            else (True, token_is_dynamic(parsed.token or ""))
        )
    elif name == "dotnet":
        result = _dotnet_project_execution(arguments)
    elif name == "uv":
        parsed = _parsed_subcommand(name, arguments)
        result = (
            (False, True)
            if parsed.ambiguous
            else (command_name(parsed.token or "") == "run", False)
        )
    elif name in _PROJECT_SUBCOMMANDS:
        result = _subcommand_project_execution(name, arguments)
    else:
        result = (name in _PROJECT_RUNNERS, False)
    return result


def _python_project_execution(arguments: list[str]) -> tuple[bool, bool]:
    parsed = parse_python_module(arguments)
    dynamic = parsed.module is not None and token_is_dynamic(parsed.module)
    if parsed.ambiguous or dynamic:
        return False, True
    return parsed.module in _PYTHON_PROJECT_MODULES, False


def _script_project_execution(name: str, arguments: list[str]) -> tuple[bool, bool]:
    parsed = _parsed_subcommand(name, arguments)
    subcommand = command_name(parsed.token or "")
    if parsed.ambiguous or token_is_dynamic(parsed.token or ""):
        return False, True
    if subcommand == "run":
        dynamic = any(token_is_dynamic(argument) for argument in parsed.arguments[:1])
        return bool(parsed.arguments), dynamic
    return subcommand in {"build", "check", "lint", "test", "typecheck"}, False


def _subcommand_project_execution(name: str, arguments: list[str]) -> tuple[bool, bool]:
    parsed = _parsed_subcommand(name, arguments)
    if parsed.ambiguous or token_is_dynamic(parsed.token or ""):
        return False, True
    return command_name(parsed.token or "") in _PROJECT_SUBCOMMANDS[name], False


def _dotnet_project_execution(arguments: list[str]) -> tuple[bool, bool]:
    parsed = _parsed_subcommand("dotnet", arguments)
    if parsed.ambiguous:
        return False, True
    subcommand = parsed.token or ""
    if not subcommand:
        return False, False
    if token_is_dynamic(subcommand):
        return False, True
    return command_name(subcommand) != "help", False


def _executing_option_payloads(
    arguments: list[str], options: frozenset[str]
) -> _CarrierPayloads:
    payloads: list[str] = []
    ambiguous = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        option, separator, value = argument.partition("=")
        execution_option = next(
            (
                candidate
                for candidate in options
                if option == candidate
                or (option.startswith("--") and candidate.startswith(option))
            ),
            None,
        )
        if execution_option is None:
            index += 1
            continue
        if separator:
            payload = value
            index += 1
        elif index + 1 < len(arguments):
            payload = arguments[index + 1]
            index += 2
        else:
            ambiguous = True
            break
        if payload:
            payloads.append(payload)
        else:
            ambiguous = True
    return _CarrierPayloads(
        tuple(dict.fromkeys(payloads)),
        recognized=True,
        ambiguous=ambiguous,
        dynamic=any(token_is_dynamic(payload) for payload in payloads),
        masks_status=bool(payloads),
    )


def _sed_payload(arguments: list[str]) -> _CarrierPayloads:
    scripts: list[str] = []
    ambiguous = False
    program_supplied = False
    options_done = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if not options_done and argument == "--":
            options_done = True
            index += 1
            continue
        if not options_done and argument in {"--expression", "-e"}:
            if index + 1 >= len(arguments):
                ambiguous = True
                break
            scripts.append(arguments[index + 1])
            program_supplied = True
            index += 2
            continue
        if not options_done and argument.startswith("--expression="):
            scripts.append(argument.partition("=")[2])
            program_supplied = True
            index += 1
            continue
        if not options_done and argument.startswith("-e") and argument != "-e":
            scripts.append(argument[2:])
            program_supplied = True
            index += 1
            continue
        if not options_done and (
            argument in {"--file", "-f"} or argument.startswith(("--file=", "-f"))
        ):
            program_supplied = True
            ambiguous = True
            index += 2 if argument in {"--file", "-f"} else 1
            continue
        if not options_done and (
            argument in _SED_FLAGS
            or argument in {"--in-place", "-i"}
            or argument.startswith(("--in-place=", "-i"))
        ):
            index += 1
            continue
        if not options_done and argument.startswith("-") and argument != "-":
            ambiguous = True
            index += 1
            continue
        if not program_supplied:
            scripts.append(argument)
            program_supplied = True
        index += 1

    payloads: list[str] = []
    for script in scripts:
        commands, script_ambiguous = _sed_script_payloads(script)
        payloads.extend(commands)
        ambiguous = ambiguous or script_ambiguous
    return _CarrierPayloads(
        tuple(dict.fromkeys(payloads)),
        recognized=True,
        ambiguous=ambiguous,
        dynamic=any(token_is_dynamic(payload) for payload in payloads),
        masks_status=bool(payloads),
    )


def _sed_script_payloads(script: str) -> tuple[tuple[str, ...], bool]:
    payloads: list[str] = []
    ambiguous = False
    for expression in (_SED_DIRECT_EXECUTION, _SED_ALT_DIRECT_EXECUTION):
        for match in expression.finditer(script):
            payload = (match.group("payload") or "").strip()
            if payload:
                payloads.append(payload)
            else:
                ambiguous = True

    for expression in (_SED_SUBSTITUTION, _SED_ALT_SUBSTITUTION):
        for match in expression.finditer(script):
            delimiter = match.group("delimiter")
            pattern = _sed_delimited_field(script, match.end(), delimiter)
            if pattern is None:
                ambiguous = True
                continue
            _, replacement_start = pattern
            replacement = _sed_delimited_field(script, replacement_start, delimiter)
            if replacement is None:
                ambiguous = True
                continue
            rendered, flags_start = replacement
            flags_match = re.match(r"[A-Za-z0-9]*", script[flags_start:])
            flags = flags_match.group(0) if flags_match is not None else ""
            if "e" not in flags:
                continue
            payload = rendered.replace(f"\\{delimiter}", delimiter).replace(
                "\\\\", "\\"
            )
            if payload:
                payloads.append(payload)
            else:
                ambiguous = True
            ambiguous = ambiguous or bool(re.search(r"(?<!\\)&|\\[0-9]", rendered))
    return tuple(dict.fromkeys(payloads)), ambiguous


def _sed_delimited_field(
    script: str, start: int, delimiter: str
) -> tuple[str, int] | None:
    escaped = False
    value: list[str] = []
    for index in range(start, len(script)):
        character = script[index]
        if escaped:
            value.extend(("\\", character))
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == delimiter:
            return "".join(value), index + 1
        value.append(character)
    return None


def _mapfile_payload(arguments: list[str]) -> _CarrierPayloads:
    payloads: list[str] = []
    ambiguous = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        if argument == "-C":
            if index + 1 >= len(arguments):
                ambiguous = True
                break
            payloads.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("-C"):
            payload = argument[2:]
            if payload:
                payloads.append(payload)
            else:
                ambiguous = True
            index += 1
            continue
        index += 1
    return _CarrierPayloads(
        tuple(dict.fromkeys(payloads)),
        recognized=True,
        ambiguous=ambiguous,
        dynamic=any(token_is_dynamic(payload) for payload in payloads),
        masks_status=bool(payloads),
    )


def _before_option_terminator(arguments: list[str]) -> list[str]:
    return arguments[: arguments.index("--")] if "--" in arguments else arguments


def _has_long_option(arguments: list[str], candidates: frozenset[str]) -> bool:
    return any(
        _abbreviates_long_option(argument.partition("=")[0], candidate)
        for argument in _before_option_terminator(arguments)
        for candidate in candidates
    )


def _has_short_option(arguments: list[str], flags: frozenset[str]) -> bool:
    return any(
        argument.startswith("-")
        and not argument.startswith("--")
        and any(flag in argument[1:] for flag in flags)
        for argument in _before_option_terminator(arguments)
    )


def _arguments_require_explicit_authority(name: str, arguments: list[str]) -> bool:
    match name:
        case "less":
            required = (
                any(
                    argument.startswith("+") and len(argument) > 1
                    for argument in arguments
                )
                or _has_short_option(arguments, frozenset({"k", "t"}))
                or _has_long_option(
                    arguments,
                    frozenset({
                        "--cmd",
                        "--lesskey-content",
                        "--lesskey-file",
                        "--lesskey-src",
                        "--tag",
                    }),
                )
            )
        case "file":
            required = _file_short_effect(arguments) or _has_long_option(
                arguments,
                frozenset({
                    "--compile",
                    "--files-from",
                    "--no-sandbox",
                    "--uncompress",
                    "--uncompress-noreport",
                }),
            )
        case "find":
            required = any(
                argument in {"-H", "-L"}
                or argument.startswith(("-files0-from", "--files0-from"))
                for argument in _before_option_terminator(arguments)
            )
        case "sort" | "wc":
            required = _has_long_option(arguments, frozenset({"--files0-from"}))
        case checksum if checksum in _CHECKSUM_MANIFEST_COMMANDS:
            required = _has_short_option(
                arguments, frozenset({"c"})
            ) or _has_long_option(arguments, frozenset({"--check"}))
        case "grep":
            required = _has_short_option(
                arguments, frozenset({"R"})
            ) or _has_long_option(arguments, frozenset({"--dereference-recursive"}))
        case "rg":
            required = _has_short_option(
                arguments, frozenset({"L"})
            ) or _has_long_option(arguments, frozenset({"--follow"}))
        case "du":
            required = (
                _has_long_option(arguments, frozenset({"--files0-from"}))
                or _has_short_option(arguments, frozenset({"D", "H", "L"}))
                or _has_long_option(
                    arguments, frozenset({"--dereference", "--dereference-args"})
                )
            )
        case "diff":
            recursive = _has_short_option(
                arguments, frozenset({"r"})
            ) or _has_long_option(arguments, frozenset({"--recursive"}))
            no_dereference = _has_long_option(
                arguments, frozenset({"--no-dereference"})
            )
            required = recursive and not no_dereference
        case "tree":
            required = _has_short_option(
                arguments, frozenset({"l"})
            ) or _has_long_option(arguments, frozenset({"--followlinks"}))
        case "tail":
            required = _tail_reopens_by_name(arguments)
        case "ls":
            required = _has_short_option(
                arguments, frozenset({"H", "L"})
            ) or _has_long_option(
                arguments,
                frozenset({
                    "--dereference",
                    "--dereference-command-line",
                    "--dereference-command-line-symlink-to-dir",
                }),
            )
        case _:
            required = False
    return required


def _tail_reopens_by_name(arguments: list[str]) -> bool:
    for argument in _before_option_terminator(arguments):
        option, separator, value = argument.partition("=")
        if _abbreviates_long_option(option, "--retry"):
            return True
        if (
            separator
            and value.casefold() == "name"
            and _abbreviates_long_option(option, "--follow")
        ):
            return True
        if (
            argument.startswith("-")
            and not argument.startswith("--")
            and "F" in argument[1:]
        ):
            return True
    return False


def _file_short_effect(arguments: list[str]) -> bool:
    for argument in _before_option_terminator(arguments):
        if not argument.startswith("-") or argument.startswith("--"):
            continue
        for flag in argument[1:]:
            if flag in {"C", "S", "Z", "f", "z"}:
                return True
            if flag in {"F", "m", "P"}:
                break
    return False


def _less_payload(arguments: list[str]) -> _CarrierPayloads:
    commands: list[str] = []
    ambiguous = _arguments_require_explicit_authority("less", arguments)
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith("+") and len(argument) > 1:
            commands.append(argument.lstrip("+"))
        else:
            option, separator, value = argument.partition("=")
            if _abbreviates_long_option(option, "--cmd"):
                if separator:
                    commands.append(value)
                elif index + 1 >= len(arguments):
                    ambiguous = True
                else:
                    commands.append(arguments[index + 1])
                    index += 1
        index += 1

    payloads: list[str] = []
    for command in commands:
        stripped = command.lstrip()
        if not stripped:
            continue
        if stripped.startswith(("/", "?")):
            if "\n" in stripped:
                ambiguous = True
            continue
        match = re.search(r"[!#|v]", stripped)
        if match is None:
            continue
        marker = match.group(0)
        if marker == "v":
            ambiguous = True
            continue
        payload_start = match.end() + (1 if marker == "|" else 0)
        payload = stripped[payload_start:].strip()
        if payload:
            payloads.append(payload)
        else:
            ambiguous = True
    return _CarrierPayloads(
        tuple(dict.fromkeys(payloads)),
        recognized=True,
        ambiguous=ambiguous,
        dynamic=any(token_is_dynamic(payload) for payload in payloads),
        masks_status=bool(payloads),
    )


def _xargs_payload(arguments: list[str]) -> _CarrierPayloads:
    spec = OptionSpec(
        flags=frozenset({
            "--exit",
            "--interactive",
            "--null",
            "--no-run-if-empty",
            "--open-tty",
            "--show-limits",
            "--verbose",
            "-0",
            "-o",
            "-p",
            "-r",
            "-t",
            "-x",
        }),
        values=frozenset({
            "--arg-file",
            "--delimiter",
            "--max-args",
            "--max-chars",
            "--max-procs",
            "--process-slot-var",
            "-E",
            "-I",
            "-L",
            "-P",
            "-a",
            "-d",
            "-n",
            "-s",
        }),
        optional_values=frozenset({
            "--eof",
            "--max-lines",
            "--replace",
            "-e",
            "-i",
            "-l",
        }),
    )
    parsed = parse_leading_command(arguments, spec)
    if parsed.ambiguous:
        return _CarrierPayloads(recognized=True, ambiguous=True)
    if parsed.token is None:
        return _CarrierPayloads(recognized=True)
    tokens = (parsed.token, *parsed.arguments)
    payload = shlex.join(tokens)
    marker = _xargs_replacement_marker(arguments)
    replacements = (_ReplacementPayload(payload, marker),) if marker is not None else ()
    return _CarrierPayloads(
        (payload,),
        recognized=True,
        dynamic=any(token_is_dynamic(token) for token in tokens),
        replacements=replacements,
        masks_status=True,
    )


def _xargs_replacement_marker(arguments: list[str]) -> str | None:
    for index, argument in enumerate(arguments):
        if argument in {"--replace", "-i"}:
            return "{}"
        if argument == "-I":
            return arguments[index + 1] if index + 1 < len(arguments) else ""
        if argument.startswith("--replace="):
            return argument.partition("=")[2]
        if argument.startswith("-I") and argument != "-I":
            return argument[2:]
        if argument.startswith("-i") and argument != "-i":
            return argument[2:]
    return None


def _find_payloads(arguments: list[str]) -> _CarrierPayloads:
    payloads: list[str] = []
    ambiguous = _arguments_require_explicit_authority("find", arguments)
    index = 0
    while index < len(arguments):
        if arguments[index] not in _FIND_EXECUTION_PREDICATES:
            index += 1
            continue
        start = index + 1
        index = start
        while index < len(arguments) and arguments[index] not in {";", "+"}:
            index += 1
        if index >= len(arguments):
            return _CarrierPayloads(tuple(payloads), recognized=True, ambiguous=True)
        if index > start:
            tokens = arguments[start:index]
            payloads.append(shlex.join(tokens))
        index += 1
    replacements = tuple(_ReplacementPayload(payload, "{}") for payload in payloads)
    return _CarrierPayloads(
        tuple(payloads),
        recognized=True,
        ambiguous=ambiguous,
        dynamic=any(token_is_dynamic(payload) for payload in payloads),
        replacements=replacements,
        masks_status=bool(payloads),
    )


def _launcher_payload(name: str, arguments: list[str]) -> _CarrierPayloads | None:
    if name == "npx" and (call_payload := _npx_call_payload(arguments)) is not None:
        return call_payload
    if name in _PACKAGE_LAUNCHERS:
        parsed = parse_leading_command(arguments, _PACKAGE_LAUNCHER_OPTIONS)
    elif manager := _package_manager_launcher(name, arguments):
        parsed = manager
    else:
        return None
    if parsed.ambiguous:
        return _CarrierPayloads(recognized=True, ambiguous=True)
    if parsed.token is None:
        return _CarrierPayloads(recognized=True)
    tokens = (parsed.token, *parsed.arguments)
    return _CarrierPayloads(
        (shlex.join(tokens),),
        recognized=True,
        dynamic=any(token_is_dynamic(token) for token in tokens),
        masks_status=True,
    )


def _package_manager_launcher(
    name: str, arguments: list[str]
) -> ParsedLeadingCommand | None:
    expected = {
        "bun": frozenset({"x"}),
        "composer": frozenset({"exec"}),
        "npm": frozenset({"exec", "x"}),
        "pipx": frozenset({"run"}),
        "pnpm": frozenset({"dlx", "exec"}),
        "poetry": frozenset({"run"}),
        "yarn": frozenset({"dlx", "exec"}),
    }.get(name)
    if expected is None:
        return None
    spec = (
        _GENERIC_SCRIPT_OPTIONS
        if name in {"bun", "composer", "pipx", "poetry"}
        else _OPTION_SPECS[name]
    )
    outer = parse_leading_command(arguments, spec)
    if outer.ambiguous:
        return ParsedLeadingCommand(None, (), ambiguous=True)
    if command_name(outer.token or "") not in expected:
        return None
    return parse_leading_command(list(outer.arguments), _PACKAGE_LAUNCHER_OPTIONS)


def _npx_call_payload(arguments: list[str]) -> _CarrierPayloads | None:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--" or not argument.startswith("-"):
            break
        option, separator, value = argument.partition("=")
        if option in _PACKAGE_LAUNCHER_OPTIONS.flags:
            index += 1
            continue
        if option in _PACKAGE_LAUNCHER_OPTIONS.values:
            index += 1 if separator else _SHORT_OPTION_LENGTH
            continue
        if option not in {"--call", "-c"}:
            break
        if separator:
            payload = value
        elif argument.startswith("-c") and argument != "-c":
            payload = argument[2:]
        elif index + 1 < len(arguments):
            payload = arguments[index + 1]
        else:
            return _CarrierPayloads(recognized=True, ambiguous=True)
        return _CarrierPayloads(
            (payload,),
            recognized=True,
            dynamic=token_is_dynamic(payload),
            masks_status=True,
        )
    return None


def _trap_payload(arguments: list[str]) -> _CarrierPayloads:
    remaining = arguments[1:] if arguments[:1] == ["--"] else arguments
    if not remaining:
        return _CarrierPayloads(recognized=True)
    option = remaining[0]
    if option.startswith("-") and option != "-":
        flags = option[1:]
        return _CarrierPayloads(
            recognized=True, ambiguous=not flags or not set(flags) <= {"l", "p"}
        )
    if len(remaining) < _SHORT_OPTION_LENGTH or option in {"", "-"}:
        return _CarrierPayloads(recognized=True)
    return _CarrierPayloads(
        (option,), recognized=True, dynamic=token_is_dynamic(option), masks_status=True
    )


def _source_payload(arguments: list[str]) -> _CarrierPayloads:
    return _CarrierPayloads(recognized=True, ambiguous=bool(arguments))


def _enable_payload(arguments: list[str]) -> _CarrierPayloads:
    ambiguous = False
    for argument in arguments:
        if argument == "--":
            continue
        if argument == "-f" or (
            argument.startswith("-f") and len(argument) > _SHORT_OPTION_LENGTH
        ):
            ambiguous = True
            break
        if argument.startswith("-") and not set(argument[1:]) <= {
            "a",
            "d",
            "n",
            "p",
            "s",
        }:
            ambiguous = True
            break
    return _CarrierPayloads(recognized=True, ambiguous=ambiguous)


def _builtin_payload(arguments: list[str]) -> _CarrierPayloads:
    remaining = arguments[1:] if arguments[:1] == ["--"] else arguments
    if not remaining:
        return _CarrierPayloads(recognized=True)
    target = command_name(remaining[0])
    if target not in _STATEFUL_SHELL_BUILTINS and target not in {
        ".",
        "builtin",
        "command",
        "enable",
        "eval",
        "exec",
        "mapfile",
        "printf",
        "readarray",
        "source",
        "trap",
    }:
        return _CarrierPayloads(recognized=True)
    return _CarrierPayloads((shlex.join(remaining),), recognized=True)


def _rg_payload(arguments: list[str]) -> _CarrierPayloads:
    ambiguous = _arguments_require_explicit_authority("rg", arguments)
    for index, argument in enumerate(arguments):
        option, separator, value = argument.partition("=")
        if not any(
            _abbreviates_long_option(option, candidate)
            for candidate in {"--hostname-bin", "--pre"}
        ):
            continue
        if separator:
            payload = value
        elif index + 1 < len(arguments):
            payload = arguments[index + 1]
        else:
            return _CarrierPayloads(recognized=True, ambiguous=True)
        if not payload:
            return _CarrierPayloads(recognized=True, ambiguous=True)
        return _CarrierPayloads(
            (payload,),
            recognized=True,
            ambiguous=ambiguous,
            dynamic=token_is_dynamic(payload),
            masks_status=True,
        )
    return _CarrierPayloads(recognized=True, ambiguous=ambiguous)


def _git_payload(arguments: list[str]) -> _CarrierPayloads:
    if arguments and all(argument in {"--help", "--version"} for argument in arguments):
        return _CarrierPayloads(recognized=True)
    parsed = parse_leading_command(arguments, _GIT_GLOBAL_OPTIONS)
    if parsed.ambiguous or parsed.token is None:
        return _CarrierPayloads(recognized=True, ambiguous=True)
    if arguments[0] != parsed.token:
        return _CarrierPayloads(recognized=True, ambiguous=True)
    return _git_subcommand_payload(command_name(parsed.token), list(parsed.arguments))


def _git_subcommand_payload(
    subcommand: str, subcommand_arguments: list[str]
) -> _CarrierPayloads:
    if subcommand == "grep":
        result = _git_grep_payload(subcommand_arguments)
    elif subcommand in {"blame", "diff", "log", "show"}:
        result = _git_diff_payload(subcommand_arguments)
    elif subcommand == "status":
        result = _CarrierPayloads(recognized=True)
    elif subcommand == "cat-file":
        result = _git_cat_file_payload(subcommand_arguments)
    elif subcommand == "submodule":
        result = _git_submodule_payload(subcommand_arguments)
    elif subcommand == "bisect" and subcommand_arguments[:1] == ["run"]:
        result = _literal_command_payload(subcommand_arguments[1:])
    elif subcommand in _GIT_TERMINAL_SUBCOMMANDS:
        result = _CarrierPayloads(recognized=True, ambiguous=True)
    else:
        result = _CarrierPayloads(recognized=True, ambiguous=True)
    return result


def _is_standalone_direct_git(
    command: str, tokens: list[str], *, depth: int = 0
) -> bool:
    root = _get_parser().parse(command.encode("utf-8")).root_node
    return bool(
        depth == 0
        and tokens[0] == "git"
        and len(root.named_children) == 1
        and root.named_children[0].type == "command"
        and _tokenize(command) == tokens
    )


def _git_help_requested(arguments: list[str] | tuple[str, ...]) -> bool:
    before_terminator = (
        arguments[: arguments.index("--")] if "--" in arguments else arguments
    )
    return any(argument in {"--help", "-h"} for argument in before_terminator)


def harden_automated_command(command: str) -> str:
    tokens = _tokenize(command)
    match tokens:
        case ["git", subcommand, *arguments] if (
            subcommand in (_AUTOMATED_GIT_INSPECTIONS)
            and not _git_help_requested(arguments)
            and _is_standalone_direct_git(command, tokens)
        ):
            pass
        case _:
            return command
    hardened = ["git", *TRUSTED_GIT_CONFIG_ARGS, subcommand]
    if subcommand in {"log", "show"}:
        hardened.extend(("--no-ext-diff", "--no-textconv"))
    elif subcommand in {"blame", "grep"}:
        hardened.append("--no-textconv")
    hardened.extend(arguments)
    return shlex.join(hardened)


def _git_grep_payload(arguments: list[str]) -> _CarrierPayloads:
    payload: str | None = None
    ambiguous = False
    for index, argument in enumerate(arguments):
        if argument == "--":
            break
        option, separator, value = argument.partition("=")
        if _abbreviates_long_option(option, "--open-files-in-pager"):
            if separator:
                payload = value
                ambiguous = not payload
            else:
                ambiguous = True
            break
        if argument == "--no-textconv":
            continue
        if _abbreviates_long_option(option, "--textconv") and option != "--text":
            ambiguous = True
            break
        if argument.startswith("-") and not argument.startswith("--"):
            offset = argument[1:].find("O")
            if offset < 0:
                continue
            payload = argument[offset + 2 :]
            if not payload and index + 1 < len(arguments):
                candidate = arguments[index + 1]
                if candidate != "--" and not candidate.startswith("-"):
                    payload = candidate
            ambiguous = not payload
            break
    payloads = (payload,) if payload else ()
    return _CarrierPayloads(
        payloads,
        recognized=True,
        ambiguous=ambiguous,
        dynamic=bool(payload and token_is_dynamic(payload)),
        masks_status=bool(payload),
    )


def _git_diff_payload(arguments: list[str]) -> _CarrierPayloads:
    before_terminator = (
        arguments[: arguments.index("--")] if "--" in arguments else arguments
    )
    effectful = ("--ext-diff", "--output", "--show-signature", "--textconv")
    ambiguous = any(
        option not in {"--no-ext-diff", "--no-textconv", "--text"}
        and any(_abbreviates_long_option(option, candidate) for candidate in effectful)
        for argument in before_terminator
        for option in (argument.partition("=")[0],)
    ) or any("%G" in argument for argument in before_terminator)
    return _CarrierPayloads(recognized=True, ambiguous=ambiguous)


def _git_cat_file_payload(arguments: list[str]) -> _CarrierPayloads:
    before_terminator = (
        arguments[: arguments.index("--")] if "--" in arguments else arguments
    )
    effectful = ("--filters", "--textconv")
    ambiguous = any(
        any(
            _abbreviates_long_option(argument.partition("=")[0], candidate)
            for candidate in effectful
        )
        for argument in before_terminator
    )
    return _CarrierPayloads(recognized=True, ambiguous=ambiguous)


def _abbreviates_long_option(option: str, candidate: str) -> bool:
    return option == candidate or bool(
        option.startswith("--")
        and len(option) > _SHORT_OPTION_LENGTH
        and candidate.startswith(option)
    )


def _git_submodule_payload(arguments: list[str]) -> _CarrierPayloads:
    try:
        index = arguments.index("foreach")
    except ValueError:
        return _CarrierPayloads(recognized=True, ambiguous=True)
    index += 1
    while index < len(arguments) and arguments[index] == "--recursive":
        index += 1
    if index < len(arguments) and arguments[index] == "--":
        index += 1
    return _literal_command_payload(arguments[index:])


def _literal_command_payload(arguments: list[str]) -> _CarrierPayloads:
    if not arguments:
        return _CarrierPayloads(recognized=True, ambiguous=True)
    payload = " ".join(arguments)
    return _CarrierPayloads(
        (payload,),
        recognized=True,
        ambiguous=True,
        dynamic=token_is_dynamic(payload),
        masks_status=True,
    )


def _multicall_payload(arguments: list[str]) -> _CarrierPayloads:
    ambiguous = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            index += 1
            break
        if not argument.startswith("-") or argument == "-":
            break
        ambiguous = True
        index += 1
    if index >= len(arguments):
        return _CarrierPayloads(recognized=True, ambiguous=ambiguous)
    return _CarrierPayloads(
        (shlex.join(arguments[index:]),),
        recognized=True,
        ambiguous=ambiguous,
        masks_status=True,
    )


def _terminal_carrier(tokens: list[str]) -> bool:
    name = command_name(tokens[0])
    if name == "command" and _is_command_lookup(tokens):
        return True
    return bool(
        tokens[1:2]
        and tokens[1] in _NON_EXECUTING_WRAPPER_OPTIONS.get(name, frozenset())
    )


_CARRIER_HANDLERS: dict[str, Callable[[list[str]], _CarrierPayloads]] = {
    ".": _source_payload,
    "builtin": _builtin_payload,
    "enable": _enable_payload,
    "git": _git_payload,
    "less": _less_payload,
    "mapfile": _mapfile_payload,
    "readarray": _mapfile_payload,
    "rg": _rg_payload,
    "sed": _sed_payload,
    "source": _source_payload,
    "trap": _trap_payload,
}


def _carrier_payloads(tokens: list[str]) -> _CarrierPayloads:
    name = command_name(tokens[0])
    arguments = tokens[1:]
    if opaque := _opaque_carrier(tokens, name, arguments):
        result = opaque
    elif handler := _CARRIER_HANDLERS.get(name):
        result = handler(arguments)
    elif options := _EXECUTING_VALUE_OPTIONS.get(name):
        result = _executing_option_payloads(arguments, options)
        if _arguments_require_explicit_authority(name, arguments):
            result = result._replace(ambiguous=True)
    elif _terminal_carrier(tokens) or name in _ARGUMENTS_ARE_DATA:
        result = _CarrierPayloads(
            recognized=True,
            ambiguous=_arguments_require_explicit_authority(name, arguments),
        )
    elif name == "coproc":
        result = _CarrierPayloads(
            (shlex.join(arguments),) if arguments else (),
            recognized=True,
            ambiguous=not arguments,
            masks_status=bool(arguments),
        )
    elif name == "find":
        result = _find_payloads(arguments)
    elif name == "xargs":
        result = _xargs_payload(arguments)
    elif name in {"busybox", "toybox"}:
        result = _multicall_payload(arguments)
    elif name == "eval":
        unwrapped = _unwrapped_payload(tokens)
        result = unwrapped._replace(masks_status=bool(unwrapped.payloads))
    elif (launcher := _launcher_payload(name, arguments)) is not None:
        result = launcher
    elif name == "uv":
        unwrapped = _unwrapped_payload(tokens)
        result = (
            unwrapped
            if unwrapped.recognized
            else _package_manager_carrier(name, arguments)
        )
    elif name in _PACKAGE_MANAGERS or name in _SYSTEM_PACKAGE_MANAGERS:
        result = _package_manager_carrier(name, arguments)
    else:
        result = _unwrapped_payload(tokens)
    return result


def _opaque_carrier(
    tokens: list[str], name: str, arguments: list[str]
) -> _CarrierPayloads | None:
    if name in _OPAQUE_WRAPPERS or _opaque_code_interpreter(name):
        return _CarrierPayloads(recognized=True, ambiguous=True)
    if name not in _STDIN_SHELLS - {"fish"}:
        return None
    payload = _unwrapped_payload(tokens)
    return payload._replace(recognized=True, ambiguous=True)


def _opaque_code_interpreter(name: str) -> bool:
    return name in _OPAQUE_CODE_INTERPRETERS or bool(
        re.fullmatch(r"(?:lua|pypy|python)(?:\d+(?:\.\d+)*)?", name)
    )


def _package_manager_carrier(name: str, arguments: list[str]) -> _CarrierPayloads:
    if arguments and all(
        argument in {"--help", "--version", "-V", "-h", "-v"} for argument in arguments
    ):
        return _CarrierPayloads(recognized=True)
    if name == "pacman":
        observational = bool(arguments) and all(
            argument.startswith(("--query", "-Q")) for argument in arguments
        )
    else:
        parsed = _parsed_subcommand(name, arguments)
        inspections = (
            _SYSTEM_PACKAGE_INSPECTION_SUBCOMMANDS
            if name in _SYSTEM_PACKAGE_MANAGERS
            else _PACKAGE_INSPECTION_SUBCOMMANDS
        )
        observational = (
            not parsed.ambiguous
            and command_name(parsed.token or "") in inspections.get(name, frozenset())
            and _package_inspection_is_observational(
                name, command_name(parsed.token or ""), parsed.arguments
            )
        )
    return _CarrierPayloads(recognized=True, ambiguous=not observational)


def _unwrapped_payload(tokens: list[str]) -> _CarrierPayloads:
    unwrapped = unwrap_command_once(tokens)
    if unwrapped.shell_payload is not None:
        return _CarrierPayloads(
            (unwrapped.shell_payload,),
            recognized=True,
            ambiguous=unwrapped.ambiguous,
            dynamic=unwrapped.dynamic,
        )
    if unwrapped.ambiguous:
        child = (
            shlex.join(unwrapped.tokens)
            if unwrapped.changed
            and unwrapped.tokens
            and tuple(unwrapped.tokens) != tuple(tokens)
            else None
        )
        return _CarrierPayloads(
            (child,) if child is not None else (),
            recognized=True,
            ambiguous=True,
            dynamic=unwrapped.dynamic,
        )
    if not unwrapped.changed:
        return _CarrierPayloads()
    if not unwrapped.tokens:
        return _CarrierPayloads(
            recognized=True, ambiguous=True, dynamic=unwrapped.dynamic
        )
    return _CarrierPayloads(
        (shlex.join(unwrapped.tokens),), recognized=True, dynamic=unwrapped.dynamic
    )


def _direct_deferral_reason(
    tokens: list[str],
    *,
    attest_executable: bool = True,
    allow_shell_builtin: bool = True,
) -> str | None:
    if not tokens:
        return None
    name = command_name(tokens[0])
    if state_reason := _shell_state_deferral(tokens):
        reason = state_reason
    elif (
        _has_startup_environment_assignment(tokens)
        or _executable_token_is_dynamic(tokens[0])
        or _opaque_code_interpreter(name)
        or _command_uses_default_path(tokens)
    ):
        reason = _AMBIGUOUS_AUTHORITY_DEFERRAL
    elif name in _PRIVILEGED_EXECUTABLES:
        reason = _PRIVILEGED_EXECUTION_DEFERRAL
    elif dependency_reason := _dependency_reason(tokens):
        reason = dependency_reason
    else:
        project, ambiguous = _project_execution(tokens)
        if project:
            reason = _PROJECT_EXECUTION_DEFERRAL
        elif attest_executable and not _executable_is_trusted(
            tokens[0], name, allow_shell_builtin=allow_shell_builtin
        ):
            reason = _UNTRUSTED_EXECUTABLE_DEFERRAL
        elif ambiguous or not _carrier_payloads(tokens).recognized:
            reason = _AMBIGUOUS_AUTHORITY_DEFERRAL
        else:
            reason = None
    return reason


def _shell_state_deferral(tokens: list[str]) -> str | None:
    name = command_name(tokens[0])
    if name in _STATEFUL_SHELL_BUILTINS:
        return _SHELL_STATE_DEFERRAL
    if name == "printf" and any(
        argument == "-v" or argument.startswith("-v") for argument in tokens[1:]
    ):
        return _SHELL_STATE_DEFERRAL
    return None


def _git_authority_deferral(
    command: str, tokens: list[str], *, depth: int
) -> str | None:
    if command_name(tokens[0]) != "git":
        return None
    parsed = parse_leading_command(tokens[1:], _GIT_GLOBAL_OPTIONS)
    if parsed.ambiguous or parsed.token is None:
        return None
    subcommand = command_name(parsed.token)
    if subcommand in _GIT_WORKTREE_INSPECTIONS:
        return _GIT_WORKTREE_INSPECTION_DEFERRAL
    if subcommand not in _AUTOMATED_GIT_INSPECTIONS:
        return _GIT_INSPECTION_DEFERRAL
    if _git_help_requested(parsed.arguments):
        return _GIT_INSPECTION_DEFERRAL
    return (
        None
        if _is_standalone_direct_git(command, tokens, depth=depth)
        else _GIT_INSPECTION_DEFERRAL
    )


def _executable_is_trusted(token: str, name: str, *, allow_shell_builtin: bool) -> bool:
    if is_windows() or not sys.platform.startswith("linux"):
        return True
    if allow_shell_builtin and token in _SHELL_BUILTINS:
        return True
    try:
        if "/" not in token:
            resolve_trusted_system_executable(token)
        else:
            validate_trusted_system_executable(token)
    except TrustedCommandError:
        return False
    return True


def _command_uses_default_path(tokens: list[str]) -> bool:
    if command_name(tokens[0]) != "command" or _is_command_lookup(tokens):
        return False
    for argument in tokens[1:]:
        if argument == "--" or not argument.startswith("-"):
            return False
        if "p" in argument[1:]:
            return True
    return False


def _payload_allows_shell_builtin(tokens: list[str]) -> bool:
    return tokens[0] in _STDIN_SHELLS | {
        ".",
        "builtin",
        "command",
        "coproc",
        "eval",
        "mapfile",
        "readarray",
        "source",
        "trap",
    }


def _has_startup_environment_assignment(tokens: list[str]) -> bool:
    if token_is_assignment(tokens[0]):
        return True
    if command_name(tokens[0]) != "env":
        return False
    return bool(unwrap_command_once(tokens).environment_assignments)


def _executable_token_is_dynamic(token: str) -> bool:
    if token_is_dynamic(token) or any(marker in token for marker in ("*", "?", "[")):
        return True
    return "{" in token and "}" in token and ("," in token or ".." in token)


def _unproven_suffix_deferral(tokens: list[str]) -> str | None:
    for suffix in (tokens[index:] for index in range(1, len(tokens))):
        if command_name(suffix[0]) in _STDIN_SHELLS or _executable_token_is_dynamic(
            suffix[0]
        ):
            return _AMBIGUOUS_AUTHORITY_DEFERRAL
        if reason := _direct_deferral_reason(suffix, attest_executable=False):
            return reason
        _, separator, assigned = suffix[0].partition("=")
        for value in (suffix[0], assigned if separator else ""):
            nested = _tokenize(value)
            if not nested:
                continue
            if command_name(nested[0]) in _STDIN_SHELLS:
                return _AMBIGUOUS_AUTHORITY_DEFERRAL
            if reason := _direct_deferral_reason(nested, attest_executable=False):
                return reason
    return None


def _tokenized_parts(command_parts: list[str]) -> tuple[list[list[str]], bool]:
    tokens: list[list[str]] = []
    for part in command_parts:
        parsed = _tokenize(part)
        if parsed is None:
            return tokens, True
        if parsed:
            tokens.append(parsed)
    return tokens, False


def _replacement_is_structural(
    tokens: list[str], marker: str, carrier: _CarrierPayloads
) -> bool:
    name = command_name(tokens[0])
    if name in _STDIN_SHELLS | {"eval"} and any(
        marker in payload for payload in carrier.payloads
    ):
        return True
    raw_count = sum(token.count(marker) for token in tokens)
    unwrapped = unwrap_command_once(tokens)
    if unwrapped.changed:
        child_count = sum(token.count(marker) for token in unwrapped.tokens)
        if raw_count > child_count:
            return True
    launcher_names = _PACKAGE_LAUNCHERS | {
        "bun",
        "composer",
        "npm",
        "pipx",
        "pnpm",
        "poetry",
        "yarn",
    }
    payload_count = sum(payload.count(marker) for payload in carrier.payloads)
    return bool(
        carrier.payloads and name in launcher_names and raw_count > payload_count
    )


def _replacement_markers_for_payload(
    payload: str, active: frozenset[str], replacements: tuple[_ReplacementPayload, ...]
) -> tuple[frozenset[str], bool]:
    markers = {marker for marker in active if marker and marker in payload}
    ambiguous = False
    for replacement in replacements:
        if not replacement.marker:
            ambiguous = True
        elif replacement.body == payload and replacement.marker in payload:
            markers.add(replacement.marker)
    return frozenset(markers), ambiguous


def _replacement_markers_are_unsafe(
    tokens: list[str], markers: frozenset[str], carrier: _CarrierPayloads
) -> bool:
    name = command_name(tokens[0])
    for marker in markers:
        if marker in tokens[0]:
            return True
        marker_visible = any(marker in token for token in tokens)
        if marker_visible and name not in _REPLACEMENT_LITERAL_ARGUMENT_COMMANDS:
            return True
        if marker_visible and (
            _replacement_is_structural(tokens, marker, carrier)
            or not carrier.recognized
            or carrier.ambiguous
            or carrier.dynamic
        ):
            return True
    return False


def _marker_can_become_executable(payload: str, markers: frozenset[str]) -> bool:
    tokens = _tokenize(payload)
    return tokens is None or bool(
        tokens and any(marker in tokens[0] for marker in markers)
    )


def _initial_policy_deferral(command: str, ambiguous: bool) -> str | None:
    if _has_shell_state_mutation(command):
        return _SHELL_STATE_DEFERRAL
    return _AMBIGUOUS_AUTHORITY_DEFERRAL if ambiguous else None


def _prefer_shell_state_deferral(
    current: str | None, candidate: str | None
) -> str | None:
    if candidate == _SHELL_STATE_DEFERRAL:
        return candidate
    return current or candidate


def analyze_command_policy(
    command: str, command_parts: list[str], *, extract_commands: ExtractCommands
) -> CommandPolicyAnalysis:
    if denial := command_analysis_preflight_denial(command):
        return CommandPolicyAnalysis(denial=denial)
    initial, ambiguous = _tokenized_parts(command_parts)
    if sum(len(tokens) for tokens in initial) > _COMMAND_ANALYSIS_TOKEN_LIMIT:
        return CommandPolicyAnalysis(denial=_COMMAND_ANALYSIS_SIZE_LIMIT_DENIAL)
    pending = [_PendingCommand(tokens) for tokens in initial]
    ambiguous = (
        ambiguous
        or normalize_bash_ansi_c(command) != command
        or _has_dynamic_shell_arguments(command)
        or _get_parser().parse(command.encode("utf-8")).root_node.has_error
    )
    shell_input = _shell_stdin_payloads(command)
    for payload in shell_input.payloads:
        nested, nested_ambiguous = _tokenized_parts(extract_commands(payload.body))
        pending.extend(_PendingCommand(tokens, depth=1) for tokens in nested)
        ambiguous = ambiguous or nested_ambiguous
    ambiguous = ambiguous or shell_input.ambiguous
    deferral = _initial_policy_deferral(command, ambiguous)
    exact_fish_version = (
        tuple(split_bash_tokens(command))
        if _is_direct_fish_version_query(command)
        else None
    )
    seen: set[tuple[tuple[str, ...], frozenset[str], int, bool]] = set()

    while pending:
        current = pending.pop()
        tokens = current.tokens
        key = (
            tuple(tokens),
            current.replacement_markers,
            current.depth,
            current.allow_shell_builtin,
        )
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > _COMMAND_ANALYSIS_NODE_LIMIT:
            return CommandPolicyAnalysis(denial=_COMMAND_ANALYSIS_LIMIT_DENIAL)

        if command_name(tokens[0]) == "fish":
            if (
                exact_fish_version is not None
                and tuple(tokens) == exact_fish_version
                and not current.replacement_markers
            ):
                deferral = deferral or _FISH_VERSION_QUERY_DEFERRAL
                continue
            return CommandPolicyAnalysis(
                denial=_FISH_EXECUTION_DENIAL, deferral=deferral
            )

        deferral = deferral or _git_authority_deferral(
            command, tokens, depth=current.depth
        )
        deferral = _prefer_shell_state_deferral(
            deferral,
            _direct_deferral_reason(
                tokens, allow_shell_builtin=current.allow_shell_builtin
            ),
        )
        carrier = _carrier_payloads(tokens)
        if _replacement_markers_are_unsafe(
            tokens, current.replacement_markers, carrier
        ):
            deferral = deferral or _AMBIGUOUS_AUTHORITY_DEFERRAL
        if carrier.ambiguous or carrier.dynamic or carrier.masks_status:
            deferral = deferral or _AMBIGUOUS_AUTHORITY_DEFERRAL
        if not carrier.recognized:
            deferral = deferral or _unproven_suffix_deferral(tokens)
        for payload in carrier.payloads:
            markers, marker_ambiguous = _replacement_markers_for_payload(
                payload, current.replacement_markers, carrier.replacements
            )
            if marker_ambiguous or (
                markers and _marker_can_become_executable(payload, markers)
            ):
                deferral = deferral or _AMBIGUOUS_AUTHORITY_DEFERRAL
            if current.depth >= _MAX_DEPTH:
                return CommandPolicyAnalysis(
                    denial=_COMMAND_ANALYSIS_LIMIT_DENIAL, deferral=deferral
                )
            nested, nested_ambiguous = _tokenized_parts(extract_commands(payload))
            pending.extend(
                _PendingCommand(
                    tokens,
                    depth=current.depth + 1,
                    replacement_markers=markers,
                    allow_shell_builtin=_payload_allows_shell_builtin(current.tokens),
                )
                for tokens in nested
            )
            if nested_ambiguous:
                deferral = deferral or _AMBIGUOUS_AUTHORITY_DEFERRAL

    return CommandPolicyAnalysis(deferral=deferral)


def execution_match_candidates(
    command: str, *, extract_commands: ExtractCommands
) -> tuple[str, ...]:
    initial, _ = _tokenized_parts(extract_commands(command))
    pending = [_PendingCommand(tokens) for tokens in initial]
    candidates = [command]
    seen: set[tuple[tuple[str, ...], int]] = set()
    while pending and len(seen) < _COMMAND_ANALYSIS_NODE_LIMIT:
        current = pending.pop()
        key = (tuple(current.tokens), current.depth)
        if key in seen:
            continue
        seen.add(key)
        tokens = current.tokens
        candidates.append(shlex.join(tokens))
        normalized = (command_name(tokens[0]), *tokens[1:])
        candidates.append(shlex.join(normalized))
        if current.depth >= _MAX_DEPTH:
            continue
        carrier = _carrier_payloads(tokens)
        for payload in carrier.payloads:
            nested, _ = _tokenized_parts(extract_commands(payload))
            pending.extend(
                _PendingCommand(tokens, depth=current.depth + 1) for tokens in nested
            )
    return tuple(dict.fromkeys(candidates))


def _masking_shell_operators(command: str) -> frozenset[str]:
    source = command.encode("utf-8")
    tree = _get_parser().parse(source)
    operators: set[str] = set()

    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.type == "command_substitution":
            operators.add("$(")
        elif node.type == "process_substitution":
            operators.add("<(")
        for child in node.children:
            if not child.is_named and child.type in _FAILURE_MASKING_OPERATORS:
                operators.add(child.type)
        for left, right in zip(node.children, node.children[1:], strict=False):
            if b"\n" in source[left.end_byte : right.start_byte]:
                operators.add("\n")
        pending.extend(reversed(node.children))

    return frozenset(operators)


def auto_approval_blocker(command: str) -> str | None:
    try:
        split_bash_tokens(command)
    except ValueError:
        return (
            "Command could not be tokenized (unbalanced quotes); the "
            "validator's view may not match what the shell executes."
        )
    operator = _auto_approval_operator(command)
    if operator is None:
        return None
    return (
        f"Command uses shell operator '{operator}'. The allowlist inspects "
        "only the leading command word, so it cannot soundly auto-approve "
        "composition or output redirection."
    )


def _auto_approval_operator(command: str) -> str | None:
    source = command.encode("utf-8")
    tree = _get_parser().parse(source)
    if tree.root_node.has_error:
        return "invalid syntax"
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if operator := _AUTO_APPROVAL_OPERATOR_NODES.get(node.type):
            return operator
        if node.type == "file_redirect":
            rendered = source[node.start_byte : node.end_byte].decode("utf-8")
            if match := _OUTPUT_REDIRECT_RE.match(rendered):
                return match.group(0).strip()
        for child in node.children:
            if not child.is_named and child.type in _AUTO_APPROVAL_UNNAMED_OPERATORS:
                return child.type
        for left, right in zip(node.children, node.children[1:], strict=False):
            if b"\n" in source[left.end_byte : right.start_byte]:
                return "newline"
        pending.extend(reversed(node.children))
    return None


def _contains_project_execution(
    command_parts: list[str], extract_commands: ExtractCommands, *, depth: int
) -> tuple[bool, bool]:
    if depth > _MAX_DEPTH:
        return False, True
    ambiguous_result = False
    for part in command_parts:
        tokens = _tokenize(part)
        if tokens is None:
            ambiguous_result = True
            continue
        project, ambiguous = _project_execution(tokens)
        if project:
            return True, ambiguous
        ambiguous_result = ambiguous_result or ambiguous
        carrier = _carrier_payloads(tokens)
        ambiguous_result = ambiguous_result or carrier.ambiguous or carrier.dynamic
        for payload in carrier.payloads:
            nested_project, nested_ambiguous = _contains_project_execution(
                extract_commands(payload), extract_commands, depth=depth + 1
            )
            if nested_project:
                return True, ambiguous_result or nested_ambiguous
            ambiguous_result = ambiguous_result or nested_ambiguous
    return False, ambiguous_result


def masked_verification_status_reason(
    command: str,
    command_parts: list[str],
    *,
    extract_commands: ExtractCommands,
    depth: int = 0,
) -> str | None:
    if depth > _MAX_DEPTH:
        return None
    shell_input = _shell_stdin_payloads(command)
    for payload in shell_input.payloads:
        if masked_verification_status_reason(
            payload.body,
            extract_commands(payload.body),
            extract_commands=extract_commands,
            depth=depth + 1,
        ):
            return _MASKED_STATUS_DENIAL
    if any(
        _part_masks_verification(part, extract_commands, depth=depth)
        for part in command_parts
    ):
        return _MASKED_STATUS_DENIAL
    if not _masking_shell_operators(command):
        return None
    project, _ = _contains_project_execution(
        command_parts, extract_commands, depth=depth
    )
    return _MASKED_STATUS_DENIAL if project else None


def _part_masks_verification(
    part: str, extract_commands: ExtractCommands, *, depth: int
) -> bool:
    tokens = _tokenize(part)
    if not tokens:
        return False
    carrier = _carrier_payloads(tokens)
    for payload in carrier.payloads:
        if masked_verification_status_reason(
            payload,
            extract_commands(payload),
            extract_commands=extract_commands,
            depth=depth + 1,
        ):
            return True
        if carrier.masks_status:
            project, _ = _contains_project_execution(
                extract_commands(payload), extract_commands, depth=depth + 1
            )
            if project:
                return True
    return False
