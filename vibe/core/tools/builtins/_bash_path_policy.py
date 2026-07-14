"""Conservative path extraction for Bash permission boundaries."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re

from tree_sitter import Language, Parser
import tree_sitter_bash as tsbash

from vibe.core.scratchpad import is_scratchpad_path
from vibe.core.tools._command_tokens import (
    command_name,
    split_bash_tokens,
    token_is_dynamic,
    unwrap_command_tokens,
)
from vibe.core.tools.utils import is_path_within_workdir

_PATH_OPERAND_COMMANDS = frozenset({
    "cat",
    "cd",
    "chmod",
    "chown",
    "cksum",
    "comm",
    "cp",
    "cut",
    "date",
    "diff",
    "du",
    "file",
    "find",
    "fmt",
    "fold",
    "git",
    "grep",
    "head",
    "join",
    "less",
    "ls",
    "md5sum",
    "mkdir",
    "more",
    "mv",
    "nl",
    "od",
    "paste",
    "readlink",
    "rm",
    "sha1sum",
    "sha256sum",
    "sha512sum",
    "shasum",
    "sort",
    "stat",
    "sum",
    "tac",
    "tail",
    "touch",
    "tree",
    "uniq",
    "wc",
})
_INPUT_FILE_REDIRECT = re.compile(r"^\s*\d*(?:<>|<(?![<&(]))")
_BRACE_EXPANSION = re.compile(r"\{[^{}]*(?:,|\.\.)[^{}]*\}")
_QUOTED_GLOB_BYTES = {ord("*"): 0x1C, ord("?"): 0x1D, ord("["): 0x1E, ord("]"): 0x1F}
_RAW_STRING_DYNAMIC_BYTES = {ord("$"): 0x18, ord("`"): 0x19}
_ATTACHED_PATH_OPTIONS = {
    "date": ("-f", "-r"),
    "file": ("-f", "-m"),
    "grep": ("-f",),
    "sort": ("-T",),
}


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


def collect_outside_dirs(
    command_parts: list[str], raw_command: str | None = None
) -> set[str]:
    directories: set[str] = set()
    for part in command_parts:
        tokens = _tokens(part)
        if not tokens or command_name(tokens[0]) not in _PATH_OPERAND_COMMANDS:
            continue
        for raw_token in tokens[1:]:
            token = _option_value(command_name(tokens[0]), raw_token)
            if token is None or (
                command_name(tokens[0]) == "chmod" and token.startswith("+")
            ):
                continue
            if directory := _outside_directory(token):
                directories.add(directory)
    if raw_command is not None:
        for token in _input_redirect_paths(raw_command):
            if directory := _outside_directory(token):
                directories.add(directory)
    return directories


def _input_redirect_paths(command: str) -> tuple[str, ...]:
    source = command.encode("utf-8")
    tree = _get_parser().parse(source)
    paths: list[str] = []

    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.type == "file_redirect":
            rendered = source[node.start_byte : node.end_byte].decode("utf-8")
            destination = node.child_by_field_name("destination")
            if _INPUT_FILE_REDIRECT.match(rendered) and destination is not None:
                raw = source[destination.start_byte : destination.end_byte].decode(
                    "utf-8"
                )
                try:
                    tokens = split_bash_tokens(_protect_quoted_literals(raw))
                except ValueError:
                    tokens = []
                if len(tokens) == 1:
                    paths.append(tokens[0])
        pending.extend(reversed(node.named_children))
    return tuple(paths)


def _outside_directory(token: str) -> str | None:
    candidate = _expand_path_token(token)
    if candidate is None:
        return None
    if candidate == os.sep and token_is_dynamic(token):
        return os.sep
    if is_path_within_workdir(candidate) or is_scratchpad_path(candidate):
        return None
    try:
        resolved = Path(candidate).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        resolved = resolved.resolve()
        return str(resolved if resolved.is_dir() else resolved.parent)
    except (OSError, RuntimeError):
        return os.sep


def _tokens(command: str) -> list[str]:
    try:
        raw = split_bash_tokens(_protect_quoted_literals(command))
    except ValueError:
        return []
    unwrapped = unwrap_command_tokens(raw)
    if not unwrapped.ambiguous and unwrapped.tokens:
        return list(unwrapped.tokens)
    return raw


def _option_value(name: str, token: str) -> str | None:
    if token == "--" or not token.startswith("-"):
        return token
    _, separator, value = token.partition("=")
    if separator:
        return value
    for option in _ATTACHED_PATH_OPTIONS.get(name, ()):
        if token.startswith(option) and token != option:
            return token[len(option) :]
    if os.sep in token:
        return token[token.index(os.sep) :]
    if token_is_dynamic(token):
        return token
    return None


def _expand_path_token(token: str) -> str | None:
    if _BRACE_EXPANSION.search(token):
        return os.sep
    if any(marker in token for marker in "*?["):
        return os.sep
    expanded = os.path.expandvars(_expand_bash_tilde(token))
    if token_is_dynamic(expanded) and "$(" not in expanded and "`" not in expanded:
        return os.sep
    path_like = (
        expanded.startswith((os.sep, "~", ".")) or os.sep in expanded or os.sep in token
    )
    if not path_like:
        return None
    return expanded


def _expand_bash_tilde(token: str) -> str:
    if token == "~+" or token.startswith("~+/"):
        return str(Path.cwd()) + token[2:]
    if token == "~-" or token.startswith("~-/"):
        oldpwd = os.environ.get("OLDPWD")
        return os.sep if oldpwd is None else oldpwd + token[2:]
    expanded = os.path.expanduser(token)
    return os.sep if expanded.startswith("~") else expanded


def _protect_quoted_literals(command: str) -> str:
    source = bytearray(command.encode("utf-8"))
    tree = _get_parser().parse(bytes(source))
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        replacements: dict[int, int] = {}
        if node.type in {"ansi_c_string", "raw_string", "string"}:
            replacements.update(_QUOTED_GLOB_BYTES)
        if node.type == "raw_string":
            replacements.update(_RAW_STRING_DYNAMIC_BYTES)
        if replacements:
            for index in range(node.start_byte, node.end_byte):
                if replacement := replacements.get(source[index]):
                    source[index] = replacement
        pending.extend(reversed(node.named_children))
    return source.decode("utf-8")
