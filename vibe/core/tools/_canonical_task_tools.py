from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from vibe.core.tools.base import BaseTool
from vibe.core.tools.builtins.bash import Bash
from vibe.core.tools.builtins.edit import Edit
from vibe.core.tools.builtins.glob import Glob
from vibe.core.tools.builtins.grep import Grep
from vibe.core.tools.builtins.lsp import Lsp
from vibe.core.tools.builtins.read import Read
from vibe.core.tools.builtins.skill import Skill
from vibe.core.tools.builtins.task import Task
from vibe.core.tools.builtins.task_checks import TaskChecks
from vibe.core.tools.builtins.todo import Todo
from vibe.core.tools.builtins.tool_search import ToolSearch
from vibe.core.tools.builtins.verify_work import VerifyWork
from vibe.core.tools.builtins.webfetch import WebFetch
from vibe.core.tools.builtins.websearch import WebSearch
from vibe.core.tools.builtins.write_file import WriteFile

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


_MANAGED_ACTIVE_ROOT_TOOLS = frozenset({
    "bash",
    "edit",
    "glob",
    "grep",
    "read",
    "skill",
    "task",
    "todo",
    "write_file",
})
_MANAGED_VERIFICATION_ROOT_TOOLS = frozenset({
    "glob",
    "grep",
    "read",
    "skill",
    "task",
    "verify_work",
})
_MANAGED_SUBAGENT_TOOLS = frozenset({
    "bash",
    "glob",
    "grep",
    "read",
    "skill",
    "task_checks",
})

_CANONICAL_TASK_TOOLS = MappingProxyType({
    tool.get_name(): tool
    for tool in (
        Bash,
        Edit,
        Glob,
        Grep,
        Lsp,
        Read,
        Skill,
        Task,
        TaskChecks,
        Todo,
        ToolSearch,
        VerifyWork,
        WebFetch,
        WebSearch,
        WriteFile,
    )
})


def canonical_task_tools(allowlist: frozenset[str]) -> dict[str, type[BaseTool]]:
    return {
        name: tool for name, tool in _CANONICAL_TASK_TOOLS.items() if name in allowlist
    }


def managed_tool_allowlist(
    config: VibeConfig, *, is_subagent: bool, task_allowlist: frozenset[str] | None
) -> frozenset[str] | None:
    recipe = config.trusted_verification_recipe
    topology = recipe.execution_topology if recipe is not None else None
    if topology is None:
        return task_allowlist

    if is_subagent:
        ceiling = _MANAGED_SUBAGENT_TOOLS
        if task_allowlist is None:
            return ceiling - {"task_checks"}
    elif topology.state == "active":
        ceiling = _MANAGED_ACTIVE_ROOT_TOOLS
    else:
        ceiling = _MANAGED_VERIFICATION_ROOT_TOOLS

    if task_allowlist is None:
        return ceiling
    return ceiling & task_allowlist


__all__ = ["canonical_task_tools", "managed_tool_allowlist"]
