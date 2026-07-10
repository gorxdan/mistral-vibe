from __future__ import annotations

from types import MappingProxyType

from vibe.core.tools.base import BaseTool
from vibe.core.tools.builtins.edit import Edit
from vibe.core.tools.builtins.glob import Glob
from vibe.core.tools.builtins.grep import Grep
from vibe.core.tools.builtins.lsp import Lsp
from vibe.core.tools.builtins.read import Read
from vibe.core.tools.builtins.skill import Skill
from vibe.core.tools.builtins.task_checks import TaskChecks
from vibe.core.tools.builtins.todo import Todo
from vibe.core.tools.builtins.tool_search import ToolSearch
from vibe.core.tools.builtins.webfetch import WebFetch
from vibe.core.tools.builtins.websearch import WebSearch
from vibe.core.tools.builtins.write_file import WriteFile

_CANONICAL_TASK_TOOLS = MappingProxyType({
    tool.get_name(): tool
    for tool in (
        Edit,
        Glob,
        Grep,
        Lsp,
        Read,
        Skill,
        TaskChecks,
        Todo,
        ToolSearch,
        WebFetch,
        WebSearch,
        WriteFile,
    )
})


def canonical_task_tools(allowlist: frozenset[str]) -> dict[str, type[BaseTool]]:
    return {
        name: tool for name, tool in _CANONICAL_TASK_TOOLS.items() if name in allowlist
    }


__all__ = ["canonical_task_tools"]
