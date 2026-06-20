from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.hooks.models import HookConfig

# Conventional component subdir names used when a manifest omits the field.
_DEFAULT_DIRS = {
    "agents": "agents",
    "skills": "skills",
    "tools": "tools",
    "workflows": "workflows",
    "prompts": "prompts",
}


class PluginManifest(BaseModel):
    """A ``plugin.toml`` at a plugin root. Component fields are dir names
    (str or list) relative to the plugin root; omitted ones default to the
    conventional subdir name (only added if it exists on disk).
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    version: str = ""
    description: str = ""
    agents: str | list[str] | None = None
    skills: str | list[str] | None = None
    tools: str | list[str] | None = None
    workflows: str | list[str] | None = None
    prompts: str | list[str] | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    hooks: str | list[dict[str, Any]] | None = None

    def component_dirs(self, component: str) -> list[str]:
        raw = getattr(self, component)
        if raw is None:
            return [_DEFAULT_DIRS[component]]
        return [raw] if isinstance(raw, str) else list(raw)


class PluginLoadResult(BaseModel):
    plugins: list[str] = Field(default_factory=list)
    agent_paths: list[Path] = Field(default_factory=list)
    skill_paths: list[Path] = Field(default_factory=list)
    tool_paths: list[Path] = Field(default_factory=list)
    workflow_paths: list[Path] = Field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    hooks: list[HookConfig] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
