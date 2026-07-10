from __future__ import annotations

import functools
import hashlib
import re
from typing import Any

import orjson

from vibe import __version__
from vibe.core.agents.models import AgentSafety, AgentType, profile_requires_isolation

_CACHE_POLICY_VERSION = 4
_CACHEABLE_PROFILES = frozenset({"explore", "planner"})
_CACHEABLE_TOOLS = frozenset({"glob", "grep", "lsp", "read"})
_DEPENDENCY_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


@functools.cache
def _trusted_cacheable_tools() -> dict[str, type[Any]]:
    from vibe.core.tools.builtins.glob import Glob
    from vibe.core.tools.builtins.grep import Grep
    from vibe.core.tools.builtins.lsp import Lsp
    from vibe.core.tools.builtins.read import Read

    return {"glob": Glob, "grep": Grep, "lsp": Lsp, "read": Read}


def _repository_fingerprint() -> str | None:
    from vibe.core._workspace_verification import workspace_fingerprint

    return workspace_fingerprint()


def _effective_model_alias(context: Any, agent: str, requested: str | None) -> str:
    if requested:
        return requested
    manager = getattr(context, "agent_manager", None)
    config = getattr(manager, "config", None)
    if config is None:
        return ""
    grunt_model = getattr(config, "grunt_model", "")
    if agent == "grunt" and grunt_model:
        return grunt_model
    if subagent_model := getattr(config, "subagent_model", ""):
        return subagent_model
    return getattr(context, "active_model", None) or getattr(config, "active_model", "")


def _model_payload(context: Any, alias: str) -> dict[str, Any] | None:
    manager = getattr(context, "agent_manager", None)
    config = getattr(manager, "config", None)
    if config is None:
        return None
    model = next(
        (
            candidate
            for candidate in getattr(config, "models", ())
            if candidate.alias == alias
        ),
        None,
    )
    if model is None:
        return None
    provider = next(
        (
            candidate
            for candidate in getattr(config, "providers", ())
            if candidate.name == model.provider
        ),
        None,
    )
    if provider is None:
        return None
    return {
        "alias": alias,
        "model": model.model_dump(mode="json"),
        "provider": provider.model_dump(mode="json"),
    }


def _routing_payload(context: Any) -> dict[str, Any] | None:
    manager = getattr(context, "agent_manager", None)
    config = getattr(manager, "config", None)
    routing = getattr(config, "model_routing", None)
    if routing is None:
        return None
    formatter_alias = routing.formatter_model
    formatter = _model_payload(context, formatter_alias) if formatter_alias else None
    if formatter_alias and formatter is None:
        return None
    semantic_alias = routing.semantic_escalation_model
    semantic = _model_payload(context, semantic_alias) if semantic_alias else None
    if semantic_alias and semantic is None:
        return None
    return {
        "policy": routing.model_dump(mode="json"),
        "formatter": formatter,
        "semantic_escalation": semantic,
    }


def _profile_payload(context: Any, agent: str) -> dict[str, Any] | None:
    manager = getattr(context, "agent_manager", None)
    if manager is None or agent not in _CACHEABLE_PROFILES:
        return None
    try:
        profile = manager.get_agent(agent)
    except ValueError:
        return None
    if (
        profile.agent_type is not AgentType.SUBAGENT
        or profile.safety is not AgentSafety.SAFE
        or profile_requires_isolation(profile)
    ):
        return None
    enabled = profile.overrides.get("enabled_tools")
    if not isinstance(enabled, list) or not enabled:
        return None
    names = frozenset(str(name) for name in enabled)
    if not names <= _CACHEABLE_TOOLS:
        return None
    return {
        "name": str(profile.name),
        "safety": profile.safety.value,
        "overrides": profile.overrides,
        "manifest_names": sorted(names),
    }


def _manifest_payload(
    context: Any, profile: dict[str, Any]
) -> list[dict[str, Any]] | None:
    manager = getattr(context, "tool_manager", None)
    if manager is None:
        return None
    registered = getattr(manager, "registered_tools", None)
    if not isinstance(registered, dict):
        return None
    trusted = _trusted_cacheable_tools()
    tools: list[dict[str, Any]] = []
    for name in profile["manifest_names"]:
        tool = registered.get(name)
        trusted_tool = trusted.get(name)
        if (
            tool is None
            or trusted_tool is None
            or tool is not trusted_tool
            or not getattr(tool, "read_only", False)
        ):
            return None
        tool_config = manager.get_tool_config(name)
        tools.append({
            "name": name,
            "implementation": f"{tool.__module__}:{tool.__qualname__}",
            "description": tool.description,
            "parameters": tool.get_parameters(),
            "policy": tool_config.model_dump(mode="json"),
        })
    return tools


def workflow_cache_context(
    context: Any,
    *,
    agent: str,
    model: str | None,
    trusted_dependency_fingerprint: str | None = None,
) -> str | None:
    if (
        context is None
        or not isinstance(trusted_dependency_fingerprint, str)
        or _DEPENDENCY_FINGERPRINT.fullmatch(trusted_dependency_fingerprint) is None
        or (repository := _repository_fingerprint()) is None
    ):
        return None
    profile = _profile_payload(context, agent)
    if profile is None:
        return None
    alias = _effective_model_alias(context, agent, model)
    model_payload = _model_payload(context, alias)
    manifest = _manifest_payload(context, profile)
    routing = _routing_payload(context)
    if model_payload is None or manifest is None or routing is None:
        return None
    payload = {
        "version": _CACHE_POLICY_VERSION,
        "complete_dependencies": trusted_dependency_fingerprint,
        "repository": repository,
        "harness": __version__,
        "agent": profile,
        "model": model_payload,
        "model_routing": routing,
        "manifest": manifest,
    }
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(encoded).hexdigest()


def workflow_cache_identity(
    context: Any,
    *,
    agent: str,
    model: str | None,
    isolation: str | None,
    then: str | None,
    contract: object | None,
    citations: object | None = None,
    trusted_dependency_fingerprint: str | None = None,
) -> str | None:
    if (
        isolation is not None
        or then is not None
        or contract is not None
        or citations is not None
    ):
        return None
    return workflow_cache_context(
        context,
        agent=agent,
        model=model,
        trusted_dependency_fingerprint=trusted_dependency_fingerprint,
    )


__all__ = ["workflow_cache_context", "workflow_cache_identity"]
