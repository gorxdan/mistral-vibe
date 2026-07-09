from __future__ import annotations

import hashlib
from typing import Any

import orjson

from vibe import __version__


def _repository_fingerprint() -> str:
    from vibe.core._workspace_verification import workspace_fingerprint

    return workspace_fingerprint() or "outside-git"


def _effective_model_alias(context: Any, agent: str, requested: str | None) -> str:
    if requested:
        return requested
    manager = getattr(context, "agent_manager", None)
    config = getattr(manager, "config", None)
    if config is None:
        return getattr(context, "active_model", None) or ""
    grunt_model = getattr(config, "grunt_model", "")
    if agent == "grunt" and grunt_model:
        return grunt_model
    if subagent_model := getattr(config, "subagent_model", ""):
        return subagent_model
    return getattr(context, "active_model", None) or getattr(config, "active_model", "")


def _model_payload(context: Any, alias: str) -> dict[str, Any]:
    manager = getattr(context, "agent_manager", None)
    config = getattr(manager, "config", None)
    if config is None:
        return {"alias": alias}
    for model in getattr(config, "models", ()):
        if model.alias == alias:
            return model.model_dump(mode="json")
    return {"alias": alias}


def _profile_payload(context: Any, agent: str) -> dict[str, Any]:
    manager = getattr(context, "agent_manager", None)
    if manager is None:
        return {}
    try:
        profile = manager.get_agent(agent)
    except ValueError:
        return {}
    safety = getattr(profile, "safety", "")
    return {
        "name": getattr(profile, "name", agent),
        "safety": getattr(safety, "value", safety),
        "overrides": getattr(profile, "overrides", {}),
    }


def _manifest_payload(context: Any) -> list[dict[str, Any]]:
    manager = getattr(context, "tool_manager", None)
    if manager is None:
        return []
    tools: list[dict[str, Any]] = []
    manifest = getattr(manager, "manifest_tools", {})
    for name, tool in sorted(manifest.items()):
        tools.append({
            "name": name,
            "description": tool.description,
            "parameters": tool.get_parameters(),
        })
    return tools


def workflow_cache_context(context: Any, *, agent: str, model: str | None) -> str:
    alias = _effective_model_alias(context, agent, model)
    payload = {
        "version": 1,
        "repository": _repository_fingerprint(),
        "harness": __version__,
        "agent": _profile_payload(context, agent),
        "model": _model_payload(context, alias),
        "manifest": _manifest_payload(context),
    }
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(encoded).hexdigest()
