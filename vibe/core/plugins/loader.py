"""Plugin/bundle manifest loader.

A plugin is a directory with a ``plugin.toml`` declaring component subdirs
(agents/skills/tools/workflows/prompts), MCP servers, and hooks. The loader is
purely additive: it resolves a manifest into the EXISTING config path lists +
mcp list, so no manager ever touches plugins directly. It is fully defensive —
malformed manifests, name collisions, and path-escapes degrade to issues and
NEVER raise, so a bad plugin can't break startup.
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import TYPE_CHECKING

from pydantic import ValidationError

from vibe.core.logger import logger
from vibe.core.plugins.models import PluginLoadResult, PluginManifest
from vibe.core.utils.matching import name_matches

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

_COMPONENTS = (
    ("agents", "agent_paths"),
    ("skills", "skill_paths"),
    ("tools", "tool_paths"),
    ("workflows", "workflow_paths"),
    ("prompts", "prompt_paths"),
)


def _candidate_manifests(
    plugin_paths: list[Path], plugin_dirs: list[Path]
) -> list[Path]:
    """Resolve explicit plugin roots/files + ``<dir>/*/plugin.toml`` entries."""
    out: list[Path] = []
    for p in plugin_paths:
        if p.is_file() and p.name == "plugin.toml":
            out.append(p)
        elif p.is_dir() and (p / "plugin.toml").is_file():
            out.append(p / "plugin.toml")
    for d in plugin_dirs:
        if not d.is_dir():
            continue
        d_resolved = d.resolve()
        for child in sorted(d.iterdir()):
            # Confinement: a symlinked child plugin whose target escapes the
            # discovered plugins dir must not be loaded — it could pull an
            # arbitrary on-disk plugin (agents/skills/tools/hooks/mcp) into the
            # harness. Mirrors the per-path confinement applied to sibling
            # discovery (tools/skills/agents/workflows).
            try:
                child_resolved = child.resolve()
            except (OSError, ValueError):
                continue
            if not _within(child_resolved, d_resolved):
                logger.warning(
                    "plugin child %s escapes plugins dir %s; skipping",
                    child,
                    d_resolved,
                )
                continue
            manifest = child / "plugin.toml"
            if manifest.is_file():
                out.append(manifest)
    # de-dupe preserving order
    seen: set[Path] = set()
    return [m for m in out if not (m in seen or seen.add(m))]


def load_plugins_from_fs(
    plugin_paths: list[Path],
    plugin_dirs: list[Path] | None = None,
    *,
    enabled: list[str] | None = None,
    disabled: list[str] | None = None,
) -> PluginLoadResult:
    result = PluginLoadResult()
    seen_names: set[str] = set()

    for manifest_path in _candidate_manifests(plugin_paths, plugin_dirs or []):
        root = manifest_path.parent
        try:
            with manifest_path.open("rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            result.issues.append(f"{manifest_path}: unreadable/invalid TOML ({e})")
            continue
        try:
            manifest = PluginManifest.model_validate(data)
        except ValidationError as e:
            result.issues.append(f"{manifest_path}: invalid manifest ({e})")
            continue

        name = manifest.name
        # enabled allowlist wins; else disabled denylist.
        if enabled:
            if not name_matches(name, enabled):
                continue
        elif disabled and name_matches(name, disabled):
            continue

        if name in seen_names:
            result.issues.append(f"plugin name collision: {name!r} (skipped)")
            continue
        seen_names.add(name)
        result.plugins.append(name)

        for component, target in _COMPONENTS:
            for rel in manifest.component_dirs(component):
                resolved = (root / rel).resolve()
                if not _within(resolved, root):
                    result.issues.append(
                        f"{name}: {component} dir {rel!r} escapes the plugin root"
                    )
                    continue
                if resolved.is_dir():
                    getattr(result, target).append(resolved)
        result.mcp_servers.extend(manifest.mcp_servers)
        _collect_hooks(manifest, root, name, result)

    return result


def _collect_hooks(
    manifest: PluginManifest, root: Path, name: str, result: PluginLoadResult
) -> None:
    """Parse a manifest's hooks (a path to a hooks.toml, or inline [[hooks]])
    into HookConfig entries on the result. Defensive: errors → issues.
    """
    from vibe.core.hooks.config import _load_hooks_file
    from vibe.core.hooks.models import HookConfig

    raw = manifest.hooks
    if raw is None:
        return
    if isinstance(raw, str):
        resolved = (root / raw).resolve()
        if not _within(resolved, root):
            result.issues.append(f"{name}: hooks path {raw!r} escapes the plugin root")
            return
        loaded = _load_hooks_file(resolved)
        result.hooks.extend(loaded.hooks)
        result.issues.extend(f"{name}: {i.message}" for i in loaded.issues)
        return
    for entry in raw:  # inline list of hook dicts
        try:
            result.hooks.append(HookConfig.model_validate(entry))
        except (ValidationError, ValueError) as e:
            result.issues.append(f"{name}: invalid inline hook ({e})")


def _within(path: Path, root: Path) -> bool:
    try:
        return path == root or path.is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def apply_plugin_result(config: VibeConfig, result: PluginLoadResult) -> None:
    """Fold a PluginLoadResult into a VibeConfig (additive). Paths are appended;
    mcp_servers are union-merged by name (existing config wins).
    """
    config.agent_paths = [*config.agent_paths, *result.agent_paths]
    config.skill_paths = [*config.skill_paths, *result.skill_paths]
    config.tool_paths = [*config.tool_paths, *result.tool_paths]
    config.workflow_paths = [*config.workflow_paths, *result.workflow_paths]
    config.prompt_paths = [*config.prompt_paths, *result.prompt_paths]

    existing = {s.name for s in config.mcp_servers}
    from vibe.core.config import MCPServer as _MCPServerAdapter

    for raw in result.mcp_servers:
        nm = raw.get("name")
        if not nm or nm in existing:
            continue
        try:
            from pydantic import TypeAdapter

            server = TypeAdapter(_MCPServerAdapter).validate_python(raw)
        except Exception as e:
            logger.warning("plugin MCP server %r invalid: %s", nm, e)
            continue
        config.mcp_servers.append(server)
        existing.add(nm)
