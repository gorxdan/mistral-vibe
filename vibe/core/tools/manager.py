from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
import difflib
import functools
import hashlib
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import re
import sys
import threading
from typing import TYPE_CHECKING, Any

from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.logger import logger
from vibe.core.paths import DEFAULT_TOOL_DIR
from vibe.core.tools.base import BaseTool, BaseToolConfig, ToolInfo, ToolPermission
from vibe.core.utils import first_sentence, name_matches, run_sync

_TOOL_SEARCH_FUZZY_MATCH_THRESHOLD = 0.25
# logger.isEnabledFor(10) is always True (logger.py forces DEBUG); gate on env instead.
_TELEMETRY_DEBUG = os.environ.get("DEBUG_MODE") == "true" or (
    os.environ.get("LOG_LEVEL", "WARNING").upper() == "DEBUG"
)

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig
    from vibe.core.tools.connectors import ConnectorRegistry
    from vibe.core.tools.mcp import MCPRegistry
    from vibe.core.tools.mcp.tools import MCPTool


@functools.cache
def _mcp_tool_base() -> type[MCPTool]:
    from vibe.core.tools.mcp.tools import MCPTool

    return MCPTool


def _try_canonical_module_name(path: Path) -> str | None:
    """Extract canonical module name for vibe package files.

    Prevents Pydantic class identity mismatches when the same module
    is imported via dynamic discovery and regular imports.
    """
    try:
        parts = path.resolve().parts
    except (OSError, ValueError):
        return None

    try:
        vibe_idx = parts.index("vibe")
    except ValueError:
        return None

    if vibe_idx + 1 >= len(parts):
        return None

    module_parts = [p.removesuffix(".py") for p in parts[vibe_idx:]]
    return ".".join(module_parts)


def _compute_module_name(path: Path) -> str:
    """Return canonical module name for vibe files, hash-based synthetic name otherwise."""
    if canonical := _try_canonical_module_name(path):
        return canonical

    resolved = path.resolve()
    path_hash = hashlib.md5(str(resolved).encode()).hexdigest()[:8]
    stem = re.sub(r"[^0-9A-Za-z_]", "_", path.stem) or "mod"
    return f"vibe_tools_discovered_{stem}_{path_hash}"


@functools.cache
def _is_available_takes_config(cls: type[BaseTool]) -> bool:
    # is_available's signature is static per class; inspect.signature is costly,
    # so memoise the "does it accept config" decision rather than re-introspecting
    # on every availability check.
    return bool(inspect.signature(cls.is_available).parameters)


class NoSuchToolError(Exception):
    """Exception raised when a tool is not found."""


class ToolManager:
    """Manages tool discovery and instantiation for an Agent.

    Discovers available tools from the provided search paths. Each Agent
    should have its own ToolManager instance.
    """

    def __init__(
        self,
        config_getter: Callable[[], VibeConfig],
        mcp_registry: MCPRegistry | None = None,
        connector_registry: ConnectorRegistry | None = None,
        *,
        defer_mcp: bool = False,
        permission_getter: Callable[[str], ToolPermission | None] | None = None,
        runtime_allowlist: frozenset[str] | None = None,
        host: bool = True,
    ) -> None:
        self._config_getter = config_getter
        self._permission_getter = permission_getter
        self._runtime_allowlist = runtime_allowlist
        self._host = host
        # None until MCP is actually needed: constructing MCPRegistry imports
        # the mcp SDK (~100ms), which must stay off the interactive cold start.
        self._mcp_registry: MCPRegistry | None = mcp_registry
        self._connector_registry = connector_registry
        self._instances: dict[str, BaseTool] = {}
        self._manifest_pins: list[str] = []
        # Sticky, never LRU-capped: sharing dynamic_pinned_tool_limit would let
        # a remote activation evict an in-use builtin. Bounded (≤5) by marking.
        self._builtin_pins: set[str] = set()
        self._search_paths: list[Path] = (
            []
            if runtime_allowlist is not None
            else self._compute_search_paths(self._config)
        )
        self._lock = threading.Lock()
        self._mcp_integrated = False

        if runtime_allowlist is None:
            self._all_tools = {
                cls.get_name(): cls
                for cls in self._iter_tool_classes(self._search_paths)
            }
        else:
            from vibe.core.tools._canonical_task_tools import canonical_task_tools

            self._all_tools = canonical_task_tools(runtime_allowlist)
        if not defer_mcp and runtime_allowlist is None:
            self.integrate_all()

    @property
    def _config(self) -> VibeConfig:
        return self._config_getter()

    @staticmethod
    def _compute_search_paths(config: VibeConfig) -> list[Path]:
        paths: list[Path] = [DEFAULT_TOOL_DIR.path]

        paths.extend(config.tool_paths)

        mgr = get_harness_files_manager()
        paths.extend(mgr.project_tools_dirs)
        paths.extend(mgr.user_tools_dirs)

        unique: list[Path] = []
        seen: set[Path] = set()
        for p in paths:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                unique.append(rp)
        return unique

    @staticmethod
    def _iter_tool_classes(search_paths: list[Path]) -> Iterator[type[BaseTool]]:
        """Iterate over all search_paths to find tool classes.

        Note: if a search path is not a directory, it is treated as a single tool file.
        """
        for base in search_paths:
            if not base.is_dir() and base.name.endswith(".py"):
                if tools := ToolManager._load_tools_from_file(base):
                    for tool in tools:
                        yield tool

            for path in base.rglob("*.py"):
                if tools := ToolManager._load_tools_from_file(path):
                    for tool in tools:
                        yield tool

    @staticmethod
    def _load_tools_from_file(file_path: Path) -> list[type[BaseTool]] | None:
        if not file_path.is_file():
            return
        name = file_path.name
        if name.startswith("_"):
            return

        canonical_name = _try_canonical_module_name(file_path)
        if canonical_name is not None:
            try:
                module = importlib.import_module(canonical_name)
            except Exception:
                return
            parent_name, _, child_name = canonical_name.rpartition(".")
            if parent := sys.modules.get(parent_name):
                setattr(parent, child_name, module)
        else:
            module_name = _compute_module_name(file_path)
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    return
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(module_name, None)
                    return

        tools = []
        for tool_obj in vars(module).values():
            if not inspect.isclass(tool_obj):
                continue
            if not issubclass(tool_obj, BaseTool) or tool_obj is BaseTool:
                continue
            if inspect.isabstract(tool_obj):
                continue
            tools.append(tool_obj)
        return tools

    @staticmethod
    def discover_tool_defaults(
        search_paths: list[Path] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if search_paths is None:
            search_paths = [DEFAULT_TOOL_DIR.path]

        defaults: dict[str, dict[str, Any]] = {}
        for cls in ToolManager._iter_tool_classes(search_paths):
            try:
                tool_name = cls.get_name()
                config_class = cls._get_tool_config_class()
                defaults[tool_name] = config_class().model_dump(exclude_none=True)
            except Exception as e:
                logger.warning(
                    "Failed to get defaults for tool %s: %s", cls.__name__, e
                )
                continue
        return defaults

    @property
    def registered_tools(self) -> dict[str, type[BaseTool]]:
        with self._lock:
            return dict(self._all_tools)

    @property
    def available_tools(self) -> dict[str, type[BaseTool]]:
        return self._filtered_available_tools()

    def _deferrable_builtin_names(self, tools: dict[str, type[BaseTool]]) -> set[str]:
        if not self._config.tool_manifest.defer_builtin_tools:
            return set()
        return {
            name
            for name, cls in tools.items()
            if cls.manifest_deferrable and not self._is_dynamic_remote_tool(cls)
        }

    @property
    def manifest_tools(self) -> dict[str, type[BaseTool]]:
        tools = self._filtered_available_tools()
        manifest = self._config.tool_manifest
        if not manifest.dynamic_subset_enabled or self._config.enabled_tools:
            return tools
        remote = {
            name for name, cls in tools.items() if self._is_dynamic_remote_tool(cls)
        }
        remote_gated = bool(remote) and len(tools) > manifest.dynamic_subset_threshold
        deferrable = self._deferrable_builtin_names(tools)
        if not remote_gated and not deferrable:
            return {name: cls for name, cls in tools.items() if name != "tool_search"}
        if "tool_search" not in tools:
            return tools
        hidden: set[str] = set()
        if remote_gated:
            hidden |= remote - set(
                self._manifest_pins[: manifest.dynamic_pinned_tool_limit]
            )
        hidden |= deferrable - self._builtin_pins
        return {name: cls for name, cls in tools.items() if name not in hidden}

    def hidden_tool_names(self) -> set[str]:
        return set(self._filtered_available_tools()) - set(self.manifest_tools)

    def deferred_builtin_stubs(self) -> list[tuple[str, str]]:
        tools = self._filtered_available_tools()
        visible = set(self.manifest_tools)
        return [
            (name, first_sentence(tools[name].description, 90))
            for name in sorted(self._deferrable_builtin_names(tools) - visible)
        ]

    def _filtered_available_tools(self) -> dict[str, type[BaseTool]]:
        with self._lock:
            runtime_available = {
                name: cls
                for name, cls in self._all_tools.items()
                if self._is_tool_available(cls)
                and (not cls.runtime_scoped or self._runtime_allowlist is not None)
                and (self._host or not cls.host_only)
            }

        # Per-source filtering first (MCP server/connector disabled flags).
        result = self._apply_per_source_filtering(runtime_available)

        # Global overrides take precedence.
        if self._runtime_allowlist is None:
            if self._config.enabled_tools:
                result = {
                    name: cls
                    for name, cls in result.items()
                    if name_matches(name, self._config.enabled_tools)
                    or (self._host and cls.host_policy_control)
                }
            elif self._config.disabled_tools:
                result = {
                    name: cls
                    for name, cls in result.items()
                    if not name_matches(name, self._config.disabled_tools)
                }
        elif self._config.disabled_tools:
            result = {
                name: cls
                for name, cls in result.items()
                if not name_matches(name, self._config.disabled_tools)
            }
        if self._runtime_allowlist is not None:
            result = {
                name: cls
                for name, cls in result.items()
                if name in self._runtime_allowlist
            }
        return result

    @staticmethod
    def _is_dynamic_remote_tool(tool_cls: type[BaseTool]) -> bool:
        # MCPTool subclasses can only exist once the proxy module is loaded;
        # skip the import so lazy startup never pulls it just to answer "no".
        if "vibe.core.tools.mcp.tools" not in sys.modules:
            return False
        return (
            issubclass(tool_cls, _mcp_tool_base())
            and tool_cls.get_name() != "tool_search"
        )

    def pin_manifest_tools(self, tool_names: list[str]) -> list[str]:
        available = self._filtered_available_tools()
        remote_names = {
            name for name, cls in available.items() if self._is_dynamic_remote_tool(cls)
        }
        builtin_sel = [
            name
            for name in tool_names
            if name in self._deferrable_builtin_names(available)
        ]
        remote_sel = [name for name in tool_names if name in remote_names]
        self._builtin_pins.update(builtin_sel)
        if remote_sel:
            limit = self._config.tool_manifest.dynamic_pinned_tool_limit
            merged = remote_sel + [
                name for name in self._manifest_pins if name not in remote_sel
            ]
            self._manifest_pins = merged[:limit]
        if not builtin_sel and not remote_sel:
            return []
        return [*sorted(self._builtin_pins), *self._manifest_pins]

    def search_tools(
        self, query: str, *, max_results: int | None = None
    ) -> list[ToolInfo]:
        available = self._filtered_available_tools()
        deferrable = self._deferrable_builtin_names(available)
        candidates = [
            (name, cls)
            for name, cls in available.items()
            if self._is_dynamic_remote_tool(cls) or name in deferrable
        ]
        # debug-only: scope difflib-vs-BM25 necessity by candidate count.
        if _TELEMETRY_DEBUG:
            logger.debug("tool_search candidates=%s query=%s", len(candidates), query)
        limit = max_results or self._config.tool_manifest.dynamic_search_results
        terms = [term for term in query.lower().split() if term]
        scored: list[tuple[float, str, type[BaseTool]]] = []
        for name, cls in candidates:
            description = str(getattr(cls, "description", ""))
            haystack = f"{name} {description}".lower()
            if not terms:
                scored.append((0.0, name, cls))
                continue
            exact_hits = sum(1 for term in terms if term in haystack)
            fuzzy = difflib.SequenceMatcher(None, query.lower(), haystack).ratio()
            score = exact_hits + fuzzy
            if exact_hits or fuzzy >= _TOOL_SEARCH_FUZZY_MATCH_THRESHOLD:
                scored.append((score, name, cls))
        scored.sort(key=lambda item: (-item[0], item[1]))
        # MCP classes resolve no prompts/*.md (loader returns None anyway); the
        # explicit gate keeps the builtin-only intent of `usage` obvious.
        return [
            ToolInfo(
                name=name,
                description=str(getattr(cls, "description", "")),
                parameters=cls.get_parameters(),
                usage=cls.get_tool_prompt() if name in deferrable else None,
            )
            for _, name, cls in scored[:limit]
        ]

    def _is_tool_available(self, cls: type[BaseTool]) -> bool:
        # Backwards-compatibility check to avoid breaking
        # existing custom tools that call is_available without parameters
        if _is_available_takes_config(cls):
            return cls.is_available(self._config)
        return cls.is_available()

    def _apply_per_source_filtering(
        self, tools: dict[str, type[BaseTool]]
    ) -> dict[str, type[BaseTool]]:
        """Filter out MCP/connector tools disabled at the server or connector level."""
        disabled_sources, per_source_disabled = self._build_source_disable_index()
        if not disabled_sources and not per_source_disabled:
            return tools

        return {
            name: cls
            for name, cls in tools.items()
            if not self._is_source_disabled(cls, disabled_sources, per_source_disabled)
        }

    def _build_source_disable_index(
        self,
    ) -> tuple[set[tuple[str, bool]], dict[tuple[str, bool], set[str]]]:
        """Return (fully_disabled, per_tool_disabled) keyed by (source_name, is_connector)."""
        disabled_sources: set[tuple[str, bool]] = set()
        per_source_disabled: dict[tuple[str, bool], set[str]] = {}

        for srv in self._config.mcp_servers:
            key = (srv.name, False)
            if srv.disabled:
                disabled_sources.add(key)
            elif srv.disabled_tools:
                per_source_disabled[key] = set(srv.disabled_tools)

        for cfg in self._config.connectors:
            if cfg.disabled_tools and not cfg.disabled:
                per_source_disabled[(cfg.name, True)] = set(cfg.disabled_tools)

        if self._connector_registry is not None:
            by_name = self._config.connectors_by_name()
            for name in self._connector_registry.get_connector_names():
                cfg = by_name.get(name)
                if cfg is None or cfg.disabled:
                    disabled_sources.add((name, True))

        return disabled_sources, per_source_disabled

    @staticmethod
    def _is_source_disabled(
        tool_cls: type[BaseTool],
        disabled_sources: set[tuple[str, bool]],
        per_source_disabled: dict[tuple[str, bool], set[str]],
    ) -> bool:
        if not issubclass(tool_cls, _mcp_tool_base()):
            return False
        server_name = tool_cls.get_server_name()
        if server_name is None:
            return False
        key = (server_name, tool_cls.is_connector())
        if key in disabled_sources:
            return True
        return tool_cls.get_remote_name() in per_source_disabled.get(key, set())

    def integrate_mcp(self, *, raise_on_failure: bool = False) -> None:
        """Discover and register MCP tools (sync wrapper).

        Idempotent: subsequent calls after a successful integration are
        no-ops to avoid redundant MCP discovery.
        """
        run_sync(self._integrate_mcp_async(raise_on_failure=raise_on_failure))

    async def _integrate_mcp_async(self, *, raise_on_failure: bool = False) -> None:
        """Async MCP discovery — canonical implementation."""
        if self._runtime_allowlist is not None:
            return
        if self._mcp_integrated:
            return
        if not self._config.mcp_servers:
            return

        try:
            mcp_tools = await self._get_mcp_registry().get_tools_async(
                self._config.mcp_servers
            )
        except Exception as exc:
            logger.warning("MCP integration failed: %s", exc)
            if raise_on_failure:
                raise
            return

        with self._lock:
            self._purge_mcp_state()
            self._all_tools = {**self._all_tools, **mcp_tools}
            self._manifest_pins = [
                name for name in self._manifest_pins if name in self._all_tools
            ]
        self._mcp_integrated = True
        logger.info(
            "MCP integration registered %d tools (via registry)", len(mcp_tools)
        )

    def _purge_connector_state(self) -> None:
        """Remove stale connector tool classes and cached instances."""
        stale_keys = [
            name
            for name, cls in self._all_tools.items()
            if issubclass(cls, _mcp_tool_base()) and cls.is_connector()
        ]
        for key in stale_keys:
            self._all_tools.pop(key, None)
            self._instances.pop(key, None)

    def _purge_mcp_state(self) -> None:
        """Remove stale MCP tool classes and cached instances."""
        stale_keys = [
            name
            for name, cls in self._all_tools.items()
            if issubclass(cls, _mcp_tool_base()) and not cls.is_connector()
        ]
        for key in stale_keys:
            self._all_tools.pop(key, None)
            self._instances.pop(key, None)

    def integrate_connectors(self) -> None:
        """Discover and register connector tools (sync wrapper)."""
        run_sync(self.integrate_connectors_async())

    async def integrate_connectors_async(self) -> None:
        """Discover and register connector tools — canonical implementation.

        Thread-safe: can be called from the deferred-init background thread.
        """
        if self._runtime_allowlist is not None or self._connector_registry is None:
            return

        try:
            connector_tools = await self._connector_registry.get_tools_async()
        except Exception as exc:
            logger.warning("Connector integration failed: %s", exc)
            with self._lock:
                self._purge_connector_state()
            return

        with self._lock:
            self._purge_connector_state()
            self._all_tools.update(connector_tools)
            self._manifest_pins = [
                name for name in self._manifest_pins if name in self._all_tools
            ]
        logger.info("Connector integration registered %s tools", len(connector_tools))

    async def refresh_remote_tools_async(self) -> None:
        """Force MCP and connector re-discovery for the current config."""
        with self._lock:
            if self._mcp_registry is not None:
                self._mcp_registry.clear()
            self._purge_mcp_state()
            self._mcp_integrated = False
            self._purge_connector_state()
            if self._connector_registry is not None:
                self._connector_registry.clear()

        await self._integrate_all_async()

    def refresh_remote_tools(self) -> None:
        """Sync wrapper for :meth:`refresh_remote_tools_async`."""
        run_sync(self.refresh_remote_tools_async())

    def integrate_all(self, *, raise_on_mcp_failure: bool = False) -> None:
        """Discover MCP and connector tools in parallel.

        Runs both async discovery paths concurrently via ``asyncio.gather``
        inside a single ``run_sync`` call.
        """
        run_sync(self._integrate_all_async(raise_on_mcp_failure=raise_on_mcp_failure))

    def _get_mcp_registry(self) -> MCPRegistry:
        if self._mcp_registry is None:
            from vibe.core.tools.mcp import MCPRegistry

            self._mcp_registry = MCPRegistry()
        return self._mcp_registry

    def set_mcp_registry(self, mcp_registry: MCPRegistry) -> None:
        self._mcp_registry = mcp_registry

    def set_connector_registry(
        self, connector_registry: ConnectorRegistry | None
    ) -> None:
        self._connector_registry = connector_registry

    async def _integrate_all_async(self, *, raise_on_mcp_failure: bool = False) -> None:
        """Run MCP and connector discovery concurrently.

        Uses ``return_exceptions=True`` so that a failing MCP server does
        not cancel in-flight connector discovery (or vice-versa).
        """
        mcp_result, connector_result = await asyncio.gather(
            self._integrate_mcp_async(raise_on_failure=raise_on_mcp_failure),
            self.integrate_connectors_async(),
            return_exceptions=True,
        )

        # Re-raise MCP errors when the caller asked for them.
        if isinstance(mcp_result, BaseException):
            if raise_on_mcp_failure:
                raise mcp_result
            logger.warning("MCP integration failed: %s", mcp_result)

        if isinstance(connector_result, BaseException):
            logger.warning("Connector integration failed: %s", connector_result)

    def get_tool_config(self, tool_name: str) -> BaseToolConfig:
        with self._lock:
            tool_class = self._all_tools.get(tool_name)

        if tool_class:
            config_class = tool_class._get_tool_config_class()
            default_config = config_class()
        else:
            config_class = BaseToolConfig
            default_config = BaseToolConfig()

        user_overrides = self._config.tools.get(tool_name)
        permission_override = (
            self._permission_getter(tool_name) if self._permission_getter else None
        )
        if user_overrides is None and permission_override is None:
            return config_class()

        merged_dict = {**default_config.model_dump(), **(user_overrides or {})}
        if permission_override is not None:
            merged_dict["permission"] = permission_override.value
        return config_class.model_validate(merged_dict)

    def get(self, tool_name: str) -> BaseTool:
        """Get a tool instance, creating it lazily on first call.

        Raises:
            NoSuchToolError: If the requested tool is not available.
        """
        if tool_name in self._instances:
            return self._instances[tool_name]

        available = self.available_tools
        if tool_name not in available:
            raise NoSuchToolError(
                f"Unknown or disabled tool: {tool_name}. "
                f"Available: {list(available.keys())}"
            )
        tool_class = available[tool_name]
        self._instances[tool_name] = tool_class.from_config(
            lambda: self.get_tool_config(tool_name)
        )
        return self._instances[tool_name]

    def reset_all(self) -> None:
        self._instances.clear()
