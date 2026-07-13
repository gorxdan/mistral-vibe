from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import replace
from enum import StrEnum, auto
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.logger import logger
from vibe.core.lsp import LSPNotConnectedError, get_lsp_manager
from vibe.core.lsp._adherence import record_lsp_call
from vibe.core.lsp._manager import current_lsp_generation
from vibe.core.lsp._pagination import (
    LspContinuationError,
    LspContinuationPage,
    LspContinuationReloadRequired,
    LspContinuationStore,
    LspQueryBinding,
)
from vibe.core.lsp._positions import (
    codepoint_position_to_utf16,
    split_lsp_lines,
    utf16_range_to_codepoint,
)
from vibe.core.lsp._symbols import NormalizedSymbol, normalize_document_symbols
from vibe.core.lsp._types import (
    LSPError,
    LSPProtocolError,
    Position,
    Range,
    path_from_uri,
    uri_from_path,
)
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.tools.utils import (
    enforce_isolated_confine,
    enforce_team_metadata_confine,
    isolated_worktree_root,
    resolve_file_tool_permission,
)
from vibe.core.types import ToolStreamEvent
from vibe.core.utils.io import read_safe_async

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig
    from vibe.core.types import ToolResultEvent

_MAX_FILE_BYTES = 10 * 1024 * 1024
_MAX_LOCATION_RESULTS = 50
_MAX_SYMBOL_RESULTS = 100
_MAX_CALL_RESULTS = 50
_MAX_CONTINUATION_TOKEN_LENGTH = 256
_METHOD_NOT_FOUND = -32601
_CALL_HIERARCHY_RETRIES = 4
_CALL_HIERARCHY_BACKOFF = (0.2, 0.4, 0.8)
# LSP SymbolKind.Class (5). pyright returns a CallHierarchyItem for a class but
# never resolves incoming/outgoing edges (it does not model instantiation as a
# call edge), so an empty follow-up on a class is a correct final result, not a
# cold-index artifact. Skipping the retry loop avoids a ~1.4s stall and a
# misleading "server was indexing" caveat on class symbols.
_SYMBOL_KIND_CLASS = 5
# Memoize repeat queries within a short window. Keyed on the queried file's
# mtime so any edit invalidates instantly; the TTL bounds cross-file staleness
# (a references/call-hierarchy result can shift when *another* file changes).
_RESULT_CACHE_TTL = 3.0
_RESULT_CACHE_MAX_ENTRIES = 128
_CALL_HIERARCHY_OPS = frozenset({
    "prepare_call_hierarchy",
    "incoming_calls",
    "outgoing_calls",
})
_PAGEABLE_OPS = frozenset({
    "go_to_definition",
    "find_references",
    "document_symbol",
    "workspace_symbol",
    "go_to_implementation",
    "prepare_call_hierarchy",
    "incoming_calls",
    "outgoing_calls",
})


@asynccontextmanager
async def _lease_server(manager: Any, file_path: str) -> AsyncIterator[Any]:
    lease = getattr(manager, "lease_server_for_file", None)
    if callable(lease):
        lease_server = cast(Callable[[str], AbstractAsyncContextManager[Any]], lease)
        async with lease_server(file_path) as server:
            yield server
        return
    yield manager.get_server_for_file(file_path)


class LspOperation(StrEnum):
    STATUS = auto()
    GO_TO_DEFINITION = auto()
    FIND_REFERENCES = auto()
    HOVER = auto()
    DOCUMENT_SYMBOL = auto()
    WORKSPACE_SYMBOL = auto()
    GO_TO_IMPLEMENTATION = auto()
    PREPARE_CALL_HIERARCHY = auto()
    INCOMING_CALLS = auto()
    OUTGOING_CALLS = auto()


class LspArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    operation: LspOperation = Field(
        description=(
            "LSP operation to perform. Position-based operations "
            "(go_to_definition, find_references, hover, go_to_implementation, "
            "prepare_call_hierarchy, incoming_calls, outgoing_calls) require "
            "line and character. status accepts an optional file_path and "
            "reports live server readiness without starting a server. "
            "document_symbol needs only file_path. "
            "workspace_symbol needs only query and may omit file_path "
            "(it is workspace-wide; all configured servers are queried and "
            "their results merged). workspace_symbol is unavailable under a "
            "path-scoped task contract."
        )
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to the source file. Required for every operation "
            "except workspace_symbol and status."
        ),
    )
    line: int | None = Field(
        default=None,
        ge=1,
        description="1-based line number. Required for position-based operations.",
    )
    character: int | None = Field(
        default=None,
        ge=1,
        description=(
            "1-based Unicode code-point column; tabs count as one character. "
            "Required for position-based operations."
        ),
    )
    query: str | None = Field(
        default=None, description="Symbol query string. Required for workspace_symbol."
    )
    continuation_token: str | None = Field(
        default=None,
        max_length=_MAX_CONTINUATION_TOKEN_LENGTH,
        description=(
            "Opaque token returned by a previous page. Repeat the exact original "
            "operation, path, position, and query when supplying it."
        ),
    )


class LspResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    operation: str
    summary: str = Field(description="Short human/machine-readable result text.")
    locations: list[dict[str, Any]] = Field(default_factory=list)
    symbol_names: list[str] = Field(default_factory=list)
    symbols: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int | None = None
    returned_count: int | None = None
    was_truncated: bool = False
    has_more: bool = False
    page_offset: int = 0
    continuation_token: str | None = None
    readiness: dict[str, Any] | None = None
    workspace_coverage: dict[str, int | bool] | None = None
    partial_coverage: bool = False


class LspConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )


class LspState(BaseToolState):
    pass


class Lsp(
    BaseTool[LspArgs, LspResult, LspConfig, LspState], ToolUIData[LspArgs, LspResult]
):
    read_only: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Query a language server for semantic code intelligence: "
        "go-to-definition, find-references, hover (type info), "
        "document/workspace symbols, go-to-implementation, call hierarchy, and "
        "live readiness by file type. "
        "Prefer this over grep when you need to resolve a symbol, trace its "
        "callers/callees, or read its type — it understands imports, overloads, "
        "and generated code that textual search cannot."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if isolated_worktree_root() is not None:
            return False
        if config is None:
            return True
        return "lsp" in getattr(config, "installed_components", [])

    def resolve_permission(self, args: LspArgs) -> PermissionContext | None:
        if args.file_path is not None:
            return resolve_file_tool_permission(
                args.file_path,
                tool_name=self.get_name(),
                allowlist=self.config.allowlist,
                denylist=self.config.denylist,
                config_permission=self.config.permission,
                sensitive_patterns=self.config.sensitive_patterns,
            )
        return PermissionContext(permission=self.config.permission)

    @staticmethod
    def _lsp_installed() -> bool:
        # Read the persisted flag without depending on InvokeContext carrying
        # VibeConfig (it doesn't). VibeConfig.load is cached and cheap.
        from vibe.core.config import VibeConfig

        return "lsp" in VibeConfig.load().installed_components

    def _ensure_manager(self) -> Any:
        if isolated_worktree_root() is not None:
            return None
        manager = get_lsp_manager()
        if manager is not None:
            return manager
        if not self._lsp_installed():
            return None
        from vibe.core.config import VibeConfig
        from vibe.core.lsp._lifecycle import setup_lsp_for_config

        config = VibeConfig.load()
        return setup_lsp_for_config(config, lambda: config, Path.cwd())

    async def run(
        self, args: LspArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LspResult, None]:
        if (
            args.continuation_token is not None
            and args.operation.value not in _PAGEABLE_OPS
        ):
            raise ToolError(f"{args.operation.value} does not return pageable results")
        if (
            ctx is not None
            and ctx.task_contract is not None
            and args.operation is LspOperation.WORKSPACE_SYMBOL
        ):
            raise ToolError(
                "workspace_symbol is unavailable under a path-scoped task contract"
            )
        manager = self._ensure_manager()
        if manager is None:
            if isolated_worktree_root() is not None:
                raise ToolError(
                    "LSP is disabled for isolated subagents until language "
                    "servers can run inside the worktree sandbox. Use read, "
                    "grep, and glob in this execution mode."
                )
            installed = self._lsp_installed()
            if installed:
                raise ToolError(
                    "LSP is enabled but no language server is running in this "
                    "session. Restart vibe, or install a server pyright/"
                    "typescript-language-server/etc. on PATH and run /lspstall."
                )
            raise ToolError("LSP is not enabled. Run /lspstall to enable it.")
        if args.operation is LspOperation.STATUS:
            file_path = (
                self._resolve_readiness_path(args.file_path)
                if args.file_path is not None
                else None
            )
            snapshot = manager.readiness(file_path)
            yield self._format_readiness(snapshot)
            return
        raw_path = args.file_path
        if raw_path is None:
            if args.operation is LspOperation.WORKSPACE_SYMBOL:
                if not args.query:
                    raise ToolError("workspace_symbol requires a non-empty query.")
                binding = self._query_binding(manager, args, None, ctx)
                result = await self._workspace_symbol(
                    manager,
                    args.query or "",
                    binding=binding,
                    continuation_token=args.continuation_token,
                )
                record_lsp_call(args.operation.value, ctx=ctx, cache_hit=False)
                yield result
                return
            raise ToolError(
                f"{args.operation.value} requires file_path. Only "
                "workspace_symbol and status may omit it."
            )
        file_path = self._resolve_path(raw_path)
        binding = self._query_binding(manager, args, file_path, ctx)
        if args.continuation_token is not None:
            resumed = self._resume_page(args, binding)
            if resumed is not None:
                record_lsp_call(args.operation.value, ctx=ctx, cache_hit=True)
                yield resumed
                return
        position_required = args.operation in {
            LspOperation.GO_TO_DEFINITION,
            LspOperation.FIND_REFERENCES,
            LspOperation.HOVER,
            LspOperation.GO_TO_IMPLEMENTATION,
            LspOperation.PREPARE_CALL_HIERARCHY,
            LspOperation.INCOMING_CALLS,
            LspOperation.OUTGOING_CALLS,
        }
        if position_required and (args.line is None or args.character is None):
            raise ToolError(
                f"{args.operation} requires both line and character (1-based)."
            )
        if args.operation is LspOperation.WORKSPACE_SYMBOL and not args.query:
            raise ToolError("workspace_symbol requires a non-empty query.")
        async with _lease_server(manager, file_path) as server:
            if server is None:
                raise ToolError(
                    f"No LSP server configured for "
                    f"{Path(file_path).suffix or 'extensionless'} files. Run "
                    "/lspstall to re-detect installed servers, or add a "
                    "[[lsp_servers]] entry with a matching language."
                )
            async for event in self._execute(
                manager, args, file_path, position_required, server, ctx, binding
            ):
                yield event

    async def _execute(
        self,
        manager: Any,
        args: LspArgs,
        file_path: str,
        position_required: bool,
        server: Any,
        ctx: InvokeContext | None,
        binding: LspQueryBinding,
    ) -> AsyncGenerator[ToolStreamEvent | LspResult, None]:
        try:
            text = await read_safe_async(Path(file_path))
            server_t0 = time.perf_counter()
            await manager.open_document(
                file_path,
                text.text,
                server.config.language_id_for(Path(file_path).suffix),
            )
            position: dict[str, int] | None = None
            if position_required:
                try:
                    self._validate_position(
                        Path(file_path), args.line or 1, args.character or 1, text.text
                    )
                except ToolError as exc:
                    # Bad guessed coords are recoverable: yield a soft steer to
                    # document_symbol instead of a hard error + dead-coord retry.
                    yield LspResult(
                        operation=str(args.operation),
                        summary=(
                            f"{exc} No symbol at line {args.line}:{args.character} — "
                            f"use document_symbol to list valid symbol positions in "
                            f"{Path(file_path).name}."
                        ),
                    )
                    return
                protocol_position = codepoint_position_to_utf16(
                    text.text,
                    Position(
                        line=(args.line or 1) - 1, character=(args.character or 1) - 1
                    ),
                )
                position = {
                    "line": protocol_position.line,
                    "character": protocol_position.character,
                }
            # Hover is scalar and memoizable; pageable ops are snapshot-backed and
            # must revalidate continuation binding (never cache token-bearing pages).
            cache_key: tuple[Any, ...] | None = None
            if args.operation is LspOperation.HOVER:
                cache_key = (
                    file_path,
                    Path(file_path).stat().st_mtime_ns,
                    str(args.operation),
                    args.line,
                    args.character,
                    args.query,
                    args.continuation_token,
                    binding.session_id,
                    binding.task_brief_hash,
                    binding.lsp_generation,
                    binding.workspace_root,
                )
                hit = self._result_cache_get(cache_key)
                if hit is not None and time.monotonic() - hit[0] < _RESULT_CACHE_TTL:
                    logger.debug(
                        "lsp %s cache hit %.1fms",
                        args.operation.value,
                        (time.perf_counter() - server_t0) * 1000.0,
                    )
                    record_lsp_call(args.operation.value, ctx=ctx, cache_hit=True)
                    yield self._scope_result(hit[1], ctx)
                    return
            if ctx is not None and args.operation.value in _CALL_HIERARCHY_OPS:
                # Call-hierarchy can block ~1s+ while a cold server indexes;
                # surface activity so the pause is not read as a hang.
                yield ToolStreamEvent(
                    tool_name="lsp",
                    tool_call_id=ctx.tool_call_id,
                    message=(
                        "Resolving call graph — the language server may still "
                        "be indexing on first use"
                    ),
                )
            result = await self._dispatch(
                manager, args, file_path, position, binding, ctx
            )
            logger.debug(
                "lsp %s %.1fms",
                args.operation.value,
                (time.perf_counter() - server_t0) * 1000.0,
            )
            record_lsp_call(args.operation.value, ctx=ctx, cache_hit=False)
            if cache_key is not None:
                self._result_cache_put(cache_key, result)
        except LSPNotConnectedError as exc:
            raise ToolError(str(exc)) from exc
        except LSPProtocolError as exc:
            if exc.code == _METHOD_NOT_FOUND:
                raise ToolError(
                    f"{server.config.name} does not support {args.operation.value}. "
                    + self._method_not_found_hint(args.operation)
                ) from exc
            raise ToolError(f"LSP request failed: {exc}") from exc
        except LSPError as exc:
            raise ToolError(f"LSP request failed: {exc}") from exc

        yield self._scope_result(result, ctx)

    def _scope_result(self, result: LspResult, ctx: InvokeContext | None) -> LspResult:
        if ctx is None or ctx.task_contract is None or not result.locations:
            return result
        locations = [
            location
            for location in result.locations
            if ctx.task_contract.allows_search_result(
                path_from_uri(
                    location.get("uri", "")
                    or (location.get("data") or {}).get("uri", "")
                )
            )
        ]
        if len(locations) == len(result.locations):
            return result
        first_line = result.summary.splitlines()[0]
        label = first_line.split(" (", 1)[0].removesuffix(":")
        scoped = self._format_locations(label, locations)
        if result.was_truncated:
            scoped.total_count = result.total_count
            scoped.returned_count = len(locations)
            scoped.was_truncated = True
            omitted = (result.total_count or len(result.locations)) - len(
                result.locations
            )
            scoped.summary += (
                f"\n  [truncated before task scoping: at least {omitted} "
                "additional result(s) omitted]"
            )
        scoped.page_offset = result.page_offset
        scoped.continuation_token = result.continuation_token
        return scoped

    @staticmethod
    def _filter_task_locations(
        locations: list[dict[str, Any]], ctx: InvokeContext | None
    ) -> list[dict[str, Any]]:
        if ctx is None or ctx.task_contract is None:
            return locations
        return [
            location
            for location in locations
            if ctx.task_contract.allows_search_result(
                path_from_uri(
                    location.get("uri", "")
                    or (location.get("data") or {}).get("uri", "")
                )
            )
        ]

    async def _dispatch(
        self,
        manager: Any,
        args: LspArgs,
        file_path: str,
        position: dict[str, int] | None,
        binding: LspQueryBinding | None = None,
        ctx: InvokeContext | None = None,
    ) -> LspResult:
        uri = uri_from_path(file_path)
        text_doc = {"textDocument": {"uri": uri}}
        simple = self._simple_dispatch_table().get(args.operation)
        if simple is not None:
            method, label, formatter, extra = simple
            if position is not None:
                raw = await self._request_at_identifier(
                    manager, file_path, text_doc, position, method, extra
                )
            else:
                params = {**text_doc, **(extra or {})}
                raw, _ = await manager.send_request(file_path, method, params)
            if formatter == self._format_locations:
                filtered = await self._filter_gitignored(self._as_location_list(raw))
                filtered = self._filter_task_locations(filtered, ctx)
                normalized = await self._normalize_location_positions(filtered)
                page = self._page_items(
                    normalized,
                    binding=binding
                    or self._query_binding(manager, args, file_path, ctx),
                    continuation_token=args.continuation_token,
                    page_size=_MAX_LOCATION_RESULTS,
                )
                return formatter(label, list(page.items), page=page)
            if formatter == self._format_symbols:
                symbols = await self._normalize_symbols(raw, uri)
                return self._page_symbol_records(
                    label,
                    symbols,
                    binding=binding
                    or self._query_binding(manager, args, file_path, ctx),
                    continuation_token=args.continuation_token,
                )
            return formatter(label, raw)
        if args.operation is LspOperation.HOVER:
            raw = await self._request_at_identifier(
                manager, file_path, text_doc, position or {}, "textDocument/hover", None
            )
            return self._format_hover(raw)
        if args.operation is LspOperation.WORKSPACE_SYMBOL:
            query = args.query or ""
            raw, _ = await manager.send_request(
                file_path, "workspace/symbol", {"query": query}
            )
            symbols = await self._normalize_symbols(raw, "")
            return self._page_symbol_records(
                f"Workspace symbols matching '{query}'",
                symbols,
                binding=binding or self._query_binding(manager, args, file_path, ctx),
                continuation_token=args.continuation_token,
                query=query,
            )
        if args.operation in {
            LspOperation.PREPARE_CALL_HIERARCHY,
            LspOperation.INCOMING_CALLS,
            LspOperation.OUTGOING_CALLS,
        }:
            return await self._call_hierarchy(
                manager, args, file_path, text_doc, position or {}, binding, ctx
            )
        raise ToolError(f"Unsupported operation: {args.operation}")

    @staticmethod
    def _method_not_found_hint(operation: LspOperation) -> str:
        # textDocument/implementation resolves concrete overrides of an abstract
        # base or interface — "caller/callee" describes call-graph ops, not this
        # one, so the generic fallback would mislead. find_references shows
        # usages and workspace_symbol locates subclasses by name.
        if operation is LspOperation.GO_TO_IMPLEMENTATION:
            return (
                "Try find_references to list usages, or workspace_symbol to "
                "locate subclasses by name."
            )
        if operation in {
            LspOperation.PREPARE_CALL_HIERARCHY,
            LspOperation.INCOMING_CALLS,
            LspOperation.OUTGOING_CALLS,
        }:
            return (
                "Try find_references to list usages. It does not preserve "
                "caller/callee direction, so confirm direction by reading the "
                "returned sites."
            )
        return "Try find_references to list usages, or document_symbol for an outline."

    @staticmethod
    def _format_readiness(snapshot: Any) -> LspResult:
        lines = [f"LSP readiness: {snapshot.state.value} — {snapshot.reason}"]
        if snapshot.route_pool is not None:
            pool = snapshot.route_pool
            lines.append(
                "  workspace roots: "
                f"{pool.resident_dynamic_roots}/{pool.max_dynamic_roots} dynamic; "
                f"{pool.retiring_servers} server(s) retiring"
            )
            lines.append(
                "  workspace_symbol coverage: "
                f"{pool.resident_roots}/{pool.known_roots} known roots resident"
            )
        if snapshot.selected_workspace_root is not None:
            lines.append(f"  selected root: {snapshot.selected_workspace_root}")
        for server in snapshot.servers:
            extensions = ", ".join(server.extensions) or "no extensions"
            operations = (
                "unknown until initialized"
                if server.operations is None
                else ", ".join(server.operations) or "no semantic providers advertised"
            )
            lines.append(
                f"  {server.name}: {server.state.value} ({extensions}); {operations}"
            )
            if server.error:
                lines.append(f"    error: {server.error}")
        return LspResult(
            operation="status",
            summary="\n".join(lines),
            readiness=snapshot.model_dump(mode="json"),
        )

    def _resume_page(self, args: LspArgs, binding: LspQueryBinding) -> LspResult | None:
        token = args.continuation_token
        if token is None:
            return None
        try:
            page = self._continuation_store().get_page(token, binding)
        except LspContinuationReloadRequired:
            return None
        except LspContinuationError as exc:
            raise ToolError(str(exc)) from exc

        items = list(page.items)
        location_label = {
            LspOperation.GO_TO_DEFINITION: "Definitions",
            LspOperation.GO_TO_IMPLEMENTATION: "Implementations",
            LspOperation.FIND_REFERENCES: "References",
            LspOperation.INCOMING_CALLS: "Incoming calls",
            LspOperation.OUTGOING_CALLS: "Outgoing calls",
        }.get(args.operation)
        if location_label is not None:
            result = self._format_locations(location_label, items, page=page)
        elif args.operation is LspOperation.DOCUMENT_SYMBOL:
            result = self._format_symbol_dicts("Document symbols", items, page=page)
        elif args.operation is LspOperation.WORKSPACE_SYMBOL:
            label = f"Workspace symbols matching '{args.query or ''}'"
            result = self._format_symbol_dicts(label, items, page=page)
        elif args.operation is LspOperation.PREPARE_CALL_HIERARCHY:
            result = self._format_call_items("Call hierarchy items", items, page=page)
        else:
            result = None
        return result

    async def _workspace_symbol(
        self,
        manager: Any,
        query: str,
        *,
        binding: LspQueryBinding,
        continuation_token: str | None,
    ) -> LspResult:
        label = f"Workspace symbols matching '{query}'"
        coverage = self._workspace_coverage(manager)
        if continuation_token is not None:
            try:
                page = self._continuation_store().get_page(continuation_token, binding)
            except LspContinuationReloadRequired:
                pass
            except LspContinuationError as exc:
                raise ToolError(str(exc)) from exc
            else:
                return self._apply_workspace_coverage(
                    self._format_symbol_dicts(label, list(page.items), page=page),
                    coverage,
                )
        servers = manager.servers
        if not servers:
            raise ToolError(
                "No LSP servers are configured. Install a language server such "
                "as pyright or typescript-language-server on PATH and run "
                "/lspstall to re-detect it, or add a matching [[lsp_servers]] entry."
            )
        # workspace/symbol is workspace-wide and a workspace may span several
        # languages (pyright + gopls + ...). With no file_path to route by
        # extension, fan out to every configured server and merge, so symbols
        # from all languages surface. A per-server failure (server down, or
        # workspace/symbol unsupported) degrades gracefully: that server
        # contributes nothing rather than failing the whole query.
        request_all = getattr(manager, "send_request_all", None)
        if callable(request_all):
            broadcast = cast(
                Callable[[str, dict[str, Any]], Awaitable[list[Any]]], request_all
            )
            batches = await broadcast("workspace/symbol", {"query": query})
        else:
            batches = await asyncio.gather(
                *(
                    server.send_request("workspace/symbol", {"query": query})
                    for server in servers.values()
                ),
                return_exceptions=True,
            )
        coverage = self._workspace_coverage(manager)
        merged, supported = self._merge_symbol_batches(batches)
        if not merged and continuation_token is None:
            if not supported:
                if coverage is not None and coverage.get("partial") is True:
                    raise ToolError(
                        "No resident server supports workspace_symbol, and the "
                        "query covered only resident roots because known roots "
                        "were retired. Use file-scoped document_symbol or raise "
                        "lsp_max_workspace_roots before concluding no symbols exist."
                    )
                raise ToolError(
                    "No configured server supports workspace_symbol. "
                    "Pass file_path to target a server, or use document_symbol "
                    "on a specific file."
                )
            return self._apply_workspace_coverage(
                LspResult(
                    operation="symbols",
                    summary=f"Workspace symbols matching '{query}': none found.",
                ),
                coverage,
            )
        symbols = await self._normalize_symbols(merged, "")
        return self._apply_workspace_coverage(
            self._page_symbol_records(
                label,
                symbols,
                binding=binding,
                continuation_token=continuation_token,
                query=query,
            ),
            coverage,
        )

    @staticmethod
    def _workspace_coverage(manager: Any) -> dict[str, int | bool] | None:
        get_readiness = getattr(manager, "readiness", None)
        if not callable(get_readiness):
            return None
        snapshot = cast(Callable[[], Any], get_readiness)()
        pool = getattr(snapshot, "route_pool", None)
        if pool is None:
            return None
        return {
            "resident_roots": pool.resident_roots,
            "known_roots": pool.known_roots,
            "partial": pool.workspace_symbol_partial,
        }

    @staticmethod
    def _apply_workspace_coverage(
        result: LspResult, coverage: dict[str, int | bool] | None
    ) -> LspResult:
        if coverage is None:
            return result
        partial = coverage.get("partial") is True
        summary = result.summary
        if partial:
            summary += (
                "\nCoverage warning: workspace_symbol queried only "
                f"{coverage.get('resident_roots', 0)} of "
                f"{coverage.get('known_roots', 0)} known workspace roots because "
                "the others were retired by the LSP root limit. Results are "
                "partial; use file-scoped document_symbol or raise "
                "lsp_max_workspace_roots for full coverage."
            )
        return result.model_copy(
            update={
                "summary": summary,
                "workspace_coverage": coverage,
                "partial_coverage": partial,
            }
        )

    @staticmethod
    def _merge_symbol_batches(batches: Any) -> tuple[list[dict[str, Any]], bool]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        supported = False
        for batch in batches:
            if isinstance(batch, BaseException):
                # -32601 method-not-found (no workspace index), crashes, and
                # timeouts are all treated as "this server contributes nothing".
                continue
            supported = True
            for sym in batch or []:
                if not isinstance(sym, dict):
                    continue
                key = Lsp._workspace_symbol_identity(sym)
                if key is not None and key in seen:
                    continue
                if key is not None:
                    seen.add(key)
                merged.append(sym)
        return merged, supported

    @staticmethod
    def _workspace_symbol_identity(symbol: dict[str, Any]) -> tuple[str, ...] | None:
        location = symbol.get("location")
        if not isinstance(location, dict):
            return None
        uri = str(location.get("uri", ""))
        range_ = location.get("range")
        if not uri or not isinstance(range_, dict):
            return None
        start = range_.get("start")
        end = range_.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            return None
        return (
            str(symbol.get("name", "")),
            uri,
            str(start.get("line", "")),
            str(start.get("character", "")),
            str(end.get("line", "")),
            str(end.get("character", "")),
            str(symbol.get("kind", "")),
            str(symbol.get("containerName", "")),
        )

    def _simple_dispatch_table(
        self,
    ) -> dict[LspOperation, tuple[str, str, Any, dict[str, Any] | None]]:
        return {
            LspOperation.GO_TO_DEFINITION: (
                "textDocument/definition",
                "Definitions",
                self._format_locations,
                None,
            ),
            LspOperation.GO_TO_IMPLEMENTATION: (
                "textDocument/implementation",
                "Implementations",
                self._format_locations,
                None,
            ),
            LspOperation.FIND_REFERENCES: (
                "textDocument/references",
                "References",
                self._format_locations,
                {"context": {"includeDeclaration": True}},
            ),
            LspOperation.DOCUMENT_SYMBOL: (
                "textDocument/documentSymbol",
                "Document symbols",
                self._format_symbols,
                None,
            ),
        }

    async def _call_hierarchy(
        self,
        manager: Any,
        args: LspArgs,
        file_path: str,
        text_doc: dict[str, Any],
        position: dict[str, int],
        binding: LspQueryBinding | None = None,
        ctx: InvokeContext | None = None,
    ) -> LspResult:
        if binding is None:
            binding = self._query_binding(manager, args, file_path, ctx)
        items = await self._prepare_call_hierarchy_at(
            manager, file_path, text_doc, position
        )
        if not items:
            # prepareCallHierarchy needs the cursor on the callable's identifier
            # (its selectionRange), not the `fn`/`def` keyword or whitespace.
            # All position ops share this constraint; resolve the deepest
            # document symbol spanning the position and retry at its
            # selectionRange.start so the request lands on the identifier.
            resolved = await self._resolve_callable_position(
                manager, file_path, text_doc, position
            )
            if resolved is not None and resolved != position:
                items = await self._prepare_call_hierarchy_at(
                    manager, file_path, text_doc, resolved
                )

        if args.operation is LspOperation.PREPARE_CALL_HIERARCHY:
            items = self._filter_task_locations(items, ctx)
            items = await self._normalize_location_positions(items)
            page = self._page_items(
                items,
                binding=binding,
                continuation_token=args.continuation_token,
                page_size=_MAX_CALL_RESULTS,
            )
            return self._format_call_items(
                "Call hierarchy items", list(page.items), page=page
            )

        label = (
            "Incoming calls"
            if args.operation is LspOperation.INCOMING_CALLS
            else "Outgoing calls"
        )
        if not items:
            return LspResult(
                operation=str(args.operation),
                summary=(
                    f"{label}: no callable at line {position.get('line', 0) + 1}. "
                    "Set character to the function/method name, or use "
                    "find_references to list usages (it does not preserve "
                    "caller/callee direction)."
                ),
            )
        method = (
            "callHierarchy/incomingCalls"
            if args.operation is LspOperation.INCOMING_CALLS
            else "callHierarchy/outgoingCalls"
        )
        # A class-kind item resolves to no edges by design (pyright does not
        # model instantiation as a call edge); the empty follow-up is final, so
        # skip the cold-index retry loop instead of stalling ~1.4s and tagging
        # a false "server was indexing" caveat.
        class_symbol = any(
            isinstance(it, dict) and it.get("kind") == _SYMBOL_KIND_CLASS
            for it in items[:5]
        )
        max_attempts = 1 if class_symbol else _CALL_HIERARCHY_RETRIES
        out: list[dict[str, Any]] = []
        retries_used = 0
        for attempt in range(max_attempts):
            out = []
            for item in items[:5]:
                raw, _ = await manager.send_request(file_path, method, {"item": item})
                for call in raw or []:
                    # LSP: CallHierarchyIncomingCall carries `from` (the caller);
                    # CallHierarchyOutgoingCall carries `to` (the callee).
                    target = (
                        call.get("from")
                        if args.operation is LspOperation.INCOMING_CALLS
                        else call.get("to")
                    )
                    if target:
                        out.append(target)
            if out:
                break
            # prepareCallHierarchy returned items (the callable exists) but the
            # follow-up is empty: the server's package graph isn't loaded yet.
            # Wait briefly and retry — gopls/pyright need indexing before they
            # can resolve caller/callee edges.
            if attempt < max_attempts - 1:
                retries_used += 1
                await asyncio.sleep(_CALL_HIERARCHY_BACKOFF[attempt])
        out = await self._filter_gitignored(out)
        out = self._filter_task_locations(out, ctx)
        out = await self._normalize_location_positions(out)
        page = self._page_items(
            out,
            binding=binding,
            continuation_token=args.continuation_token,
            page_size=_MAX_CALL_RESULTS,
        )
        result = self._format_locations(label, list(page.items), page=page)
        if retries_used:
            # The server was still indexing (call edges resolved only after
            # backoff). Flag it so the caller does not mistake a thin or empty
            # result for "no callers/callees" — re-running once warm fills it in.
            result.summary += (
                f" [server was indexing, retried {retries_used}x — result may "
                "be incomplete; re-run if it looks thin]"
            )
        return result

    async def _prepare_call_hierarchy_at(
        self,
        manager: Any,
        file_path: str,
        text_doc: dict[str, Any],
        position: dict[str, int],
    ) -> list[dict[str, Any]]:
        raw, _ = await manager.send_request(
            file_path,
            "textDocument/prepareCallHierarchy",
            {**text_doc, "position": position},
        )
        return self._as_item_list(raw)

    @staticmethod
    def _as_item_list(raw: Any) -> list[dict[str, Any]]:
        if not raw:
            return []
        if isinstance(raw, dict):
            return [raw]
        return [x for x in raw if isinstance(x, dict)]

    async def _resolve_callable_position(
        self,
        manager: Any,
        file_path: str,
        text_doc: dict[str, Any],
        position: dict[str, int],
    ) -> dict[str, int] | None:
        raw, _ = await manager.send_request(
            file_path, "textDocument/documentSymbol", text_doc
        )
        if not raw:
            return None
        node = self._deepest_symbol_at(raw, position)
        if node is None:
            return None
        # DocumentSymbol carries selectionRange (the identifier); SymbolInformation
        # carries a location.range instead.
        if isinstance(node.get("selectionRange"), dict):
            start = (node.get("selectionRange") or {}).get("start") or {}
        else:
            start = ((node.get("location") or {}).get("range") or {}).get("start") or {}
        line = start.get("line")
        character = start.get("character")
        if line is None or character is None:
            return None
        return {"line": int(line), "character": int(character)}

    async def _request_at_identifier(
        self,
        manager: Any,
        file_path: str,
        text_doc: dict[str, Any],
        position: dict[str, int],
        method: str,
        extra: dict[str, Any] | None,
    ) -> Any:
        # Position ops (definition/references/hover/implementation) resolve only
        # when the cursor sits on the symbol's identifier; one on the
        # `def`/`class` keyword or leading whitespace returns empty. If the first
        # attempt yields nothing, resolve the deepest document symbol spanning
        # the position and retry at its selectionRange.start -- the same recovery
        # call hierarchy uses. A non-empty first result short-circuits, so this
        # never alters a query that already resolved.
        params: dict[str, Any] = {**text_doc, **(extra or {}), "position": position}
        raw, _ = await manager.send_request(file_path, method, params)
        if raw:
            return raw
        resolved = await self._resolve_callable_position(
            manager, file_path, text_doc, position
        )
        if resolved is None or resolved == position:
            return raw
        retry, _ = await manager.send_request(
            file_path, method, {**text_doc, **(extra or {}), "position": resolved}
        )
        return retry

    @classmethod
    def _deepest_symbol_at(
        cls, symbols: list[Any], position: dict[str, int]
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for sym in symbols:
            if not isinstance(sym, dict):
                continue
            if "selectionRange" in sym:
                rng = sym.get("range")
            else:
                rng = (sym.get("location") or {}).get("range")
            if not isinstance(rng, dict) or not cls._range_contains(rng, position):
                continue
            best = sym
            children = sym.get("children")
            if isinstance(children, list):
                deeper = cls._deepest_symbol_at(children, position)
                if deeper is not None:
                    best = deeper
        return best

    @staticmethod
    def _range_contains(rng: dict[str, Any], position: dict[str, int]) -> bool:
        start = rng.get("start") or {}
        end = rng.get("end") or {}
        pl, pc = position.get("line", 0), position.get("character", 0)
        sl, sc = start.get("line", 0), start.get("character", 0)
        el, ec = end.get("line", 0), end.get("character", 0)
        if pl < sl or pl > el:
            return False
        if pl == sl and pc < sc:
            return False
        if pl == el and pc > ec:
            return False
        return True

    def _format_locations(
        self, label: str, raw: Any, *, page: LspContinuationPage | None = None
    ) -> LspResult:
        items = self._as_location_list(raw)
        if not items:
            return LspResult(
                operation="locations",
                summary=f"{label}: none found.",
                total_count=0,
                returned_count=0,
            )
        returned = items if page is not None else items[:_MAX_LOCATION_RESULTS]
        total = page.total_count if page is not None else len(items)
        offset = page.offset if page is not None else 0
        lines: list[str] = [
            self._counted_header(label, total, len(returned), offset=offset)
        ]
        for loc in returned:
            path = path_from_uri(loc.get("uri", ""))
            start = (loc.get("range") or {}).get("start") or {}
            encoding_note = (
                " [UTF-16 column]" if loc.get("position_encoding") == "utf-16" else ""
            )
            lines.append(
                f"  {path}:{start.get('line', 0) + 1}:"
                f"{start.get('character', 0) + 1}{encoding_note}"
            )
        self._append_page_notice(lines, total, len(returned), offset, page)
        return LspResult(
            operation="locations",
            summary="\n".join(lines),
            locations=returned,
            total_count=total,
            returned_count=len(returned),
            was_truncated=offset > 0 or len(returned) < total,
            has_more=page.has_more if page is not None else False,
            page_offset=offset,
            continuation_token=page.continuation_token if page is not None else None,
        )

    def _format_hover(self, raw: Any) -> LspResult:
        if not raw:
            return LspResult(operation="hover", summary="No hover information.")
        contents = raw.get("contents")
        text = self._extract_markup(contents)
        return LspResult(operation="hover", summary=f"Hover:\n{text}")

    def _format_symbols(
        self, label: str, raw: Any, query: str | None = None
    ) -> LspResult:
        return self._format_symbol_records(
            label, normalize_document_symbols(raw, ""), query=query
        )

    def _format_symbol_records(
        self, label: str, records: list[NormalizedSymbol], query: str | None = None
    ) -> LspResult:
        if not records:
            return LspResult(
                operation="symbols",
                summary=f"{label}: none found.",
                total_count=0,
                returned_count=0,
            )
        items = list(records)
        if query:
            items = sorted(
                items,
                key=lambda symbol: self._normalized_symbol_sort_key(symbol, query),
            )
        symbols = [self._symbol_dict(symbol) for symbol in items]
        return self._format_symbol_dicts(label, symbols, query=None)

    def _page_symbol_records(
        self,
        label: str,
        records: list[NormalizedSymbol],
        *,
        binding: LspQueryBinding,
        continuation_token: str | None,
        query: str | None = None,
    ) -> LspResult:
        items = list(records)
        if query:
            items.sort(
                key=lambda symbol: self._normalized_symbol_sort_key(symbol, query)
            )
        page = self._page_items(
            [self._symbol_dict(symbol) for symbol in items],
            binding=binding,
            continuation_token=continuation_token,
            page_size=_MAX_SYMBOL_RESULTS,
        )
        return self._format_symbol_dicts(label, list(page.items), page=page)

    def _format_symbol_dicts(
        self,
        label: str,
        symbols: list[dict[str, Any]],
        *,
        query: str | None = None,
        page: LspContinuationPage | None = None,
    ) -> LspResult:
        if query:
            symbols = sorted(
                symbols, key=lambda symbol: self._symbol_rank(symbol, query)
            )
        returned = symbols if page is not None else symbols[:_MAX_SYMBOL_RESULTS]
        total = page.total_count if page is not None else len(symbols)
        offset = page.offset if page is not None else 0
        names = [str(symbol.get("name", "")) for symbol in returned]
        lines = [self._counted_header(label, total, len(returned), offset=offset)]
        for symbol in returned:
            container = symbol.get("container_name")
            suffix = f" in {container}" if container else ""
            path = path_from_uri(str(symbol.get("uri", "")))
            selection_range = symbol.get("selection_range")
            if isinstance(selection_range, dict):
                start = selection_range.get("start") or {}
                encoding_note = (
                    " [UTF-16 column]"
                    if symbol.get("position_encoding") == "utf-16"
                    else ""
                )
                coord = (
                    f" at {path}:{int(start.get('line', 0)) + 1}:"
                    f"{int(start.get('character', 0)) + 1}{encoding_note}"
                )
            else:
                coord = f" at {path}" if path else " (position unavailable)"
            indent = "  " * (int(symbol.get("depth", 0)) + 1)
            lines.append(f"{indent}{symbol.get('name', '')}{suffix}{coord}")
        self._append_page_notice(lines, total, len(returned), offset, page)
        return LspResult(
            operation="symbols",
            summary="\n".join(lines),
            symbol_names=names,
            symbols=returned,
            total_count=total,
            returned_count=len(returned),
            was_truncated=offset > 0 or len(returned) < total,
            has_more=page.has_more if page is not None else False,
            page_offset=offset,
            continuation_token=page.continuation_token if page is not None else None,
        )

    @staticmethod
    def _symbol_dict(symbol: NormalizedSymbol) -> dict[str, Any]:
        selection_range = symbol.selection_range
        return {
            "name": symbol.name,
            "kind": symbol.kind,
            "detail": symbol.detail,
            "uri": symbol.uri,
            "selection_range": (
                {
                    "start": {
                        "line": selection_range.start.line,
                        "character": selection_range.start.character,
                    },
                    "end": {
                        "line": selection_range.end.line,
                        "character": selection_range.end.character,
                    },
                }
                if selection_range is not None
                else None
            ),
            "depth": symbol.depth,
            "container_path": list(symbol.container_path),
            "container_name": symbol.container_name,
            "hierarchical": symbol.hierarchical,
            "position_encoding": symbol.position_encoding,
        }

    async def _normalize_symbols(
        self, raw: Any, document_uri: str
    ) -> list[NormalizedSymbol]:
        symbols = normalize_document_symbols(raw, document_uri)
        texts: dict[str, str | None] = {}
        normalized: list[NormalizedSymbol] = []
        for symbol in symbols:
            uri = symbol.uri
            if uri not in texts:
                path = Path(path_from_uri(uri))
                if not uri.startswith("file:") or not path.is_file():
                    texts[uri] = None
                else:
                    try:
                        texts[uri] = (await read_safe_async(path)).text
                    except OSError:
                        texts[uri] = None
            text = texts[uri]
            if text is None or symbol.selection_range is None:
                normalized.append(symbol)
                continue
            try:
                selection_range = utf16_range_to_codepoint(text, symbol.selection_range)
            except (TypeError, ValueError):
                normalized.append(symbol)
                continue
            normalized.append(
                replace(
                    symbol,
                    selection_range=selection_range,
                    position_encoding="unicode-codepoint",
                )
            )
        return normalized

    def _format_call_items(
        self, label: str, raw: Any, *, page: LspContinuationPage | None = None
    ) -> LspResult:
        if not raw:
            return LspResult(
                operation="call_hierarchy",
                summary=f"{label}: none at position.",
                total_count=0,
                returned_count=0,
            )
        items = list(raw)
        returned = items if page is not None else items[:_MAX_CALL_RESULTS]
        total = page.total_count if page is not None else len(items)
        offset = page.offset if page is not None else 0
        lines = [self._counted_header(label, total, len(returned), offset=offset)]
        for item in returned:
            name = item.get("name", "?")
            uri = item.get("uri") or (item.get("data") or {}).get("uri", "")
            rng = item.get("range") or {}
            start = rng.get("start") or {}
            lines.append(f"  {name} at {path_from_uri(uri)}:{start.get('line', 0) + 1}")
        self._append_page_notice(lines, total, len(returned), offset, page)
        return LspResult(
            operation="call_hierarchy",
            summary="\n".join(lines),
            locations=returned,
            total_count=total,
            returned_count=len(returned),
            was_truncated=offset > 0 or len(returned) < total,
            has_more=page.has_more if page is not None else False,
            page_offset=offset,
            continuation_token=page.continuation_token if page is not None else None,
        )

    @staticmethod
    def _counted_header(
        label: str, total: int, returned: int, *, offset: int = 0
    ) -> str:
        if total == returned and offset == 0:
            return f"{label} ({total}):"
        if returned == 0:
            return f"{label} ({total} total; page at offset {offset} is empty):"
        return f"{label} ({total} total; showing {offset + 1}–{offset + returned}):"

    @staticmethod
    def _append_truncation_notice(lines: list[str], total: int, returned: int) -> None:
        omitted = total - returned
        if omitted > 0:
            lines.append(
                f"  [truncated: {omitted} omitted; do not assume complete coverage]"
            )

    @classmethod
    def _append_page_notice(
        cls,
        lines: list[str],
        total: int,
        returned: int,
        offset: int,
        page: LspContinuationPage | None,
    ) -> None:
        if page is None:
            cls._append_truncation_notice(lines, total, returned)
            return
        if page.continuation_token is not None:
            remaining = total - offset - returned
            lines.append(
                f"  [{remaining} more result(s); repeat the exact query with "
                "continuation_token to continue]"
            )

    @staticmethod
    def _as_location_list(raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            if "uri" in raw:
                return [raw]
            target = raw.get("targetUri") or raw.get("targetUri")
            if target:
                return [
                    {
                        "uri": target,
                        "range": raw.get("targetSelectionRange")
                        or raw.get("targetRange")
                        or {},
                    }
                ]
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            out.extend(Lsp._as_location_list(item))
        return out

    async def _normalize_location_positions(
        self, locations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        texts: dict[str, str | None] = {}
        normalized: list[dict[str, Any]] = []
        for location in locations:
            uri = str(
                location.get("uri", "") or (location.get("data") or {}).get("uri", "")
            )
            range_keys = [
                key
                for key in ("range", "selectionRange")
                if isinstance(location.get(key), dict)
            ]
            if not uri or not range_keys:
                normalized.append(location)
                continue
            if uri not in texts:
                path = Path(path_from_uri(uri))
                if not uri.startswith("file:") or not path.is_file():
                    texts[uri] = None
                else:
                    try:
                        texts[uri] = (await read_safe_async(path)).text
                    except OSError:
                        texts[uri] = None
            text = texts[uri]
            if text is None:
                normalized.append({**location, "position_encoding": "utf-16"})
                continue
            converted_location = dict(location)
            try:
                for key in range_keys:
                    converted = utf16_range_to_codepoint(
                        text, Range.from_lsp(location.get(key))
                    )
                    converted_location[key] = {
                        "start": {
                            "line": converted.start.line,
                            "character": converted.start.character,
                        },
                        "end": {
                            "line": converted.end.line,
                            "character": converted.end.character,
                        },
                    }
            except (TypeError, ValueError):
                normalized.append({**location, "position_encoding": "utf-16"})
                continue
            converted_location["position_encoding"] = "unicode-codepoint"
            normalized.append(converted_location)
        return normalized

    def _result_cache_get(self, key: tuple[Any, ...]) -> tuple[float, LspResult] | None:
        cache = getattr(self, "_result_cache_store", None)
        if cache is None:
            return None
        return cache.get(key)

    def _result_cache_put(self, key: tuple[Any, ...], result: LspResult) -> None:
        cache = getattr(self, "_result_cache_store", None)
        if cache is None:
            cache = {}
            try:
                self._result_cache_store = cache
            except AttributeError:
                return
        if key not in cache and len(cache) >= _RESULT_CACHE_MAX_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = (time.monotonic(), result)

    def _continuation_store(self) -> LspContinuationStore:
        store = getattr(self, "_lsp_continuation_store", None)
        if store is None:
            store = LspContinuationStore()
            self._lsp_continuation_store = store
        return store

    @staticmethod
    def _query_binding(
        manager: Any, args: LspArgs, file_path: str | None, ctx: InvokeContext | None
    ) -> LspQueryBinding:
        task_hash = (
            ctx.task_contract.brief_hash
            if ctx is not None and ctx.task_contract is not None
            else None
        )
        root_path = getattr(manager, "root_path", None)
        return LspQueryBinding(
            operation=args.operation.value,
            file_path=file_path,
            line=args.line,
            character=args.character,
            query=args.query,
            session_id=ctx.session_id if ctx is not None else None,
            task_brief_hash=task_hash,
            lsp_generation=getattr(manager, "generation", current_lsp_generation()),
            workspace_root=str(root_path) if root_path is not None else str(Path.cwd()),
        )

    def _page_items(
        self,
        items: list[dict[str, Any]],
        *,
        binding: LspQueryBinding,
        continuation_token: str | None,
        page_size: int,
    ) -> LspContinuationPage:
        store = self._continuation_store()
        try:
            if continuation_token is None:
                return store.first_page(binding, items, page_size=page_size)
            try:
                return store.get_page(continuation_token, binding)
            except LspContinuationReloadRequired:
                return store.get_page(continuation_token, binding, reloaded_items=items)
        except LspContinuationError as exc:
            raise ToolError(str(exc)) from exc
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    _GIT_CHECK_BATCH = 50
    _GIT_CHECK_TIMEOUT = 5.0
    _REVPARSE_TIMEOUT = 3.0

    async def _filter_gitignored(
        self, locations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not locations:
            return locations
        cwd = Path.cwd()
        repo_root = await self._repo_toplevel(cwd)
        # Per-path ignore verdicts persist across calls: gitignore rules are
        # static for the session, so a definition resolving into site-packages
        # or a vendored dir is paid for once, not on every query.
        ignore_cache: dict[str, bool] = getattr(self, "_ignore_cache_store", {})
        kept: list[dict[str, Any]] = []
        for i in range(0, len(locations), self._GIT_CHECK_BATCH):
            batch = locations[i : i + self._GIT_CHECK_BATCH]
            batch_paths = [path_from_uri(loc.get("uri", "")) for loc in batch]
            uncached = [p for p in batch_paths if p not in ignore_cache]
            if uncached:
                verdicts = await self._check_ignore(cwd, repo_root, uncached)
                ignore_cache.update(zip(uncached, verdicts, strict=False))
            for loc, path in zip(batch, batch_paths, strict=False):
                if not ignore_cache.get(path, False):
                    kept.append(loc)
        try:
            self._ignore_cache_store = ignore_cache
        except AttributeError:
            pass
        return kept

    async def _repo_toplevel(self, cwd: Path) -> Path | None:
        cached = getattr(self, "_cached_repo_root", None)
        if cached is not None and cached[0] == cwd:
            return cached[1]
        root: Path | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "--show-toplevel",
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self._REVPARSE_TIMEOUT
            )
            if proc.returncode == 0:
                top = stdout.decode("utf-8", "replace").strip()
                if top:
                    root = Path(top)
        except (OSError, FileNotFoundError, TimeoutError):
            root = None
        try:
            self._cached_repo_root = (cwd, root)
        except AttributeError:
            pass
        return root

    async def _check_ignore(
        self, cwd: Path, repo_root: Path | None, paths: list[str]
    ) -> list[bool]:
        # Partition before invoking git: a single out-of-repo path (common —
        # any definition resolving into site-packages/typeshed) makes
        # ``git check-ignore`` abort with exit 128 and empty stdout, which
        # would otherwise read as "nothing ignored" and leak the whole batch.
        # Only in-repo paths can be ignored by this repo; out-of-repo paths
        # default to not-ignored. Membership is compared lexically (no symlink
        # resolution): git applies ignore rules to the path as written, so an
        # in-repo symlink whose target is outside still counts as in-repo and
        # is handed to git, which knows how to ignore it.
        verdicts: dict[str, bool] = {}
        to_check: list[str] = []
        for p in paths:
            if not Path(p).exists():
                verdicts[p] = False
            elif repo_root is not None and not self._path_within(repo_root, Path(p)):
                verdicts[p] = False
            else:
                to_check.append(p)
        if not to_check:
            return [verdicts.get(p, False) for p in paths]
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "check-ignore",
                "--no-index",
                *to_check,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError):
            return [verdicts.get(p, False) for p in paths]
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self._GIT_CHECK_TIMEOUT
            )
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return [verdicts.get(p, False) for p in paths]
        # check-ignore exits 0 if any path is ignored, 1 if none match. Any
        # other code (e.g. 128 fatal) means the output is not trustworthy;
        # fail open rather than read empty stdout as "kept".
        if proc.returncode not in {0, 1}:
            return [verdicts.get(p, False) for p in paths]
        ignored_set = {
            Path(line.decode("utf-8", "replace").strip().split(":", 1)[-1]).resolve()
            for line in stdout.splitlines()
            if line.strip()
        }
        return [verdicts.get(p, Path(p).resolve() in ignored_set) for p in paths]

    @staticmethod
    def _path_within(root: Path, path: Path) -> bool:
        try:
            Path(path).relative_to(root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _extract_markup(contents: Any) -> str:
        if isinstance(contents, str):
            return contents.strip()
        if isinstance(contents, dict):
            if "value" in contents:
                return str(contents["value"]).strip()
            if "kind" in contents:
                return str(contents.get("value", "")).strip()
        if isinstance(contents, list):
            parts: list[str] = []
            for entry in contents:
                if isinstance(entry, str):
                    parts.append(entry)
                elif isinstance(entry, dict) and "value" in entry:
                    parts.append(str(entry["value"]))
            return "\n".join(p.strip() for p in parts if p.strip())
        return str(contents).strip()

    def _resolve_path(self, raw_path: str) -> str:
        path = Path(self._resolve_readiness_path(raw_path))
        if not path.exists():
            raise ToolError(f"File not found at: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory, not a file: {path}")
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise ToolError(
                f"File is {size / 1024 / 1024:.1f} MiB; LSP rejects files over "
                f"{_MAX_FILE_BYTES / 1024 / 1024:.0f} MiB to avoid stalling the server."
            )
        return str(path)

    @staticmethod
    def _resolve_readiness_path(raw_path: str) -> str:
        if not raw_path.strip():
            raise ToolError("file_path cannot be empty")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        enforce_team_metadata_confine(path)
        enforce_isolated_confine(path)
        return str(path)

    @staticmethod
    def _validate_position(path: Path, line: int, character: int, text: str) -> None:
        lines = split_lsp_lines(text)
        line_count = len(lines)
        if line < 1 or line > line_count:
            raise ToolError(
                f"line {line} is out of range: {path.name} has "
                f"{line_count} line{'s' if line_count != 1 else ''}."
            )
        target = lines[line - 1]
        line_len = len(target)
        if character < 1 or character > line_len + 1:
            raise ToolError(
                f"column {character} is out of range: line {line} of "
                f"{path.name} has {line_len} character"
                f"{'s' if line_len != 1 else ''}."
            )

    @staticmethod
    def _symbol_rank(sym: Any, query: str) -> tuple[int, str]:
        name = str(sym.get("name", "")) if isinstance(sym, dict) else str(sym)
        lower = name.lower()
        ql = query.lower()
        if lower == ql:
            tier = 0
        elif lower.startswith(ql):
            tier = 1
        elif ql in lower:
            tier = 2
        else:
            tier = 3
        if lower.startswith("test_") or lower.startswith("test "):
            tier += 10
        return tier, lower

    @classmethod
    def _normalized_symbol_sort_key(
        cls, symbol: NormalizedSymbol, query: str
    ) -> tuple[int, str, str, int, int, str, int]:
        tier, name = cls._symbol_rank(symbol.name, query)
        start = (
            symbol.selection_range.start if symbol.selection_range is not None else None
        )
        return (
            tier,
            name,
            symbol.uri,
            start.line if start is not None else -1,
            start.character if start is not None else -1,
            symbol.container_name or "",
            symbol.kind if symbol.kind is not None else -1,
        )

    @classmethod
    def format_call_display(cls, args: LspArgs) -> ToolCallDisplay:
        target = args.file_path or "(workspace)"
        return ToolCallDisplay(summary=f"LSP {args.operation.value} {target}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, LspResult):
            return ToolResultDisplay(success=False, message=event.error or "No result")
        first_line = event.result.summary.split("\n", 1)[0]
        suffix = f"{event.duration * 1000:.0f}ms" if event.duration else ""
        return ToolResultDisplay(success=True, message=first_line, suffix=suffix)

    @classmethod
    def get_status_text(cls) -> str:
        return "Querying language server"
