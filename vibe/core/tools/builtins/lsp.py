from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from enum import StrEnum, auto
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from vibe.core.lsp import LSPNotConnectedError, get_lsp_manager
from vibe.core.lsp._types import (
    LSPError,
    LSPProtocolError,
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
from vibe.core.types import ToolStreamEvent
from vibe.core.utils.io import read_safe_async

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig
    from vibe.core.types import ToolResultEvent

_MAX_FILE_BYTES = 10 * 1024 * 1024
_METHOD_NOT_FOUND = -32601
_CALL_HIERARCHY_RETRIES = 4
_CALL_HIERARCHY_BACKOFF = (0.2, 0.4, 0.8)
# Memoize repeat queries within a short window. Keyed on the queried file's
# mtime so any edit invalidates instantly; the TTL bounds cross-file staleness
# (a references/call-hierarchy result can shift when *another* file changes).
_RESULT_CACHE_TTL = 3.0
_CALL_HIERARCHY_OPS = frozenset({
    "prepare_call_hierarchy",
    "incoming_calls",
    "outgoing_calls",
})


class LspOperation(StrEnum):
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
    operation: LspOperation = Field(
        description=(
            "LSP operation to perform. Position-based operations "
            "(go_to_definition, find_references, hover, go_to_implementation, "
            "prepare_call_hierarchy, incoming_calls, outgoing_calls) require "
            "line and character. document_symbol needs only file_path. "
            "workspace_symbol needs only query and may omit file_path "
            "(it is workspace-wide; the first configured server is queried)."
        )
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to the source file. Required for every operation "
            "except workspace_symbol."
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
        description="1-based character column. Required for position-based operations.",
    )
    query: str | None = Field(
        default=None, description="Symbol query string. Required for workspace_symbol."
    )


class LspResult(BaseModel):
    operation: str
    summary: str = Field(description="Short human/machine-readable result text.")
    locations: list[dict[str, Any]] = Field(default_factory=list)
    symbol_names: list[str] = Field(default_factory=list)


class LspConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class LspState(BaseToolState):
    pass


class Lsp(
    BaseTool[LspArgs, LspResult, LspConfig, LspState], ToolUIData[LspArgs, LspResult]
):
    read_only: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Query a language server for semantic code intelligence: "
        "go-to-definition, find-references, hover (type info), "
        "document/workspace symbols, go-to-implementation, and call hierarchy. "
        "Prefer this over grep when you need to resolve a symbol, trace its "
        "callers/callees, or read its type — it understands imports, overloads, "
        "and generated code that textual search cannot."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if config is None:
            return True
        return "lsp" in getattr(config, "installed_components", [])

    def resolve_permission(self, args: LspArgs) -> PermissionContext | None:
        return PermissionContext(permission=self.config.permission)

    @staticmethod
    def _lsp_installed() -> bool:
        # Read the persisted flag without depending on InvokeContext carrying
        # VibeConfig (it doesn't). VibeConfig.load is cached and cheap.
        from vibe.core.config import VibeConfig

        return "lsp" in VibeConfig.load().installed_components

    def _ensure_manager(self) -> Any:
        """Return the process LSP manager, lazy-initializing if needed.

        A session that started before /lspstall was run (or before a server
        binary landed on PATH) never had setup_lsp_for_config called, so the
        singleton is None even though installed_components says "lsp". Calling
        the tool then self-heals: if the flag is set, initialize on first use.
        """
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
        manager = self._ensure_manager()
        if manager is None:
            installed = self._lsp_installed()
            if installed:
                raise ToolError(
                    "LSP is enabled but no language server is running in this "
                    "session. Restart vibe, or install a server pyright/"
                    "typescript-language-server/etc. on PATH and run /lspstall."
                )
            raise ToolError("LSP is not enabled. Run /lspstall to enable it.")
        # workspace_symbol is the only operation that may omit file_path: it is
        # workspace-wide and routes to a server without a specific document.
        raw_path = args.file_path
        if raw_path is None:
            if args.operation is LspOperation.WORKSPACE_SYMBOL:
                if not args.query:
                    raise ToolError("workspace_symbol requires a non-empty query.")
                yield await self._workspace_symbol(manager, args.query or "")
                return
            raise ToolError(
                f"{args.operation.value} requires file_path. Only "
                "workspace_symbol may omit it (workspace/symbol is workspace-wide)."
            )
        file_path = self._resolve_path(raw_path)
        server = manager.get_server_for_file(file_path)
        if server is None:
            raise ToolError(
                f"No LSP server configured for {Path(file_path).suffix or 'extensionless'} files. "
                "Run /lspstall to re-detect installed servers, or add a "
                "[[lsp_servers]] entry with a matching language."
            )

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

        async for event in self._execute(
            manager, args, file_path, position_required, server, ctx
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
    ) -> AsyncGenerator[ToolStreamEvent | LspResult, None]:
        try:
            text = await read_safe_async(Path(file_path))
            await manager.open_document(
                file_path,
                text.text,
                server.config.language_id_for(Path(file_path).suffix),
            )
            position = (
                {"line": (args.line or 1) - 1, "character": (args.character or 1) - 1}
                if position_required
                else None
            )
            if position is not None:
                self._validate_position(
                    Path(file_path), args.line or 1, args.character or 1, text.text
                )
            # Memoize repeat queries on an unchanged file. workspace_symbol is
            # global (no single-file mtime to invalidate on), so never cache it.
            cache_key: tuple[Any, ...] | None = None
            if args.operation is not LspOperation.WORKSPACE_SYMBOL:
                cache_key = (
                    file_path,
                    Path(file_path).stat().st_mtime_ns,
                    str(args.operation),
                    args.line,
                    args.character,
                )
                hit = self._result_cache_get(cache_key)
                if hit is not None and time.monotonic() - hit[0] < _RESULT_CACHE_TTL:
                    yield hit[1]
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
            result = await self._dispatch(manager, args, file_path, position)
            if cache_key is not None:
                self._result_cache_put(cache_key, result)
        except LSPNotConnectedError as exc:
            raise ToolError(str(exc)) from exc
        except LSPProtocolError as exc:
            if exc.code == _METHOD_NOT_FOUND:
                raise ToolError(
                    f"{server.config.name} does not support {args.operation.value}. "
                    "Try find_references as a fallback — it returns the same "
                    "caller/callee info via a different method."
                ) from exc
            raise ToolError(f"LSP request failed: {exc}") from exc
        except LSPError as exc:
            raise ToolError(f"LSP request failed: {exc}") from exc

        yield result

    async def _dispatch(
        self,
        manager: Any,
        args: LspArgs,
        file_path: str,
        position: dict[str, int] | None,
    ) -> LspResult:
        uri = uri_from_path(file_path)
        text_doc = {"textDocument": {"uri": uri}}
        simple = self._simple_dispatch_table().get(args.operation)
        if simple is not None:
            method, label, formatter, extra = simple
            params = {**text_doc, **(extra or {})}
            if position is not None:
                params["position"] = position
            raw, _ = await manager.send_request(file_path, method, params)
            if formatter is self._format_locations:
                filtered = await self._filter_gitignored(self._as_location_list(raw))
                return formatter(label, filtered)
            return formatter(label, raw)
        if args.operation is LspOperation.HOVER:
            raw, _ = await manager.send_request(
                file_path, "textDocument/hover", {**text_doc, "position": position}
            )
            return self._format_hover(raw)
        if args.operation is LspOperation.WORKSPACE_SYMBOL:
            query = args.query or ""
            raw, _ = await manager.send_request(
                file_path, "workspace/symbol", {"query": query}
            )
            return self._format_symbols(
                f"Workspace symbols matching '{query}'", raw, query=query
            )
        if args.operation in {
            LspOperation.PREPARE_CALL_HIERARCHY,
            LspOperation.INCOMING_CALLS,
            LspOperation.OUTGOING_CALLS,
        }:
            return await self._call_hierarchy(
                manager, args, file_path, text_doc, position or {}
            )
        raise ToolError(f"Unsupported operation: {args.operation}")

    async def _workspace_symbol(self, manager: Any, query: str) -> LspResult:
        servers = manager.servers
        if not servers:
            raise ToolError(
                "No LSP servers are configured. Run /lspstall to install one, "
                "or pass file_path to route to a server by file extension."
            )
        # workspace/symbol is workspace-wide; without a file_path to route by
        # extension, query the first configured server (declaration order). Pass
        # a file_path to target a specific language server in a multi-language
        # workspace.
        server = next(iter(servers.values()))
        try:
            raw = await server.send_request("workspace/symbol", {"query": query})
        except LSPNotConnectedError as exc:
            raise ToolError(str(exc)) from exc
        except LSPProtocolError as exc:
            if exc.code == _METHOD_NOT_FOUND:
                raise ToolError(
                    "No server supports workspace_symbol in this session. "
                    "Pass file_path to target a server, or use document_symbol "
                    "on a specific file."
                ) from exc
            raise ToolError(f"LSP request failed: {exc}") from exc
        except LSPError as exc:
            raise ToolError(f"LSP request failed: {exc}") from exc
        return self._format_symbols(
            f"Workspace symbols matching '{query}'", raw or [], query=query
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
    ) -> LspResult:
        items = await self._prepare_call_hierarchy_at(
            manager, file_path, text_doc, position
        )
        if not items:
            # prepareCallHierarchy is stricter than hover/references: it needs
            # the cursor exactly on the callable's identifier (its
            # selectionRange). An agent passing a column on the `fn`/`def`
            # keyword or leading whitespace gets an empty result. Resolve the
            # deepest document symbol spanning the position and retry at its
            # selectionRange.start so the request lands on the identifier.
            resolved = await self._resolve_callable_position(
                manager, file_path, text_doc, position
            )
            if resolved is not None and resolved != position:
                items = await self._prepare_call_hierarchy_at(
                    manager, file_path, text_doc, resolved
                )

        if args.operation is LspOperation.PREPARE_CALL_HIERARCHY:
            return self._format_call_items("Call hierarchy items", items)

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
                    "find_references (carries the same caller info) as a fallback."
                ),
            )
        method = (
            "callHierarchy/incomingCalls"
            if args.operation is LspOperation.INCOMING_CALLS
            else "callHierarchy/outgoingCalls"
        )
        out: list[dict[str, Any]] = []
        retries_used = 0
        for attempt in range(_CALL_HIERARCHY_RETRIES):
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
            if attempt < _CALL_HIERARCHY_RETRIES - 1:
                retries_used += 1
                await asyncio.sleep(_CALL_HIERARCHY_BACKOFF[attempt])
        out = await self._filter_gitignored(out)
        result = self._format_locations(label, out)
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

    def _format_locations(self, label: str, raw: Any) -> LspResult:
        items = self._as_location_list(raw)
        if not items:
            return LspResult(operation="locations", summary=f"{label}: none found.")
        lines: list[str] = [f"{label} ({len(items)}):"]
        for loc in items[:50]:
            path = path_from_uri(loc.get("uri", ""))
            start = (loc.get("range") or {}).get("start") or {}
            lines.append(
                f"  {path}:{start.get('line', 0) + 1}:{start.get('character', 0) + 1}"
            )
        return LspResult(
            operation="locations", summary="\n".join(lines), locations=items[:50]
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
        if not raw:
            return LspResult(operation="symbols", summary=f"{label}: none found.")
        items = list(raw)
        if query:
            items = sorted(items, key=lambda s: self._symbol_rank(s, query))
        names: list[str] = []
        lines: list[str] = [f"{label} ({len(items)}):"]
        for sym in items[:100]:
            if "name" in sym:
                name = str(sym.get("name", ""))
                names.append(name)
                container = sym.get("containerName")
                suffix = f" in {container}" if container else ""
                loc = sym.get("location") or {}
                start = ((loc.get("range") or {}).get("start")) or {}
                coord = (
                    f" at {path_from_uri(loc.get('uri', ''))}:"
                    f"{start.get('line', 0) + 1}"
                    if loc
                    else ""
                )
                lines.append(f"  {name}{suffix}{coord}")
            elif "selectionRange" in sym:
                name = str(sym.get("name", ""))
                names.append(name)
                rng = sym.get("selectionRange") or {}
                start = rng.get("start") or {}
                lines.append(f"  {name} at :{start.get('line', 0) + 1}")
        return LspResult(
            operation="symbols", summary="\n".join(lines), symbol_names=names
        )

    def _format_call_items(self, label: str, raw: Any) -> LspResult:
        if not raw:
            return LspResult(
                operation="call_hierarchy", summary=f"{label}: none at position."
            )
        lines = [f"{label} ({len(raw)}):"]
        for item in raw[:50]:
            name = item.get("name", "?")
            uri = item.get("uri") or (item.get("data") or {}).get("uri", "")
            rng = item.get("range") or {}
            start = rng.get("start") or {}
            lines.append(f"  {name} at {path_from_uri(uri)}:{start.get('line', 0) + 1}")
        return LspResult(operation="call_hierarchy", summary="\n".join(lines))

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
        cache[key] = (time.monotonic(), result)

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
        """Resolve the repo root containing ``cwd``. ``None`` if not a git
        repo or git is unavailable. A path outside this root is not subject
        to the repo's ignore rules and is reported not-ignored by the caller.
        Cached per-cwd on the tool instance.
        """
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
        if not raw_path.strip():
            raise ToolError("file_path cannot be empty")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
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
    def _validate_position(path: Path, line: int, character: int, text: str) -> None:
        """Reject out-of-bounds line/character before sending the request.

        Servers reject bad positions with opaque messages ("column is beyond
        end of file"); validate here so the agent gets an actionable error
        naming the actual file bounds.
        """
        lines = text.splitlines()
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
        """Sort key for workspace_symbol relevance. Lower sorts first.

        Tier: 0 exact name, 1 prefix, 2 substring, 3 no match. Test symbols
        (name starts with test_ or Test) get +10 so real definitions surface
        ahead of test noise even when both are substring matches.
        """
        name = str(sym.get("name", "")) if isinstance(sym, dict) else ""
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
    def format_call_display(cls, args: LspArgs) -> ToolCallDisplay:
        target = args.file_path or "(workspace)"
        return ToolCallDisplay(summary=f"LSP {args.operation.value} {target}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, LspResult):
            return ToolResultDisplay(success=False, message=event.error or "No result")
        first_line = event.result.summary.split("\n", 1)[0]
        return ToolResultDisplay(success=True, message=first_line)

    @classmethod
    def get_status_text(cls) -> str:
        return "Querying language server"
