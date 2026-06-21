from __future__ import annotations

from collections.abc import AsyncGenerator
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from vibe.core.lsp import LSPNotConnectedError, get_lsp_manager
from vibe.core.lsp._types import LSPError, path_from_uri, uri_from_path
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
            "workspace_symbol needs only query."
        )
    )
    file_path: str = Field(description="Absolute path to the source file.")
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
        file_path = self._resolve_path(args.file_path)
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
            result = await self._dispatch(manager, args, file_path, position)
        except LSPNotConnectedError as exc:
            raise ToolError(str(exc)) from exc
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
            return self._format_symbols(f"Workspace symbols matching '{query}'", raw)
        if args.operation in {LspOperation.INCOMING_CALLS, LspOperation.OUTGOING_CALLS}:
            return await self._call_hierarchy(
                manager, args, file_path, text_doc, position or {}
            )
        raise ToolError(f"Unsupported operation: {args.operation}")

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
            LspOperation.PREPARE_CALL_HIERARCHY: (
                "textDocument/prepareCallHierarchy",
                "Call hierarchy items",
                self._format_call_items,
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
        items, _ = await manager.send_request(
            file_path,
            "textDocument/prepareCallHierarchy",
            {**text_doc, "position": position},
        )
        if not items:
            label = (
                "Incoming calls"
                if args.operation is LspOperation.INCOMING_CALLS
                else "Outgoing calls"
            )
            return LspResult(
                operation=str(args.operation),
                summary=f"{label}: no call hierarchy at position.",
            )
        method = (
            "callHierarchy/incomingCalls"
            if args.operation is LspOperation.INCOMING_CALLS
            else "callHierarchy/outgoingCalls"
        )
        out: list[dict[str, Any]] = []
        for item in items[:5]:
            raw, _ = await manager.send_request(file_path, method, {"item": item})
            for call in raw or []:
                target = (
                    call.get("to")
                    if args.operation is LspOperation.INCOMING_CALLS
                    else call.get("from")
                )
                if target:
                    out.append(target)
        label = (
            "Incoming calls"
            if args.operation is LspOperation.INCOMING_CALLS
            else "Outgoing calls"
        )
        return self._format_locations(label, out)

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

    def _format_symbols(self, label: str, raw: Any) -> LspResult:
        if not raw:
            return LspResult(operation="symbols", summary=f"{label}: none found.")
        names: list[str] = []
        lines: list[str] = [f"{label} ({len(raw)}):"]
        for sym in raw[:100]:
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
        return str(path)

    @classmethod
    def format_call_display(cls, args: LspArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"LSP {args.operation.value} {args.file_path}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, LspResult):
            return ToolResultDisplay(success=False, message=event.error or "No result")
        first_line = event.result.summary.split("\n", 1)[0]
        return ToolResultDisplay(success=True, message=first_line)

    @classmethod
    def get_status_text(cls) -> str:
        return "Querying language server"
