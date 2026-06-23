from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum, auto
from pathlib import Path
from typing import Any


class LSPError(Exception):
    """Base for the LSP subsystem."""


class LSPNotConnectedError(LSPError):
    """No usable language server for the request."""


class LSPServerCrashedError(LSPError):
    """The server process exited before responding."""


class LSPTimeoutError(LSPError):
    """A request or the startup handshake exceeded its deadline."""


class LSPProtocolError(LSPError):
    """The server returned a JSON-RPC error response.

    ``code`` is the JSON-RPC error code (e.g. -32601 MethodNotFound) when
    available, so callers can distinguish unsupported methods from transient
    errors.
    """

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class ServerState(StrEnum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    ERRORED = auto()


CONTENT_MODIFIED_CODE = -32801
REQUEST_CANCELLED_CODE = -32800


class DiagnosticSeverity(IntEnum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


_SEVERITY_LABEL = {
    DiagnosticSeverity.ERROR: "error",
    DiagnosticSeverity.WARNING: "warning",
    DiagnosticSeverity.INFORMATION: "info",
    DiagnosticSeverity.HINT: "hint",
}


def severity_label(severity: int | DiagnosticSeverity) -> str:
    try:
        return _SEVERITY_LABEL[DiagnosticSeverity(int(severity))]
    except (ValueError, KeyError):
        return "issue"


@dataclass(frozen=True)
class Position:
    line: int
    character: int

    @classmethod
    def from_lsp(cls, data: dict[str, Any] | None) -> Position:
        data = data or {}
        return cls(
            line=int(data.get("line", 0)), character=int(data.get("character", 0))
        )


@dataclass(frozen=True)
class Range:
    start: Position
    end: Position

    @classmethod
    def from_lsp(cls, data: dict[str, Any] | None) -> Range:
        data = data or {}
        return cls(
            start=Position.from_lsp(data.get("start")),
            end=Position.from_lsp(data.get("end")),
        )


@dataclass(frozen=True)
class Location:
    uri: str
    range: Range

    @classmethod
    def from_lsp(cls, data: dict[str, Any] | None) -> Location:
        data = data or {}
        return cls(
            uri=str(data.get("uri", "")), range=Range.from_lsp(data.get("range"))
        )


@dataclass(frozen=True)
class Diagnostic:
    range: Range
    severity: DiagnosticSeverity
    message: str
    source: str | None = None
    code: str | int | None = None

    @classmethod
    def from_lsp(cls, data: dict[str, Any] | None) -> Diagnostic:
        data = data or {}
        raw_severity = data.get("severity", DiagnosticSeverity.ERROR)
        try:
            severity = DiagnosticSeverity(int(raw_severity))
        except (TypeError, ValueError):
            severity = DiagnosticSeverity.ERROR
        return cls(
            range=Range.from_lsp(data.get("range")),
            severity=severity,
            message=str(data.get("message", "")),
            source=data.get("source"),
            code=data.get("code"),
        )

    @property
    def label(self) -> str:
        return _SEVERITY_LABEL.get(self.severity, "issue")

    @property
    def dedup_key(self) -> str:
        start = self.range.start
        code = "" if self.code is None else self.code
        source = self.source or ""
        return f"{self.message}|{source}|{code}|{start.line}:{start.character}"


def uri_from_path(path: str | Path) -> str:
    p = Path(path)
    return p.as_uri() if not str(path).startswith("file:") else str(path)


def path_from_uri(uri: str) -> str:
    from urllib.parse import unquote, urlparse

    if not uri.startswith("file:"):
        return uri
    parsed = urlparse(uri)
    return unquote(parsed.path)


_UTF16_SURROGATE_THRESHOLD = 0xFFFF


def utf16_column(line_text: str, python_char_offset: int) -> int:
    clamped = max(0, min(python_char_offset, len(line_text)))
    return sum(
        2 if ord(ch) > _UTF16_SURROGATE_THRESHOLD else 1 for ch in line_text[:clamped]
    )
