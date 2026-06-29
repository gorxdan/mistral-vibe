"""Stale-diagnostic filters: prove a published diagnostic wrong against the
live tree before the registry stages it.

A language server caches import-resolution results; when a concurrent session
adds a module via git, the server's cached 'not found' is stale and would
otherwise reach the model as a phantom error. Each filter is a strategy bound
to the server source that produces the diagnostic — keeping pyright's message
format out of the generic :class:`~vibe.core.lsp._registry.DiagnosticRegistry`.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Protocol

from vibe.core.lsp._types import Diagnostic, DiagnosticSeverity


class StaleDiagnosticFilter(Protocol):
    """Proves a single diagnostic stale against the workspace root."""

    def is_stale(self, diagnostic: Diagnostic, root: Path | None) -> bool: ...


def _module_resolves_on_disk(module_path: str, root: Path) -> bool:
    """True when a dotted module name maps to an existing file under root.

    The disk-evidence check that backs stale suppression: if the module the
    server cannot resolve actually exists, its cached 'not found' is stale.
    """
    parts = module_path.split(".")
    if not parts:
        return False
    rel = Path(*parts)
    return any(
        p.is_file() for p in (root / rel.with_suffix(".py"), root / rel / "__init__.py")
    )


class PyrightStaleFilter:
    """Suppresses pyright import-resolution errors whose target exists on disk.

    Pyright's ``markFilesDirty`` no-ops on never-loaded files, so a module
    added by a concurrent session stays cached as unresolvable. The regex
    matches pyright's import-resolution message format.
    """

    _RE = re.compile(
        r'(?:Import|import)\s+"([a-zA-Z_][\w.]*?)"\s+.*'
        r"(?:could not be resolved|cannot be resolved|cannot be imported)"
    )

    def is_stale(self, diagnostic: Diagnostic, root: Path | None) -> bool:
        if diagnostic.severity != DiagnosticSeverity.ERROR:
            return False
        if root is None:
            return False
        match = self._RE.search(diagnostic.message)
        if match is None:
            return False
        return _module_resolves_on_disk(match.group(1), root)


#: Filters by server source name. Servers not listed here pass diagnostics
#: through unchanged. Add an entry when a new server needs its own stale-proof.
FILTERS_BY_SOURCE: dict[str, StaleDiagnosticFilter] = {"pyright": PyrightStaleFilter()}


__all__ = ["FILTERS_BY_SOURCE", "PyrightStaleFilter", "StaleDiagnosticFilter"]
