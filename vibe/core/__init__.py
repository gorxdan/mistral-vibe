from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe.core.programmatic import (
        ProgrammaticOptions as ProgrammaticOptions,
        run_programmatic as run_programmatic,
    )


def __getattr__(name: str) -> object:
    if name == "run_programmatic":
        from vibe.core.programmatic import run_programmatic

        return run_programmatic
    if name == "ProgrammaticOptions":
        from vibe.core.programmatic import ProgrammaticOptions

        return ProgrammaticOptions

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ProgrammaticOptions", "run_programmatic"]
