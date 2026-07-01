from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

VIBE_ROOT = Path(__file__).parent


def _resolve_version() -> str:
    try:
        return version("mistral-vibe")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _resolve_version()

__all__ = ["VIBE_ROOT", "__version__"]
