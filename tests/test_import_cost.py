"""Guard against re-introducing eager imports of heavy optional dependencies.

Importing the TUI app must NOT pull in `sounddevice` (loads native PortAudio
and enumerates devices) or the `mcp` SDK (~100ms of pydantic model construction
in `mcp.types`). Both are loaded lazily on first use; this test locks that in so
a stray top-level import does not silently regress cold-start time.

Runs in a fresh subprocess because once any other test imports these modules,
the in-process ``sys.modules`` is polluted for the whole session.
"""

from __future__ import annotations

import subprocess
import sys

# Prefixes that must be absent from sys.modules after importing the app. Using
# prefixes catches submodules (e.g. mcp.types, mcp.client.*) too.
_FORBIDDEN_PREFIXES = ("sounddevice", "mcp")

_PROBE = """
import sys
import vibe.cli.textual_ui.app  # noqa: F401
forbidden = ("sounddevice", "mcp")
leaked = sorted(
    name
    for name in sys.modules
    if any(name == p or name.startswith(p + ".") for p in forbidden)
)
print(",".join(leaked))
"""


def test_app_import_does_not_pull_heavy_optional_deps() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"probe failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, (
        "importing vibe.cli.textual_ui.app eagerly imported heavy optional "
        f"dependencies (must be lazy): {leaked}. "
        f"Allowed prefixes are loaded only on first use: {_FORBIDDEN_PREFIXES}."
    )
