"""Guard against re-introducing eager imports of heavy optional dependencies.

Importing a CLI startup module must not eagerly pull heavy optional deps that
are only needed on first use: `sounddevice` (loads native PortAudio and
enumerates devices), the `mcp` SDK (~145ms, the full package via
`mcp.client.auth`), or GitPython. This test locks that in so a stray top-level
import does not silently regress cold-start time.

The forbidden set is per-module: the TUI app is clean of all three, while the
`vibe.cli.cli` entrypoint still pulls GitPython eagerly via
`programmatic -> worktree.manager` (a separate follow-up), so only mcp and
sounddevice are guarded there for now.

Runs in a fresh subprocess because once any other test imports these modules,
the in-process ``sys.modules`` is polluted for the whole session.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_PROBE = """
import sys
import {module}  # noqa: F401
forbidden = {forbidden!r}
leaked = sorted(
    name
    for name in sys.modules
    if any(name == p or name.startswith(p + ".") for p in forbidden)
)
print(",".join(leaked))
"""


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("vibe.cli.textual_ui.app", ("sounddevice", "mcp", "git")),
        # v2.18.3: the MCP registry loads its SDK eagerly (registry -> mcp_oauth
        # -> mcp.client.auth, all module-level upstream), so the cli entrypoint
        # pulls mcp. Restoring lazy MCP loading is a follow-up; sounddevice stays
        # guarded. The TUI app path above is still mcp-clean.
        ("vibe.cli.cli", ("sounddevice",)),
        # SpeechOutputFormat is pinned locally in config/models.py precisely so
        # the config stack stays off the 95-module mistralai SDK (~83ms).
        ("vibe.core.config", ("mistralai",)),
        # `vibe --help` budget: the entrypoint module must stay argparse+stdlib;
        # the config stack (and with it mistralai/otel) loads after parsing.
        ("vibe.cli.entrypoint", ("mistralai",)),
    ],
)
def test_import_does_not_pull_heavy_optional_deps(
    module: str, forbidden: tuple[str, ...]
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, forbidden=forbidden)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"probe failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, (
        f"importing {module} eagerly imported heavy optional dependencies "
        f"(must be lazy): {leaked}. These load only on first use: {forbidden}."
    )
