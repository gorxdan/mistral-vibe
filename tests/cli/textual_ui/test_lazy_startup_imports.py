from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_importing_entrypoint_does_not_import_interactive_ui_modules() -> None:
    code = """
import sys
import vibe.cli.entrypoint

blocked = [
    "vibe.cli.cli",
    "vibe.setup.trusted_folders.trust_folder_dialog",
    "textual",
    "git",
    "vibe.core.config",
    "rich",
    "httpx",
]
loaded = [name for name in blocked if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected entrypoint modules loaded: {loaded}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_importing_tui_app_does_not_import_deferred_startup_modules() -> None:
    code = """
import sys
import vibe.cli.textual_ui.app

blocked = [
    "vibe.cli.textual_ui.widgets.connector_auth_app",
    "vibe.cli.textual_ui.widgets.mcp_app",
    "vibe.core.agent_loop",
    "vibe.core.tools.connectors.connector_registry",
    "vibe.core.tools.mcp.tools",
    "mcp",
    "git",
]
loaded = [name for name in blocked if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected startup modules loaded: {loaded}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_importing_agent_loop_does_not_import_remote_tool_modules() -> None:
    code = """
import sys
import vibe.core.agent_loop

blocked = [
    "vibe.core.tools.connectors.connector_registry",
    "vibe.core.tools.mcp.tools",
    "vibe.core.teleport.git",
    "vibe.core.teleport.teleport",
    "mcp",
    "git",
]
loaded = [name for name in blocked if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected agent loop modules loaded: {loaded}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_importing_connector_registry_does_not_import_mcp_runtime() -> None:
    code = """
import sys
import vibe.core.tools.connectors.connector_registry

blocked = [
    "vibe.core.tools.mcp.tools",
    "mcp",
]
loaded = [name for name in blocked if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected connector registry modules loaded: {loaded}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_importing_mcp_app_does_not_import_mcp_runtime() -> None:
    code = """
import sys
import vibe.cli.textual_ui.widgets.mcp_app

blocked = [
    "vibe.core.tools.mcp.tools",
    "mcp",
]
loaded = [name for name in blocked if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected mcp app modules loaded: {loaded}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr or result.stdout


_AGENT_LOOP_MCP_PROBE = Path(__file__).parent / "_agent_loop_mcp_probe.py"


@pytest.mark.parametrize(
    "probe_args", [["--defer"], []], ids=["deferred", "headless_eager"]
)
def test_constructing_agent_loop_without_mcp_servers_does_not_import_mcp_package(
    tmp_path: Path, probe_args: list[str]
) -> None:
    result = subprocess.run(
        [sys.executable, str(_AGENT_LOOP_MCP_PROBE), *probe_args],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "VIBE_HOME": str(tmp_path),
            "VIBE_TEST_DISABLE_KEYRING": "1",
        },
    )

    assert result.returncode == 0, result.stderr or result.stdout
