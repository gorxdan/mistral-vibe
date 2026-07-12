from __future__ import annotations

from vibe.cli.textual_ui.app import VibeApp


def test_lsp_probe_failure_detail_never_includes_process_stderr() -> None:
    assert VibeApp._lsp_probe_failure_detail(None) == (
        "probe failed before the process exited"
    )
    assert VibeApp._lsp_probe_failure_detail(17) == "probe exited with status 17"
