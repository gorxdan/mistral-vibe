"""Guards against structural divergence from the mistralai/mistral-vibe fork.

See scripts/check_upstream_divergence.py for the rationale. The un-split gate is
filesystem-only so it runs even in shallow CI checkouts; the broader drift guard
needs the baseline commit and skips when it is absent.
"""

from __future__ import annotations

import pytest

from scripts import check_upstream_divergence as guard


def test_no_new_structural_divergence() -> None:
    if not guard.baseline_available():
        pytest.skip(
            f"baseline {guard.baseline_ref()[:12]} not in history (shallow clone); "
            "add fetch-depth: 0 to enable"
        )
    unexpected = guard.unexpected_divergences()
    assert unexpected == [], (
        "New files deleted that upstream still ships — each is a permanent "
        "modify/delete merge conflict. Use a sidecar file, or add to "
        f"_ACCEPTED_DIVERGENCE with a reason: {unexpected}"
    )


def test_agent_loop_is_single_file_matching_upstream() -> None:
    # Un-split landed: agent_loop.py is a single file at upstream's path (not a
    # package), so upstream's edits to it 3-way merge instead of modify/delete.
    root = guard.repo_root()
    assert (root / "vibe/core/agent_loop.py").is_file()
    assert not (root / "vibe/core/agent_loop").is_dir()
