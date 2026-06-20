"""Verify the persist <-> restore mechanism the TUI/ACP resume paths rely on.

ACP resume: resume_existing_session loads SessionMetadata (incl. loops) from
disk via model_validate_json, then the per-session scheduler restores from it.
TUI resume: ScheduledLoopRunner.restore_from_session reads session_metadata.loops.
Both depend on (a) loops surviving the JSON round-trip and (b) LoopManager
persisting into / restoring from session_metadata.loops.
"""

from __future__ import annotations

import pytest

from vibe.core.loop import LoopManager
from vibe.core.types import ScheduledLoop, SessionMetadata


def _metadata() -> SessionMetadata:
    return SessionMetadata(
        session_id="s1",
        start_time="2026-06-20T00:00:00",
        end_time=None,
        git_commit=None,
        git_branch=None,
        environment={},
        username="tester",
    )


class _Logger:
    """Fake session logger holding a real SessionMetadata (persist is in-memory)."""

    def __init__(self, metadata: SessionMetadata) -> None:
        self.session_metadata = metadata

    async def persist_loops(self) -> None:  # disk write is a no-op in the test
        pass


def test_scheduled_loop_survives_metadata_json_round_trip() -> None:
    # load_metadata uses SessionMetadata.model_validate_json — loops (incl. the
    # recurring flag) must survive it, or ACP/TUI resume restores nothing.
    meta = _metadata()
    meta.loops = [
        ScheduledLoop(
            id="a1",
            interval_seconds=300,
            prompt="check CI",
            next_fire_at=123.0,
            created_at=1.0,
            recurring=True,
        ),
        ScheduledLoop(
            id="b2",
            interval_seconds=60,
            prompt="ping once",
            next_fire_at=200.0,
            created_at=2.0,
            recurring=False,
        ),
    ]
    restored = SessionMetadata.model_validate_json(meta.model_dump_json())
    assert [lp.id for lp in restored.loops] == ["a1", "b2"]
    assert restored.loops[0].recurring is True
    assert restored.loops[1].recurring is False
    assert restored.loops[0].prompt == "check CI"


@pytest.mark.asyncio
async def test_loopmanager_persist_then_restore_round_trip() -> None:
    meta = _metadata()
    mgr = LoopManager(_Logger(meta))  # type: ignore[arg-type]
    await mgr.add_loop(300, "recurring task", recurring=True)
    await mgr.add_loop(60, "one-shot task", recurring=False)

    # _persist wrote the live loops into session_metadata.loops.
    assert len(meta.loops) == 2

    # A fresh manager (resume) restores from that metadata — the source the
    # TUI runner and the ACP per-session scheduler both read.
    resumed = LoopManager(_Logger(meta))  # type: ignore[arg-type]
    resumed.restore(list(meta.loops))
    assert {lp.prompt for lp in resumed.loops} == {"recurring task", "one-shot task"}
    assert {lp.recurring for lp in resumed.loops} == {True, False}


@pytest.mark.asyncio
async def test_restore_is_empty_when_metadata_has_no_loops() -> None:
    # The programmatic path: a fresh session has empty metadata, so restore is a
    # harmless no-op (loops created in the run still fire via --keep-alive).
    meta = _metadata()
    mgr = LoopManager(_Logger(meta))  # type: ignore[arg-type]
    mgr.restore(list(meta.loops))
    assert mgr.loops == []
