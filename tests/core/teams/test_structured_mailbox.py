from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from vibe.core.teams._escalate import EscalationDenied, escalate_to_lead
from vibe.core.teams.mailbox import Mailbox
from vibe.core.teams.models import Message, MessageKind

# ---------------------------------------------------------------------------
# Mailbox: structured kinds round-trip + backward compat
# ---------------------------------------------------------------------------


def test_send_with_structured_kind_round_trips(tmp_path: Path) -> None:
    box = Mailbox(tmp_path)
    sent = box.send(
        "alice",
        "lead",
        "approve rm -rf",
        kind=MessageKind.PERMISSION_REQUEST,
        payload={"request_id": "req-1", "tool": "bash", "description": "rm -rf /"},
    )
    assert sent.kind is MessageKind.PERMISSION_REQUEST
    assert sent.payload["request_id"] == "req-1"

    received = box.read("lead", mark_read=False)
    assert len(received) == 1
    assert received[0].kind is MessageKind.PERMISSION_REQUEST
    assert received[0].payload["tool"] == "bash"


def test_send_default_kind_is_text(tmp_path: Path) -> None:
    """Legacy callers (positional content) still get TEXT + empty payload."""
    box = Mailbox(tmp_path)
    sent = box.send("alice", "lead", "hello there")
    assert sent.kind is MessageKind.TEXT
    assert sent.payload == {}


def test_old_inbox_file_without_kind_loads_as_text(tmp_path: Path) -> None:
    """An inbox file written before kind/payload existed must still load."""
    # Hand-write a legacy message shape (no kind, no payload).
    inbox = tmp_path / "mailbox" / "lead"
    inbox.mkdir(parents=True)
    legacy = Message(
        id="legacy-1",
        from_name="alice",
        to_name="lead",
        content="legacy prose",
        timestamp=1.0,
    )
    (inbox / "legacy-1.json").write_text(legacy.model_dump_json(indent=2))

    box = Mailbox(tmp_path)
    received = box.read("lead", mark_read=False)
    assert len(received) == 1
    assert received[0].content == "legacy prose"
    assert received[0].kind is MessageKind.TEXT
    assert received[0].payload == {}


def test_permission_response_round_trips_with_payload(tmp_path: Path) -> None:
    box = Mailbox(tmp_path)
    box.send(
        "lead",
        "alice",
        "denied",
        kind=MessageKind.PERMISSION_RESPONSE,
        payload={"request_id": "req-1", "decision": "deny", "reason": "destructive"},
    )
    received = box.read("alice", mark_read=False)
    assert received[0].kind is MessageKind.PERMISSION_RESPONSE
    assert received[0].payload["decision"] == "deny"


# ---------------------------------------------------------------------------
# Escalate helper (teammate side)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_without_team_env_raises(tmp_path: Path, monkeypatch) -> None:
    # Wipe team env so the helper knows it's not in a teammate subprocess.
    for var in ("VIBE_TEAM_DIR", "VIBE_TEAM_NAME", "VIBE_TEAMMATE_NAME"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(EscalationDenied, match="No active team"):
        await escalate_to_lead("bash", "rm -rf /tmp")


@pytest.mark.asyncio
async def test_escalate_times_out_when_no_response(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "alice")
    monkeypatch.setenv("VIBE_TEAM_NAME", "team-x")
    with pytest.raises(EscalationDenied, match="timed out"):
        await escalate_to_lead("bash", "rm", timeout_s=0.2)


@pytest.mark.asyncio
async def test_escalate_unblocks_on_allow_response(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "alice")
    monkeypatch.setenv("VIBE_TEAM_NAME", "team-x")

    async def approver() -> None:
        # Wait for the request to land in the lead's inbox.
        box = Mailbox(tmp_path)
        for _ in range(40):
            unread = box.get_unread("lead")
            if unread:
                request = unread[0]
                box.send(
                    "lead",
                    "alice",
                    "allowed",
                    kind=MessageKind.PERMISSION_RESPONSE,
                    payload={
                        "request_id": request.payload["request_id"],
                        "decision": "allow",
                    },
                )
                return
            await asyncio.sleep(0.05)

    approver_task = asyncio.create_task(approver())
    try:
        # Should resolve without raising once the allow response arrives.
        await escalate_to_lead("bash", "rm -rf /tmp/safe", timeout_s=2.0)
    finally:
        await approver_task


@pytest.mark.asyncio
async def test_escalate_raises_on_deny_response(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "alice")
    monkeypatch.setenv("VIBE_TEAM_NAME", "team-x")

    async def denier() -> None:
        box = Mailbox(tmp_path)
        for _ in range(40):
            unread = box.get_unread("lead")
            if unread:
                request = unread[0]
                box.send(
                    "lead",
                    "alice",
                    "no",
                    kind=MessageKind.PERMISSION_RESPONSE,
                    payload={
                        "request_id": request.payload["request_id"],
                        "decision": "deny",
                        "reason": "too destructive",
                    },
                )
                return
            await asyncio.sleep(0.05)

    denier_task = asyncio.create_task(denier())
    try:
        with pytest.raises(EscalationDenied, match="too destructive"):
            await escalate_to_lead("bash", "rm -rf /", timeout_s=2.0)
    finally:
        await denier_task
