from __future__ import annotations

import signal

import pytest

from vibe.core.utils._process_groups import signal_owned_process_group


@pytest.mark.parametrize(("process_group_id", "session_id"), [(41, 99), (99, 41)])
def test_group_signal_requires_child_owned_session_and_group(
    monkeypatch: pytest.MonkeyPatch, process_group_id: int, session_id: int
) -> None:
    calls: list[tuple[int, int | signal.Signals]] = []
    monkeypatch.setattr("os.getpgid", lambda _pid: process_group_id)
    monkeypatch.setattr("os.getsid", lambda _pid: session_id)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: calls.append((pgid, sig)))

    signaled = signal_owned_process_group(41, signal.SIGTERM)

    assert signaled is False
    assert calls == []


def test_group_signal_targets_verified_child_owned_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int | signal.Signals]] = []
    monkeypatch.setattr("os.getpgid", lambda _pid: 41)
    monkeypatch.setattr("os.getsid", lambda _pid: 41)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: calls.append((pgid, sig)))

    signaled = signal_owned_process_group(41, signal.SIGKILL)

    assert signaled is True
    assert calls == [(41, signal.SIGKILL)]
