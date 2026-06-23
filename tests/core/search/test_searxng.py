from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from vibe.core.search import searxng
from vibe.core.search.searxng import SearxngSettings

_SETTINGS = SearxngSettings(
    url="http://localhost:8888",
    container_name="vibe-searxng",
    port=8888,
    image="searxng/searxng:latest",
    health_timeout=1,
)


@pytest.fixture(autouse=True)
def _reset_state():
    searxng.reset_state()
    yield
    searxng.reset_state()


def test_default_url():
    assert searxng.default_url(8888) == "http://localhost:8888"


def test_effective_url_uses_port_when_url_unset():
    assert SearxngSettings(port=7000).effective_url == "http://localhost:7000"


def test_start_command_mentions_engine_and_ports():
    cmd = _SETTINGS.start_command("podman")
    assert cmd.startswith("podman run -d --name vibe-searxng")
    assert "-p 127.0.0.1:8888:8080" in cmd


def test_start_command_binds_loopback_not_all_interfaces():
    # The managed SearXNG container must only be reachable from localhost: a
    # 0.0.0.0 bind exposes it to the LAN. Vibe only ever talks to localhost.
    cmd = _SETTINGS.start_command("docker")
    assert "127.0.0.1:8888:8080" in cmd
    assert "0.0.0.0:8888" not in cmd
    # Bare port form (which docker binds to 0.0.0.0) must not appear either.
    assert " -p 8888:8080" not in cmd


@pytest.mark.asyncio
async def test_create_container_argv_binds_loopback(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(side_effect=[False, True]))
    run_argv: list[list[str]] = []

    async def fake_run_cmd(argv, *, timeout):
        run_argv.append(argv)
        if argv[1] == "inspect":
            return (1, "", "")  # absent -> triggers `run`
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    await searxng.ensure_running(_SETTINGS, engine="docker")
    create = next(a for a in run_argv if a[1] == "run")
    assert "-p" in create
    port_flag = create[create.index("-p") + 1]
    assert port_flag == "127.0.0.1:8888:8080"


def test_detect_engine_prefers_docker(monkeypatch):
    monkeypatch.setattr(
        searxng.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == "docker" else None,
    )
    assert searxng.detect_engine() == "docker"


def test_detect_engine_falls_back_to_podman(monkeypatch):
    monkeypatch.setattr(
        searxng.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )
    assert searxng.detect_engine() == "podman"


def test_detect_engine_none(monkeypatch):
    monkeypatch.setattr(searxng.shutil, "which", lambda name: None)
    assert searxng.detect_engine() is None


def test_session_skip_toggles():
    assert searxng.session_skipped() is False
    searxng.skip_session()
    assert searxng.session_skipped() is True


@pytest.mark.asyncio
async def test_health_check_ok():
    with respx.mock() as mock:
        mock.get("http://x/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        assert await searxng.health_check("http://x") is True


@pytest.mark.asyncio
async def test_health_check_down():
    with respx.mock() as mock:
        mock.get("http://x/search").mock(side_effect=httpx.ConnectError("refused"))
        assert await searxng.health_check("http://x") is False


@pytest.mark.asyncio
async def test_ensure_running_already_up(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(return_value=True))
    outcome = await searxng.ensure_running(_SETTINGS, engine="docker")
    assert outcome.ok is True
    assert outcome.already_running is True
    assert outcome.started is False
    assert "vibe-searxng" not in searxng._started_by_us


@pytest.mark.asyncio
async def test_ensure_running_no_engine(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(return_value=False))
    monkeypatch.setattr(searxng, "detect_engine", lambda: None)
    outcome = await searxng.ensure_running(_SETTINGS)
    assert outcome.ok is False
    assert outcome.attempted is False
    assert "engine" in outcome.detail


@pytest.mark.asyncio
async def test_ensure_running_creates_absent_container(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(side_effect=[False, True]))
    calls: list[list[str]] = []

    async def fake_run_cmd(argv, *, timeout):
        calls.append(argv)
        if argv[1] == "inspect":
            return (1, "", "no such container")  # absent
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    outcome = await searxng.ensure_running(_SETTINGS, engine="docker")
    assert outcome.ok is True
    assert outcome.started is True
    assert "vibe-searxng" in searxng._started_by_us
    assert any(a[1] == "run" for a in calls)


@pytest.mark.asyncio
async def test_ensure_running_restarts_exited_container(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(side_effect=[False, True]))
    calls: list[list[str]] = []

    async def fake_run_cmd(argv, *, timeout):
        calls.append(argv)
        if argv[1] == "inspect":
            return (0, "false\n", "")  # exited
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    outcome = await searxng.ensure_running(_SETTINGS, engine="docker")
    assert outcome.started is True
    assert any(a[1] == "start" for a in calls)


@pytest.mark.asyncio
async def test_ensure_running_already_running_not_claimed(monkeypatch):
    # Container is up (inspect true) but slow to become healthy: we wait, but
    # because we did not start it, we must not claim ownership.
    monkeypatch.setattr(searxng, "health_check", AsyncMock(side_effect=[False, True]))

    async def fake_run_cmd(argv, *, timeout):
        if argv[1] == "inspect":
            return (0, "true\n", "")
        raise AssertionError("must not start a container that is already running")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    outcome = await searxng.ensure_running(_SETTINGS, engine="docker")
    assert outcome.ok is True
    assert outcome.started is False
    assert "vibe-searxng" not in searxng._started_by_us


@pytest.mark.asyncio
async def test_ensure_running_tracks_started_container_even_if_unhealthy(monkeypatch):
    # We created the container (run succeeded) but it never becomes healthy.
    # It must still be tracked so exit cleanup stops it instead of leaking it.
    settings = SearxngSettings(
        url="http://localhost:8888", container_name="vibe-searxng", health_timeout=0
    )
    monkeypatch.setattr(searxng, "health_check", AsyncMock(return_value=False))

    async def fake_run_cmd(argv, *, timeout):
        if argv[1] == "inspect":
            return (1, "", "")  # absent
        return (0, "", "")  # run -d succeeds

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    outcome = await searxng.ensure_running(settings, engine="docker")
    assert outcome.ok is False
    assert outcome.started is False  # not "started" for UX since unhealthy
    assert "vibe-searxng" in searxng._started_by_us  # but owned, so it gets cleaned up


@pytest.mark.asyncio
async def test_ensure_running_start_failure_reports_detail(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(return_value=False))

    async def fake_run_cmd(argv, *, timeout):
        if argv[1] == "inspect":
            return (1, "", "")  # absent
        return (125, "", "port is already allocated")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    outcome = await searxng.ensure_running(_SETTINGS, engine="docker")
    assert outcome.ok is False
    assert outcome.attempted is True
    assert "port is already allocated" in outcome.detail


@pytest.mark.asyncio
async def test_stop_all_started_stops_only_owned(monkeypatch):
    searxng._started_by_us.add("vibe-searxng")
    stopped: list[list[str]] = []

    async def fake_run_cmd(argv, *, timeout):
        stopped.append(argv)
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    await searxng.stop_all_started(engine="docker", enabled=True)
    assert ["docker", "stop", "vibe-searxng"] in stopped
    assert searxng._started_by_us == set()


@pytest.mark.asyncio
async def test_stop_all_started_disabled_leaves_container(monkeypatch):
    searxng._started_by_us.add("vibe-searxng")
    called: list[list[str]] = []

    async def fake_run_cmd(argv, *, timeout):
        called.append(argv)
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    await searxng.stop_all_started(engine="docker", enabled=False)
    assert called == []
    assert searxng._started_by_us == set()


@pytest.mark.asyncio
async def test_ensure_json_format_noop_when_already_present(monkeypatch):
    calls: list[list[str]] = []

    async def fake_run_cmd(argv, *, timeout):
        calls.append(argv)
        return (0, "", "")  # grep finds json

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    await searxng._ensure_json_format("docker", "vibe-searxng")
    assert any(a[1] == "exec" for a in calls)
    assert not any(a[1] == "restart" for a in calls)


@pytest.mark.asyncio
async def test_ensure_json_format_patches_and_restarts_when_missing(monkeypatch):
    calls: list[list[str]] = []
    grep_calls = 0

    async def fake_run_cmd(argv, *, timeout):
        calls.append(argv)
        nonlocal grep_calls
        if argv[1] == "exec" and "grep" in argv[-1]:
            grep_calls += 1
            # first grep (readiness): json absent; second grep (verify): present
            return (1 if grep_calls == 1 else 0, "", "")
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    await searxng._ensure_json_format("docker", "vibe-searxng")
    scripts = [a[-1] for a in calls if a[1] == "exec"]
    verbs = [a[1] for a in calls]
    assert any("sed" in s for s in scripts)
    assert "restart" in verbs
    # restart follows the patch, never the other way around
    assert verbs.index("restart") > verbs.index("exec")


@pytest.mark.asyncio
async def test_ensure_json_format_skipped_when_container_not_ready(monkeypatch):
    async def fake_run_cmd(argv, *, timeout):
        if argv[1] == "exec":
            return (125, "", "container not running")  # docker-level error
        raise AssertionError("unexpected command while not ready")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    monkeypatch.setattr(searxng.asyncio, "sleep", AsyncMock())
    await searxng._ensure_json_format("docker", "vibe-searxng")  # must not raise


@pytest.mark.asyncio
async def test_ensure_running_patches_json_on_create(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(side_effect=[False, True]))
    calls: list[list[str]] = []
    grep_calls = 0

    async def fake_run_cmd(argv, *, timeout):
        calls.append(argv)
        nonlocal grep_calls
        if argv[1] == "inspect":
            return (1, "", "")  # absent
        if argv[1] == "exec" and "grep" in argv[-1]:
            grep_calls += 1
            # readiness grep: json missing from fresh image; verify grep: present
            return (1 if grep_calls == 1 else 0, "", "")
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    outcome = await searxng.ensure_running(_SETTINGS, engine="docker")
    assert outcome.ok is True
    assert outcome.started is True
    assert "vibe-searxng" in searxng._started_by_us
    assert any(a[1] == "restart" for a in calls)


@pytest.mark.asyncio
async def test_ensure_json_format_no_restart_when_patch_does_not_take(
    monkeypatch, caplog
):
    # The sed matches nothing (e.g. non-standard indentation) so the verify grep
    # still misses json: a restart would not help, so none must happen and a
    # diagnosable warning is logged instead.
    calls: list[list[str]] = []

    async def fake_run_cmd(argv, *, timeout):
        calls.append(argv)
        if argv[1] == "exec" and "grep" in argv[-1]:
            return (1, "", "")  # json absent before and after the sed
        return (0, "", "")  # sed exits 0 (no match is still success)

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    with caplog.at_level(logging.WARNING, logger="vibe"):
        await searxng._ensure_json_format("docker", "vibe-searxng")

    assert not any(a[1] == "restart" for a in calls)
    assert any("did not take" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "image, mutable",
    [
        ("searxng/searxng:latest", True),
        ("searxng/searxng", True),  # no tag -> resolves to :latest
        ("registry.example.com/searxng/searxng:latest", True),
        ("registry.example.com/searxng/searxng", True),
        ("searxng/searxng:2024.1", False),
        ("searxng/searxng@sha256:abc123", False),  # digest-pinned
        ("registry.example.com/searxng/searxng:2024.1@sha256:abc123", False),
    ],
)
def test_has_mutable_tag(image, mutable):
    assert searxng._has_mutable_tag(image) is mutable


@pytest.mark.asyncio
async def test_ensure_running_warns_on_mutable_tag(monkeypatch, caplog):
    mutable_settings = SearxngSettings(
        url="http://localhost:8888",
        container_name="vibe-searxng",
        health_timeout=1,
        image="searxng/searxng:latest",
    )
    monkeypatch.setattr(searxng, "health_check", AsyncMock(side_effect=[False, True]))

    async def fake_run_cmd(argv, *, timeout):
        if argv[1] == "inspect":
            return (1, "", "")  # absent
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)

    with caplog.at_level(logging.WARNING, logger="vibe"):
        await searxng.ensure_running(mutable_settings, engine="docker")

    assert any("mutable image tag" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_ensure_running_no_warning_on_pinned_tag(monkeypatch, caplog):
    pinned_settings = SearxngSettings(
        url="http://localhost:8888",
        container_name="vibe-searxng",
        health_timeout=1,
        image="searxng/searxng@sha256:abc123",
    )
    monkeypatch.setattr(searxng, "health_check", AsyncMock(side_effect=[False, True]))

    async def fake_run_cmd(argv, *, timeout):
        if argv[1] == "inspect":
            return (1, "", "")  # absent
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)

    with caplog.at_level(logging.WARNING, logger="vibe"):
        await searxng.ensure_running(pinned_settings, engine="docker")

    assert not any("mutable image tag" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_health_check_does_not_follow_redirect():
    # SearXNG must not follow redirects: a compromised/redirecting instance
    # could otherwise redirect to an internal address.
    with respx.mock() as mock:
        mock.get("http://x/search").mock(
            return_value=httpx.Response(
                302, headers={"Location": "http://127.0.0.1/secret"}
            )
        )
        # A 302 is not 200, so health_check returns False; the redirect is NOT followed.
        assert await searxng.health_check("http://x") is False
        # Exactly one request — to the SearXNG endpoint, never to the redirect target.
        assert len(mock.calls) == 1


@pytest.mark.asyncio
async def test_wait_for_health_bounds_total_time_with_slow_checks(monkeypatch):
    # Each health_check call itself takes time; the total wait must be bounded by
    # total_timeout, not by total_timeout scaled by (check_time / sleep_time).
    # The old accumulator counted only the sleep interval, so ~20 iterations ran
    # for a 0.2s budget with 0.01s sleeps; the deadline-bound version makes only
    # a handful.
    call_count = 0

    async def slow_health(url: str, *, timeout: float = 3.0) -> bool:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)  # the check itself spends time
        return False

    monkeypatch.setattr(searxng, "health_check", slow_health)
    monkeypatch.setattr(searxng, "_HEALTH_POLL_INTERVAL", 0.01)

    ok = await searxng._wait_for_health("http://x", total_timeout=0.2)
    assert ok is False
    # Deadline-bound: far fewer calls than the buggy ~20; allow generous headroom.
    assert call_count < 10


# --- autostart readiness gate ---


@pytest.mark.asyncio
async def test_autostart_gate_ready_by_default_does_not_block():
    searxng.reset_state()
    # No begin_autostart() called -> gate stays set -> returns immediately.
    await asyncio.wait_for(searxng.wait_for_autostart(), timeout=1.0)


@pytest.mark.asyncio
async def test_autostart_gate_blocks_until_signaled():
    searxng.reset_state()
    searxng.begin_autostart()
    released = asyncio.Event()

    async def releaser() -> None:
        await asyncio.sleep(0.05)
        searxng.signal_autostart_done()
        released.set()

    asyncio.create_task(releaser())
    await asyncio.wait_for(searxng.wait_for_autostart(), timeout=1.0)
    assert released.is_set()


@pytest.mark.asyncio
async def test_autostart_gate_resets_with_state():
    searxng.begin_autostart()
    searxng.reset_state()
    # After reset the gate is recreated set, so a stale cleared state from a
    # prior test does not leak and block the next caller.
    await asyncio.wait_for(searxng.wait_for_autostart(), timeout=1.0)


# --- disable_engines_in_settings (pure, comment-preserving YAML surgery) ---


def _engine_block(text: str, name: str) -> str:
    """Return the text from `- name: <name>` up to (not incl.) the next stanza."""
    marker = f"- name: {name}"
    after = text.split(marker, 1)[1]
    parts = after.split("\n  - name: ", 1)
    return marker + parts[0]


def test_disable_engines_adds_flag_to_named_engine():
    text = dedent("""\
        search:
          formats:
            - html
        engines:
          - name: google
            engine: google
            shortcut: go

          - name: wikipedia
            engine: wikipedia
            shortcut: wp
        """)
    new_text, changed = searxng.disable_engines_in_settings(text, ["google"])

    assert changed == ["google"]
    assert "disabled: true" in _engine_block(new_text, "google")
    assert "disabled" not in _engine_block(new_text, "wikipedia")


def test_disable_engines_idempotent_when_already_disabled():
    text = dedent("""\
        engines:
          - name: google
            engine: google
            disabled: true
        """)
    new_text, changed = searxng.disable_engines_in_settings(text, ["google"])

    assert changed == []
    assert new_text == text


def test_disable_engines_preserves_comments():
    text = dedent("""\
        # top comment
        engines:
          # this is google
          - name: google
            engine: google  # inline
        """)
    new_text, changed = searxng.disable_engines_in_settings(text, ["google"])

    assert changed == ["google"]
    assert "# top comment" in new_text
    assert "# this is google" in new_text
    assert "# inline" in new_text


def test_disable_engines_exact_name_not_substring():
    text = dedent("""\
        engines:
          - name: google
            engine: google
          - name: google images
            engine: google
        """)
    new_text, changed = searxng.disable_engines_in_settings(text, ["google"])

    assert changed == ["google"]
    assert "disabled" not in _engine_block(new_text, "google images")


def test_disable_engines_unknown_name_is_noop():
    text = "engines:\n  - name: google\n    engine: google\n"
    new_text, changed = searxng.disable_engines_in_settings(text, ["nonexistent"])

    assert changed == []
    assert new_text == text


def test_disable_engines_multiple_targets():
    text = dedent("""\
        engines:
          - name: google
            engine: google
          - name: brave
            engine: brave
          - name: wikipedia
            engine: wikipedia
        """)
    new_text, changed = searxng.disable_engines_in_settings(text, ["google", "brave"])

    assert set(changed) == {"google", "brave"}
    assert "disabled" not in _engine_block(new_text, "wikipedia")


def test_disable_engines_missing_engines_key_is_noop():
    text = "search:\n  formats:\n    - html\n"
    new_text, changed = searxng.disable_engines_in_settings(text, ["google"])

    assert changed == []
    assert new_text == text


# --- _apply_disabled_engines (docker cp orchestration) ---


@pytest.mark.asyncio
async def test_apply_disabled_engines_skips_when_empty(monkeypatch):
    settings = SearxngSettings(container_name="vibe-searxng", disabled_engines=())

    called: list = []

    async def fake_run_cmd(argv, *, timeout):
        called.append(argv)
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    await searxng._apply_disabled_engines(settings)

    assert called == []


@pytest.mark.asyncio
async def test_apply_disabled_engines_copies_patches_restarts_on_change(monkeypatch):
    settings = SearxngSettings(
        container_name="vibe-searxng", disabled_engines=("google",)
    )
    monkeypatch.setattr(searxng, "detect_engine", lambda: "docker")
    payload = "engines:\n  - name: google\n    engine: google\n    shortcut: go\n"

    calls: list = []

    async def fake_run_cmd(argv, *, timeout):
        calls.append(list(argv))
        # cp OUT: source is the container (contains ':')
        if argv[1] == "cp" and ":" in argv[2]:
            Path(argv[3]).write_text(payload, encoding="utf-8")
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    await searxng._apply_disabled_engines(settings)

    verbs = [a[1] for a in calls]
    assert verbs.count("cp") == 2  # out then in
    assert "restart" in verbs


@pytest.mark.asyncio
async def test_apply_disabled_engines_no_restart_when_unchanged(monkeypatch):
    settings = SearxngSettings(
        container_name="vibe-searxng", disabled_engines=("google",)
    )
    monkeypatch.setattr(searxng, "detect_engine", lambda: "docker")
    # google already disabled -> patch is a no-op -> must not restart.
    payload = "engines:\n  - name: google\n    engine: google\n    disabled: true\n"

    calls: list = []

    async def fake_run_cmd(argv, *, timeout):
        calls.append(list(argv))
        if argv[1] == "cp" and ":" in argv[2]:
            Path(argv[3]).write_text(payload, encoding="utf-8")
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    await searxng._apply_disabled_engines(settings)

    assert not any(a[1] == "restart" for a in calls)


@pytest.mark.asyncio
async def test_apply_disabled_engines_warns_when_cp_out_fails(monkeypatch, caplog):
    settings = SearxngSettings(
        container_name="vibe-searxng", disabled_engines=("google",)
    )
    monkeypatch.setattr(searxng, "detect_engine", lambda: "docker")

    async def fake_run_cmd(argv, *, timeout):
        return (1, "", "no such container")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    with caplog.at_level(logging.WARNING, logger="vibe"):
        await searxng._apply_disabled_engines(settings)

    assert any("Could not fetch" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_apply_disabled_engines_warns_when_no_engine(monkeypatch, caplog):
    settings = SearxngSettings(
        container_name="vibe-searxng", disabled_engines=("google",)
    )
    monkeypatch.setattr(searxng, "detect_engine", lambda: None)
    with caplog.at_level(logging.WARNING, logger="vibe"):
        await searxng._apply_disabled_engines(settings)

    assert any("no docker/podman" in r.message for r in caplog.records)


# --- ensure_running reconciliation ---


@pytest.mark.asyncio
async def test_ensure_running_reconciles_engines_when_already_up(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(return_value=True))
    applied: list = []

    async def fake_apply(settings):
        applied.append(settings.disabled_engines)

    monkeypatch.setattr(searxng, "_apply_disabled_engines", fake_apply)
    settings = SearxngSettings(
        url="http://localhost:8888", disabled_engines=("google",)
    )
    outcome = await searxng.ensure_running(settings, engine="docker")

    assert outcome.already_running is True
    assert applied == [("google",)]


@pytest.mark.asyncio
async def test_ensure_running_applies_engines_after_create(monkeypatch):
    monkeypatch.setattr(searxng, "health_check", AsyncMock(side_effect=[False, True]))
    applied: list = []

    async def fake_apply(settings):
        applied.append(settings.disabled_engines)

    monkeypatch.setattr(searxng, "_apply_disabled_engines", fake_apply)

    async def fake_run_cmd(argv, *, timeout):
        if argv[1] == "inspect":
            return (1, "", "")  # absent
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    settings = SearxngSettings(url="http://localhost:8888", disabled_engines=("brave",))
    outcome = await searxng.ensure_running(settings, engine="docker")

    assert outcome.ok is True
    assert applied == [("brave",)]


@pytest.mark.asyncio
async def test_ensure_running_default_settings_does_not_reconcile_engines(monkeypatch):
    # Default (empty) disabled_engines must not trigger _apply_disabled_engines
    # at all -- zero overhead for users who don't configure the knob.
    monkeypatch.setattr(searxng, "health_check", AsyncMock(return_value=True))
    applied: list = []

    async def fake_apply(settings):
        applied.append(settings)

    monkeypatch.setattr(searxng, "_apply_disabled_engines", fake_apply)
    await searxng.ensure_running(_SETTINGS, engine="docker")

    assert applied == []
