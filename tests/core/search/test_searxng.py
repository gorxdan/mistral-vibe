from __future__ import annotations

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
    assert "-p 8888:8080" in cmd


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

    async def fake_run_cmd(argv, *, timeout):
        calls.append(argv)
        if argv[1] == "exec" and "grep" in argv[-1]:
            return (1, "", "")  # json absent
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

    async def fake_run_cmd(argv, *, timeout):
        calls.append(argv)
        if argv[1] == "inspect":
            return (1, "", "")  # absent
        if argv[1] == "exec" and "grep" in argv[-1]:
            return (1, "", "")  # json missing from fresh image
        return (0, "", "")

    monkeypatch.setattr(searxng, "_run_cmd", fake_run_cmd)
    outcome = await searxng.ensure_running(_SETTINGS, engine="docker")
    assert outcome.ok is True
    assert outcome.started is True
    assert "vibe-searxng" in searxng._started_by_us
    assert any(a[1] == "restart" for a in calls)
