"""Lifecycle management for a local SearXNG web-search backend.

This module is pure logic with no UI dependencies. It knows how to:

* detect a container engine (docker/podman),
* health-check a SearXNG instance,
* start one (creating or restarting a container) and wait for it to come up,
* stop the containers *this process* started — and only those.

The "only stop what we started" guarantee is provided by the process-level
``_started_by_us`` registry: :func:`ensure_running` records a container name when
it actually launches it, and :func:`stop_all_started` stops exactly that set. Both
the session-start autostart path (the TUI app) and the lazy mid-search recovery
path (the ``web_search`` tool) funnel through here, so there is a single source of
truth for ownership.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import shutil
from typing import Any

import httpx

from vibe.core.logger import logger
from vibe.core.utils.async_subprocess import kill_async_subprocess
from vibe.core.utils.http import build_ssl_context

DEFAULT_IMAGE = "searxng/searxng:latest"
DEFAULT_CONTAINER_NAME = "vibe-searxng"
DEFAULT_PORT = 8888
# SearXNG always listens on 8080 inside the container; we map a host port to it.
_INTERNAL_PORT = 8080

_HTTP_OK = 200
_QUICK_HTTP_TIMEOUT = 3.0
_QUICK_CMD_TIMEOUT = 10.0
_START_CMD_TIMEOUT = 90.0
_HEALTH_POLL_INTERVAL = 1.0

# Container names this process started, so exit cleanup only stops what we own.
_started_by_us: set[str] = set()

# Set once the user opts out of SearXNG for the session via the mid-search
# prompt; suppresses further SearXNG attempts until the process restarts.
_session_skip = False


def default_url(port: int = DEFAULT_PORT) -> str:
    return f"http://localhost:{port}"


@dataclass(frozen=True)
class SearxngSettings:
    """Resolved SearXNG configuration, read from ``[tools.web_search]``."""

    url: str | None = None
    manage: bool = True
    image: str = DEFAULT_IMAGE
    container_name: str = DEFAULT_CONTAINER_NAME
    port: int = DEFAULT_PORT
    autostart: bool = True
    stop_on_exit: bool = True
    health_timeout: int = 30

    @property
    def effective_url(self) -> str:
        return self.url or default_url(self.port)

    def start_command(self, engine: str | None = None) -> str:
        """A copy-pasteable command to launch SearXNG, for error/help text."""
        return (
            f"{engine or 'docker'} run -d --name {self.container_name} "
            f"-p {self.port}:{_INTERNAL_PORT} {self.image}"
        )

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None, *, env_url: str | None = None
    ) -> SearxngSettings:
        m: Mapping[str, Any] = mapping or {}

        def _str(key: str, default: str) -> str:
            value = m.get(key, default)
            return value if isinstance(value, str) and value else default

        def _bool(key: str, default: bool) -> bool:
            value = m.get(key, default)
            return bool(value) if isinstance(value, bool) else default

        def _int(key: str, default: int) -> int:
            value = m.get(key, default)
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool)
                else default
            )

        url = m.get("searxng_url") or env_url
        return cls(
            url=url if isinstance(url, str) and url else None,
            manage=_bool("searxng_manage", True),
            image=_str("searxng_image", DEFAULT_IMAGE),
            container_name=_str("searxng_container_name", DEFAULT_CONTAINER_NAME),
            port=_int("searxng_port", DEFAULT_PORT),
            autostart=_bool("searxng_autostart", True),
            stop_on_exit=_bool("searxng_stop_on_exit", True),
            health_timeout=_int("searxng_timeout", 30),
        )


@dataclass(frozen=True)
class StartOutcome:
    ok: bool
    already_running: bool = False
    started: bool = False
    attempted: bool = False
    detail: str = ""


def session_skipped() -> bool:
    return _session_skip


def skip_session() -> None:
    global _session_skip
    _session_skip = True


def reset_state() -> None:
    """Clear process-level state. Intended for tests."""
    global _session_skip
    _started_by_us.clear()
    _session_skip = False


def detect_engine() -> str | None:
    for engine in ("docker", "podman"):
        if shutil.which(engine):
            return engine
    return None


async def health_check(url: str, *, timeout: float = _QUICK_HTTP_TIMEOUT) -> bool:
    """True when the instance answers a JSON search query with HTTP 200."""
    target = f"{url.rstrip('/')}/search"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, verify=build_ssl_context(), timeout=timeout
        ) as client:
            response = await client.get(target, params={"q": "ping", "format": "json"})
            return response.status_code == _HTTP_OK
    except httpx.HTTPError:
        return False


async def ensure_running(
    settings: SearxngSettings, *, engine: str | None = None
) -> StartOutcome:
    """Make SearXNG reachable, starting a container if needed.

    Records the container in ``_started_by_us`` only when this call actually
    launches it (created or restarted a stopped container) and it becomes
    healthy — so a user-owned, already-running instance is never claimed.
    """
    url = settings.effective_url
    if await health_check(url):
        return StartOutcome(ok=True, already_running=True, detail="already running")

    engine = engine or detect_engine()
    if engine is None:
        return StartOutcome(
            ok=False,
            attempted=False,
            detail="no container engine (docker or podman) found",
        )

    error, we_started = await _start_container(engine, settings)
    if error is not None:
        return error

    # Track ownership the moment we launch a container, regardless of health:
    # a container we created/restarted must be cleaned up on exit even if it
    # never became healthy, otherwise we leak it.
    if we_started:
        _started_by_us.add(settings.container_name)

    ok = await _wait_for_health(url, settings.health_timeout)
    return StartOutcome(
        ok=ok,
        started=we_started and ok,
        attempted=True,
        detail="ready" if ok else "did not become healthy in time",
    )


async def _start_container(
    engine: str, settings: SearxngSettings
) -> tuple[StartOutcome | None, bool]:
    """Create or restart the container as needed.

    Returns ``(error, we_started)`` — ``error`` is a failure ``StartOutcome`` (or
    None on success), ``we_started`` is True only when this call launched it.
    """
    name = settings.container_name
    try:
        state = await _container_state(engine, name)
        if state == "running":
            return None, False
        if state == "absent":
            verb = "run"
            rc, _, err = await _run_cmd(
                [
                    engine,
                    "run",
                    "-d",
                    "--name",
                    name,
                    "-p",
                    f"{settings.port}:{_INTERNAL_PORT}",
                    settings.image,
                ],
                timeout=_START_CMD_TIMEOUT,
            )
        else:  # "exited"
            verb = "start"
            rc, _, err = await _run_cmd(
                [engine, "start", name], timeout=_START_CMD_TIMEOUT
            )
    except FileNotFoundError:
        return StartOutcome(
            ok=False, attempted=False, detail=f"{engine} not found"
        ), False
    except TimeoutError:
        return (
            StartOutcome(
                ok=False, attempted=True, detail=f"{engine} timed out starting SearXNG"
            ),
            False,
        )
    if rc != 0:
        return (
            StartOutcome(
                ok=False,
                attempted=True,
                detail=f"`{engine} {verb}` failed: {err.strip() or rc}",
            ),
            False,
        )
    return None, True


async def stop(engine: str, name: str, *, timeout: float = _QUICK_CMD_TIMEOUT) -> None:
    try:
        await _run_cmd([engine, "stop", name], timeout=timeout)
    except (FileNotFoundError, TimeoutError) as exc:
        logger.warning("Failed to stop SearXNG container %s: %s", name, exc)


async def stop_all_started(*, engine: str | None = None, enabled: bool = True) -> None:
    """Stop every container this process started.

    When ``enabled`` is False (``searxng_stop_on_exit = false``) the registry is
    cleared without stopping anything, leaving the container running.
    """
    names = list(_started_by_us)
    _started_by_us.clear()
    if not enabled or not names:
        return
    engine = engine or detect_engine()
    if engine is None:
        return
    for name in names:
        await stop(engine, name)


async def _container_state(engine: str, name: str) -> str:
    """One of "running", "exited", or "absent"."""
    try:
        rc, out, _ = await _run_cmd(
            [engine, "inspect", "-f", "{{.State.Running}}", name],
            timeout=_QUICK_CMD_TIMEOUT,
        )
    except (FileNotFoundError, TimeoutError):
        return "absent"
    if rc != 0:
        return "absent"
    return "running" if out.strip() == "true" else "exited"


async def _wait_for_health(url: str, total_timeout: float) -> bool:
    elapsed = 0.0
    while elapsed < total_timeout:
        if await health_check(url):
            return True
        await asyncio.sleep(_HEALTH_POLL_INTERVAL)
        elapsed += _HEALTH_POLL_INTERVAL
    return await health_check(url)


async def _run_cmd(argv: list[str], *, timeout: float) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError):
        await kill_async_subprocess(proc, kill_process_group=False)
        raise
    return (
        proc.returncode or 0,
        out.decode(errors="replace"),
        err.decode(errors="replace"),
    )
