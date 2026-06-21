from __future__ import annotations

import asyncio
from dataclasses import dataclass
import shutil

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
_session_skip = False


def default_url(port: int = DEFAULT_PORT) -> str:
    return f"http://localhost:{port}"


@dataclass(frozen=True)
class SearxngSettings:
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
        return (
            f"{engine or 'docker'} run -d --name {self.container_name} "
            f"-p {self.port}:{_INTERNAL_PORT} {self.image}"
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
    global _session_skip
    _started_by_us.clear()
    _session_skip = False


def detect_engine() -> str | None:
    for engine in ("docker", "podman"):
        if shutil.which(engine):
            return engine
    return None


async def health_check(url: str, *, timeout: float = _QUICK_HTTP_TIMEOUT) -> bool:
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

    # Track ownership the moment we launch a container, regardless of health: a
    # container we created/restarted must be cleaned up on exit even if it never
    # became healthy, otherwise we leak it.
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
    name = settings.container_name
    try:
        match await _container_state(engine, name):
            case "running":
                return None, False
            case "absent":
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
            case _:  # "exited"
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
