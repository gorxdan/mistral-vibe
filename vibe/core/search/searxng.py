from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import httpx
import yaml

from vibe.core.logger import logger
from vibe.core.utils.async_subprocess import kill_async_subprocess
from vibe.core.utils.http import build_ssl_context
from vibe.core.utils.io import read_safe

DEFAULT_IMAGE = "searxng/searxng:latest"
DEFAULT_CONTAINER_NAME = "vibe-searxng"
DEFAULT_PORT = 8888
# SearXNG always listens on 8080 inside the container; we map a host port to it.
_INTERNAL_PORT = 8080
# Bind the host-side port to loopback only: vibe talks to SearXNG over
# localhost, and a 0.0.0.0 bind (docker's default for `-p port:port`) would
# expose the unauthenticated instance to the LAN.
_BIND_ADDRESS = "127.0.0.1"

_HTTP_OK = 200
_QUICK_HTTP_TIMEOUT = 3.0
_QUICK_CMD_TIMEOUT = 10.0
_START_CMD_TIMEOUT = 90.0
_HEALTH_POLL_INTERVAL = 1.0
# After bringing up a container, ensure it can answer JSON queries (the upstream
# image ships `formats: [html]` only, which 403s our health probe).
_PATCH_READINESS_TRIES = 4
_PATCH_READINESS_DELAY = 1.0
# `docker exec` returns >= 125 for docker-level failures (container not running
# yet); real command exit codes (e.g. grep's 0/1) fall below this threshold.
_DOCKER_ERROR_RC = 125

# Container names this process started, so exit cleanup only stops what we own.
_started_by_us: set[str] = set()
_session_skip = False
# Cleared while session-start autostart is bringing up / reconciling the managed
# SearXNG container, so an early search waits rather than racing a container that
# is mid-(re)start and surfacing a confusing "SearXNG is down" prompt. Lazily
# created and set by default, so paths that never go through autostart never block.
_autostart_gate: asyncio.Event | None = None


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
    health_timeout: int = 60
    disabled_engines: tuple[str, ...] = ()

    @property
    def effective_url(self) -> str:
        return self.url or default_url(self.port)

    def start_command(self, engine: str | None = None) -> str:
        return (
            f"{engine or 'docker'} run -d --name {self.container_name} "
            f"-p {_BIND_ADDRESS}:{self.port}:{_INTERNAL_PORT} {self.image}"
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
    global _session_skip, _autostart_gate
    _started_by_us.clear()
    _session_skip = False
    _autostart_gate = None


def _gate() -> asyncio.Event:
    global _autostart_gate
    if _autostart_gate is None:
        _autostart_gate = asyncio.Event()
        _autostart_gate.set()
    return _autostart_gate


def begin_autostart() -> None:
    """Mark session-start autostart in progress: searches wait until
    :func:`signal_autostart_done` releases the gate.
    """
    _gate().clear()


def signal_autostart_done() -> None:
    """Release the autostart gate once startup/reconciliation has settled."""
    _gate().set()


async def wait_for_autostart() -> None:
    """Block until session-start autostart has finished (a no-op once done)."""
    await _gate().wait()


def detect_engine() -> str | None:
    for engine in ("docker", "podman"):
        if shutil.which(engine):
            return engine
    return None


def _has_mutable_tag(image: str) -> bool:
    """Return True for images pinned to :latest or carrying no tag (which
    resolves to :latest). Digest-pinned and explicitly version-tagged images
    are considered stable.
    """
    # A digest pin (name@sha256:...) is immutable regardless of tag.
    if "@" in image:
        return False
    # The path component(s) precede the repo; only the last segment carries tag.
    repo = image.rsplit("/", 1)[-1]
    if ":" not in repo:
        return True  # no tag -> resolves to :latest
    tag = repo.rsplit(":", 1)[1]
    return tag == "latest"


async def health_check(url: str, *, timeout: float = _QUICK_HTTP_TIMEOUT) -> bool:
    target = f"{url.rstrip('/')}/search"
    try:
        async with httpx.AsyncClient(
            follow_redirects=False, verify=build_ssl_context(), timeout=timeout
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
        # Reconcile engine config so a change to disabled_engines takes effect
        # without recreating the container. Gated here for zero overhead in the
        # common (unconfigured) case.
        if settings.disabled_engines:
            await _apply_disabled_engines(settings)
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
    if ok and settings.disabled_engines:
        await _apply_disabled_engines(settings)
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
                if _has_mutable_tag(settings.image):
                    logger.warning(
                        "Starting SearXNG from a mutable image tag (%s); pin a "
                        "digest or version tag via tools.web_search.searxng_image "
                        "to avoid supply-chain drift.",
                        settings.image,
                    )
                rc, _, err = await _run_cmd(
                    [
                        engine,
                        "run",
                        "-d",
                        "--name",
                        name,
                        "-p",
                        f"{_BIND_ADDRESS}:{settings.port}:{_INTERNAL_PORT}",
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
    await _ensure_json_format(engine, name)
    return None, True


_JSON_FORMAT_PRESENT = r"grep -q '^    - json' /etc/searxng/settings.yml"
_ADD_JSON_FORMAT = (
    r"sed -i 's/^    - html$/    - html\n    - json/' /etc/searxng/settings.yml"
)


async def _exec_in_container(
    engine: str, name: str, script: str, *, readiness: bool = False
) -> int | None:
    for _ in range(_PATCH_READINESS_TRIES if readiness else 1):
        try:
            rc, _, _ = await _run_cmd(
                [engine, "exec", name, "sh", "-c", script], timeout=_QUICK_CMD_TIMEOUT
            )
        except (FileNotFoundError, TimeoutError):
            return None
        # rc >= _DOCKER_ERROR_RC is a docker-level error (container still
        # starting); real command results (grep's 0/1) fall below it.
        if not readiness or rc < _DOCKER_ERROR_RC:
            return rc
        await asyncio.sleep(_PATCH_READINESS_DELAY)
    return None


# The upstream image ships `formats: [html]` only, which 403s every JSON
# request -- including our own health probe. Enable json on a container we just
# brought up so it can answer. Idempotent; settings reload on the restart.
async def _ensure_json_format(engine: str, name: str) -> None:
    rc = await _exec_in_container(engine, name, _JSON_FORMAT_PRESENT, readiness=True)
    if rc is None:
        logger.warning("SearXNG %s not ready; skipped json-format patch", name)
        return
    if rc == 0:
        return
    try:
        await _exec_in_container(engine, name, _ADD_JSON_FORMAT)
        # Confirm the sed actually matched before restarting: a settings.yml with
        # non-standard indentation leaves json absent, so a restart would neither
        # help nor change anything -- better to warn than spin a needless restart
        # and then fail health probing on an instance that still 403s JSON.
        verify = await _exec_in_container(engine, name, _JSON_FORMAT_PRESENT)
        if verify != 0:
            logger.warning(
                "SearXNG json-format patch did not take in %s; settings.yml may "
                "use non-standard indentation. Add 'json' under search.formats "
                "manually and restart the container.",
                name,
            )
            return
        await _run_cmd([engine, "restart", name], timeout=_START_CMD_TIMEOUT)
    except (FileNotFoundError, TimeoutError) as exc:
        logger.warning("Could not enable SearXNG json format in %s: %s", name, exc)


_YAML_TRUTHY = frozenset("true yes on y".split())


def disable_engines_in_settings(text: str, engines: list[str]) -> tuple[str, list[str]]:
    # Comment-preserving, surgical edit: SearXNG's settings.yml is large and
    # operator-authored; a full yaml.safe_dump round-trip would nuke comments and
    # reformat every line. Instead, compose the YAML to get line-accurate node
    # marks, then insert a single `disabled: true` line into each target engine
    # stanza that lacks one. Idempotent: engines already truthy-disabled are left
    # alone and not reported as changed.
    if not engines:
        return text, []
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return text, []
    if not isinstance(root, yaml.MappingNode):
        return text, []

    engines_node: yaml.Node | None = None
    for key_node, value_node in root.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == "engines":
            engines_node = value_node
            break
    if not isinstance(engines_node, yaml.SequenceNode):
        return text, []

    targets = set(engines)
    insert_at: list[int] = []  # 0-indexed line of each target's `- name:` line
    changed: list[str] = []
    for item in engines_node.value:
        if not isinstance(item, yaml.MappingNode):
            continue
        name: str | None = None
        already_disabled = False
        for k, v in item.value:
            if not isinstance(k, yaml.ScalarNode) or not isinstance(v, yaml.ScalarNode):
                continue
            if k.value == "name":
                name = v.value
            elif k.value == "disabled" and str(v.value).lower() in _YAML_TRUTHY:
                already_disabled = True
        if name in targets and name is not None and not already_disabled:
            insert_at.append(item.start_mark.line)
            changed.append(name)

    if not insert_at:
        return text, []

    lines = text.splitlines(keepends=True)
    for line_no in sorted(insert_at, reverse=True):
        dash_line = lines[line_no]
        leading = len(dash_line) - len(dash_line.lstrip(" "))
        indent = " " * (leading + 2)
        lines.insert(line_no + 1, f"{indent}disabled: true\n")
    return "".join(lines), changed


# Fragile upstream engines (rate-limit/CAPTCHA-prone) are the common reason a
# self-hosted SearXNG returns empty results. When `disabled_engines` is set, the
# managed container's settings.yml is patched to mark them disabled, then the
# container is restarted so the change takes effect. Idempotent: a container
# already in the desired state is left untouched (no restart). Best-effort: any
# failure logs a warning and returns so search still works with current engines.
async def _apply_disabled_engines(settings: SearxngSettings) -> None:
    if not settings.disabled_engines:
        return
    engine = detect_engine()
    if engine is None:
        logger.warning("Cannot apply SearXNG disabled engines: no docker/podman found.")
        return
    name = settings.container_name
    # mkdtemp (not TemporaryDirectory): the test scratchpad fixture patches
    # tempfile.mkdtemp on the module singleton with a 1-arg signature.
    tmpdir = tempfile.mkdtemp(prefix="vibe-searxng-")
    try:
        local = Path(tmpdir) / "settings.yml"
        rc, _, err = await _run_cmd(
            [engine, "cp", f"{name}:/etc/searxng/settings.yml", str(local)],
            timeout=_QUICK_CMD_TIMEOUT,
        )
        if rc != 0:
            logger.warning(
                "Could not fetch SearXNG settings from %s: %s", name, err.strip()
            )
            return
        text = read_safe(local).text
        new_text, changed = disable_engines_in_settings(
            text, list(settings.disabled_engines)
        )
        if not changed:
            return
        local.write_text(new_text, encoding="utf-8")
        rc, _, err = await _run_cmd(
            [engine, "cp", str(local), f"{name}:/etc/searxng/settings.yml"],
            timeout=_QUICK_CMD_TIMEOUT,
        )
        if rc != 0:
            logger.warning(
                "Could not write SearXNG settings to %s: %s", name, err.strip()
            )
            return
        rc, _, err = await _run_cmd(
            [engine, "restart", name], timeout=_START_CMD_TIMEOUT
        )
        if rc != 0:
            logger.warning(
                "Could not restart SearXNG %s after disabling engines: %s",
                name,
                err.strip(),
            )
            return
        logger.info("Disabled SearXNG engines in %s: %s", name, ", ".join(changed))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    # Bound the *total* wall-clock wait, including the time each health_check
    # call spends (up to _QUICK_HTTP_TIMEOUT). Counting only the sleep interval
    # lets a slow-answering container overrun the stated timeout by roughly the
    # check-time / sleep-time ratio.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_timeout
    while loop.time() < deadline:
        if await health_check(url):
            return True
        await asyncio.sleep(_HEALTH_POLL_INTERVAL)
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
