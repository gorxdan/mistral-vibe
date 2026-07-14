from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path
import sys
from typing import Any

import keyring
from keyring.backend import KeyringBackend
import keyring.errors
import pytest
import tomli_w

from tests.cli.plan_offer.adapters.fake_whoami_gateway import FakeWhoAmIGateway
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry
from tests.stubs.fake_voice_manager import FakeVoiceManager
from tests.update_notifier.adapters.fake_update_cache_repository import (
    FakeUpdateCacheRepository,
)
from tests.update_notifier.adapters.fake_update_gateway import FakeUpdateGateway
from vibe.cli.plan_offer.ports.whoami_gateway import WhoAmIPlanType, WhoAmIResponse
from vibe.cli.textual_ui.app import CORE_VERSION, StartupOptions, VibeApp
from vibe.core.agent_loop import AgentLoop, AgentLoopParams
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import (
    DEFAULT_MODELS,
    ModelConfig,
    SessionLoggingConfig,
    SpendConfig,
    VibeConfig,
)
from vibe.core.config.harness_files import (
    HarnessFilesManager,
    init_harness_files_manager,
    reset_harness_files_manager,
)
from vibe.core.llm.types import BackendLike
from vibe.core.utils import keyring as keyring_utils


def sandbox_e2e_available() -> bool:
    # detect_backend only returns 'bwrap' when the capability probe passes, so an
    # unshare/none result means user namespaces are unavailable (skip sandbox e2e).
    from vibe.core.tools.sandbox import detect_backend

    return detect_backend("auto") in {"bwrap", "sandbox-exec"}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "sandbox_e2e: needs a usable OS sandbox backend; skipped when user "
        "namespaces are unavailable",
    )
    config.addinivalue_line(
        "markers",
        "process_e2e: exercises real process-tree teardown or descendant "
        "lifecycle; opt-in only on a disposable non-graphical host with xdist "
        "disabled",
    )
    if not config.getoption("--run-process-e2e"):
        return
    if os.environ.get("VIBE_PROCESS_E2E_DISPOSABLE") != "1":
        raise pytest.UsageError(
            "--run-process-e2e requires VIBE_PROCESS_E2E_DISPOSABLE=1 on a "
            "disposable non-graphical host"
        )
    if config.getoption("numprocesses") not in {None, 0, "0"}:
        raise pytest.UsageError("--run-process-e2e requires -n0")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("vibe safety").addoption(
        "--run-process-e2e",
        action="store_true",
        default=False,
        help=(
            "run real process-tree teardown or descendant-lifecycle probes "
            "(requires -n0 and a disposable non-graphical host)"
        ),
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("process_e2e") and not item.config.getoption(
        "--run-process-e2e"
    ):
        pytest.skip("real process-tree teardown probes require --run-process-e2e")
    if item.get_closest_marker("sandbox_e2e") and not sandbox_e2e_available():
        pytest.skip("no usable OS sandbox backend (user namespaces unavailable)")


class _EmptyKeyring(KeyringBackend):
    """A keyring backend that stores nothing, used to keep tests off the real OS keyring."""

    priority = 1  # type: ignore[assignment]

    def get_password(self, service: str, username: str) -> str | None:
        return None

    def set_password(self, service: str, username: str, password: str) -> None:
        return None

    def delete_password(self, service: str, username: str) -> None:
        raise keyring.errors.PasswordDeleteError()


@pytest.fixture(autouse=True, scope="session")
def _isolate_prod_file_log() -> Generator[None, None, None]:
    """Keep test log records out of the production log file.

    ``vibe.core.logger`` attaches a ``RotatingFileHandler`` to the ``vibe``
    logger at import time. Without this, pytest workers (separate processes)
    sink their WARNING/ERROR records — including stub-induced failures such as
    'backend down' and 'consolidator build failed' — into
    ``~/.vibe/logs/vibe.log``, where they masquerade as live-session failures.
    Detach file handlers for the test session and restore them on teardown.
    """
    import logging as _logging

    from vibe.core.logger import logger as vibe_logger

    saved = [h for h in vibe_logger.handlers if isinstance(h, _logging.FileHandler)]
    for handler in saved:
        vibe_logger.removeHandler(handler)
    try:
        yield
    finally:
        for handler in saved:
            vibe_logger.addHandler(handler)


@pytest.fixture(autouse=True)
def _disable_os_keyring(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Keep the suite off the real OS keyring.

    ``resolve_api_key`` and ``VibeConfig._check_api_key`` now consult the keyring, so
    without this every config construction would touch the real Keychain. We install an
    empty backend (rather than patching ``keyring.get_password``) so tests that swap in
    their own backend via ``keyring.set_keyring`` still work. Tests that exercise keyring
    behaviour opt in by patching ``keyring.get_password`` / ``set_password`` directly.
    """
    original = keyring.get_keyring()
    monkeypatch.setattr(keyring_utils, "_should_use_macos_security", lambda: False)
    keyring.set_keyring(_EmptyKeyring())
    keyring_utils.clear_api_key_keyring_cache()
    try:
        yield
    finally:
        keyring_utils.clear_api_key_keyring_cache()
        keyring.set_keyring(original)


def get_base_config() -> dict[str, Any]:
    return {
        "active_model": "devstral-latest",
        "providers": [
            {
                "name": "mistral",
                "api_base": "https://api.mistral.ai/v1",
                "api_key_env_var": "MISTRAL_API_KEY",
                "browser_auth_base_url": "https://console.mistral.ai",
                "browser_auth_api_base_url": "https://console.mistral.ai/api",
                "backend": "mistral",
            }
        ],
        "models": [
            {
                "name": "mistral-vibe-cli-latest",
                "provider": "mistral",
                "alias": "devstral-latest",
            }
        ],
        "enable_telemetry": False,
    }


@pytest.fixture(autouse=True)
def tmp_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    tmp_working_directory = tmp_path_factory.mktemp("test_cwd")
    monkeypatch.chdir(tmp_working_directory)
    return tmp_working_directory


@pytest.fixture(autouse=True)
def config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    tmp_path = tmp_path_factory.mktemp("vibe")
    config_dir = tmp_path / ".vibe"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(tomli_w.dumps(get_base_config()), encoding="utf-8")

    monkeypatch.setattr("vibe.core.paths._vibe_home._DEFAULT_VIBE_HOME", config_dir)
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("vibe.core.paths._agents_home._DEFAULT_AGENTS_HOME", agents_dir)

    # Re-evaluate PLAN agent overrides so the allowlist uses the monkeypatched path
    from vibe.core.agents.models import PLAN, _plan_overrides

    object.__setattr__(PLAN, "overrides", _plan_overrides())

    return config_dir


@pytest.fixture(autouse=True)
def _reset_trusted_folders_manager(config_dir: Path) -> None:
    """Prevent the singleton from writing to the real ~/.vibe/trusted_folders.toml.

    The module-level ``trusted_folders_manager`` captures its file path at import
    time (before any monkeypatch), so it would otherwise target the real home
    directory.  Redirect it to the temp config dir used by the ``config_dir``
    fixture.
    """
    from vibe.core.trusted_folders import trusted_folders_manager

    trusted_folders_manager._file_path = config_dir / "trusted_folders.toml"
    trusted_folders_manager._trusted = []
    trusted_folders_manager._untrusted = []
    trusted_folders_manager._session_trusted = []


@pytest.fixture(autouse=True)
def _init_harness_files_manager():
    reset_harness_files_manager()
    init_harness_files_manager("user", "project")
    yield
    reset_harness_files_manager()


@pytest.fixture(autouse=True)
def _scratchpad_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Generator[Path]:
    import vibe.core.scratchpad as scratchpad_mod

    scratchpad_mod._active_scratchpads.clear()

    scratchpad_root = tmp_path_factory.mktemp("scratchpad")
    _counter = 0

    def _fake_mkdtemp(prefix: str = "") -> str:
        nonlocal _counter
        _counter += 1
        d = scratchpad_root / f"{prefix}{_counter}"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr("vibe.core.scratchpad.tempfile.mkdtemp", _fake_mkdtemp)

    yield scratchpad_root

    scratchpad_mod._active_scratchpads.clear()


@pytest.fixture(autouse=True)
def _no_live_model_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local-model discovery offline by default.

    Auto-detection probes a local ollama endpoint on every model-picker open;
    without this, tests would non-deterministically pick up models from a
    server that happens to be running on the dev machine. Tests that exercise
    discovery override this by monkeypatching fetch_model_ids themselves.
    """

    async def _no_models(*_args: object, **_kwargs: object) -> list[str]:
        return []

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _no_models)


@pytest.fixture(autouse=True)
def _mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "mock")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock")
    monkeypatch.setenv("OPENAI_API_KEY", "mock")
    monkeypatch.setenv("VERTEX_API_KEY", "mock")
    monkeypatch.setenv("REASONING_API_KEY", "mock")


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock platform to be Linux with /bin/sh shell for consistent test behavior.

    This ensures that platform-specific system prompt generation is consistent
    across all tests regardless of the actual platform running the tests.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SHELL", "/bin/sh")


@pytest.fixture(autouse=True)
def _mock_update_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vibe.cli.update_notifier.update.UPDATE_COMMANDS", ["true"])


@pytest.fixture(autouse=True)
def _disable_feedback_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vibe.core.feedback.FEEDBACK_PROBABILITY", 0)


@pytest.fixture(autouse=True)
def _disable_input_grace_periods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vibe.cli.textual_ui.widgets.approval_app._INPUT_GRACE_PERIOD_S", 0
    )
    monkeypatch.setattr(
        "vibe.cli.textual_ui.widgets.question_app._INPUT_GRACE_PERIOD_S", 0
    )
    monkeypatch.setattr("vibe.cli.textual_ui.app._DEFAULT_TYPING_DEBOUNCE_MS", 0)
    monkeypatch.delenv("VIBE_TYPING_GRACE_PERIOD_MS", raising=False)


@pytest.fixture(autouse=True)
def telemetry_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def record_telemetry(
        self: Any,
        event_name: str,
        properties: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        merged = self.build_client_event_metadata() | properties
        event: dict[str, Any] = {"event_name": event_name, "properties": merged}
        if correlation_id is not None:
            event["correlation_id"] = correlation_id
        events.append(event)

    monkeypatch.setattr(
        "vibe.core.telemetry.send.TelemetryClient.send_telemetry_event",
        record_telemetry,
    )
    return events


@pytest.fixture
def mock_prompts_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    project = tmp_path / "project" / ".vibe" / "prompts"
    user = tmp_path / "home" / ".vibe" / "prompts"
    project.mkdir(parents=True)
    user.mkdir(parents=True)

    class _MockManager(HarnessFilesManager):
        @property
        def project_prompts_dirs(self) -> list[Path]:
            return [project]

        @property
        def user_prompts_dirs(self) -> list[Path]:
            return [user]

    monkeypatch.setattr(
        "vibe.core.prompts.get_harness_files_manager",
        lambda: _MockManager(sources=("user",)),
    )
    return project, user


@pytest.fixture
def vibe_app() -> VibeApp:
    return build_test_vibe_app()


@pytest.fixture
def agent_loop() -> AgentLoop:
    return build_test_agent_loop()


@pytest.fixture
def vibe_config() -> VibeConfig:
    return build_test_vibe_config()


def make_test_models(auto_compact_threshold: int) -> list[ModelConfig]:
    return [
        m.model_copy(update={"auto_compact_threshold": auto_compact_threshold})
        for m in DEFAULT_MODELS
    ]


def build_test_vibe_config(**kwargs) -> VibeConfig:
    session_logging = kwargs.pop("session_logging", None)
    resolved_session_logging = (
        SessionLoggingConfig(enabled=False)
        if session_logging is None
        else session_logging
    )
    enable_update_checks = kwargs.pop("enable_update_checks", None)
    resolved_enable_update_checks = (
        False if enable_update_checks is None else enable_update_checks
    )
    if kwargs.get("models"):
        kwargs.setdefault("active_model", kwargs["models"][0].alias)
    # Connectors trigger a real HTTP discovery on agent construction; off by
    # default so tests don't pay for it. Connector tests pass enable_connectors=True.
    kwargs.setdefault("enable_connectors", False)
    # Use the lightweight test system prompt unless a test asks for a real one.
    kwargs.setdefault("system_prompt_id", "tests")
    # Keep the test prompt minimal: skip project-context discovery and prompt
    # detail unless a test opts in.
    kwargs.setdefault("include_project_context", False)
    kwargs.setdefault("include_prompt_detail", False)
    # Preserve the historical enforcing default: existing tests rely on spend
    # limits blocking calls. Real sessions default to advisory-only tracking.
    kwargs.setdefault("spend", SpendConfig(enforce_limits=True))
    return VibeConfig(
        session_logging=resolved_session_logging,
        enable_update_checks=resolved_enable_update_checks,
        **kwargs,
    )


def build_test_agent_loop(
    *,
    config: VibeConfig | None = None,
    agent_name: str = BuiltinAgentName.DEFAULT,
    backend: BackendLike | None = None,
    enable_streaming: bool = False,
    **kwargs,
) -> AgentLoop:

    resolved_config = config or build_test_vibe_config()

    return AgentLoop(
        resolved_config,
        backend=backend or FakeBackend(),
        params=AgentLoopParams(
            agent_name=agent_name,
            enable_streaming=enable_streaming,
            mcp_registry=kwargs.pop("mcp_registry", FakeMCPRegistry()),
            **kwargs,
        ),
    )


def build_test_vibe_app(
    *, config: VibeConfig | None = None, agent_loop: AgentLoop | None = None, **kwargs
) -> VibeApp:
    app_config = config or build_test_vibe_config()

    resolved_agent_loop = agent_loop or build_test_agent_loop(config=app_config)

    update_notifier = kwargs.pop("update_notifier", None)
    resolved_update_notifier = (
        FakeUpdateGateway() if update_notifier is None else update_notifier
    )
    update_cache_repository = kwargs.pop("update_cache_repository", None)
    resolved_update_cache_repository = (
        FakeUpdateCacheRepository()
        if update_cache_repository is None
        else update_cache_repository
    )
    plan_offer_gateway = kwargs.pop("plan_offer_gateway", None)
    resolved_plan_offer_gateway = (
        FakeWhoAmIGateway(
            WhoAmIResponse(
                plan_type=WhoAmIPlanType.CHAT,
                plan_name="INDIVIDUAL",
                prompt_switching_to_pro_plan=False,
            )
        )
        if plan_offer_gateway is None
        else plan_offer_gateway
    )
    current_version = kwargs.pop("current_version", None)
    resolved_current_version = (
        CORE_VERSION if current_version is None else current_version
    )
    voice_manager = kwargs.pop("voice_manager", FakeVoiceManager())

    return VibeApp(
        agent_loop=resolved_agent_loop,
        startup=StartupOptions(initial_prompt=kwargs.pop("initial_prompt", None)),
        current_version=resolved_current_version,
        update_notifier=resolved_update_notifier,
        update_cache_repository=resolved_update_cache_repository,
        plan_offer_gateway=resolved_plan_offer_gateway,
        voice_manager=voice_manager,
        **kwargs,
    )
