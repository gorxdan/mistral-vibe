from __future__ import annotations

import asyncio
from pathlib import Path

from acp import CreateTerminalResponse
from acp.schema import EnvVariable, TerminalOutputResponse, WaitForTerminalExitResponse
import pytest

from tests.mock.utils import collect_result
from vibe.acp.tools.builtins.bash import AcpBashState, Bash
from vibe.acp.tools.events import ToolTerminalOpenedEvent
from vibe.core.tools.base import (
    InvokeContext,
    ToolAuthorizationSource,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.builtins.bash import (
    Bash as CoreBashTool,
    BashArgs,
    BashResult,
    BashToolConfig,
)
from vibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    PermissionStore,
    RequiredPermission,
    authorization_context_fingerprint,
)
from vibe.core.types import ToolResultEvent


class MockTerminalHandle:
    def __init__(
        self,
        terminal_id: str = "test_terminal_123",
        exit_code: int | None = 0,
        output: str = "test output",
        wait_delay: float = 0.01,
    ) -> None:
        self.id = terminal_id
        self._exit_code = exit_code
        self._output = output
        self._wait_delay = wait_delay
        self._killed = False

    async def wait_for_exit(self) -> WaitForTerminalExitResponse:
        await asyncio.sleep(self._wait_delay)
        return WaitForTerminalExitResponse(exit_code=self._exit_code)

    async def current_output(self) -> TerminalOutputResponse:
        return TerminalOutputResponse(output=self._output, truncated=False)

    async def kill(self) -> None:
        self._killed = True

    async def release(self) -> None:
        pass


class MockClient:
    def __init__(self, terminal_handle: MockTerminalHandle | None = None) -> None:
        self._terminal_handle = terminal_handle or MockTerminalHandle()
        self._create_terminal_called = False
        self._session_update_called = False
        self._create_terminal_error: Exception | None = None
        self._last_create_params: dict[
            str, str | list[str] | list[EnvVariable] | int | None
        ] = {}

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list | None = None,
        output_byte_limit: int | None = None,
        **kwargs,
    ) -> CreateTerminalResponse:
        self._create_terminal_called = True
        self._last_create_params = {
            "command": command,
            "session_id": session_id,
            "args": args,
            "cwd": cwd,
            "env": env,
            "output_byte_limit": output_byte_limit,
        }
        if self._create_terminal_error:
            raise self._create_terminal_error
        return CreateTerminalResponse(terminal_id=self._terminal_handle.id)

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs
    ) -> TerminalOutputResponse:
        return await self._terminal_handle.current_output()

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs
    ) -> WaitForTerminalExitResponse:
        return await self._terminal_handle.wait_for_exit()

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs
    ) -> None:
        await self._terminal_handle.release()

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs) -> None:
        await self._terminal_handle.kill()

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        self._session_update_called = True


@pytest.fixture
def mock_client() -> MockClient:
    return MockClient()


@pytest.fixture
def acp_bash_tool(mock_client: MockClient) -> Bash:
    config = BashToolConfig()
    # Use model_construct to bypass Pydantic validation for testing
    state = AcpBashState.model_construct(
        client=mock_client, session_id="test_session_123", cwd=str(Path.cwd())
    )
    return Bash(config_getter=lambda: config, state=state)


class TestAcpBashBasic:
    def test_get_name(self) -> None:
        assert Bash.get_name() == "bash"

    def test_get_summary_simple_command(self) -> None:
        args = BashArgs(command="ls")
        display = Bash.get_summary(args)
        assert display == "ls"

    def test_get_summary_with_timeout(self) -> None:
        args = BashArgs(command="ls", timeout=10)
        display = Bash.get_summary(args)
        assert display == "ls (timeout 10s)"


class TestAcpBashExecution:
    @pytest.mark.asyncio
    async def test_editor_terminal_uses_bound_workspace(
        self,
        acp_bash_tool: Bash,
        mock_client: MockClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace = tmp_path / "workspace"
        other = tmp_path / "other"
        workspace.mkdir()
        other.mkdir()
        acp_bash_tool.state.cwd = str(workspace)
        monkeypatch.chdir(other)

        await collect_result(acp_bash_tool.run(BashArgs(command="echo ok")))

        assert mock_client._last_create_params["cwd"] == str(workspace)

    @pytest.mark.asyncio
    async def test_automated_authorization_uses_core_execution(
        self, acp_bash_tool: Bash, mock_client: MockClient, monkeypatch
    ) -> None:
        calls: list[str] = []

        async def core_run(_self, args, _ctx=None):
            calls.append(args.command)
            yield BashResult(
                command=args.command, stdout="core", stderr="", returncode=0
            )

        monkeypatch.setattr(CoreBashTool, "run", core_run)
        ctx = InvokeContext(
            tool_call_id="automated",
            authorization_source=ToolAuthorizationSource.POLICY,
        )

        result = await collect_result(acp_bash_tool.run(BashArgs(command="cat x"), ctx))

        assert result.stdout == "core"
        assert calls == ["cat x"]
        assert mock_client._create_terminal_called is False

    @pytest.mark.asyncio
    async def test_automated_authorization_requires_bound_workspace(
        self, acp_bash_tool: Bash, mock_client: MockClient, monkeypatch
    ) -> None:
        async def fail_if_run(*_args, **_kwargs):
            raise AssertionError("core execution must not be reached")

        acp_bash_tool.state.cwd = None
        monkeypatch.setattr(CoreBashTool, "run", fail_if_run)
        ctx = InvokeContext(
            tool_call_id="missing-workspace",
            authorization_source=ToolAuthorizationSource.POLICY,
        )

        with pytest.raises(ToolError, match="workspace is unavailable"):
            await collect_result(acp_bash_tool.run(BashArgs(command="cat x"), ctx))

        assert mock_client._create_terminal_called is False

    @pytest.mark.asyncio
    async def test_automated_authorization_rejects_process_workspace_drift(
        self,
        acp_bash_tool: Bash,
        mock_client: MockClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fail_if_run(*_args, **_kwargs):
            raise AssertionError("core execution must not be reached")

        bound = tmp_path / "bound"
        drifted = tmp_path / "drifted"
        bound.mkdir()
        drifted.mkdir()
        acp_bash_tool.state.cwd = str(bound)
        monkeypatch.chdir(drifted)
        monkeypatch.setattr(CoreBashTool, "run", fail_if_run)
        ctx = InvokeContext(
            tool_call_id="drifted-workspace",
            authorization_source=ToolAuthorizationSource.POLICY,
        )

        with pytest.raises(ToolError, match="workspace changed"):
            await collect_result(acp_bash_tool.run(BashArgs(command="cat x"), ctx))

        assert mock_client._create_terminal_called is False

    @pytest.mark.asyncio
    async def test_user_authorization_drift_is_rejected_before_terminal(
        self, acp_bash_tool: Bash, mock_client: MockClient, monkeypatch
    ) -> None:
        permission = PermissionContext(
            permission=ToolPermission.NEVER, reason="disabled after approval"
        )
        monkeypatch.setattr(
            acp_bash_tool, "resolve_permission", lambda _args: permission
        )
        ctx = InvokeContext(
            tool_call_id="user-drift",
            authorization_source=ToolAuthorizationSource.USER,
            authorization_fingerprint=authorization_context_fingerprint(
                "bash", BashArgs(command="cat x"), permission
            ),
        )

        with pytest.raises(ToolError, match="disabled after approval"):
            await collect_result(acp_bash_tool.run(BashArgs(command="cat x"), ctx))

        assert mock_client._create_terminal_called is False

    @pytest.mark.asyncio
    async def test_stored_authorization_drift_is_rejected_before_terminal(
        self, acp_bash_tool: Bash, mock_client: MockClient, monkeypatch
    ) -> None:
        required = RequiredPermission(
            scope=PermissionScope.COMMAND_PATTERN,
            invocation_pattern="cat x",
            session_pattern="cat *",
            label="cat *",
        )
        monkeypatch.setattr(
            acp_bash_tool,
            "resolve_permission",
            lambda _args: PermissionContext(
                permission=ToolPermission.ASK, required_permissions=[required]
            ),
        )
        ctx = InvokeContext(
            tool_call_id="stored-drift",
            permission_store=PermissionStore(),
            authorization_source=ToolAuthorizationSource.STORED_USER,
            authorization_fingerprint=authorization_context_fingerprint(
                "bash",
                BashArgs(command="cat x"),
                PermissionContext(
                    permission=ToolPermission.ASK, required_permissions=[required]
                ),
            ),
        )

        with pytest.raises(ToolError, match="no longer covers"):
            await collect_result(acp_bash_tool.run(BashArgs(command="cat x"), ctx))

        assert mock_client._create_terminal_called is False

    @pytest.mark.asyncio
    async def test_direct_hard_denial_is_rejected_before_terminal(
        self, acp_bash_tool: Bash, mock_client: MockClient, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            acp_bash_tool,
            "resolve_permission",
            lambda _args: PermissionContext(
                permission=ToolPermission.NEVER, reason="command denied"
            ),
        )

        with pytest.raises(ToolError, match="command denied"):
            await collect_result(acp_bash_tool.run(BashArgs(command="cat x")))

        assert mock_client._create_terminal_called is False

    @pytest.mark.asyncio
    async def test_run_success(
        self, acp_bash_tool: Bash, mock_client: MockClient
    ) -> None:
        from pathlib import Path

        args = BashArgs(command="echo hello")
        result = await collect_result(acp_bash_tool.run(args))

        assert isinstance(result, BashResult)
        assert result.stdout == "test output"
        assert result.stderr == ""
        assert result.returncode == 0
        assert mock_client._create_terminal_called

        # Verify create_terminal was called correctly
        params = mock_client._last_create_params
        assert params["session_id"] == "test_session_123"
        assert params["command"] == "echo hello"
        assert params["cwd"] == str(Path.cwd())  # effective_workdir defaults to cwd

    @pytest.mark.asyncio
    async def test_run_creates_terminal_with_env_vars(
        self, mock_client: MockClient
    ) -> None:
        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session", cwd=str(Path.cwd())
            ),
        )

        args = BashArgs(command="NODE_ENV=test npm run build")
        await collect_result(tool.run(args))

        params = mock_client._last_create_params
        assert params["command"] == "NODE_ENV=test npm run build"

    @pytest.mark.asyncio
    async def test_run_with_nonzero_exit_code(self, mock_client: MockClient) -> None:
        custom_handle = MockTerminalHandle(
            terminal_id="custom_terminal", exit_code=1, output="error: command failed"
        )
        mock_client._terminal_handle = custom_handle

        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session", cwd=str(Path.cwd())
            ),
        )

        args = BashArgs(command="test_command")
        with pytest.raises(ToolError) as exc_info:
            await collect_result(tool.run(args))

        assert (
            str(exc_info.value)
            == "Command failed: 'test_command'\nReturn code: 1\nStdout: error: command failed"
        )

    @pytest.mark.asyncio
    async def test_run_create_terminal_failure(self, mock_client: MockClient) -> None:
        mock_client._create_terminal_error = RuntimeError("Connection failed")

        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="test")
        with pytest.raises(ToolError) as exc_info:
            await collect_result(tool.run(args))

        assert (
            str(exc_info.value)
            == "Failed to create terminal: RuntimeError('Connection failed')"
        )

    @pytest.mark.asyncio
    async def test_run_without_client(self) -> None:
        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(client=None, session_id="test_session"),
        )

        args = BashArgs(command="test")
        with pytest.raises(ToolError) as exc_info:
            await collect_result(tool.run(args))

        assert (
            str(exc_info.value)
            == "Client not available in tool state. This tool can only be used within an ACP session."
        )

    @pytest.mark.asyncio
    async def test_run_without_session_id(self) -> None:
        mock_client = MockClient()
        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(client=mock_client, session_id=None),
        )

        args = BashArgs(command="test")
        with pytest.raises(ToolError) as exc_info:
            await collect_result(tool.run(args))

        assert (
            str(exc_info.value)
            == "Session ID not available in tool state. This tool can only be used within an ACP session."
        )

    @pytest.mark.asyncio
    async def test_run_with_none_exit_code(self, mock_client: MockClient) -> None:
        custom_handle = MockTerminalHandle(
            terminal_id="none_exit_terminal", exit_code=None, output="output"
        )
        mock_client._terminal_handle = custom_handle

        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="test_command")
        result = await collect_result(tool.run(args))

        assert result.returncode == 0
        assert result.stdout == "output"


class TestAcpBashTimeout:
    @pytest.mark.asyncio
    async def test_run_with_timeout_raises_error_and_kills(
        self, mock_client: MockClient
    ) -> None:
        custom_handle = MockTerminalHandle(
            terminal_id="timeout_terminal",
            output="partial output",
            wait_delay=20,  # Longer than the 1 second timeout
        )
        mock_client._terminal_handle = custom_handle

        # Use a config with different default timeout to verify args timeout overrides it
        tool = Bash(
            config_getter=lambda: BashToolConfig(default_timeout=30),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="slow_command", timeout=1)
        with pytest.raises(ToolError) as exc_info:
            await collect_result(tool.run(args))

        assert str(exc_info.value) == "Command timed out after 1s: 'slow_command'"
        assert custom_handle._killed

    @pytest.mark.asyncio
    async def test_run_timeout_bounded_when_kill_hangs(
        self, mock_client: MockClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "vibe.acp.tools.builtins.bash._TERMINAL_CLEANUP_TIMEOUT", 0.05
        )

        custom_handle = MockTerminalHandle(
            terminal_id="hanging_kill_terminal", wait_delay=20
        )
        mock_client._terminal_handle = custom_handle

        async def hanging_kill() -> None:
            await asyncio.sleep(30)

        custom_handle.kill = hanging_kill

        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="slow_command", timeout=1)
        with pytest.raises(ToolError) as exc_info:
            await asyncio.wait_for(collect_result(tool.run(args)), timeout=5)

        assert str(exc_info.value) == "Command timed out after 1s: 'slow_command'"

    @pytest.mark.asyncio
    async def test_run_timeout_handles_kill_failure(
        self, mock_client: MockClient
    ) -> None:
        custom_handle = MockTerminalHandle(
            terminal_id="kill_failure_terminal",
            wait_delay=20,  # Longer than the 1 second timeout
        )
        mock_client._terminal_handle = custom_handle

        async def failing_kill() -> None:
            raise RuntimeError("Kill failed")

        custom_handle.kill = failing_kill

        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="slow_command", timeout=1)
        # Should still raise timeout error even if kill fails
        with pytest.raises(ToolError) as exc_info:
            await collect_result(tool.run(args))

        assert str(exc_info.value) == "Command timed out after 1s: 'slow_command'"


class TestAcpBashTerminalOpenedEvent:
    @pytest.mark.asyncio
    async def test_run_yields_terminal_opened_event(
        self, mock_client: MockClient
    ) -> None:
        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session", cwd=str(Path.cwd())
            ),
        )

        args = BashArgs(command="test")
        events: list[ToolTerminalOpenedEvent] = []
        async for item in tool.run(args, InvokeContext(tool_call_id="test_call")):
            if isinstance(item, ToolTerminalOpenedEvent):
                events.append(item)

        assert len(events) == 1
        assert events[0].terminal_id == mock_client._terminal_handle.id
        assert events[0].tool_call_id == "test_call"
        assert events[0].tool_name == "bash"


class TestAcpBashConcurrentInvocations:
    @pytest.mark.asyncio
    async def test_concurrent_invocations_yield_distinct_tool_call_ids(self) -> None:
        mock_client = MockClient(MockTerminalHandle(terminal_id="t", wait_delay=0.05))
        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session", cwd=str(Path.cwd())
            ),
        )

        async def run_and_collect_ids(tool_call_id: str) -> list[str]:
            ids: list[str] = []
            async for item in tool.run(
                BashArgs(command="echo hi"), InvokeContext(tool_call_id=tool_call_id)
            ):
                if isinstance(item, ToolTerminalOpenedEvent):
                    ids.append(item.tool_call_id)
            return ids

        results = await asyncio.gather(
            run_and_collect_ids("T1"), run_and_collect_ids("T2")
        )

        assert results[0] == ["T1"]
        assert results[1] == ["T2"]


class TestAcpBashConfig:
    @pytest.mark.asyncio
    async def test_run_uses_config_default_timeout(
        self, mock_client: MockClient
    ) -> None:
        custom_handle = MockTerminalHandle(
            terminal_id="config_timeout_terminal",
            wait_delay=0.01,  # Shorter than config timeout
        )
        mock_client._terminal_handle = custom_handle

        tool = Bash(
            config_getter=lambda: BashToolConfig(default_timeout=30),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="fast", timeout=None)
        result = await collect_result(tool.run(args))

        # Should succeed with config timeout
        assert result.returncode == 0


class TestAcpBashCleanup:
    @pytest.mark.asyncio
    async def test_run_releases_terminal_on_success(
        self, mock_client: MockClient
    ) -> None:
        custom_handle = MockTerminalHandle(terminal_id="cleanup_terminal")
        mock_client._terminal_handle = custom_handle

        release_called = False

        async def mock_release() -> None:
            nonlocal release_called
            release_called = True

        custom_handle.release = mock_release

        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="test")
        await collect_result(tool.run(args))

        assert release_called

    @pytest.mark.asyncio
    async def test_run_skips_release_after_timeout_kill(
        self, mock_client: MockClient
    ) -> None:
        # On timeout _wait_for_terminal_exit force-kills the terminal. A
        # subsequent release_terminal would target an already-destroyed
        # terminal and fail on the real backend, so release must be skipped
        # once kill has finalized the terminal.
        custom_handle = MockTerminalHandle(
            terminal_id="timeout_cleanup_terminal",
            wait_delay=2.0,  # Longer than the 1 second timeout
        )
        mock_client._terminal_handle = custom_handle

        release_called = False

        async def mock_release() -> None:
            nonlocal release_called
            release_called = True

        custom_handle.release = mock_release

        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="slow", timeout=1)
        try:
            await collect_result(tool.run(args))
        except ToolError:
            pass

        assert custom_handle._killed
        assert not release_called

    @pytest.mark.asyncio
    async def test_run_handles_release_failure(self, mock_client: MockClient) -> None:
        custom_handle = MockTerminalHandle(terminal_id="release_failure_terminal")

        async def failing_release() -> None:
            raise RuntimeError("Release failed")

        custom_handle.release = failing_release
        mock_client._terminal_handle = custom_handle

        tool = Bash(
            config_getter=lambda: BashToolConfig(),
            state=AcpBashState.model_construct(
                client=mock_client, session_id="test_session"
            ),
        )

        args = BashArgs(command="test")
        # Should not raise, release failure is silently ignored
        result = await collect_result(tool.run(args))

        assert result is not None
        assert result.stdout == "test output"


class TestAcpBashToolResultSessionUpdate:
    def test_success_reports_completed(self) -> None:
        event = ToolResultEvent(
            tool_name="bash",
            tool_call_id="call_1",
            tool_class=Bash,
            result=BashResult(command="echo ok", stdout="ok", stderr="", returncode=0),
        )

        update = Bash.tool_result_session_update(event)

        assert update is not None
        assert update.status == "completed"

    def test_user_rejection_reports_failed(self) -> None:
        event = ToolResultEvent(
            tool_name="bash",
            tool_call_id="call_1",
            tool_class=Bash,
            skipped=True,
            skip_reason="User rejected the tool call, provide an alternative plan",
        )

        update = Bash.tool_result_session_update(event)

        assert update is not None
        assert update.status == "failed"

    def test_error_reports_failed(self) -> None:
        event = ToolResultEvent(
            tool_name="bash", tool_call_id="call_1", tool_class=Bash, error="boom"
        )

        update = Bash.tool_result_session_update(event)

        assert update is not None
        assert update.status == "failed"
