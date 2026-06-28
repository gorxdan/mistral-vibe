from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import (
    BUILTIN_AGENTS,
    AgentType,
    profile_requires_isolation,
)
from vibe.core.telemetry.types import TerminalEmulator
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError, ToolPermission
from vibe.core.tools.builtins.task import Task, TaskArgs, TaskResult, TaskToolConfig
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.safety_judge import JudgeVerdict
from vibe.core.types import AssistantEvent, LLMMessage, Role


@pytest.fixture
def task_tool() -> Task:
    return Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())


class TestTaskConcurrencyGating:
    """call_is_read_only decides whether a task call fans out concurrently
    (read-only in-process subagent) or serializes with the writers.
    """

    @pytest.fixture
    def manager(self) -> AgentManager:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        return AgentManager(lambda: config)

    def test_read_only_profile_is_concurrent_safe(self, manager: AgentManager) -> None:
        args = TaskArgs(task="x", agent="explore")
        assert Task.call_is_read_only(args, agent_manager=manager) is True

    def test_write_capable_profile_serializes(self, manager: AgentManager) -> None:
        writer = next(
            (p.name for p in manager.get_subagents() if profile_requires_isolation(p)),
            None,
        )
        assert writer is not None, "expected a write-capable builtin subagent"
        args = TaskArgs(task="x", agent=writer)
        assert Task.call_is_read_only(args, agent_manager=manager) is False

    def test_async_run_serializes(self, manager: AgentManager) -> None:
        args = TaskArgs(task="x", agent="explore", async_run=True)
        assert Task.call_is_read_only(args, agent_manager=manager) is False

    def test_no_agent_manager_serializes(self) -> None:
        args = TaskArgs(task="x", agent="explore")
        assert Task.call_is_read_only(args, agent_manager=None) is False

    def test_task_is_marked_subagent_spawner(self) -> None:
        assert Task.is_subagent_spawner is True

    def test_base_default_mirrors_static_read_only_flag(self) -> None:
        from vibe.core.tools.builtins.edit import Edit
        from vibe.core.tools.builtins.grep import Grep

        # Default impl ignores args and mirrors the class read_only flag.
        assert Grep.call_is_read_only(MagicMock()) is True
        assert Edit.call_is_read_only(MagicMock()) is False
        assert Grep.is_subagent_spawner is False


class TestTaskArgs:
    def test_default_agent_is_explore(self) -> None:
        args = TaskArgs(task="do something")
        assert args.agent == "explore"

    def test_custom_values(self) -> None:
        args = TaskArgs(task="do something", agent="explore")
        assert args.task == "do something"
        assert args.agent == "explore"


class TestTaskToolValidation:
    @pytest.fixture
    def ctx(self) -> InvokeContext:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        manager = AgentManager(lambda: config)
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            terminal_emulator=TerminalEmulator.VSCODE,
        )

    @pytest.mark.asyncio
    async def test_rejects_primary_agent(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        args = TaskArgs(task="do something", agent="default")

        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))

        assert "agent" in str(exc_info.value).lower()
        assert "subagent" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_agent(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        args = TaskArgs(task="do something", agent="nonexistent")

        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))

        assert "Unknown agent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_requires_agent_manager_in_context(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent="explore")
        ctx = InvokeContext(tool_call_id="test-call-id")  # No agent_manager

        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))

        assert "agent_manager" in str(exc_info.value).lower()

    def test_explore_agent_is_valid_subagent(self) -> None:
        agent = BUILTIN_AGENTS["explore"]
        assert agent.agent_type == AgentType.SUBAGENT


class TestTaskToolModelRouting:
    @pytest.fixture
    def ctx(self) -> InvokeContext:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        manager = AgentManager(lambda: config)
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            terminal_emulator=TerminalEmulator.VSCODE,
        )

    def test_model_defaults_to_none(self) -> None:
        assert TaskArgs(task="do something").model is None

    @pytest.mark.asyncio
    async def test_unknown_model_alias_rejected(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        args = TaskArgs(task="review", agent="explore", model="not-a-real-model")
        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))
        msg = str(exc_info.value)
        assert "Unknown model alias 'not-a-real-model'" in msg
        # The error lists the configured aliases so the host can self-correct.
        assert ctx.agent_manager.config.active_model in msg

    @pytest.mark.asyncio
    async def test_valid_model_threaded_into_in_process_loop(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        valid_alias = ctx.agent_manager.config.active_model

        async def mock_act(task: str):
            yield AssistantEvent(content="ok")

        with (
            patch("vibe.core.tools.builtins.task.AgentLoop") as mock_loop_class,
            patch(
                "vibe.core.tools.builtins.task.VibeConfig.load",
                return_value=ctx.agent_manager.config,
            ) as mock_load,
        ):
            mock_loop = MagicMock()
            mock_loop.act = mock_act
            mock_loop.messages = []
            mock_loop.set_approval_callback = MagicMock()
            mock_loop_class.return_value = mock_loop

            args = TaskArgs(task="review", agent="explore", model=valid_alias)
            await collect_result(task_tool.run(args, ctx))

            assert mock_load.call_args.kwargs.get("active_model") == valid_alias

    @pytest.mark.asyncio
    async def test_omitted_model_inherits_parent_in_process_loop(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        # When the caller omits a model, the subagent must inherit the parent's
        # resolved active_model. Otherwise a fresh VibeConfig.load() falls back to
        # the hardcoded mistral default and dies with "Missing MISTRAL_API_KEY".
        async def mock_act(task: str):
            yield AssistantEvent(content="ok")

        with (
            patch("vibe.core.tools.builtins.task.AgentLoop") as mock_loop_class,
            patch(
                "vibe.core.tools.builtins.task.VibeConfig.load",
                return_value=ctx.agent_manager.config,
            ) as mock_load,
        ):
            mock_loop = MagicMock()
            mock_loop.act = mock_act
            mock_loop.messages = []
            mock_loop.set_approval_callback = MagicMock()
            mock_loop_class.return_value = mock_loop

            args = TaskArgs(task="review", agent="explore")
            await collect_result(task_tool.run(args, ctx))

            assert (
                mock_load.call_args.kwargs.get("active_model")
                == ctx.agent_manager.config.active_model
            )

    @pytest.mark.asyncio
    async def test_omitted_model_prefers_parent_effective_model(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        # ctx.active_model carries the parent's effective (running) model, incl. a
        # runtime failover override; it must win over the configured active_model.
        ctx.active_model = "parent-running-model"

        async def mock_act(task: str):
            yield AssistantEvent(content="ok")

        with (
            patch("vibe.core.tools.builtins.task.AgentLoop") as mock_loop_class,
            patch(
                "vibe.core.tools.builtins.task.VibeConfig.load",
                return_value=ctx.agent_manager.config,
            ) as mock_load,
        ):
            mock_loop = MagicMock()
            mock_loop.act = mock_act
            mock_loop.messages = []
            mock_loop.set_approval_callback = MagicMock()
            mock_loop_class.return_value = mock_loop

            args = TaskArgs(task="review", agent="explore")
            await collect_result(task_tool.run(args, ctx))

            assert (
                mock_load.call_args.kwargs.get("active_model") == "parent-running-model"
            )

    @pytest.mark.asyncio
    async def test_valid_model_threaded_into_isolated_spawn(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        valid_alias = ctx.agent_manager.config.active_model

        class _FakeIsolatedResult:
            output = "done"
            worktree_path = None
            branch = None

        async def fake_run(*a, **kw):
            return _FakeIsolatedResult()

        args = TaskArgs(task="review", agent="worker", model=valid_alias)
        with (
            patch(
                "vibe.core.tools.builtins.task.profile_requires_isolation",
                return_value=True,
            ),
            patch(
                "vibe.core.tools.builtins.task.run_isolated_agent", side_effect=fake_run
            ) as mock_run,
        ):
            await collect_result(task_tool.run(args, ctx))

        assert mock_run.call_args.kwargs.get("model") == valid_alias

    @pytest.mark.asyncio
    async def test_omitted_model_inherits_parent_isolated_spawn(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        class _FakeIsolatedResult:
            output = "done"
            worktree_path = None
            branch = None

        async def fake_run(*a, **kw):
            return _FakeIsolatedResult()

        args = TaskArgs(task="review", agent="worker")
        with (
            patch(
                "vibe.core.tools.builtins.task.profile_requires_isolation",
                return_value=True,
            ),
            patch(
                "vibe.core.tools.builtins.task.run_isolated_agent", side_effect=fake_run
            ) as mock_run,
        ):
            await collect_result(task_tool.run(args, ctx))

        assert (
            mock_run.call_args.kwargs.get("model")
            == ctx.agent_manager.config.active_model
        )


class TestTaskToolResolvePermission:
    def test_explore_allowed_by_default(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent="explore")
        result = task_tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_unknown_agent_returns_none(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent="custom_agent")
        result = task_tool.resolve_permission(args)
        assert result is None

    def test_denylist_takes_precedence(self) -> None:
        config = TaskToolConfig(allowlist=["explore"], denylist=["explore"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent="explore")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_glob_pattern_in_allowlist(self) -> None:
        config = TaskToolConfig(allowlist=["exp*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent="explore")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_glob_pattern_in_denylist(self) -> None:
        config = TaskToolConfig(denylist=["danger*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent="dangerous_agent")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_empty_lists_returns_none(self) -> None:
        config = TaskToolConfig(allowlist=[], denylist=[])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent="explore")
        result = tool.resolve_permission(args)
        assert result is None

    def test_default_config_has_explore_in_allowlist(self) -> None:
        config = TaskToolConfig()
        assert "explore" in config.allowlist


class TestTaskToolExecution:
    @pytest.fixture
    def ctx(self) -> InvokeContext:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        manager = AgentManager(lambda: config)
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            terminal_emulator=TerminalEmulator.VSCODE,
        )

    @pytest.mark.asyncio
    async def test_happy_path_returns_subagent_response(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        """Test that task tool successfully runs a subagent and returns its response."""
        mock_messages = [
            LLMMessage(role=Role.system, content="system"),
            LLMMessage(role=Role.user, content="task"),
            LLMMessage(role=Role.assistant, content="response 1"),
            LLMMessage(role=Role.assistant, content="response 2"),
        ]

        async def mock_act(task: str):
            yield AssistantEvent(content="Hello from subagent!")
            yield AssistantEvent(content=" More content.")

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = mock_messages
            mock_agent_loop.set_approval_callback = MagicMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            args = TaskArgs(task="explore the codebase", agent="explore")
            result = await collect_result(task_tool.run(args, ctx))

            assert isinstance(result, TaskResult)
            assert result.response == "Hello from subagent! More content."
            assert result.turns_used == 2  # 2 assistant messages in mock_messages
            assert result.completed is True
            assert (
                mock_agent_loop_class.call_args.kwargs["terminal_emulator"]
                is TerminalEmulator.VSCODE
            )

    @pytest.mark.asyncio
    async def test_handles_stopped_by_middleware(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        """Test that task tool reports incomplete when stopped by middleware."""
        mock_messages = [
            LLMMessage(role=Role.system, content="system"),
            LLMMessage(role=Role.assistant, content="partial"),
        ]

        async def mock_act(task: str):
            yield AssistantEvent(content="Partial response", stopped_by_middleware=True)

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = mock_messages
            mock_agent_loop.set_approval_callback = MagicMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            args = TaskArgs(task="do something", agent="explore")
            result = await collect_result(task_tool.run(args, ctx))

            assert isinstance(result, TaskResult)
            assert result.completed is False

    @pytest.mark.asyncio
    async def test_handles_subagent_exception(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        """Test that task tool gracefully handles exceptions from subagent."""
        mock_messages = [LLMMessage(role=Role.system, content="system")]

        async def mock_act(task: str):
            yield AssistantEvent(content="Starting...")
            raise RuntimeError("Simulated error")

        with patch("vibe.core.tools.builtins.task.AgentLoop") as mock_agent_loop_class:
            mock_agent_loop = MagicMock()
            mock_agent_loop.act = mock_act
            mock_agent_loop.messages = mock_messages
            mock_agent_loop.set_approval_callback = MagicMock()
            mock_agent_loop_class.return_value = mock_agent_loop

            args = TaskArgs(task="do something", agent="explore")
            result = await collect_result(task_tool.run(args, ctx))

            assert isinstance(result, TaskResult)
            assert result.completed is False
            assert "Simulated error" in result.response


class TestIsolatedSpawnJudgeGate:
    @pytest.fixture
    def ctx(self) -> InvokeContext:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        manager = AgentManager(lambda: config)
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            terminal_emulator=TerminalEmulator.VSCODE,
        )

    @staticmethod
    def _ctx_with_factory(
        ctx: InvokeContext, factory, approval_callback=None
    ) -> InvokeContext:
        from dataclasses import replace

        return replace(
            ctx, safety_judge_factory=factory, approval_callback=approval_callback
        )

    @pytest.mark.asyncio
    async def test_no_judge_factory_proceeds(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        # No safety_judge_factory -> fail open (proceed), so the gate returns None.
        gate = task_tool._judge_isolated_spawn("prompt", "worker", ctx)
        assert await gate is None

    @pytest.mark.asyncio
    async def test_judge_safe_proceeds(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        from vibe.core.tools.safety_judge import JudgeVerdict

        class _SafeJudge:
            async def judge(self, kind, prompt, context):
                return JudgeVerdict(safe=True, reason="ok")

        ctx = self._ctx_with_factory(ctx, lambda: _SafeJudge())
        assert await task_tool._judge_isolated_spawn("prompt", "worker", ctx) is None

    @pytest.mark.asyncio
    async def test_judge_deny_without_callback_blocks(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        from vibe.core.tools.safety_judge import JudgeVerdict

        class _DenyJudge:
            async def judge(self, kind, prompt, context):
                return JudgeVerdict(safe=False, reason="rm -rf detected")

        cctx = self._ctx_with_factory(ctx, lambda: _DenyJudge(), approval_callback=None)
        reason = await task_tool._judge_isolated_spawn("prompt", "worker", cctx)
        assert reason == "rm -rf detected"

    @pytest.mark.asyncio
    async def test_judge_deny_then_user_approves_proceeds(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        from vibe.core.tools.safety_judge import JudgeVerdict
        from vibe.core.types import ApprovalResponse

        class _DenyJudge:
            async def judge(self, kind, prompt, context):
                return JudgeVerdict(safe=False, reason="looks risky")

        async def approve(*args, **kwargs):
            return ApprovalResponse.YES, None, None

        cctx = self._ctx_with_factory(
            ctx, lambda: _DenyJudge(), approval_callback=approve
        )
        assert await task_tool._judge_isolated_spawn("prompt", "worker", cctx) is None

    @pytest.mark.asyncio
    async def test_factory_raising_fails_open(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        def boom():
            raise RuntimeError("judge down")

        cctx = self._ctx_with_factory(ctx, boom)
        # Factory raised -> treat as no judge -> proceed.
        assert await task_tool._judge_isolated_spawn("prompt", "worker", cctx) is None

    @pytest.mark.asyncio
    async def test_denied_isolated_spawn_does_not_spawn_subprocess(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        # End-to-end: when the judge denies, run_isolated_agent is never called
        # and the TaskResult reports the denial in-band (completed=False).
        from vibe.core.tools.safety_judge import JudgeVerdict

        class _DenyJudge:
            async def judge(self, kind, prompt, context):
                return JudgeVerdict(safe=False, reason="destructive task")

        cctx = self._ctx_with_factory(ctx, lambda: _DenyJudge(), approval_callback=None)
        args = TaskArgs(task="rm -rf everything", agent="worker")

        with (
            patch("vibe.core.tools.builtins.task.run_isolated_agent") as mock_run,
            patch(
                "vibe.core.tools.builtins.task.profile_requires_isolation",
                return_value=True,
            ),
        ):
            result = await collect_result(task_tool.run(args, cctx))
            assert mock_run.call_count == 0
            assert isinstance(result, TaskResult)
            assert result.completed is False
            assert "destructive task" in result.response


class TestAsyncRun:
    """async_run=true — non-blocking isolated delegation."""

    @pytest.fixture
    def ctx(self) -> InvokeContext:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        manager = AgentManager(lambda: config)
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            terminal_emulator=TerminalEmulator.VSCODE,
        )

    @pytest.mark.asyncio
    async def test_async_run_rejected_for_read_only_profile(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        # explore is read-only and in-process; async_run is not meaningful.
        args = TaskArgs(task="do something", agent="explore", async_run=True)
        with pytest.raises(ToolError) as exc:
            await collect_result(task_tool.run(args, ctx))
        msg = str(exc.value)
        assert "async_run=True requires an isolated" in msg
        assert "launch_workflow" in msg

    @pytest.mark.asyncio
    async def test_async_run_returns_immediately_with_task_id(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        from vibe.core.tools.background import BackgroundRegistry

        registry = BackgroundRegistry()
        ctx_with_registry = replace(ctx, background_registry=registry)

        async def fake_run():
            return None

        args = TaskArgs(task="do work", agent="worker", async_run=True)
        with (
            patch(
                "vibe.core.tools.builtins.task.profile_requires_isolation",
                return_value=True,
            ),
            patch(
                "vibe.core.tools.builtins.task.run_isolated_agent", side_effect=fake_run
            ) as mock_run,
        ):
            result = await collect_result(task_tool.run(args, ctx_with_registry))

        assert isinstance(result, TaskResult)
        assert result.task_id is not None
        assert result.task_id.startswith("asub-")
        assert result.completed is False
        assert result.isolated is True
        # The isolated runner was invoked (the asyncio.Task was scheduled).
        await asyncio.sleep(0.01)
        assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_async_run_without_registry_falls_back_to_blocking(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        # ctx has no background_registry wired — the tool must fall back to the
        # synchronous isolated path rather than failing hard.
        args = TaskArgs(task="do work", agent="worker", async_run=True)

        class _FakeIsolatedResult:
            output = "isolated result"
            returncode = 0
            worktree_path = None
            branch = None

        async def fake_run(*a, **kw):
            return _FakeIsolatedResult()

        with (
            patch(
                "vibe.core.tools.builtins.task.profile_requires_isolation",
                return_value=True,
            ),
            patch(
                "vibe.core.tools.builtins.task.run_isolated_agent", side_effect=fake_run
            ),
        ):
            result = await collect_result(task_tool.run(args, ctx))

        assert isinstance(result, TaskResult)
        # Blocking path returned the isolated result, not a task_id.
        assert result.task_id is None
        assert "isolated result" in result.response

    @pytest.mark.asyncio
    async def test_async_run_judge_denial_returns_incomplete(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        from vibe.core.tools.background import BackgroundRegistry

        registry = BackgroundRegistry()

        class _DenyJudge:
            async def judge(self, kind, prompt, context):
                return JudgeVerdict(safe=False, reason="too destructive")

        ctx_with_judge = replace(
            ctx, background_registry=registry, safety_judge_factory=lambda: _DenyJudge()
        )
        args = TaskArgs(task="rm -rf everything", agent="worker", async_run=True)
        with (
            patch(
                "vibe.core.tools.builtins.task.profile_requires_isolation",
                return_value=True,
            ),
            patch("vibe.core.tools.builtins.task.run_isolated_agent") as mock_run,
        ):
            result = await collect_result(task_tool.run(args, ctx_with_judge))

        assert isinstance(result, TaskResult)
        assert result.completed is False
        assert "too destructive" in result.response
        assert result.task_id is None
        assert mock_run.call_count == 0
