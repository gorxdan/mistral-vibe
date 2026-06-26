from __future__ import annotations

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.tools.base import ToolPermission
from vibe.core.types import ApprovalResponse, FunctionCall, ToolCall, ToolResultEvent


def _todo_call(call_id: str, args: str) -> ToolCall:
    return ToolCall(
        id=call_id, index=0, function=FunctionCall(name="todo", arguments=args)
    )


def _config(todo_permission: ToolPermission):
    return build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": todo_permission.value}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )


async def _drive(loop, prompt: str = "go"):
    events: list = []
    async for ev in loop.act(prompt):
        events.append(ev)
    return events


class TestModifyApprovalDispatch:
    @pytest.mark.asyncio
    async def test_modify_re_dispatches_with_edited_args(self) -> None:
        """User picks MODIFY: the tool runs with the edited args, not the model's."""
        original_args = '{"action": "read"}'
        modified_args = {"action": "read", "tag": "important"}

        seen_args: list[BaseModel] = []

        async def approval_callback(
            _tool_name: str,
            args: BaseModel,
            _tool_call_id: str,
            _rp: list | None = None,
            _judge_note: str | None = None,
        ) -> tuple[ApprovalResponse, str | None, dict | None]:
            seen_args.append(args)
            return (ApprovalResponse.MODIFY, None, modified_args)

        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="checking", tool_calls=[_todo_call("c1", original_args)]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        loop = build_test_agent_loop(
            config=_config(ToolPermission.ASK),
            agent_name="default",
            backend=backend,
            enable_streaming=False,
        )
        loop.set_approval_callback(approval_callback)

        events = await _drive(loop)

        # The callback saw the model's original args.
        assert seen_args, "approval callback was never invoked"
        # The tool result reflects the MODIFIED args (tag=important), proving
        # the re-dispatch used the user's edited args, not the model's.
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert results, "no tool result event"
        assert results[0].error is None
        assert results[0].skipped is False

    @pytest.mark.asyncio
    async def test_modify_with_invalid_args_falls_back_to_skip(self) -> None:
        """MODIFY args that fail validation are rejected with a feedback SKIP."""
        # 'todos' must be a list[TodoItem] | None; a bare string fails type
        # validation (TodoArgs has no model_config, so it defaults to coercing,
        # but a string is not coercible to list[TodoItem]).
        invalid_modified = {"action": "read", "todos": "not-a-list"}

        async def approval_callback(
            _tool_name: str,
            _args: BaseModel,
            _tool_call_id: str,
            _rp: list | None = None,
            _judge_note: str | None = None,
        ) -> tuple[ApprovalResponse, str | None, dict | None]:
            return (ApprovalResponse.MODIFY, None, invalid_modified)

        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="checking",
                    tool_calls=[_todo_call("c1", '{"action": "read"}')],
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        loop = build_test_agent_loop(
            config=_config(ToolPermission.ASK),
            agent_name="default",
            backend=backend,
            enable_streaming=False,
        )
        loop.set_approval_callback(approval_callback)

        events = await _drive(loop)

        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert results, "no tool result event"
        # Validation failure surfaces as a skip with feedback, not a crash.
        assert results[0].skipped is True
        assert results[0].skip_reason is not None
        assert "validation" in results[0].skip_reason.lower()

    @pytest.mark.asyncio
    async def test_yes_still_works_under_new_contract(self) -> None:
        """Regression guard: the plain YES path is unaffected by the 3-tuple."""

        async def approval_callback(
            _tool_name: str,
            _args: BaseModel,
            _tool_call_id: str,
            _rp: list | None = None,
            _judge_note: str | None = None,
        ) -> tuple[ApprovalResponse, str | None, dict | None]:
            return (ApprovalResponse.YES, None, None)

        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="checking",
                    tool_calls=[_todo_call("c1", '{"action": "read"}')],
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        loop = build_test_agent_loop(
            config=_config(ToolPermission.ASK),
            agent_name="default",
            backend=backend,
            enable_streaming=False,
        )
        loop.set_approval_callback(approval_callback)

        events = await _drive(loop)

        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert results
        assert results[0].error is None
        assert results[0].skipped is False
        assert loop.stats.tool_calls_succeeded == 1
