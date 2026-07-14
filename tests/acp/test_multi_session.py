from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from acp import PROTOCOL_VERSION, RequestError
from acp.schema import TextContentBlock
import pytest
from pytest import raises

from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.acp.acp_agent_loop import VibeAcpAgentLoop
from vibe.acp.exceptions import InvalidRequestError
from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.types import Role


class TestMultiSessionCore:
    @pytest.mark.asyncio
    async def test_process_rejects_a_second_workspace(
        self, acp_agent_loop: VibeAcpAgentLoop, tmp_path: Path
    ) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()

        await acp_agent_loop.new_session(cwd=str(first), mcp_servers=[])

        with pytest.raises(InvalidRequestError, match="different workspace"):
            await acp_agent_loop.new_session(cwd=str(second), mcp_servers=[])

        assert Path.cwd().resolve() == first.resolve()

    @pytest.mark.asyncio
    async def test_process_rejects_different_additional_roots(
        self, acp_agent_loop: VibeAcpAgentLoop, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "workspace"
        first = tmp_path / "first"
        second = tmp_path / "second"
        workspace.mkdir()
        first.mkdir()
        second.mkdir()

        await acp_agent_loop.new_session(
            cwd=str(workspace), additional_directories=[str(first)], mcp_servers=[]
        )

        with pytest.raises(InvalidRequestError, match="different workspace"):
            await acp_agent_loop.new_session(
                cwd=str(workspace), additional_directories=[str(second)], mcp_servers=[]
            )

        assert get_harness_files_manager().additional_dirs == (first.resolve(),)

    @pytest.mark.asyncio
    async def test_additional_root_order_does_not_change_workspace_identity(
        self, acp_agent_loop: VibeAcpAgentLoop, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "workspace"
        first = tmp_path / "first"
        second = tmp_path / "second"
        workspace.mkdir()
        first.mkdir()
        second.mkdir()

        await acp_agent_loop.new_session(
            cwd=str(workspace),
            additional_directories=[str(second), str(first)],
            mcp_servers=[],
        )
        await acp_agent_loop.new_session(
            cwd=str(workspace),
            additional_directories=[str(first), str(second)],
            mcp_servers=[],
        )

        assert get_harness_files_manager().additional_dirs == (
            first.resolve(),
            second.resolve(),
        )

    @pytest.mark.asyncio
    async def test_prompt_fails_closed_after_process_cwd_drift(
        self,
        acp_agent_loop: VibeAcpAgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace = tmp_path / "workspace"
        other = tmp_path / "other"
        workspace.mkdir()
        other.mkdir()
        response = await acp_agent_loop.new_session(cwd=str(workspace), mcp_servers=[])
        monkeypatch.chdir(other)

        with pytest.raises(InvalidRequestError, match="working directory drifted"):
            await acp_agent_loop.prompt(
                session_id=response.session_id,
                prompt=[TextContentBlock(type="text", text="Hello")],
            )

    @pytest.mark.asyncio
    async def test_different_sessions_use_different_agents(
        self, acp_agent_loop: VibeAcpAgentLoop
    ) -> None:
        await acp_agent_loop.initialize(protocol_version=PROTOCOL_VERSION)
        session1_response = await acp_agent_loop.new_session(
            cwd=str(Path.cwd()), mcp_servers=[]
        )
        session1 = acp_agent_loop.sessions[session1_response.session_id]
        session2_response = await acp_agent_loop.new_session(
            cwd=str(Path.cwd()), mcp_servers=[]
        )
        session2 = acp_agent_loop.sessions[session2_response.session_id]

        assert session1.id != session2.id
        # Each agent loop should be independent
        assert session1.agent_loop is not session2.agent_loop
        assert id(session1.agent_loop) != id(session2.agent_loop)

    @pytest.mark.asyncio
    async def test_error_on_nonexistent_session(
        self, acp_agent_loop: VibeAcpAgentLoop
    ) -> None:
        await acp_agent_loop.initialize(protocol_version=PROTOCOL_VERSION)
        await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])

        fake_session_id = "fake-session-id-" + str(uuid4())

        with raises(RequestError) as exc_info:
            await acp_agent_loop.prompt(
                session_id=fake_session_id,
                prompt=[TextContentBlock(type="text", text="Hello, world!")],
            )

        assert isinstance(exc_info.value, RequestError)
        assert exc_info.value.code == -32602
        assert "Session not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_simultaneous_message_processing(
        self, acp_agent_loop: VibeAcpAgentLoop, backend: FakeBackend
    ) -> None:
        await acp_agent_loop.initialize(protocol_version=PROTOCOL_VERSION)
        session1_response = await acp_agent_loop.new_session(
            cwd=str(Path.cwd()), mcp_servers=[]
        )
        session1 = acp_agent_loop.sessions[session1_response.session_id]
        session2_response = await acp_agent_loop.new_session(
            cwd=str(Path.cwd()), mcp_servers=[]
        )
        session2 = acp_agent_loop.sessions[session2_response.session_id]

        backend._streams = [
            [mock_llm_chunk(content="Response 1")],
            [mock_llm_chunk(content="Response 2")],
        ]

        async def run_session1():
            await acp_agent_loop.prompt(
                session_id=session1.id,
                prompt=[TextContentBlock(type="text", text="Prompt for session 1")],
            )

        async def run_session2():
            await acp_agent_loop.prompt(
                session_id=session2.id,
                prompt=[TextContentBlock(type="text", text="Prompt for session 2")],
            )

        await asyncio.gather(run_session1(), run_session2())

        user_message1 = next(
            (msg for msg in session1.agent_loop.messages if msg.role == Role.USER), None
        )
        assert user_message1 is not None
        assert user_message1.content == "Prompt for session 1"
        user_message2 = next(
            (msg for msg in session2.agent_loop.messages if msg.role == Role.USER), None
        )
        assert user_message2 is not None
        assert user_message2.content == "Prompt for session 2"

        # Backend stream order is non-deterministic under asyncio.gather, so
        # assert that both sessions received distinct responses from the
        # expected set rather than pinning a specific assignment.
        assistant_message1 = next(
            (msg for msg in session1.agent_loop.messages if msg.role == Role.ASSISTANT),
            None,
        )
        assistant_message2 = next(
            (msg for msg in session2.agent_loop.messages if msg.role == Role.ASSISTANT),
            None,
        )
        assert assistant_message1 is not None
        assert assistant_message2 is not None
        assert {assistant_message1.content, assistant_message2.content} == {
            "Response 1",
            "Response 2",
        }
