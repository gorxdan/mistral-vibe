"""codex (openai-chatgpt) prompt-cache routing: identity headers + the
`x-codex-turn-state` sticky token replay/reset state machine on AgentLoop.
"""

from __future__ import annotations

from tests.conftest import build_test_agent_loop
from vibe.core.config import ProviderConfig


def _codex_provider() -> ProviderConfig:
    return ProviderConfig(
        name="openai-chatgpt",
        api_base="https://chatgpt.com/backend-api/codex",
        api_style="openai-chatgpt",
    )


def _plain_provider() -> ProviderConfig:
    return ProviderConfig(name="kimi", api_base="https://api.kimi.com/coding/v1")


def test_codex_provider_gets_session_and_thread_id_headers() -> None:
    loop = build_test_agent_loop()
    headers = loop._get_extra_headers(_codex_provider())
    # Both routing headers carry the conversation's session id (== the body
    # prompt_cache_key) so the backend pins the conversation to one partition.
    assert headers["session-id"] == loop.session_id
    assert headers["thread-id"] == loop.session_id
    assert headers["x-affinity"] == loop.session_id


def test_non_codex_provider_gets_no_routing_headers() -> None:
    loop = build_test_agent_loop()
    headers = loop._get_extra_headers(_plain_provider())
    assert "session-id" not in headers
    assert "thread-id" not in headers


def test_turn_state_reset_on_user_prompt_and_replayed_on_secondary() -> None:
    loop = build_test_agent_loop()
    provider = _codex_provider()

    # A stale token from a previous turn must NOT leak into a new user turn.
    loop._codex_turn_state = "stale-token"
    loop._is_user_prompt_call = True
    headers, sink = loop._codex_routing(provider)
    assert "x-codex-turn-state" not in headers  # reset, not replayed
    assert loop._codex_turn_state is None
    assert sink == {}  # codex provider => capture sink present

    # The server hands back a token on the first request of the turn.
    loop._capture_codex_turn_state({"x-codex-turn-state": "turn-1"})
    assert loop._codex_turn_state == "turn-1"

    # Subsequent tool-loop calls in the same turn replay it to stay sticky.
    loop._is_user_prompt_call = False
    headers, sink = loop._codex_routing(provider)
    assert headers["x-codex-turn-state"] == "turn-1"
    assert sink == {}


def test_codex_routing_no_sink_for_non_codex_provider() -> None:
    loop = build_test_agent_loop()
    loop._codex_turn_state = "turn-1"
    headers, sink = loop._codex_routing(_plain_provider())
    # Non-codex providers never replay the codex token, but the sink is still
    # allocated so x-ratelimit-* headers can be captured for /status limits.
    assert sink is not None
    assert "x-codex-turn-state" not in headers


def test_capture_ignores_empty_or_missing_token() -> None:
    loop = build_test_agent_loop()
    loop._codex_turn_state = "keep"
    loop._capture_codex_turn_state(None)
    loop._capture_codex_turn_state({})
    loop._capture_codex_turn_state({"other": "x"})
    assert loop._codex_turn_state == "keep"
