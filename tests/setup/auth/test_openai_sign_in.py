from __future__ import annotations

import asyncio
import base64
import json
import socket
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx

from vibe.core.auth import openai_oauth as oauth
from vibe.setup.auth.openai_sign_in import OpenAISignInError, OpenAISignInService


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _b64url(data: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")


def _fake_jwt(account_id: str = "acct_123") -> str:
    return (
        f"{_b64url({'alg': 'none'})}."
        f"{_b64url({'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}."
        "sig"
    )


async def _send_callback(port: int, *, code: str, state: str | None) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    query = f"code={code}" + (f"&state={state}" if state is not None else "")
    writer.write(
        f"GET /auth/callback?{query} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
    )
    await writer.drain()
    await reader.read(1024)
    writer.close()


@pytest.mark.asyncio
@respx.mock
async def test_authenticate_success_persists_tokens() -> None:
    port = _free_port()
    route = respx.post(oauth.OPENAI_TOKEN_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "access-tok",
                "refresh_token": "refresh-tok",
                "id_token": _fake_jwt("acct_abc"),
                "expires_in": 3600,
            },
        )
    )

    def driver(url: str) -> bool:
        state = parse_qs(urlparse(url).query)["state"][0]
        asyncio.get_running_loop().create_task(
            _send_callback(port, code="auth-code", state=state)
        )
        return True

    service = OpenAISignInService(open_browser=driver, port=port)
    tokens = await service.authenticate()

    assert route.called
    assert tokens.access_token == "access-tok"
    assert tokens.account_id == "acct_abc"
    # Persisted to the token store.
    loaded = oauth.load_tokens()
    assert loaded is not None
    assert loaded.refresh_token == "refresh-tok"


@pytest.mark.asyncio
@respx.mock
async def test_authenticate_sends_pkce_verifier_form_encoded() -> None:
    port = _free_port()
    captured: dict[str, str] = {}

    def driver(url: str) -> bool:
        captured["authorize_url"] = url
        state = parse_qs(urlparse(url).query)["state"][0]
        asyncio.get_running_loop().create_task(
            _send_callback(port, code="c", state=state)
        )
        return True

    route = respx.post(oauth.OPENAI_TOKEN_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "a",
                "refresh_token": "r",
                "id_token": _fake_jwt(),
                "expires_in": 3600,
            },
        )
    )

    await OpenAISignInService(open_browser=driver, port=port).authenticate()

    body = parse_qs(route.calls.last.request.content.decode())
    assert body["grant_type"] == ["authorization_code"]
    assert body["code"] == ["c"]
    assert body["code_verifier"]  # present
    assert body["client_id"] == [oauth.OPENAI_OAUTH_CLIENT_ID]
    # Challenge in the authorize URL is the S256 hash of the sent verifier.
    challenge = parse_qs(urlparse(captured["authorize_url"]).query)["code_challenge"][0]
    expected = (
        base64
        .urlsafe_b64encode(
            __import__("hashlib").sha256(body["code_verifier"][0].encode()).digest()
        )
        .decode()
        .rstrip("=")
    )
    assert challenge == expected


@pytest.mark.asyncio
@respx.mock
async def test_authenticate_state_mismatch_aborts() -> None:
    port = _free_port()
    respx.post(oauth.OPENAI_TOKEN_URL).mock(return_value=httpx.Response(200, json={}))

    def driver(url: str) -> bool:
        asyncio.get_running_loop().create_task(
            _send_callback(port, code="c", state="WRONG-STATE")
        )
        return True

    with pytest.raises(OpenAISignInError, match="state mismatch"):
        await OpenAISignInService(open_browser=driver, port=port).authenticate()


@pytest.mark.asyncio
async def test_authenticate_port_in_use_raises() -> None:
    port = _free_port()
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        with pytest.raises(OpenAISignInError, match="already in use"):
            await OpenAISignInService(
                open_browser=lambda _u: True, port=port
            ).authenticate()
    finally:
        blocker.close()


def test_build_authorize_url_params() -> None:
    url = oauth.build_authorize_url(code_challenge="chal", state="st")
    params = parse_qs(urlparse(url).query)
    assert params["response_type"] == ["code"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["redirect_uri"] == [oauth.OPENAI_OAUTH_REDIRECT_URI]
    assert params["client_id"] == [oauth.OPENAI_OAUTH_CLIENT_ID]


def test_generate_pkce_pair_is_s256() -> None:
    import hashlib

    verifier, challenge = oauth.generate_pkce_pair()
    expected = (
        base64
        .urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    assert challenge == expected
