from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
import json
import stat

import httpx
import pytest
import respx

from vibe.core.auth import openai_oauth as oauth


def _b64url(data: dict[str, object]) -> str:
    raw = json.dumps(data).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _fake_jwt(account_id: str = "acct_123") -> str:
    header = _b64url({"alg": "none", "typ": "JWT"})
    payload = _b64url({
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
        "exp": 9999999999,
    })
    return f"{header}.{payload}.sig"


def _tokens(
    *, expires_in: int = 3600, account_id: str = "acct_123"
) -> oauth.OpenAIOAuthTokens:
    return oauth.OpenAIOAuthTokens(
        access_token="access-1",
        refresh_token="refresh-1",
        account_id=account_id,
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
        id_token=_fake_jwt(account_id),
    )


def test_account_id_from_token() -> None:
    assert oauth.account_id_from_token(_fake_jwt("acct_xyz")) == "acct_xyz"


def test_account_id_from_token_missing_claim() -> None:
    bad = f"{_b64url({'alg': 'none'})}.{_b64url({'sub': 'u'})}.sig"
    assert oauth.account_id_from_token(bad) is None


def test_decode_jwt_malformed_raises() -> None:
    with pytest.raises(oauth.OpenAIOAuthError):
        oauth.decode_jwt_claims("not-a-jwt")


def test_save_and_load_roundtrip() -> None:
    tokens = _tokens()
    oauth.save_tokens(tokens)

    loaded = oauth.load_tokens()
    assert loaded is not None
    assert loaded.access_token == tokens.access_token
    assert loaded.refresh_token == tokens.refresh_token
    assert loaded.account_id == tokens.account_id


def test_saved_token_store_is_private() -> None:
    oauth.save_tokens(_tokens())
    mode = stat.S_IMODE(oauth.token_store_path().stat().st_mode)
    assert mode == 0o600


def test_load_tokens_absent_returns_none() -> None:
    assert oauth.load_tokens() is None


def test_needs_refresh_window() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    fresh = oauth.OpenAIOAuthTokens(
        access_token="a",
        refresh_token="r",
        account_id="acct",
        expires_at=now + timedelta(minutes=30),
    )
    stale = oauth.OpenAIOAuthTokens(
        access_token="a",
        refresh_token="r",
        account_id="acct",
        expires_at=now + timedelta(minutes=2),  # inside the 5m margin
    )
    assert fresh.needs_refresh(now=now) is False
    assert stale.needs_refresh(now=now) is True


@pytest.mark.asyncio
async def test_resolve_without_store_raises() -> None:
    with pytest.raises(oauth.OpenAINotAuthenticatedError):
        await oauth.resolve_chatgpt_credentials()


@pytest.mark.asyncio
async def test_resolve_fresh_token_skips_network() -> None:
    oauth.save_tokens(_tokens(expires_in=3600))
    # No respx mock installed: any network call would raise, proving we skip it.
    creds = await oauth.resolve_chatgpt_credentials()
    assert creds.access_token == "access-1"
    assert creds.account_id == "acct_123"


@pytest.mark.asyncio
@respx.mock
async def test_resolve_refreshes_when_expired() -> None:
    oauth.save_tokens(_tokens(expires_in=10))  # inside refresh margin
    new_jwt = _fake_jwt("acct_123")
    route = respx.post(oauth.OPENAI_TOKEN_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "access-2",
                "refresh_token": "refresh-2",
                "id_token": new_jwt,
                "expires_in": 3600,
            },
        )
    )

    creds = await oauth.resolve_chatgpt_credentials()

    assert route.called
    assert creds.access_token == "access-2"
    # Refreshed tokens are persisted.
    reloaded = oauth.load_tokens()
    assert reloaded is not None
    assert reloaded.access_token == "access-2"
    assert reloaded.refresh_token == "refresh-2"


@pytest.mark.asyncio
@respx.mock
async def test_refresh_preserves_account_id_when_no_new_id_token() -> None:
    oauth.save_tokens(_tokens(expires_in=10, account_id="acct_keep"))
    respx.post(oauth.OPENAI_TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "access-2", "expires_in": 3600}
        )
    )

    creds = await oauth.resolve_chatgpt_credentials()
    assert creds.account_id == "acct_keep"
    reloaded = oauth.load_tokens()
    assert reloaded is not None
    assert reloaded.refresh_token == "refresh-1"  # old refresh token reused


@pytest.mark.asyncio
@respx.mock
async def test_refresh_http_error_raises() -> None:
    oauth.save_tokens(_tokens(expires_in=10))
    respx.post(oauth.OPENAI_TOKEN_URL).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"})
    )
    with pytest.raises(oauth.OpenAIRefreshError):
        await oauth.resolve_chatgpt_credentials()


def test_auth_headers_present() -> None:
    headers = oauth.ChatGPTCredentials("tok", "acct_123").auth_headers()
    assert headers["ChatGPT-Account-ID"] == "acct_123"
    assert headers["originator"] == oauth.OPENAI_ORIGINATOR
    assert headers["version"] == oauth.OPENAI_CODEX_VERSION
