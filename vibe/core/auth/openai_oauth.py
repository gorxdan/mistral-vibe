"""ChatGPT-subscription OAuth credentials for the OpenAI provider.

This implements the "Sign in with ChatGPT" credential model used by OpenAI's
Codex CLI so a ChatGPT Plus/Pro/Team subscription can drive the harness without
per-token API billing. The flow and request shapes are reverse-engineered from
the public ``openai/codex`` source; the ChatGPT backend it targets
(``chatgpt.com/backend-api/codex``) is undocumented and may change without
notice. Keep this isolated behind the opt-in OpenAI-ChatGPT provider.

This module owns only the *credential* side: the on-disk token store, JWT
account-id extraction, refresh, and the per-request resolver consumed by the
LLM backend. The interactive browser login lives in
``vibe.setup.auth.openai_sign_in`` and writes the token store this module reads.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import os
from pathlib import Path
import secrets
from typing import Any, Final
import urllib.parse

import httpx
import orjson

from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME
from vibe.core.utils.http import build_ssl_context

# --- Codex OAuth client constants (from openai/codex codex-rs/login) ---------
# The public, first-party Codex client id. Reusing it is what makes a ChatGPT
# subscription usable here; it is also why this path is OpenAI-ToS-grey.
OPENAI_OAUTH_CLIENT_ID: Final = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_OAUTH_ISSUER: Final = "https://auth.openai.com"
OPENAI_AUTHORIZE_URL: Final = f"{OPENAI_OAUTH_ISSUER}/oauth/authorize"
OPENAI_TOKEN_URL: Final = f"{OPENAI_OAUTH_ISSUER}/oauth/token"
# Codex registered this exact redirect (host + port + path) with the client id,
# so it must be reproduced verbatim; the loopback listener binds 127.0.0.1.
OPENAI_OAUTH_REDIRECT_PORT: Final = 1455
OPENAI_OAUTH_REDIRECT_URI: Final = (
    f"http://localhost:{OPENAI_OAUTH_REDIRECT_PORT}/auth/callback"
)
OPENAI_OAUTH_SCOPES: Final = "openid profile email offline_access"
# The JWT claim (in the id_token / access_token) that carries the account id.
_AUTH_CLAIM: Final = "https://api.openai.com/auth"
_ACCOUNT_ID_KEY: Final = "chatgpt_account_id"

# Refresh slightly ahead of expiry, matching codex's safety margin.
_REFRESH_MARGIN: Final = timedelta(minutes=5)

# api_style that selects the ChatGPT backend + this credential resolver.
OPENAI_CHATGPT_API_STYLE: Final = "openai-chatgpt"
# Default base url for the ChatGPT (subscription) backend.
OPENAI_CHATGPT_API_BASE: Final = "https://chatgpt.com/backend-api/codex"

# Identity headers the ChatGPT backend expects. originator must be a known Codex
# originator; the backend may reject unknown values, so we present as codex_cli_rs.
# Overridable for forward-compat when OpenAI bumps the accepted version.
OPENAI_ORIGINATOR: Final = os.getenv("CHATON_CODEX_ORIGINATOR", "codex_cli_rs")
OPENAI_CODEX_VERSION: Final = os.getenv("CHATON_CODEX_VERSION", "0.142.0")


class OpenAIOAuthError(RuntimeError):
    """Base error for the ChatGPT-subscription credential flow."""


class OpenAINotAuthenticatedError(OpenAIOAuthError):
    def __init__(self) -> None:
        super().__init__(
            "Not signed in to ChatGPT. Run `chaton --setup` and pick "
            "'Sign in with ChatGPT', or remove the openai-chatgpt provider."
        )


class OpenAIRefreshError(OpenAIOAuthError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            f"Could not refresh ChatGPT credentials: {reason}. "
            "Run `chaton --setup` and sign in with ChatGPT again."
        )


def token_store_path() -> Path:
    """Location of the ChatGPT OAuth token store (``$VIBE_HOME/auth/openai.json``)."""
    return VIBE_HOME.path / "auth" / "openai.json"


@dataclass(frozen=True, slots=True)
class OpenAIOAuthTokens:
    access_token: str
    refresh_token: str
    account_id: str
    expires_at: datetime
    id_token: str = ""

    def needs_refresh(self, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        return now >= self.expires_at - _REFRESH_MARGIN

    def to_json(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "account_id": self.account_id,
            "expires_at": self.expires_at.astimezone(UTC).isoformat(),
            "id_token": self.id_token,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OpenAIOAuthTokens:
        try:
            expires_at = datetime.fromisoformat(data["expires_at"])
        except (KeyError, ValueError) as exc:
            raise OpenAIOAuthError("Token store is missing a valid expiry.") from exc
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        try:
            return cls(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                account_id=data["account_id"],
                expires_at=expires_at,
                id_token=data.get("id_token", ""),
            )
        except KeyError as exc:
            raise OpenAIOAuthError(f"Token store is missing field {exc}.") from exc


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT payload WITHOUT signature verification.

    The token is already trusted (it came straight from OpenAI's token endpoint
    over TLS); we only need to read its claims, mirroring codex which also does
    not verify the signature locally.
    """
    _JWT_MIN_PARTS = 2  # JWT = header.payload.signature
    parts = token.split(".")
    if len(parts) < _JWT_MIN_PARTS:
        raise OpenAIOAuthError("Malformed JWT: expected at least two segments.")
    try:
        payload = orjson.loads(_b64url_decode(parts[1]))
    except (ValueError, orjson.JSONDecodeError) as exc:
        raise OpenAIOAuthError("Could not decode JWT payload.") from exc
    if not isinstance(payload, dict):
        raise OpenAIOAuthError("JWT payload is not a JSON object.")
    return payload


def account_id_from_token(token: str) -> str | None:
    """Extract ``chatgpt_account_id`` from an id_token's auth claim."""
    claims = decode_jwt_claims(token)
    auth = claims.get(_AUTH_CLAIM)
    if isinstance(auth, dict):
        account_id = auth.get(_ACCOUNT_ID_KEY)
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def load_tokens() -> OpenAIOAuthTokens | None:
    path = token_store_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OpenAIOAuthError(f"Could not read token store: {exc}") from exc
    return OpenAIOAuthTokens.from_json(orjson.loads(raw))


def save_tokens(tokens: OpenAIOAuthTokens) -> None:
    """Persist tokens to ``$VIBE_HOME/auth/openai.json`` with 0600 perms."""
    path = token_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = orjson.dumps(tokens.to_json(), option=orjson.OPT_INDENT_2)
    # Create the file private from the start (umask-independent), then write.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    # Tighten perms even if the file pre-existed with looser bits.
    os.chmod(path, 0o600)


def _tokens_from_response(
    data: dict[str, Any], *, previous: OpenAIOAuthTokens | None, now: datetime
) -> OpenAIOAuthTokens:
    access_token = data.get("access_token")
    if not access_token:
        raise OpenAIRefreshError("token response had no access_token")
    expires_in = int(data.get("expires_in", 3600))
    expires_at = now + timedelta(seconds=expires_in)
    id_token = data.get("id_token") or (previous.id_token if previous else "")
    # A refresh response may omit the refresh_token (reuse the old one) and may
    # not re-assert the account id; fall back to what we already had.
    refresh_token = data.get("refresh_token") or (
        previous.refresh_token if previous else ""
    )
    account_id = (account_id_from_token(id_token) if id_token else None) or (
        previous.account_id if previous else None
    )
    if not refresh_token:
        raise OpenAIOAuthError("token response had no refresh_token")
    if not account_id:
        raise OpenAIOAuthError(
            "Could not determine the ChatGPT account id from the token."
        )
    return OpenAIOAuthTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        account_id=account_id,
        expires_at=expires_at,
        id_token=id_token,
    )


def generate_pkce_pair() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for the S256 PKCE method."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def generate_state() -> str:
    return secrets.token_urlsafe(32)


def build_authorize_url(*, code_challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "redirect_uri": OPENAI_OAUTH_REDIRECT_URI,
        "scope": OPENAI_OAUTH_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        # Codex-specific flags so the issued id_token carries org + account id.
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{OPENAI_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


async def exchange_code_for_tokens(
    code: str, code_verifier: str, *, now: datetime | None = None
) -> OpenAIOAuthTokens:
    """Exchange an authorization code for tokens (form-encoded, per codex)."""
    now = now or datetime.now(UTC)
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": OPENAI_OAUTH_REDIRECT_URI,
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "code_verifier": code_verifier,
    }
    async with httpx.AsyncClient(timeout=30.0, verify=build_ssl_context()) as client:
        try:
            resp = await client.post(OPENAI_TOKEN_URL, data=data)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OpenAIOAuthError(
                f"Token exchange failed: HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise OpenAIOAuthError(f"Token exchange failed: {exc}") from exc
    return _tokens_from_response(resp.json(), previous=None, now=now)


async def _refresh(tokens: OpenAIOAuthTokens, *, now: datetime) -> OpenAIOAuthTokens:
    # Codex refreshes with a JSON body (not form-encoded) at the token endpoint.
    body = {
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": tokens.refresh_token,
        "scope": OPENAI_OAUTH_SCOPES,
    }
    async with httpx.AsyncClient(timeout=30.0, verify=build_ssl_context()) as client:
        try:
            resp = await client.post(OPENAI_TOKEN_URL, json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OpenAIRefreshError(
                f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise OpenAIRefreshError(str(exc)) from exc
    return _tokens_from_response(resp.json(), previous=tokens, now=now)


@dataclass(frozen=True, slots=True)
class ChatGPTCredentials:
    access_token: str
    account_id: str

    def auth_headers(self) -> dict[str, str]:
        """Identity headers the ChatGPT backend requires alongside the bearer."""
        return {
            "ChatGPT-Account-ID": self.account_id,
            "originator": OPENAI_ORIGINATOR,
            "version": OPENAI_CODEX_VERSION,
            "User-Agent": f"{OPENAI_ORIGINATOR}/{OPENAI_CODEX_VERSION}",
        }


async def resolve_chatgpt_credentials(
    *, now: datetime | None = None
) -> ChatGPTCredentials:
    """Load tokens, refreshing (and persisting) if near expiry.

    Raises ``OpenAINotAuthenticatedError`` if there is no token store yet.
    """
    now = now or datetime.now(UTC)
    tokens = load_tokens()
    if tokens is None:
        raise OpenAINotAuthenticatedError()
    if tokens.needs_refresh(now=now):
        logger.debug("Refreshing ChatGPT OAuth access token")
        tokens = await _refresh(tokens, now=now)
        save_tokens(tokens)
    return ChatGPTCredentials(
        access_token=tokens.access_token, account_id=tokens.account_id
    )
