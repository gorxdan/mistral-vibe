from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from vibe.core.auth.openai_oauth import (
    OpenAINotAuthenticatedError,
    resolve_chatgpt_credentials,
)
from vibe.core.logger import logger
from vibe.core.utils.http import build_ssl_context

# Codex/ChatGPT usage endpoint. Path mirrors codex-rs/backend-client:
#   {base}/api/codex/usage   (PathStyle::CodexApi)
# base defaults to OPENAI_CHATGPT_API_BASE from vibe.core.auth.openai_oauth.
_USAGE_PATH = "/api/codex/usage"
# A /status render should never hang on the quota fetch. Codex uses a short
# timeout too; we match that posture rather than the chat-completion timeout.
_FETCH_TIMEOUT = 6.0
_HTTP_OK = 200


class CodexQuotaWindow(BaseModel):
    """One rolling window (e.g. the 5h primary or weekly secondary)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    used_percent: float
    window_minutes: int | None = None
    resets_at: int | None = None  # unix seconds

    @property
    def percent_left(self) -> float:
        return max(0.0, min(100.0, 100.0 - self.used_percent))


class CodexCredits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    has_credits: bool = False
    unlimited: bool = False
    balance: str | None = None


class CodexMonthlyLimit(BaseModel):
    """Workspace spend-control individual monthly limit (enterprise plans)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    used: str
    limit: str
    remaining_percent: int
    resets_at: int

    @property
    def percent_left(self) -> float:
        return max(0.0, min(100.0, float(self.remaining_percent)))


class CodexQuotaSnapshot(BaseModel):
    """The Codex/ChatGPT plan's rolling quota state at one point in time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    captured_at: float
    primary: CodexQuotaWindow | None = None
    secondary: CodexQuotaWindow | None = None
    credits: CodexCredits | None = None
    monthly_limit: CodexMonthlyLimit | None = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in ("primary", "secondary", "credits", "monthly_limit")
        )


def _window(raw: Any) -> CodexQuotaWindow | None:
    if not isinstance(raw, dict):
        return None
    try:
        return CodexQuotaWindow.model_validate(raw)
    except ValidationError:
        return None


async def fetch_codex_quota(api_base: str) -> CodexQuotaSnapshot | None:
    """Fetch the Codex/ChatGPT plan usage snapshot, or None on any failure.

    Returns None when: not signed in, network error, non-2xx, or unparseable
    body. The /status card treats None as "section not shown" — the fetch is
    best-effort and must never block or crash the status render.
    """
    try:
        creds = await resolve_chatgpt_credentials()
    except OpenAINotAuthenticatedError:
        return None
    except Exception as e:
        logger.error("Codex quota auth failed: %s", e)
        return None

    url = f"{api_base.rstrip('/')}{_USAGE_PATH}"
    headers = {
        "Authorization": f"Bearer {creds.access_token}",
        **creds.auth_headers(),
    }
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT, verify=build_ssl_context()
        ) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != _HTTP_OK:
                return None
            payload = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.error("Codex quota fetch failed: %s", e)
        return None

    import time

    return _parse_payload(payload, captured_at=time.time())


def _parse_payload(payload: Any, *, captured_at: float) -> CodexQuotaSnapshot | None:
    """Map the /api/codex/usage JSON into a snapshot.

    Shape (from codex RateLimitStatusPayload): top-level ``rate_limit`` holds
    ``primary_window`` / ``secondary_window``; ``credits`` and
    ``spend_control.individual_limit`` are sibling objects. We take the primary
    codex limit only (additional_rate_limits is out of scope for /status).
    """
    if not isinstance(payload, dict):
        return None
    rate_limit = payload.get("rate_limit")
    primary = secondary = None
    if isinstance(rate_limit, dict):
        primary = _window(rate_limit.get("primary_window"))
        secondary = _window(rate_limit.get("secondary_window"))

    credits = None
    raw_credits = payload.get("credits")
    if isinstance(raw_credits, dict):
        try:
            credits = CodexCredits.model_validate(raw_credits)
        except ValidationError:
            credits = None

    monthly = None
    spend_control = payload.get("spend_control")
    if isinstance(spend_control, dict):
        raw_individual = spend_control.get("individual_limit")
        if isinstance(raw_individual, dict):
            try:
                monthly = CodexMonthlyLimit.model_validate(raw_individual)
            except ValidationError:
                monthly = None

    snap = CodexQuotaSnapshot(
        captured_at=captured_at,
        primary=primary,
        secondary=secondary,
        credits=credits,
        monthly_limit=monthly,
    )
    return None if snap.is_empty() else snap
