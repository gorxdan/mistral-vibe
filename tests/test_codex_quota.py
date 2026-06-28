from __future__ import annotations

from httpx import Response
import pytest
import respx

from vibe.core.usage import (
    CodexMonthlyLimit,
    CodexQuotaSnapshot,
    CodexQuotaWindow,
    fetch_codex_quota,
)
from vibe.core.usage._codex_quota import _parse_payload


def _payload() -> dict:
    return {
        "rate_limit": {
            "primary_window": {
                "used_percent": 45.0,
                "window_minutes": 300,
                "resets_at": 2_000_000_000,
            },
            "secondary_window": {
                "used_percent": 30.0,
                "window_minutes": 10080,
                "resets_at": 2_000_050_000,
            },
        },
        "credits": {"has_credits": True, "unlimited": False, "balance": "38"},
        "spend_control": {
            "individual_limit": {
                "used": "1200",
                "limit": "5000",
                "remaining_percent": 76,
                "resets_at": 2_000_100_000,
            }
        },
    }


class TestParsePayload:
    def test_full_payload(self):
        snap = _parse_payload(_payload(), captured_at=1.0)
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 45.0
        assert snap.primary.window_minutes == 300
        assert snap.primary.percent_left == 55.0
        assert snap.secondary is not None
        assert snap.secondary.window_minutes == 10080
        assert snap.credits is not None
        assert snap.credits.balance == "38"
        assert snap.monthly_limit is not None
        assert snap.monthly_limit.remaining_percent == 76

    def test_empty_payload_returns_none(self):
        assert _parse_payload({}, captured_at=1.0) is None

    def test_garbage_payload_returns_none(self):
        assert _parse_payload("not a dict", captured_at=1.0) is None
        assert _parse_payload(None, captured_at=1.0) is None

    def test_partial_windows(self):
        snap = _parse_payload(
            {"rate_limit": {"primary_window": {"used_percent": 10.0}}},
            captured_at=1.0,
        )
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.window_minutes is None
        assert snap.secondary is None

    def test_unlimited_credits(self):
        snap = _parse_payload(
            {"credits": {"has_credits": True, "unlimited": True}},
            captured_at=1.0,
        )
        assert snap is not None
        assert snap.credits is not None
        assert snap.credits.unlimited is True

    def test_no_credits_when_has_credits_false(self):
        # credits present but has_credits=False → snapshot still built (windows
        # may be present), but the renderer skips the credits line.
        snap = _parse_payload(
            {
                "rate_limit": {"primary_window": {"used_percent": 5.0}},
                "credits": {"has_credits": False},
            },
            captured_at=1.0,
        )
        assert snap is not None
        assert snap.credits is not None
        assert snap.credits.has_credits is False


class _FakeCreds:
    access_token = "tok"
    account_id = "acct"

    def auth_headers(self) -> dict:
        return {"ChatGPT-Account-ID": "acct"}


async def _fake_resolve() -> _FakeCreds:
    return _FakeCreds()


class TestFetchCodexQuota:
    @pytest.mark.asyncio
    @respx.mock
    async def test_success(self, monkeypatch) -> None:
        url = "https://chatgpt.com/backend-api/codex/api/codex/usage"
        respx.get(url).mock(return_value=Response(200, json=_payload()))
        # The fetch calls the module-level binding in _codex_quota; patch that.
        monkeypatch.setattr(
            "vibe.core.usage._codex_quota.resolve_chatgpt_credentials",
            _fake_resolve,
        )
        snap = await fetch_codex_quota("https://chatgpt.com/backend-api/codex")
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 45.0

    @pytest.mark.asyncio
    async def test_not_signed_in_returns_none(self, monkeypatch) -> None:
        from vibe.core.auth.openai_oauth import OpenAINotAuthenticatedError

        async def _raise() -> None:
            raise OpenAINotAuthenticatedError()

        monkeypatch.setattr(
            "vibe.core.usage._codex_quota.resolve_chatgpt_credentials", _raise
        )
        assert await fetch_codex_quota("https://x") is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_non_200_returns_none(self, monkeypatch) -> None:
        url = "https://x/api/codex/usage"
        respx.get(url).mock(return_value=Response(429))
        monkeypatch.setattr(
            "vibe.core.usage._codex_quota.resolve_chatgpt_credentials",
            _fake_resolve,
        )
        assert await fetch_codex_quota("https://x") is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_unparseable_body_returns_none(self, monkeypatch) -> None:
        url = "https://x/api/codex/usage"
        respx.get(url).mock(return_value=Response(200, text="not json"))
        monkeypatch.setattr(
            "vibe.core.usage._codex_quota.resolve_chatgpt_credentials",
            _fake_resolve,
        )
        assert await fetch_codex_quota("https://x") is None


def test_snapshot_is_empty_for_blank():
    snap = CodexQuotaSnapshot(captured_at=1.0)
    assert snap.is_empty()
    snap2 = CodexQuotaSnapshot(
        captured_at=1.0, primary=CodexQuotaWindow(used_percent=10.0)
    )
    assert not snap2.is_empty()


def test_window_percent_left_clamps():
    w = CodexQuotaWindow(used_percent=150.0)
    assert w.percent_left == 0.0
    w2 = CodexQuotaWindow(used_percent=-5.0)
    assert w2.percent_left == 100.0


def test_monthly_percent_left_clamps():
    m = CodexMonthlyLimit(
        used="1", limit="10", remaining_percent=150, resets_at=1
    )
    assert m.percent_left == 100.0
