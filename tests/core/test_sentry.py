from __future__ import annotations

import sys

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core import sentry
from vibe.core.telemetry.types import EntrypointMetadata

_METADATA = EntrypointMetadata(
    agent_entrypoint="cli",
    agent_version="0.0.0",
    client_name="test",
    client_version="0.0.0",
)


def test_init_sentry_without_dsn_short_circuits_before_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sentry, "_SENTRY_DSN", None)
    monkeypatch.delitem(sys.modules, "sentry_sdk", raising=False)
    config = build_test_vibe_config(enable_telemetry=True)

    result = sentry.init_sentry(config, headless=True, entrypoint_metadata=_METADATA)

    assert result is False
    assert "sentry_sdk" not in sys.modules


def test_init_sentry_disabled_telemetry_returns_false() -> None:
    config = build_test_vibe_config(enable_telemetry=False)

    assert (
        sentry.init_sentry(config, headless=True, entrypoint_metadata=_METADATA)
        is False
    )
