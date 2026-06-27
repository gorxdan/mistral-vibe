from __future__ import annotations

import pytest

from tests.constants import ANTHROPIC_BASE_URL
from vibe.core.config import OtelSpanExporterConfig, ProviderConfig, VibeConfig
from vibe.core.types import Backend


class TestOtelSpanExporterConfig:
    def test_none_without_explicit_endpoint_with_mistral_provider(
        self, vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Remote export is opt-in: a configured Mistral provider + API key must
        # NOT auto-derive a remote endpoint. Traces stay local-only.
        monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
        config = vibe_config.model_copy(
            update={
                "providers": [
                    ProviderConfig(
                        name="mistral",
                        api_base="https://customer.mistral.ai/v1",
                        backend=Backend.MISTRAL,
                    )
                ]
            }
        )
        assert config.otel_span_exporter_config is None

    def test_none_with_default_providers(
        self, vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "sk-default")
        assert vibe_config.otel_span_exporter_config is None

    def test_none_without_mistral_provider(
        self, vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "sk-fallback")
        config = vibe_config.model_copy(
            update={
                "providers": [
                    ProviderConfig(
                        name="anthropic", api_base=f"{ANTHROPIC_BASE_URL}/v1"
                    )
                ]
            }
        )
        assert config.otel_span_exporter_config is None

    def test_no_warning_when_local_only(
        self,
        vibe_config: VibeConfig,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with caplog.at_level("WARNING"):
            assert vibe_config.otel_span_exporter_config is None
        assert "OTEL tracing enabled" not in caplog.text

    def test_explicit_otel_endpoint_appends_default_traces_path(
        self, vibe_config: VibeConfig
    ) -> None:
        config = vibe_config.model_copy(
            update={"otel_endpoint": "https://my-collector:4318"}
        )
        result = config.otel_span_exporter_config
        assert result is not None
        assert result == OtelSpanExporterConfig(
            endpoint="https://my-collector:4318/v1/traces"
        )
        assert result.headers is None

    def test_explicit_otel_endpoint_preserves_path_prefix(
        self, vibe_config: VibeConfig
    ) -> None:
        config = vibe_config.model_copy(
            update={"otel_endpoint": "https://my-collector:4318/api/public/otel"}
        )
        result = config.otel_span_exporter_config
        assert result is not None
        assert result == OtelSpanExporterConfig(
            endpoint="https://my-collector:4318/api/public/otel/v1/traces"
        )
        assert result.headers is None
