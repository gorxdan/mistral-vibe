from __future__ import annotations

from vibe.core.llm.backend.adapter_port import PreparedRequest, RequestParams
from vibe.core.llm.backend.anthropic import AnthropicAdapter

DEFAULT_BEDROCK_REGION = "us-east-1"


def build_bedrock_base_url(region: str) -> str:
    return f"https://bedrock-mantle.{region}.api.aws/anthropic"


class BedrockAnthropicAdapter(AnthropicAdapter):
    """Amazon Bedrock Mantle adapter.

    The Bedrock Mantle endpoint serves Claude through the Anthropic Messages API
    shape (standard SSE streaming) at ``https://bedrock-mantle.{region}.api.aws/
    anthropic/v1/messages``. Auth is a bearer token in ``x-api-key``. The parent
    ``AnthropicAdapter`` already builds exactly that wire shape; this subclass only
    pins the region-aware base URL.
    """

    def prepare_request(self, params: RequestParams) -> PreparedRequest:
        req = super().prepare_request(params)
        region = params.provider.region or DEFAULT_BEDROCK_REGION
        return req._replace(base_url=build_bedrock_base_url(region))
