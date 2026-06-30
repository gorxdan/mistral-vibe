from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from vibe.core.llm.backend.adapter_port import PreparedRequest, RequestParams
from vibe.core.llm.backend.anthropic import AnthropicAdapter
from vibe.core.logger import logger

if TYPE_CHECKING:
    from botocore.credentials import Credentials

DEFAULT_BEDROCK_REGION = "us-east-1"
# Bedrock Mantle SigV4 signing service name (matches codex-rs / the Mantle docs).
BEDROCK_MANTLE_SERVICE = "bedrock-mantle"


class BedrockAuthError(RuntimeError):
    """Raised when neither a bearer token nor AWS SDK credentials resolve."""


def build_bedrock_base_url(region: str) -> str:
    return f"https://bedrock-mantle.{region}.api.aws/anthropic"


class _BedrockCredentials:
    """Lazily resolves and caches the AWS SDK credential chain via botocore.

    Covers the standard chain (env vars, shared ``~/.aws`` config, SSO, assumed
    roles, ECS/IMDS, ``credential_process``) -- everything botocore resolves, so
    users with ``aws configure`` / SSO / IAM roles work without a bearer token.
    Refreshable credentials self-refresh through ``get_frozen_credentials()``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._creds: Credentials | None = None

    def resolve(self) -> Credentials | None:
        with self._lock:
            if self._creds is None:
                from botocore.session import Session

                self._creds = Session().get_credentials()
            return self._creds


_BEDROCK_CREDS = _BedrockCredentials()


def _sign_sigv4(
    *, method: str, url: str, headers: dict[str, str], body: bytes, region: str
) -> dict[str, str]:
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    creds_obj = _BEDROCK_CREDS.resolve()
    if creds_obj is None:
        raise BedrockAuthError(
            "No Amazon Bedrock credentials found: set AWS_BEARER_TOKEN_BEDROCK, "
            "or configure the AWS SDK credential chain (e.g. `aws configure`)."
        )
    creds = creds_obj.get_frozen_credentials()
    aws_req = AWSRequest(method=method, url=url, data=body, headers=dict(headers))
    SigV4Auth(creds, BEDROCK_MANTLE_SERVICE, region).add_auth(aws_req)
    return {
        k: v.decode() if isinstance(v, bytes) else v for k, v in aws_req.headers.items()
    }


class BedrockAnthropicAdapter(AnthropicAdapter):
    """Amazon Bedrock Mantle adapter.

    The Bedrock Mantle endpoint serves Claude through the Anthropic Messages API
    shape (standard SSE streaming) at ``https://bedrock-mantle.{region}.api.aws/
    anthropic/v1/messages``. The parent ``AnthropicAdapter`` already builds that
    exact wire shape; this subclass pins the region-aware base URL and resolves
    auth:

    1. Bearer token (``AWS_BEARER_TOKEN_BEDROCK``) -> ``x-api-key`` (inherited).
    2. AWS SDK credential chain -> SigV4 ``Authorization`` (fallback).
    """

    def prepare_request(self, params: RequestParams) -> PreparedRequest:
        req = super().prepare_request(params)
        region = params.provider.region or DEFAULT_BEDROCK_REGION
        base_url = build_bedrock_base_url(region)
        if params.api_key:
            # Bearer-token path: x-api-key already set by the parent.
            return req._replace(base_url=base_url)
        logger.debug("Bedrock bearer token absent; signing with AWS credentials")
        url = f"{base_url}{req.endpoint}"
        signed_headers = _sign_sigv4(
            method="POST", url=url, headers=req.headers, body=req.body, region=region
        )
        return req._replace(base_url=base_url, headers=signed_headers)
