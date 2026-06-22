from __future__ import annotations

import types

import httpx
from pydantic import ValidationError

from vibe.core.telemetry.types import TeleportFailureDetails
from vibe.core.teleport.errors import ServiceTeleportError
from vibe.core.teleport.models import NuageRequest, NuageResponse
from vibe.core.utils.http import build_ssl_context


class NuageClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout

    async def __aenter__(self) -> NuageClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout), verify=build_ssl_context()
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None

    @property
    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout), verify=build_ssl_context()
            )
            self._owns_client = True
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def start(self, request: NuageRequest) -> NuageResponse:
        response = await self._http_client.post(
            f"{self._base_url}/api/v1/code/sessions",
            headers=self._headers(),
            json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        if not response.is_success:
            raise ServiceTeleportError(
                f"Vibe Code Web start failed "
                f"(status {response.status_code}): {response.text}",
                telemetry_details=TeleportFailureDetails(
                    failure_kind="http_error", http_status_code=response.status_code
                ),
            )

        try:
            return NuageResponse.model_validate(response.json())
        except ValidationError as e:
            raise ServiceTeleportError(
                "Vibe Code Web response was invalid",
                telemetry_details=TeleportFailureDetails(
                    failure_kind="invalid_schema", http_status_code=response.status_code
                ),
            ) from e
        except ValueError as e:
            raise ServiceTeleportError(
                "Vibe Code Web response was not valid JSON",
                telemetry_details=TeleportFailureDetails(
                    failure_kind="invalid_json", http_status_code=response.status_code
                ),
            ) from e
