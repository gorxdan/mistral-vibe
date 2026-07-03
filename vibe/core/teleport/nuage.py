from __future__ import annotations

import asyncio
import types

import httpx
from pydantic import ValidationError

from vibe.core.telemetry.types import TeleportFailureDetails
from vibe.core.teleport.errors import ServiceTeleportError
from vibe.core.teleport.models import NuageRequest, NuageResponse
from vibe.core.utils.http import build_ssl_context

_AMBIGUOUS_CREATE_STATUS_CODES = frozenset({504})
_AMBIGUOUS_REQUEST_ERRORS: tuple[type[httpx.RequestError], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


class NuageClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        max_start_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._max_start_attempts = max(1, max_start_attempts)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)

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
        response: httpx.Response | None = None
        for attempt in range(self._max_start_attempts):
            try:
                response = await self._http_client.post(
                    f"{self._base_url}/api/v1/code/sessions",
                    headers=self._headers(),
                    json=request.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                )
            except _AMBIGUOUS_REQUEST_ERRORS as e:
                if attempt < self._max_start_attempts - 1:
                    await asyncio.sleep(self._retry_delay_seconds)
                    continue
                raise self._ambiguous_create_error() from e

            if (
                response.status_code in _AMBIGUOUS_CREATE_STATUS_CODES
                and attempt < self._max_start_attempts - 1
            ):
                await asyncio.sleep(self._retry_delay_seconds)
                continue

            break

        if response is None:
            raise self._ambiguous_create_error()

        if response.status_code in _AMBIGUOUS_CREATE_STATUS_CODES:
            raise self._ambiguous_create_error(http_status_code=response.status_code)

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

    @staticmethod
    def _ambiguous_create_error(
        http_status_code: int | None = None,
    ) -> ServiceTeleportError:
        details = TeleportFailureDetails(failure_kind="ambiguous_create")
        if http_status_code is not None:
            details["http_status_code"] = http_status_code
        return ServiceTeleportError(
            "Vibe Code Web did not confirm session creation after retrying. "
            "Check Vibe Code Web before trying again.",
            telemetry_details=details,
        )
