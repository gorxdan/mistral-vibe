from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import functools
import ipaddress
import socket
from typing import TYPE_CHECKING, ClassVar, final
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolStreamEvent
from vibe.core.utils.http import build_ssl_context

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent


_HONEST_USER_AGENT = "vibe-cli"
_HTTP_FORBIDDEN = 403
_MAX_REDIRECTS = 5


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True for private, loopback, link-local, reserved, or multicast IPs.

    Also explicitly blocks the well-known cloud metadata endpoint.
    """
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        return True
    if str(ip) == "169.254.169.254":
        return True
    return False


@functools.cache
def _make_converter_class() -> type:
    from markdownify import MarkdownConverter

    class _Converter(MarkdownConverter):
        convert_script = convert_style = convert_noscript = convert_iframe = (
            convert_object
        ) = convert_embed = lambda *_, **__: ""

    return _Converter


class WebFetchArgs(BaseModel):
    url: str = Field(description="URL to fetch (http/https)")
    timeout: int | None = Field(
        default=None, description="Timeout in seconds (max 120)"
    )


class WebFetchResult(BaseModel):
    url: str
    content: str
    content_type: str
    was_truncated: bool = False


class WebFetchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK

    default_timeout: int = Field(default=30, description="Default timeout in seconds.")
    max_timeout: int = Field(default=120, description="Maximum allowed timeout.")
    max_content_bytes: int = Field(
        default=120_000,
        description="Maximum content size in bytes returned to the model.",
    )
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        description="User agent string for requests.",
    )


class WebFetch(
    BaseTool[WebFetchArgs, WebFetchResult, WebFetchConfig, BaseToolState],
    ToolUIData[WebFetchArgs, WebFetchResult],
):
    description: ClassVar[str] = (
        "Fetch content from a URL. Converts HTML to markdown for readability."
    )

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalise a URL to always have an http(s) scheme.

        Handles protocol-relative URLs (//example.com) and bare URLs (example.com).
        """
        raw = url.lstrip("/") if url.startswith("//") else url
        return raw if raw.startswith(("http://", "https://")) else "https://" + raw

    async def _validate_url(self, url: str) -> None:
        """Reject URLs that resolve to private/loopback/link-local IPs.

        This prevents SSRF attacks against cloud metadata endpoints,
        internal services, and the local machine.
        """
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            raise ToolError("Invalid URL: no host found")

        # Bare IP address (IPv4 or IPv6)
        try:
            ip = ipaddress.ip_address(host)
            if _is_blocked_ip(ip):
                raise ToolError(f"SSRF blocked: {ip} is a private/reserved IP address")
            return
        except ValueError:
            pass

        # Hostname – resolve asynchronously to avoid blocking the event loop
        try:
            loop = asyncio.get_running_loop()
            infos = await loop.run_in_executor(
                None, socket.getaddrinfo, host, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        except (socket.gaierror, OSError):
            return  # Let the HTTP request fail naturally with a clearer error

        for info in infos:
            addr = info[4][0]
            # Strip IPv6 scope suffix if present
            if "%" in addr:
                addr = addr.split("%")[0]
            try:
                ip = ipaddress.ip_address(addr)
                if _is_blocked_ip(ip):
                    raise ToolError(
                        f"SSRF blocked: {host} resolves to {ip}"
                    )
            except ValueError:
                continue

    def resolve_permission(self, args: WebFetchArgs) -> PermissionContext | None:
        if self.config.permission in {ToolPermission.ALWAYS, ToolPermission.NEVER}:
            return PermissionContext(permission=self.config.permission)

        parsed = urlparse(self._normalize_url(args.url))
        domain = parsed.netloc or parsed.path.split("/")[0]
        if not domain:
            return None

        return PermissionContext(
            permission=ToolPermission.ASK,
            required_permissions=[
                RequiredPermission(
                    scope=PermissionScope.URL_PATTERN,
                    invocation_pattern=domain,
                    session_pattern=domain,
                    label=f"fetching from {domain}",
                )
            ],
        )

    @final
    async def run(
        self, args: WebFetchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WebFetchResult, None]:
        self._validate_args(args)

        url = self._normalize_url(args.url)
        timeout = self._resolve_timeout(args.timeout)

        await self._validate_url(url)
        content, content_type = await self._fetch_url(url, timeout)

        if "text/html" in content_type:
            content = _html_to_markdown(content)

        content_bytes = content.encode("utf-8")
        was_truncated = len(content_bytes) > self.config.max_content_bytes
        if was_truncated:
            content = content_bytes[: self.config.max_content_bytes].decode(
                "utf-8", errors="ignore"
            )
            content += "\n\n[Content truncated due to size limit]"

        yield WebFetchResult(
            url=url,
            content=content,
            content_type=content_type,
            was_truncated=was_truncated,
        )

    def _validate_args(self, args: WebFetchArgs) -> None:
        if not args.url.strip():
            raise ToolError("URL cannot be empty")

        parsed = urlparse(args.url)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            raise ToolError(
                f"Invalid URL scheme: {parsed.scheme}. Must be http or https."
            )

        if args.timeout is not None:
            if args.timeout <= 0:
                raise ToolError("Timeout must be a positive number")
            if args.timeout > self.config.max_timeout:
                raise ToolError(
                    f"Timeout cannot exceed {self.config.max_timeout} seconds"
                )

    def _resolve_timeout(self, timeout: int | None) -> int:
        if timeout is None:
            return self.config.default_timeout
        return min(timeout, self.config.max_timeout)

    async def _fetch_url(self, url: str, timeout: int) -> tuple[str, str]:
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            response = await self._do_fetch(url, timeout, headers)
        except httpx.TimeoutException:
            raise ToolError(f"Request timed out after {timeout} seconds")
        except httpx.RequestError as e:
            raise ToolError(f"Failed to fetch URL: {e}")

        if response.is_error:
            raise ToolError(
                f"HTTP error {response.status_code}: {response.reason_phrase}"
            )

        content_type = response.headers.get("Content-Type", "text/plain")

        return response.text, content_type

    async def _do_fetch(
        self, url: str, timeout: int, headers: dict[str, str]
    ) -> httpx.Response:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            verify=build_ssl_context(),
        ) as client:
            response = await client.get(url, headers=headers)

            # Manually follow redirects so every hop can be SSRF-validated
            redirect_count = 0
            while response.is_redirect and redirect_count < _MAX_REDIRECTS:
                location = response.headers.get("location")
                if not location:
                    break
                next_url = urljoin(str(response.url), location)
                await self._validate_url(next_url)
                response = await client.get(next_url, headers=headers)
                redirect_count += 1

            # In case we are hitting bot detection retry once honestly
            if (
                response.status_code == _HTTP_FORBIDDEN
                and response.headers.get("cf-mitigated") == "challenge"
            ):
                headers["User-Agent"] = _HONEST_USER_AGENT
                response = await client.get(str(response.url), headers=headers)

            return response

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if event.args is None:
            return ToolCallDisplay(summary="webfetch")
        if not isinstance(event.args, WebFetchArgs):
            return ToolCallDisplay(summary="webfetch")

        parsed = urlparse(event.args.url)
        domain = parsed.netloc or event.args.url[:50]
        summary = f"Fetching: {domain}"

        if event.args.timeout:
            summary += f" (timeout {event.args.timeout}s)"

        return ToolCallDisplay(summary=summary)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, WebFetchResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        content_len = len(event.result.content)
        content_type = event.result.content_type.split(";")[0]
        message = f"Fetched {event.result.url} ({content_len:,} chars, {content_type})"
        if event.result.was_truncated:
            message += " [truncated]"

        return ToolResultDisplay(success=True, message=message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Fetching URL"


def _html_to_markdown(html: str) -> str:
    converter_class = _make_converter_class()
    return converter_class(heading_style="ATX", bullets="-").convert(html)
