from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import errno
from typing import Any, Final
import urllib.parse

import anyio.to_thread
import httpx
import keyring
import keyring.backends.fail
import keyring.errors
from mcp.client.auth import (
    OAuthClientProvider,
    OAuthFlowError,
    OAuthTokenError,
    TokenStorage,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl, BaseModel, ConfigDict

from vibe.core.config import MCPHttp, MCPOAuth, MCPStreamableHttp
from vibe.core.logger import logger
from vibe.core.utils.http import build_ssl_context

_SERVICE: Final = "ai.mistral.vibe"
_USERNAME_PREFIX: Final = "mcp-oauth"
_CLIENT_NAME: Final = "Chaton"
_LOGIN_TIMEOUT_SECONDS: Final = 300.0
_MIN_REQUEST_LINE_PARTS: Final = 2
_HEADER_TERMINATORS: Final = frozenset({b"\r\n", b"\n", b""})
_PUBLIC_AUTH_METHODS: Final = frozenset({None, "none"})
_TOKEN_OK_STATUS: Final = 200
_TOKEN_OK_UBOUND: Final = 300


class MCPOAuthError(Exception):
    def _fmt(self) -> str:
        return self.__class__.__name__


class MCPOAuthPortInUse(MCPOAuthError):
    def __init__(self, *, port: int, server_alias: str) -> None:
        self.port = port
        self.server_alias = server_alias
        super().__init__(self._fmt())

    def _fmt(self) -> str:
        return (
            f"Loopback callback port {self.port} is already in use; cannot complete "
            f"OAuth login for MCP server {self.server_alias!r}. "
            "Set `auth.redirect_port` to a free port in this server's config and retry."
        )


class MCPOAuthHeadlessError(MCPOAuthError):
    def __init__(self, *, server_alias: str) -> None:
        self.server_alias = server_alias
        super().__init__(self._fmt())

    def _fmt(self) -> str:
        return (
            f"No OS keyring backend is available; cannot store OAuth tokens for "
            f"MCP server {self.server_alias!r}. "
            'Switch this server to `auth.type = "static"` with `api_key_env` for '
            "headless or CI environments."
        )


class MCPOAuthInvalidGrant(MCPOAuthError):
    def __init__(self, *, server_alias: str, reason: str) -> None:
        self.server_alias = server_alias
        self.reason = reason
        super().__init__(self._fmt())

    def _fmt(self) -> str:
        return (
            f"OAuth refresh failed for MCP server {self.server_alias!r}: {self.reason}. "
            f"Run `/mcp login {self.server_alias}` to re-authenticate."
        )


class MCPOAuthLoginFailed(MCPOAuthError):
    def __init__(self, *, server_alias: str, reason: str) -> None:
        self.server_alias = server_alias
        self.reason = reason
        super().__init__(self._fmt())

    def _fmt(self) -> str:
        return (
            f"OAuth login failed for MCP server {self.server_alias!r}: {self.reason}. "
            f"Run `/mcp login {self.server_alias}` to retry."
        )


def _kr_username(alias: str, kind: str) -> str:
    return f"{_USERNAME_PREFIX}:{alias}:{kind}"


async def _kr_get(username: str) -> str | None:
    return await anyio.to_thread.run_sync(keyring.get_password, _SERVICE, username)


async def _kr_set(username: str, value: str) -> None:
    await anyio.to_thread.run_sync(keyring.set_password, _SERVICE, username, value)


async def _kr_delete(username: str) -> None:
    try:
        await anyio.to_thread.run_sync(keyring.delete_password, _SERVICE, username)
    except keyring.errors.PasswordDeleteError:
        pass


class Fingerprint(BaseModel):
    """Config-drift detection marker for OAuth MCP servers.

    Captures the server URL, normalized scopes, and client identity marker
    (client_id, client_metadata_url, or "<dcr>" for DCR flow).
    Two fingerprints are equal if all fields match; changes indicate the config
    has changed and re-authentication is needed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    scopes_sorted: tuple[str, ...]
    client_marker: str

    @classmethod
    def compute(cls, server: MCPHttp | MCPStreamableHttp) -> Fingerprint:
        auth = server.auth
        if not isinstance(auth, MCPOAuth):
            raise TypeError(
                "Fingerprint.compute requires an OAuth-configured MCP server; "
                f"server {server.name!r} uses auth.type={type(auth).__name__}"
            )
        scopes = tuple(sorted({s.strip() for s in auth.scopes if s.strip()}))
        if auth.client_id:
            marker = auth.client_id
        elif auth.client_metadata_url:
            marker = str(auth.client_metadata_url)
        else:
            marker = "<dcr>"
        return cls(url=server.url, scopes_sorted=scopes, client_marker=marker)

    @classmethod
    async def load(cls, alias: str) -> Fingerprint | None:
        raw = await _kr_get(_kr_username(alias, "fingerprint"))
        if raw is None:
            return None
        return cls.model_validate_json(raw)

    async def save(self, alias: str) -> None:
        await _kr_set(_kr_username(alias, "fingerprint"), self.model_dump_json())


class KeyringTokenStorage(TokenStorage):
    def __init__(
        self,
        alias: str,
        *,
        fallback_client_info: OAuthClientInformationFull | None = None,
    ) -> None:
        try:
            backend = keyring.get_keyring()
        except (ImportError, keyring.errors.KeyringError) as exc:
            raise MCPOAuthHeadlessError(server_alias=alias) from exc
        if isinstance(backend, keyring.backends.fail.Keyring):
            raise MCPOAuthHeadlessError(server_alias=alias)
        self._alias = alias
        self._fallback_client_info = fallback_client_info

    async def get_tokens(self) -> OAuthToken | None:
        raw = await _kr_get(_kr_username(self._alias, "tokens"))
        if raw is None:
            return None
        return OAuthToken.model_validate_json(raw)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await _kr_set(_kr_username(self._alias, "tokens"), tokens.model_dump_json())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = await _kr_get(_kr_username(self._alias, "client_info"))
        if raw is None:
            return self._fallback_client_info
        return OAuthClientInformationFull.model_validate_json(raw)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await _kr_set(
            _kr_username(self._alias, "client_info"), client_info.model_dump_json()
        )


_LOGO_SVG: Final = (
    '<svg class="mark" viewBox="0 0 162 162" xmlns="http://www.w3.org/2000/svg" '
    'aria-hidden="true">'
    '<path d="M50.9987 32.0001H30.9987V50.0001H50.9987V32.0001Z"/>'
    '<path d="M130.999 32.0001H110.999V50.0001H130.999V32.0001Z"/>'
    '<path d="M90.9988 92.0002H70.9988V110H90.9988V92.0002Z"/>'
    '<path d="M50.9987 92.0002H30.9987V110H50.9987V92.0002Z"/>'
    '<path d="M130.999 92.0002H110.999V110H130.999V92.0002Z"/>'
    '<path d="M70.9987 52.0004H30.9987V70.0004H70.9987V52.0004Z"/>'
    '<path d="M71.0002 112H11V130.001H71.0002V112Z"/>'
    '<path d="M151 112H90.9998V130.001H151V112Z"/>'
    '<path d="M130.999 72.0002H30.9987V90.0003H130.999V72.0002Z"/>'
    '<path d="M131 52.0004H90.9998V70.0004H131V52.0004Z"/>'
    "</svg>"
)

_FAVICON_DATA_URL: Final = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 162 162' fill='%23FA500F'>"
    "<path d='M50.9987 32.0001H30.9987V50.0001H50.9987V32.0001Z'/>"
    "<path d='M130.999 32.0001H110.999V50.0001H130.999V32.0001Z'/>"
    "<path d='M90.9988 92.0002H70.9988V110H90.9988V92.0002Z'/>"
    "<path d='M50.9987 92.0002H30.9987V110H50.9987V92.0002Z'/>"
    "<path d='M130.999 92.0002H110.999V110H130.999V92.0002Z'/>"
    "<path d='M70.9987 52.0004H30.9987V70.0004H70.9987V52.0004Z'/>"
    "<path d='M71.0002 112H11V130.001H71.0002V112Z'/>"
    "<path d='M151 112H90.9998V130.001H151V112Z'/>"
    "<path d='M130.999 72.0002H30.9987V90.0003H130.999V72.0002Z'/>"
    "<path d='M131 52.0004H90.9998V70.0004H131V52.0004Z'/>"
    "</svg>"
)

_BASE_STYLE: Final = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       display: flex; align-items: center; justify-content: center;
       min-height: 100vh; margin: 0;
       background: light-dark(#FBFBF8, #171722);
       color: light-dark(#15202b, #FBFBF8); }
.card { padding: 2.5rem 3rem; border-radius: 12px;
        background: light-dark(#F5F4EF, #242433);
        box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 8px 24px rgba(0,0,0,.12);
        text-align: center; max-width: 28rem; }
.mark { width: 64px; height: 64px; display: block; margin: 0 auto 1.25rem; }
.mark path { fill: #FA500F; }
h1 { font-size: 1.4rem; margin: 0 0 .5rem; font-weight: 600; }
p { margin: 0; opacity: .7; }
"""


def _render_page(*, title: str, heading: str, body: str) -> bytes:
    return (
        f"<!DOCTYPE html>\n"
        f'<html lang="en">\n'
        f"<head>\n"
        f'<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f'<link rel="icon" href="{_FAVICON_DATA_URL}">\n'
        f"<style>{_BASE_STYLE}</style>\n"
        f"</head>\n"
        f"<body>\n"
        f'<div class="card">\n'
        f"{_LOGO_SVG}\n"
        f"<h1>{heading}</h1>\n"
        f"<p>{body}</p>\n"
        f"</div>\n"
        f"</body>\n"
        f"</html>\n"
    ).encode()


_SUCCESS_HTML: Final = _render_page(
    title="Chaton - Login complete",
    heading="Login complete",
    body="You can close this tab and return to Chaton.",
)

_ERROR_HTML: Final = _render_page(
    title="Chaton - Login failed",
    heading="Login failed",
    body="The authorization server did not return an authorization code. Return to Chaton and try again.",
)


def _http_response(status_line: bytes, body: bytes) -> bytes:
    return (
        status_line
        + b"Content-Type: text/html; charset=utf-8\r\n"
        + b"Connection: close\r\n"
        + b"Cache-Control: no-store\r\n"
        + b"Content-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\n\r\n"
        + body
    )


class LoopbackCallbackHandler:
    def __init__(self, *, port: int, server_alias: str) -> None:
        self._port = port
        self._server_alias = server_alias

    async def _fail(
        self,
        writer: asyncio.StreamWriter,
        msg: str,
        *,
        future: asyncio.Future[tuple[str, str | None]],
    ) -> None:
        writer.write(_http_response(b"HTTP/1.1 400 Bad Request\r\n", _ERROR_HTML))
        await writer.drain()
        if not future.done():
            future.set_exception(
                MCPOAuthError(f"OAuth callback for {self._server_alias!r} {msg}")
            )

    async def serve_once(self) -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, str | None]] = loop.create_future()

        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                request_line = await reader.readline()
                while True:
                    line = await reader.readline()
                    if line in _HEADER_TERMINATORS:
                        break
                parts = request_line.split(b" ", 2)
                if len(parts) < _MIN_REQUEST_LINE_PARTS:
                    await self._fail(
                        writer, "received a malformed HTTP request", future=future
                    )
                    return
                path = parts[1].decode("latin-1", errors="replace")
                query = urllib.parse.urlparse(path).query
                params = urllib.parse.parse_qs(query)
                code_values = params.get("code") or []
                state_values = params.get("state") or []
                if not code_values:
                    await self._fail(
                        writer, "missing 'code' query parameter", future=future
                    )
                    return
                writer.write(_http_response(b"HTTP/1.1 200 OK\r\n", _SUCCESS_HTML))
                await writer.drain()
                if not future.done():
                    future.set_result((
                        code_values[0],
                        state_values[0] if state_values else None,
                    ))
            except BaseException as exc:
                if not future.done():
                    future.set_exception(exc)
                raise
            finally:
                writer.close()
                with _suppress_close_errors():
                    await writer.wait_closed()

        try:
            server = await asyncio.start_server(
                handle, host="127.0.0.1", port=self._port
            )
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise MCPOAuthPortInUse(
                    port=self._port, server_alias=self._server_alias
                ) from exc
            raise

        try:
            return await future
        finally:
            server.close()
            with _suppress_close_errors():
                await server.wait_closed()


class _suppress_close_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        return exc_type is not None and issubclass(
            exc_type, (ConnectionError, OSError, asyncio.CancelledError)
        )


class _ConfidentialClientOAuthProvider(OAuthClientProvider):
    """Provider tolerant of authorization servers that register confidential clients.

    Some servers (notably Supabase's hosted MCP) return a ``client_secret`` in the
    DCR response while omitting ``token_endpoint_auth_method``. The SDK then treats
    the client as public (auth method ``None``) and omits the secret at token
    exchange, which the server rejects with ``422 Required parameter: client_secret``.
    This subclass coerces such clients to ``client_secret_post`` (RFC 7591 §2)
    before every token exchange and refresh, and accepts any 2xx token response
    (Supabase answers ``201 Created``) rather than only exactly ``200``.
    """

    @staticmethod
    def _coerce_confidential(client_info: OAuthClientInformationFull) -> None:
        if client_info.client_secret and (
            client_info.token_endpoint_auth_method in _PUBLIC_AUTH_METHODS
        ):
            client_info.token_endpoint_auth_method = "client_secret_post"

    @staticmethod
    def _accept_2xx(response: httpx.Response) -> None:
        # The SDK only treats exactly 200 as success; normalize any 2xx so a 201
        # token response is not misread as a failure. Non-2xx is left untouched.
        if _TOKEN_OK_STATUS <= response.status_code < _TOKEN_OK_UBOUND:
            response.status_code = _TOKEN_OK_STATUS

    async def _exchange_token_authorization_code(
        self,
        auth_code: str,
        code_verifier: str,
        *,
        token_data: dict[str, Any] | None = None,
    ) -> httpx.Request:
        if ci := self.context.client_info:
            self._coerce_confidential(ci)
        return await super()._exchange_token_authorization_code(
            auth_code, code_verifier, token_data=token_data
        )

    async def _refresh_token(self) -> httpx.Request:
        if ci := self.context.client_info:
            self._coerce_confidential(ci)
        return await super()._refresh_token()

    async def _handle_token_response(self, response: httpx.Response) -> None:
        self._accept_2xx(response)
        await super()._handle_token_response(response)

    async def _handle_refresh_response(self, response: httpx.Response) -> bool:
        self._accept_2xx(response)
        return await super()._handle_refresh_response(response)


def build_oauth_provider(
    server: MCPHttp | MCPStreamableHttp,
    *,
    redirect_handler: Callable[[str], Awaitable[None]],
    callback_handler: Callable[[], Awaitable[tuple[str, str | None]]],
) -> OAuthClientProvider:
    auth = server.auth
    if not isinstance(auth, MCPOAuth):
        raise TypeError(
            "build_oauth_provider requires an OAuth-configured MCP server; "
            f"server {server.name!r} uses auth.type={type(auth).__name__}"
        )
    redirect_uri = AnyUrl(f"http://127.0.0.1:{auth.redirect_port}/callback")
    scope = " ".join(s for s in auth.scopes if s) or None
    metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        scope=scope,
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        client_name=_CLIENT_NAME,
    )
    client_metadata_url = (
        str(auth.client_metadata_url) if auth.client_metadata_url else None
    )
    fallback_client_info = (
        OAuthClientInformationFull(client_id=auth.client_id, **metadata.model_dump())
        if auth.client_id
        else None
    )
    return _ConfidentialClientOAuthProvider(
        server_url=server.url,
        client_metadata=metadata,
        storage=KeyringTokenStorage(
            alias=server.name, fallback_client_info=fallback_client_info
        ),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        client_metadata_url=client_metadata_url,
    )


async def perform_oauth_login(
    server: MCPHttp | MCPStreamableHttp, *, on_url: Callable[[str], Awaitable[None]]
) -> None:
    auth = server.auth
    if not isinstance(auth, MCPOAuth):
        raise TypeError(
            "perform_oauth_login requires an OAuth-configured MCP server; "
            f"server {server.name!r} uses auth.type={type(auth).__name__}"
        )
    handler = LoopbackCallbackHandler(port=auth.redirect_port, server_alias=server.name)
    provider = build_oauth_provider(
        server, redirect_handler=on_url, callback_handler=handler.serve_once
    )
    try:
        async with httpx.AsyncClient(
            auth=provider, timeout=_LOGIN_TIMEOUT_SECONDS, verify=build_ssl_context()
        ) as client:
            await client.get(server.url)
    except (OAuthTokenError, OAuthFlowError) as exc:
        raise MCPOAuthLoginFailed(server_alias=server.name, reason=str(exc)) from exc
    await Fingerprint.compute(server).save(server.name)


async def is_logged_in(server: MCPHttp | MCPStreamableHttp) -> bool:
    """Return True when stored OAuth credentials match the current server config.

    Headless environments (no OS keyring backend) report False and log the
    static/PAT guidance instead of raising.
    """
    try:
        storage = KeyringTokenStorage(alias=server.name)
    except MCPOAuthHeadlessError as exc:
        logger.info("%s", exc)
        return False
    if await storage.get_tokens() is None:
        return False
    saved = await Fingerprint.load(server.name)
    return saved == Fingerprint.compute(server)


def make_non_interactive_handlers(
    alias: str,
) -> tuple[
    Callable[[str], Awaitable[None]], Callable[[], Awaitable[tuple[str, str | None]]]
]:
    """Handlers for OAuth providers used outside an interactive login.

    Discovery and per-call tool execution never open a browser: if the token
    cannot be refreshed silently, the handlers raise so the caller surfaces a
    clear "run /mcp login <alias>" error instead of hanging or spawning UI.
    """

    async def _redirect(url: str) -> None:
        raise MCPOAuthLoginFailed(
            server_alias=alias,
            reason=(
                f"interactive login required to authorize {url}; "
                f"run `/mcp login {alias}`"
            ),
        )

    async def _callback() -> tuple[str, str | None]:
        raise MCPOAuthLoginFailed(
            server_alias=alias,
            reason=f"interactive login required; run `/mcp login {alias}`",
        )

    return _redirect, _callback


def build_non_interactive_provider(
    server: MCPHttp | MCPStreamableHttp,
) -> OAuthClientProvider:
    """Build an OAuth provider whose handlers fail loudly outside a login flow.

    Used by the registry (discovery) and the proxy tool (per-call): the
    provider refreshes silently when tokens are valid, and raises a clear error
    when re-authorization is needed.
    """
    redirect, callback = make_non_interactive_handlers(server.name)
    return build_oauth_provider(
        server, redirect_handler=redirect, callback_handler=callback
    )


async def clear_stored_credentials(alias: str) -> None:
    for kind in ("tokens", "client_info", "fingerprint"):
        await _kr_delete(_kr_username(alias, kind))
