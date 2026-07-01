"""Interactive "Sign in with ChatGPT" browser flow for the OpenAI provider.

Drives the loopback PKCE OAuth dance (authorize in the browser, capture the
redirect on ``127.0.0.1:1455``, exchange the code for tokens) and persists the
result via :mod:`vibe.core.auth.openai_oauth`. This is the reverse-engineered
Codex flow; see that module's docstring for the ToS / stability caveats.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import errno
import urllib.parse
import webbrowser

from vibe.core.auth.openai_oauth import (
    OPENAI_OAUTH_REDIRECT_PORT,
    OpenAIOAuthError,
    OpenAIOAuthTokens,
    build_authorize_url,
    exchange_code_for_tokens,
    generate_pkce_pair,
    generate_state,
    save_tokens,
)
from vibe.core.logger import logger

_LOGIN_TIMEOUT_SECONDS = 300.0
_MIN_REQUEST_LINE_PARTS = 2
_HEADER_TERMINATORS = frozenset({b"\r\n", b"\n", b""})

BrowserOpener = Callable[[str], bool]
NowFn = Callable[[], datetime]
UrlCallback = Callable[[str], None]

_SUCCESS_BODY = (
    b"<!DOCTYPE html><html><head><meta charset='utf-8'>"
    b"<title>Mistral Vibe - signed in</title></head>"
    b"<body style='font-family:system-ui;text-align:center;margin-top:4rem'>"
    b"<h1>Signed in to ChatGPT</h1>"
    b"<p>You can close this tab and return to Mistral Vibe.</p></body></html>"
)
_ERROR_BODY = (
    b"<!DOCTYPE html><html><head><meta charset='utf-8'>"
    b"<title>Mistral Vibe - sign-in failed</title></head>"
    b"<body style='font-family:system-ui;text-align:center;margin-top:4rem'>"
    b"<h1>Sign-in failed</h1>"
    b"<p>No authorization code was returned. Return to Mistral Vibe and try again.</p>"
    b"</body></html>"
)


class OpenAISignInError(OpenAIOAuthError):
    """Raised when the interactive ChatGPT sign-in cannot complete."""


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


class _LoopbackCallbackServer:
    """One-shot loopback HTTP server that captures the OAuth redirect.

    Binds ``127.0.0.1`` on the fixed Codex redirect port. The redirect target is
    registered as ``http://localhost:1455/auth/callback``; ``localhost`` resolves
    to the loopback address, so binding ``127.0.0.1`` receives it.
    """

    def __init__(self, port: int) -> None:
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._future: asyncio.Future[tuple[str, str | None]] | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._future = loop.create_future()
        try:
            self._server = await asyncio.start_server(
                self._handle, host="127.0.0.1", port=self._port
            )
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise OpenAISignInError(
                    f"Port {self._port} is already in use, so the ChatGPT "
                    "sign-in redirect cannot be captured. Close any running "
                    "Codex/ChatGPT login (or whatever owns that port) and retry."
                ) from exc
            raise

    async def wait(self) -> tuple[str, str | None]:
        assert self._future is not None and self._server is not None
        try:
            return await asyncio.wait_for(self._future, timeout=_LOGIN_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            raise OpenAISignInError(
                "Timed out waiting for the ChatGPT sign-in to complete."
            ) from exc
        finally:
            self._server.close()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        future = self._future
        assert future is not None
        try:
            request_line = await reader.readline()
            while True:
                line = await reader.readline()
                if line in _HEADER_TERMINATORS:
                    break
            parts = request_line.split(b" ", 2)
            if len(parts) < _MIN_REQUEST_LINE_PARTS:
                await self._fail(writer, future)
                return
            query = urllib.parse.urlparse(parts[1].decode("latin-1")).query
            params = urllib.parse.parse_qs(query)
            code = (params.get("code") or [""])[0]
            state = (params.get("state") or [None])[0]
            if not code:
                await self._fail(writer, future)
                return
            writer.write(_http_response(b"HTTP/1.1 200 OK\r\n", _SUCCESS_BODY))
            await writer.drain()
            if not future.done():
                future.set_result((code, state))
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
        finally:
            writer.close()

    async def _fail(
        self,
        writer: asyncio.StreamWriter,
        future: asyncio.Future[tuple[str, str | None]],
    ) -> None:
        writer.write(_http_response(b"HTTP/1.1 400 Bad Request\r\n", _ERROR_BODY))
        await writer.drain()
        if not future.done():
            future.set_exception(
                OpenAISignInError("Authorization server returned no code.")
            )


@dataclass
class OpenAISignInService:
    open_browser: BrowserOpener = field(default=webbrowser.open)
    now: NowFn = field(default=lambda: datetime.now(UTC))
    # Fixed by the Codex client registration in production; overridable in tests.
    port: int = OPENAI_OAUTH_REDIRECT_PORT

    async def authenticate(
        self, *, on_url: UrlCallback | None = None
    ) -> OpenAIOAuthTokens:
        """Run the full sign-in and return (and persist) the resulting tokens."""
        verifier, challenge = generate_pkce_pair()
        state = generate_state()
        authorize_url = build_authorize_url(code_challenge=challenge, state=state)

        server = _LoopbackCallbackServer(self.port)
        # Bind before opening the browser so the redirect is never missed.
        await server.start()

        if on_url is not None:
            on_url(authorize_url)
        self._open_browser(authorize_url)

        code, returned_state = await server.wait()
        if returned_state != state:
            raise OpenAISignInError(
                "OAuth state mismatch; sign-in aborted to prevent a CSRF replay."
            )

        tokens = await exchange_code_for_tokens(code, verifier, now=self.now())
        save_tokens(tokens)
        return tokens

    def _open_browser(self, url: str) -> None:
        try:
            self.open_browser(url)
        except Exception:
            logger.debug("Failed to open browser for ChatGPT sign-in", exc_info=True)
