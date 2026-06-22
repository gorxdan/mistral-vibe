from __future__ import annotations

from enum import StrEnum, auto

from vibe.setup.auth.browser_sign_in_gateway_port import (
    BrowserSignInGateway,
    BrowserSignInPollResult,
    BrowserSignInProcess,
)

__all__ = [
    "BrowserSignInError",
    "BrowserSignInErrorCode",
    "BrowserSignInGateway",
    "BrowserSignInPollResult",
    "BrowserSignInProcess",
]


class BrowserSignInErrorCode(StrEnum):
    START_FAILED = auto()
    POLL_FAILED = auto()
    UNKNOWN_STATE = auto()
    EXCHANGE_FAILED = auto()
    MISSING_API_KEY = auto()
    MISSING_EXCHANGE_TOKEN = auto()
    EXPIRED = auto()
    DENIED = auto()
    PROVIDER_ERROR = auto()
    TIMED_OUT = auto()
    OPEN_BROWSER_FAILED = auto()


class BrowserSignInError(Exception):
    def __init__(
        self, message: str, *, code: BrowserSignInErrorCode | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
