from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol


@dataclass
class BrowserSignInProcess:
    process_id: str
    sign_in_url: str
    poll_url: str
    expires_at: datetime


@dataclass
class BrowserSignInPollResult:
    status: Literal["pending", "completed", "expired", "denied", "error"]
    exchange_token: str | None = None
    message: str | None = None


class BrowserSignInGateway(Protocol):
    async def create_process(self, code_challenge: str) -> BrowserSignInProcess: ...

    async def poll(self, poll_url: str) -> BrowserSignInPollResult: ...

    async def exchange(
        self, process_id: str, exchange_token: str, code_verifier: str
    ) -> str: ...

    async def aclose(self) -> None: ...
