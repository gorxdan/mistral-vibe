from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from vibe.cli.turn_summary.models import TurnSummaryResult
from vibe.core.types import BaseEvent


class TurnSummaryPort(ABC):
    @property
    @abstractmethod
    def generation(self) -> int: ...

    @property
    @abstractmethod
    def on_summary(self) -> Callable[[TurnSummaryResult], None] | None: ...

    @on_summary.setter
    @abstractmethod
    def on_summary(self, value: Callable[[TurnSummaryResult], None] | None) -> None: ...

    @abstractmethod
    def start_turn(self, user_message: str) -> None: ...

    @abstractmethod
    def track(self, event: BaseEvent) -> None: ...

    @abstractmethod
    def set_error(self, message: str) -> None: ...

    @abstractmethod
    def cancel_turn(self) -> None: ...

    @abstractmethod
    def end_turn(self) -> Callable[[], bool] | None: ...

    @abstractmethod
    async def close(self) -> None: ...
