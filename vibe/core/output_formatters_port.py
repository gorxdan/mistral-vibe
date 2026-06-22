from __future__ import annotations

from abc import ABC, abstractmethod
import sys
from typing import TextIO

from vibe.core.types import BaseEvent, LLMMessage


class OutputFormatter(ABC):
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self._messages: list[LLMMessage] = []
        self._final_response: str | None = None

    @abstractmethod
    def on_message_added(self, message: LLMMessage) -> None:
        pass

    @abstractmethod
    def on_event(self, event: BaseEvent) -> None:
        pass

    @abstractmethod
    def finalize(self) -> str | None:
        """Finalize output and return any final text to be printed.

        Returns:
            String to print, or None if formatter handles its own output
        """
        pass
