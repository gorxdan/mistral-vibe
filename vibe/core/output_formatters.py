from __future__ import annotations

import sys
from typing import TextIO

import orjson

from vibe.core.output_formatters_port import OutputFormatter
from vibe.core.teleport.types import (
    TeleportCheckingGitEvent,
    TeleportCompleteEvent,
    TeleportPushingEvent,
    TeleportPushRequiredEvent,
    TeleportStartingWorkflowEvent,
)
from vibe.core.types import AssistantEvent, BaseEvent, LLMMessage, OutputFormat

__all__ = ["OutputFormatter", "create_formatter"]


class TextOutputFormatter(OutputFormatter):
    def on_message_added(self, message: LLMMessage) -> None:
        self._messages.append(message)

    def _print(self, text: str) -> None:
        print(text, file=self.stream)

    def on_event(self, event: BaseEvent) -> None:
        match event:
            case AssistantEvent():
                self._final_response = event.content
            case TeleportCheckingGitEvent():
                self._print("Preparing workspace...")
            case TeleportPushRequiredEvent(unpushed_count=count):
                self._print(f"Pushing {count} commit(s)...")
            case TeleportPushingEvent():
                self._print("Syncing with remote...")
            case TeleportStartingWorkflowEvent():
                self._print("Teleporting...")
            case TeleportCompleteEvent():
                self._final_response = event.url

    def finalize(self) -> str | None:
        return self._final_response


class JsonOutputFormatter(OutputFormatter):
    def on_message_added(self, message: LLMMessage) -> None:
        self._messages.append(message)

    def on_event(self, event: BaseEvent) -> None:
        pass

    def finalize(self) -> str | None:
        messages_data = [msg.model_dump(mode="json") for msg in self._messages]
        self.stream.write(
            orjson.dumps(messages_data, option=orjson.OPT_INDENT_2).decode("utf-8")
        )
        self.stream.write("\n")
        self.stream.flush()
        return None


class StreamingJsonOutputFormatter(OutputFormatter):
    def on_message_added(self, message: LLMMessage) -> None:
        self.stream.write(orjson.dumps(message.model_dump(mode="json")).decode("utf-8"))
        self.stream.write("\n")
        self.stream.flush()

    def on_event(self, event: BaseEvent) -> None:
        pass

    def finalize(self) -> str | None:
        return None


def create_formatter(
    format_type: OutputFormat, stream: TextIO = sys.stdout
) -> OutputFormatter:
    formatters = {
        OutputFormat.TEXT: TextOutputFormatter,
        OutputFormat.JSON: JsonOutputFormatter,
        OutputFormat.STREAMING: StreamingJsonOutputFormatter,
    }

    formatter_class = formatters.get(format_type, TextOutputFormatter)
    return formatter_class(stream)
