from __future__ import annotations

from typing import Any


def apply_openai_chat_thinking(
    payload: dict[str, Any], *, provider_name: str, level: str
) -> bool:
    if provider_name != "longcat":
        return False

    if "thinking" not in payload and "reasoning_effort" not in payload:
        payload["thinking"] = {"type": "disabled" if level == "off" else "enabled"}
    return True
