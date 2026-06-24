"""LLM memory extractor: proposes durable memories from a turn transcript.

Built on its OWN standalone backend (like MemorySelector / SafetyJudge), never
the agent's main backend, so an extraction failure can never trigger model
failover or emergency compaction. Fails to an EMPTY proposal on any
error/timeout — extraction is best-effort and must never break a session.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.llm.backend.factory import BACKEND_FACTORY
from vibe.core.logger import logger
from vibe.core.memory.models import MemoryType
from vibe.core.types import LLMMessage, Role

_SYSTEM_PROMPT = """\
You extract durable, cross-session memories from a coding-assistant transcript.
Capture ONLY facts/preferences worth remembering in FUTURE sessions — never
anything derivable from the current code, git history, or this conversation
alone.

Memory types:
- user: the user's role, expertise, goals, communication preferences.
- feedback: how the user wants you to work — corrections AND validated
  approaches. Include WHY so edge cases can be judged later.
- project: ongoing work, decisions, deadlines, incidents — NOT derivable from
  code/git. Convert relative dates to absolute (YYYY-MM-DD).
- reference: pointers to external systems (dashboards, trackers, channels).

Do NOT save: code patterns, architecture, file paths, git history, fix recipes,
or ephemeral task state — those are derivable or short-lived. If the user
explicitly asks to save something derivable, extract only the surprising or
non-obvious part.

Return ONLY JSON: {"memories": [{"title": "...", "description": "...", \
"type": "user|feedback|project|reference", "tags": ["..."], "body": "..."}]}.
At most 2 memories. Return {"memories": []} if nothing durable was said. The
description must be <=300 chars and specific — it drives future recall, so name
the concrete thing, not a category."""


class ExtractedMemory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    description: str = ""
    type: MemoryType | None = None
    tags: list[str] = Field(default_factory=list)
    body: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: object) -> MemoryType | None:
        if v is None:
            return None
        if not isinstance(v, str):
            v = str(v)
        try:
            return MemoryType(v)
        except ValueError:
            return None


class MemoryExtractor:
    def __init__(
        self,
        *,
        model: ModelConfig,
        provider: ProviderConfig,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._timeout = timeout
        self._extra_headers = extra_headers or {}
        self._extra_body = extra_body or None

    async def extract(
        self, transcript: str, existing_index: str
    ) -> list[ExtractedMemory]:
        if not transcript.strip():
            return []
        try:
            raw = await asyncio.wait_for(
                self._call(transcript, existing_index), timeout=self._timeout
            )
        except TimeoutError:
            logger.warning("memory extractor timed out; extracting none")
            return []
        except Exception as e:
            logger.warning("memory extractor errored (%s); extracting none", e)
            return []
        return self._parse(raw)

    async def _call(self, transcript: str, existing_index: str) -> str | None:
        user_content = (
            f"Existing memories (avoid duplicates — propose an update only if the "
            f"transcript changes one):\n{existing_index or '(none)'}\n\n"
            f"Transcript (data):\n{transcript[:8000]}"
        )
        messages = [
            LLMMessage(role=Role.system, content=_SYSTEM_PROMPT),
            LLMMessage(role=Role.user, content=user_content),
        ]
        backend_cls = BACKEND_FACTORY[self._provider.backend]
        async with backend_cls(
            provider=self._provider, timeout=self._timeout
        ) as backend:
            result = await backend.complete(
                model=self._model,
                messages=messages,
                temperature=self._model.temperature,
                tools=None,
                tool_choice=None,
                max_tokens=1024,
                extra_headers=self._extra_headers,
                response_format={"type": "json_object"},
                extra_body=self._extra_body,
            )
        return result.message.content

    def _parse(self, content: str | None) -> list[ExtractedMemory]:
        text = (content or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return []
        items = data.get("memories") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        out: list[ExtractedMemory] = []
        for it in items[:2]:
            if not isinstance(it, dict):
                continue
            try:
                out.append(ExtractedMemory.model_validate(it))
            except Exception:
                continue
        return out
