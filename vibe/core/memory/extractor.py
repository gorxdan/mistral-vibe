"""LLM memory extractor: proposes durable memories from a turn transcript.

Built on its OWN standalone backend (like MemorySelector / SafetyJudge), never
the agent's main backend, so an extraction failure can never trigger model
failover or emergency compaction. Fails to an EMPTY proposal on any
error/timeout — extraction is best-effort and must never break a session.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.logger import logger
from vibe.core.memory._llm_client import _MemoryLLMClient
from vibe.core.memory.models import MemoryType
from vibe.core.types import LLMMessage, Role
from vibe.core.usage import CallKind, SpendPurpose, UsageMeter
from vibe.core.usage._session import SessionSpendAdapter

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

Do NOT save one-shot project status: closed audits, "PR is merge-ready",
"cards resolved", pre-existing test footnotes, or CI notes that will not
matter next session. Project memories must be resume pointers (ongoing work,
decisions, deadlines) — not a diary of finished tasks.

Every create MUST include type AND a specific description (<=300 chars). Title
should be short (a few words); the description carries the recall key. Prefer
updating an existing memory over creating a near-duplicate.

When the transcript REFINES or CORRECTS a memory already in the index, emit
"action": "update" with "id" set to that EXISTING memory's id (copy it exactly
from the index) and "body" holding only the new/changed detail. Otherwise omit
"action" (it defaults to "create"). Never invent an id for a create, and never
reuse a create's slug as an update target — an update must name a real id from
the index or it is dropped.

Return ONLY JSON: {"memories": [{"title": "...", "description": "...", \
"type": "user|feedback|project|reference", "tags": ["..."], "body": "...", \
"action": "create|update", "id": "<existing id, update only>"}]}.
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
    # "create" writes a new memory (default); "update" merges into the memory
    # whose id matches `id` (must already exist or the proposal is dropped).
    action: Literal["create", "update"] = "create"
    id: str | None = None

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


def merge_memory_body(existing: str, addition: str, today: str) -> str:
    # Append the new detail as a dated addendum so an update never destroys the
    # prior text. A future consolidation pass can reconcile the combined body;
    # until then the history stays recoverable and auditable.
    add = (addition or "").strip()
    if not add:
        return existing
    return f"{(existing or '').rstrip()}\n\n--- Updated {today} ---\n{add}"


class MemoryExtractor(_MemoryLLMClient):
    def __init__(
        self,
        *,
        model: ModelConfig,
        provider: ProviderConfig,
        timeout: float = 30.0,
        usage_meter: UsageMeter | None = None,
        spend_adapter: SessionSpendAdapter | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            provider=provider,
            timeout=timeout,
            call_kind=CallKind.MEMORY_EXTRACT,
            spend_purpose=SpendPurpose.MEMORY_EXTRACT,
            usage_meter=usage_meter,
            spend_adapter=spend_adapter,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

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
            LLMMessage(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
            LLMMessage(role=Role.USER, content=user_content),
        ]
        return await self._complete_json(
            messages, max_tokens=1024, temperature=self._model.temperature
        )

    def _parse(self, content: str | None) -> list[ExtractedMemory]:
        text = (content or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            data = orjson.loads(text[start : end + 1])
        except (orjson.JSONDecodeError, ValueError):
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
