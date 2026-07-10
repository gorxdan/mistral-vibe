from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import orjson

from vibe.core.tools.tool_result_store import truncate_middle_chars
from vibe.core.utils.io import write_safe

BACKGROUND_COMPLETION_PREVIEW_CHARS = 4_000
BACKGROUND_BATCH_PREVIEW_CHARS = 12_000
BACKGROUND_ARTIFACT_RECORD_CHARS = 384
BACKGROUND_OUTCOME_RECORD_CHARS = 1_500
_BACKGROUND_PATH_CHARS = 128
_ARTIFACT_PREFIX = "BACKGROUND_ARTIFACT_JSON:"
_OUTCOME_PREFIX = "TASK_OUTCOME_JSON:"
_RESERVED_BACKGROUND_PREFIXES = ("BACKGROUND_ARTIFACT_JSON:", "TASK_OUTCOME_JSON:")


@dataclass(frozen=True, slots=True)
class BackgroundArtifact:
    path: str | None
    sha256: str
    size_bytes: int


def prepare_background_completion(
    response: str, path: Path | None
) -> tuple[str, BackgroundArtifact]:
    encoded = response.encode("utf-8")
    persisted_path: Path | None = None
    if path is not None:
        try:
            write_safe(path, response)
        except OSError:
            pass
        else:
            persisted_path = path
    artifact = BackgroundArtifact(
        path=str(persisted_path) if persisted_path is not None else None,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )
    if len(response) <= BACKGROUND_COMPLETION_PREVIEW_CHARS:
        return response, artifact

    preview = truncate_middle_chars(response, BACKGROUND_COMPLETION_PREVIEW_CHARS)
    if persisted_path is None:
        return (
            f"{preview}\n\n"
            f"...[Background result truncated from {len(response):,} characters; "
            "full output could not be persisted.]",
            artifact,
        )
    return (
        f"{preview}\n\n"
        f"...[Full background result ({len(response):,} characters) persisted to "
        f"{persisted_path}; use the `read` tool to retrieve it.]",
        artifact,
    )


def compact_background_completion(response: str, path: Path | None) -> str:
    return prepare_background_completion(response, path)[0]


def escape_background_body(response: str) -> str:
    return "\n".join(
        f"WORKER_TEXT_{line}"
        if line.startswith(_RESERVED_BACKGROUND_PREFIXES)
        else line
        for line in response.splitlines()
    )


def format_task_outcome_record(task_id: str, outcome: dict[str, Any]) -> str:
    raw = orjson.dumps(outcome, option=orjson.OPT_SORT_KEYS)
    payload = {
        "diagnostics": _bounded_items(outcome.get("diagnostics")),
        "digest": hashlib.sha256(raw).hexdigest(),
        "evidence": _bounded_items(outcome.get("evidence")),
        "status": str(outcome.get("status", "unknown")),
        "summary": str(outcome.get("summary", "")),
        "task_id": task_id,
        "truncated": True,
    }
    return _fit_outcome_record(payload, BACKGROUND_OUTCOME_RECORD_CHARS)


def format_background_artifact_record(
    *,
    path: str | None,
    sha256: str | None,
    size_bytes: int | None,
    status: str,
    task_id: str,
) -> str:
    payload = {
        "path": path,
        "sha256": sha256 or "",
        "size_bytes": size_bytes,
        "status": status,
        "task_id": task_id,
    }
    return _fit_artifact_record(payload, BACKGROUND_ARTIFACT_RECORD_CHARS)


def compact_background_batch(sections: Sequence[str]) -> str:
    if not sections:
        return ""
    joined = "\n\n".join(sections)
    if len(joined) <= BACKGROUND_BATCH_PREVIEW_CHARS:
        return joined
    split = [_split_background_section(section) for section in sections]
    separator_chars = 2 * (len(split) - 1)
    section_budget, extra = divmod(
        BACKGROUND_BATCH_PREVIEW_CHARS - separator_chars, len(split)
    )
    bounded = [
        _compact_background_section(
            artifacts, outcomes, body, section_budget + (1 if index < extra else 0)
        )
        for index, (artifacts, outcomes, body) in enumerate(split)
    ]
    return "\n\n".join(bounded)[:BACKGROUND_BATCH_PREVIEW_CHARS]


def _split_background_section(section: str) -> tuple[list[str], list[str], str]:
    artifacts: list[str] = []
    outcomes: list[str] = []
    body: list[str] = []
    for line in section.splitlines():
        if line.startswith("BACKGROUND_ARTIFACT_JSON:"):
            artifacts.append(line)
        elif line.startswith("TASK_OUTCOME_JSON:"):
            outcomes.append(line)
        else:
            body.append(line)
    return artifacts, outcomes, "\n".join(body)


def _compact_background_section(
    artifacts: Sequence[str], outcomes: Sequence[str], body: str, limit: int
) -> str:
    blocks: list[str] = []
    outcome_minimum = _minimal_outcome_lines(outcomes)
    outcome_reserve = len(outcome_minimum) + bool(artifacts and outcomes)
    artifact_limit = max(limit - outcome_reserve, 0) if artifacts else 0
    artifact_block = _bounded_artifact_lines(artifacts, artifact_limit)
    if artifact_block:
        blocks.append(artifact_block)

    used = sum(len(block) for block in blocks) + max(len(blocks) - 1, 0)
    outcome_limit = max(limit - used - bool(blocks and outcomes), 0)
    outcome_block = _bounded_outcome_lines(outcomes, outcome_limit)
    if outcome_block:
        blocks.append(outcome_block)

    used = sum(len(block) for block in blocks) + max(len(blocks) - 1, 0)
    body_limit = max(limit - used - bool(blocks and body), 0)
    if body and body_limit:
        preview = escape_background_body(_bounded_text(body, body_limit))[:body_limit]
        if preview:
            blocks.append(preview)
    return "\n".join(blocks)[:limit]


def _bounded_artifact_lines(lines: Sequence[str], limit: int) -> str:
    if not lines or limit <= 0:
        return ""
    separator_chars = len(lines) - 1
    if separator_chars >= limit:
        return ""
    per_line = (limit - separator_chars) // len(lines)
    bounded = [
        record
        for line in lines
        if (
            record := _fit_artifact_record(
                _record_payload(line, _ARTIFACT_PREFIX), per_line
            )
        )
    ]
    return "\n".join(bounded)[:limit]


def _bounded_outcome_lines(lines: Sequence[str], limit: int) -> str:
    if not lines or limit <= 0:
        return ""
    raw = "\n".join(lines)
    if len(raw) <= limit:
        return raw
    separator_chars = len(lines) - 1
    if separator_chars >= limit:
        return ""
    per_line = (limit - separator_chars) // len(lines)
    bounded: list[str] = []
    for line in lines:
        encoded = line.removeprefix(_OUTCOME_PREFIX)
        payload = _record_payload(line, _OUTCOME_PREFIX)
        payload.setdefault("digest", hashlib.sha256(encoded.encode()).hexdigest())
        payload["truncated"] = True
        record = _fit_outcome_record(payload, per_line)
        if record:
            bounded.append(record)
    return "\n".join(bounded)[:limit]


def _minimal_outcome_lines(lines: Sequence[str]) -> str:
    return "\n".join(
        _minimal_outcome_record(_record_payload(line, _OUTCOME_PREFIX))
        for line in lines
    )


def _record_payload(line: str, prefix: str) -> dict[str, Any]:
    encoded = line.removeprefix(prefix)
    try:
        value = orjson.loads(encoded)
    except orjson.JSONDecodeError:
        return {"digest": hashlib.sha256(encoded.encode()).hexdigest()}
    return value if isinstance(value, dict) else {}


def _fit_artifact_record(payload: dict[str, Any], limit: int) -> str:
    raw_path = payload.get("path")
    path = str(raw_path) if raw_path is not None else None
    value: dict[str, Any] = {
        "path": path if path is None or len(path) <= _BACKGROUND_PATH_CHARS else None,
        "sha256": str(payload.get("sha256", ""))[:64],
        "size_bytes": payload.get("size_bytes"),
        "status": str(payload.get("status", ""))[:24],
        "task_id": str(payload.get("task_id", ""))[:48],
    }
    if path is not None and value["path"] is None:
        value["path_sha256"] = hashlib.sha256(path.encode()).hexdigest()

    def render() -> str:
        return _ARTIFACT_PREFIX + orjson.dumps(value).decode()

    record = render()
    if len(record) <= limit:
        return record
    value.pop("status", None)
    record = render()
    if len(record) <= limit:
        return record
    if value.get("path") is not None:
        value["path_sha256"] = hashlib.sha256(str(value["path"]).encode()).hexdigest()
        value["path"] = None
        record = render()
    return record if len(record) <= limit else ""


def _fit_outcome_record(payload: dict[str, Any], limit: int) -> str:
    value: dict[str, Any] = {
        "diagnostics": _bounded_items(payload.get("diagnostics")),
        "digest": str(payload.get("digest", ""))[:64],
        "evidence": _bounded_items(payload.get("evidence")),
        "status": str(payload.get("status", "unknown"))[:32],
        "summary": str(payload.get("summary", ""))[:512],
        "task_id": str(payload.get("task_id", ""))[:128],
        "truncated": bool(payload.get("truncated", True)),
    }

    def render() -> str:
        return _OUTCOME_PREFIX + orjson.dumps(value).decode()

    record = render()
    while len(record) > limit:
        if value["evidence"]:
            value["evidence"].pop()
        elif value["diagnostics"]:
            value["diagnostics"].pop()
        elif value["summary"]:
            value["summary"] = value["summary"][: len(value["summary"]) // 2]
        else:
            return _minimal_outcome_record(value, limit=limit)
        record = render()
    return record


def _minimal_outcome_record(
    payload: dict[str, Any], *, limit: int | None = None
) -> str:
    value = {
        "digest": str(payload.get("digest", ""))[:64],
        "status": str(payload.get("status", "unknown"))[:24],
        "task_id": str(payload.get("task_id", ""))[:48],
    }
    record = _OUTCOME_PREFIX + orjson.dumps(value).decode()
    return record if limit is None or len(record) <= limit else ""


def _bounded_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return [] if value is None or value == "" else [str(value)[:256]]
    return [str(item)[:256] for item in value[:4]]


def _bounded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    if limit <= len(marker):
        return text[:limit]
    content_chars = limit - len(marker)
    head = content_chars * 3 // 4
    tail = content_chars - head
    return f"{text[:head]}{marker}{text[-tail:]}"


__all__ = [
    "BACKGROUND_ARTIFACT_RECORD_CHARS",
    "BACKGROUND_BATCH_PREVIEW_CHARS",
    "BACKGROUND_COMPLETION_PREVIEW_CHARS",
    "BACKGROUND_OUTCOME_RECORD_CHARS",
    "BackgroundArtifact",
    "compact_background_batch",
    "compact_background_completion",
    "escape_background_body",
    "format_background_artifact_record",
    "format_task_outcome_record",
    "prepare_background_completion",
]
