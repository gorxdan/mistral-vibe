"""Deterministic local-first selection for durable memories."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import re

from vibe.core.memory.models import MemoryEntry

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DESCRIPTION_PHRASE_MAX_TERMS = 4
_STOP_WORDS = frozenset({
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "memory",
    "my",
    "of",
    "on",
    "or",
    "please",
    "remember",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
})


@dataclass(frozen=True, slots=True)
class LocalMemorySelection:
    ids: tuple[str, ...]
    ambiguous: bool
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _MemoryRecord:
    id: str
    title: str
    description: str
    tags: tuple[str, ...]
    index_line: str
    position: int


class LocalMemorySelector:
    def __init__(
        self, *, max_selected: int, min_score: float, ambiguity_margin: float
    ) -> None:
        self._max_selected = max_selected
        self._min_score = min_score
        self._ambiguity_margin = ambiguity_margin

    def select(
        self,
        entries: list[MemoryEntry],
        user_message: str,
        *,
        already_surfaced: set[str] | None = None,
    ) -> LocalMemorySelection:
        records = tuple(
            _MemoryRecord(
                id=entry.id,
                title=entry.metadata.title,
                description=entry.metadata.description,
                tags=tuple(entry.metadata.tags),
                index_line=entry.index_line(),
                position=position,
            )
            for position, entry in enumerate(entries)
        )
        fingerprint = _fingerprint(records)
        query = " ".join(_tokens(user_message))
        surfaced = tuple(sorted(already_surfaced or ()))
        ids, ambiguous = _select_cached(
            query,
            fingerprint,
            records,
            surfaced,
            self._max_selected,
            self._min_score,
            self._ambiguity_margin,
        )
        return LocalMemorySelection(
            ids=ids, ambiguous=ambiguous, fingerprint=fingerprint
        )


def _fingerprint(records: tuple[_MemoryRecord, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.index_line.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token
            for token in _TOKEN_RE.findall(text.lower())
            if token not in _STOP_WORDS
        )
    )


def _field_tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _score_record(
    record: _MemoryRecord,
    query_terms: tuple[str, ...],
    query: str,
    inverse_frequency: dict[str, float],
) -> float:
    fields = (
        (_field_tokens(record.id), 8.0),
        (_field_tokens(record.title), 6.0),
        (_field_tokens(" ".join(record.tags)), 5.0),
        (_field_tokens(record.description), 3.0),
        (_field_tokens(record.index_line), 1.0),
    )
    score = 0.0
    for term in query_terms:
        weight = max(
            (weight for tokens, weight in fields if term in tokens), default=0.0
        )
        score += weight * inverse_frequency.get(term, 1.0)

    normalized_id = " ".join(_tokens(record.id))
    normalized_title = " ".join(_tokens(record.title))
    normalized_tags = tuple(" ".join(_tokens(tag)) for tag in record.tags)
    if _contains_phrase(query, normalized_id):
        score += 8.0
    if _contains_phrase(query, normalized_title):
        score += 6.0
    score += sum(4.0 for tag in normalized_tags if _contains_phrase(query, tag))
    if len(query_terms) <= _DESCRIPTION_PHRASE_MAX_TERMS:
        normalized_description = " ".join(_tokens(record.description))
        if _contains_phrase(normalized_description, query):
            score += 3.0
    return score


@lru_cache(maxsize=256)
def _select_cached(
    query: str,
    fingerprint: str,
    records: tuple[_MemoryRecord, ...],
    surfaced: tuple[str, ...],
    max_selected: int,
    min_score: float,
    ambiguity_margin: float,
) -> tuple[tuple[str, ...], bool]:
    query_terms = _tokens(query)
    if not query_terms or not records or max_selected <= 0:
        return (), False

    document_frequency = {
        term: sum(term in _field_tokens(record.index_line) for record in records)
        for term in query_terms
    }
    inverse_frequency = {
        term: 1.0 + math.log((len(records) + 1) / (frequency + 1))
        for term, frequency in document_frequency.items()
    }
    surfaced_set = frozenset(surfaced)
    scored: list[tuple[str, float, int, bool]] = []
    for record in records:
        raw_score = _score_record(record, query_terms, query, inverse_frequency)
        if raw_score < min_score:
            continue
        was_surfaced = record.id in surfaced_set
        rank_score = raw_score * (0.9 if was_surfaced else 1.0)
        scored.append((record.id, rank_score, record.position, was_surfaced))

    scored.sort(key=lambda item: (-item[1], item[3], item[2], item[0]))
    selected = tuple(item[0] for item in scored[:max_selected])
    if len(scored) <= max_selected:
        return selected, False

    cutoff_score = scored[max_selected - 1][1]
    next_score = scored[max_selected][1]
    relative_gap = (cutoff_score - next_score) / max(cutoff_score, 1.0)
    return selected, relative_gap <= ambiguity_margin
