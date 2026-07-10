from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
import re


class MemorySignalKind(StrEnum):
    EXPLICIT_REMEMBER = auto()
    EXPLICIT_FORGET = auto()
    USER_PREFERENCE = auto()
    USER_CORRECTION = auto()
    DURABLE_DECISION = auto()


@dataclass(frozen=True, slots=True)
class MemorySignal:
    kind: MemorySignalKind
    evidence: str
    message_index: int


_SIGNAL_PATTERNS: tuple[tuple[MemorySignalKind, re.Pattern[str]], ...] = (
    (
        MemorySignalKind.EXPLICIT_FORGET,
        re.compile(
            r"\b(?:forget|do not remember|don't remember|remove that from memory)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemorySignalKind.EXPLICIT_REMEMBER,
        re.compile(
            r"\b(?:remember(?: that)?|please remember|keep (?:this|that) in mind)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemorySignalKind.USER_PREFERENCE,
        re.compile(
            r"\b(?:i prefer|my preference is|when working with me|please always|please never)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MemorySignalKind.USER_CORRECTION,
        re.compile(
            r"(?:^|[.!?]\s+)(?:actually|correction:|to correct that|that's not right)",
            re.IGNORECASE,
        ),
    ),
    (
        MemorySignalKind.DURABLE_DECISION,
        re.compile(
            r"\b(?:we decided(?: that)?|the decision is|from now on we|going forward we)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_memory_signals(text: str, *, message_index: int) -> tuple[MemorySignal, ...]:
    evidence = " ".join(text.strip().split())[:1_200]
    if not evidence:
        return ()
    detected: list[MemorySignal] = []
    for kind, pattern in _SIGNAL_PATTERNS:
        if pattern.search(evidence) is not None:
            detected.append(
                MemorySignal(kind=kind, evidence=evidence, message_index=message_index)
            )
    return tuple(detected)


def extractable_signals(signals: tuple[MemorySignal, ...]) -> tuple[MemorySignal, ...]:
    return tuple(
        signal
        for signal in signals
        if signal.kind is not MemorySignalKind.EXPLICIT_FORGET
    )


__all__ = [
    "MemorySignal",
    "MemorySignalKind",
    "detect_memory_signals",
    "extractable_signals",
]
