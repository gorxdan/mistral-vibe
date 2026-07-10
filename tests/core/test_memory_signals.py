from __future__ import annotations

from vibe.core.config import ModelPurpose, PurposeModelRoutingConfig
from vibe.core.memory._signals import (
    MemorySignalKind,
    detect_memory_signals,
    extractable_signals,
)


def test_detects_explicit_durable_signals() -> None:
    signals = detect_memory_signals(
        "Actually, please remember that I prefer concise status updates.",
        message_index=7,
    )

    assert {signal.kind for signal in signals} == {
        MemorySignalKind.EXPLICIT_REMEMBER,
        MemorySignalKind.USER_PREFERENCE,
        MemorySignalKind.USER_CORRECTION,
    }
    assert all(signal.message_index == 7 for signal in signals)


def test_routine_task_text_produces_no_signal() -> None:
    assert (
        detect_memory_signals(
            "Run the parser tests and fix the failure.", message_index=2
        )
        == ()
    )


def test_forget_intent_never_enters_extraction() -> None:
    signals = detect_memory_signals(
        "Forget that preference and remove that from memory.", message_index=3
    )

    assert {signal.kind for signal in signals} == {MemorySignalKind.EXPLICIT_FORGET}
    assert extractable_signals(signals) == ()


def test_model_routing_requires_explicit_aliases() -> None:
    routing = PurposeModelRoutingConfig(
        formatter_model=" cheap ",
        retrieval_model="reranker",
        mechanical_model="grunt",
        semantic_escalation_model="strong",
    )

    assert routing.alias_for(ModelPurpose.FORMATTER) == "cheap"
    assert routing.alias_for(ModelPurpose.RETRIEVAL) == "reranker"
    assert routing.alias_for(ModelPurpose.MECHANICAL) == "grunt"
    assert routing.alias_for(ModelPurpose.SEMANTIC_ESCALATION) == "strong"
    assert PurposeModelRoutingConfig().alias_for(ModelPurpose.RETRIEVAL) is None
