from __future__ import annotations

from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, field_validator


class ModelPurpose(StrEnum):
    FORMATTER = auto()
    RETRIEVAL = auto()
    MECHANICAL = auto()
    SEMANTIC_ESCALATION = auto()


class PurposeModelRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    formatter_model: str = ""
    retrieval_model: str = ""
    mechanical_model: str = ""
    semantic_escalation_model: str = ""

    @field_validator(
        "formatter_model",
        "retrieval_model",
        "mechanical_model",
        "semantic_escalation_model",
    )
    @classmethod
    def normalize_alias(cls, value: str) -> str:
        return value.strip()

    def alias_for(self, purpose: ModelPurpose) -> str | None:
        match purpose:
            case ModelPurpose.FORMATTER:
                alias = self.formatter_model
            case ModelPurpose.RETRIEVAL:
                alias = self.retrieval_model
            case ModelPurpose.MECHANICAL:
                alias = self.mechanical_model
            case ModelPurpose.SEMANTIC_ESCALATION:
                alias = self.semantic_escalation_model
        return alias or None


__all__ = ["ModelPurpose", "PurposeModelRoutingConfig"]
