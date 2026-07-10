from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TrustedVerificationCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float = Field(default=300.0, gt=0, le=3_600)

    @field_validator("name", "cwd")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip() or "\0" in value:
            raise ValueError("verification check text must be nonempty")
        return value

    @field_validator("argv")
    @classmethod
    def _validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument or "\0" in argument for argument in value):
            raise ValueError("verification check argv entries must be nonempty")
        return value


class TrustedVerificationRecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe_version: str = Field(min_length=1)
    task_brief: str = Field(min_length=1)
    acceptance_contract: str = Field(min_length=1)
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    checks: tuple[TrustedVerificationCheckConfig, ...] = Field(min_length=1)

    @field_validator("recipe_version", "task_brief", "acceptance_contract")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trusted verification recipe text must be nonempty")
        return value

    @field_validator("allowed_paths")
    @classmethod
    def _validate_allowed_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            normalized = pattern.strip().replace("\\", "/")
            path = PurePosixPath(normalized)
            if not normalized or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"invalid allowed-path pattern: {pattern!r}")
        return value

    @model_validator(mode="after")
    def _validate_check_names(self) -> TrustedVerificationRecipeConfig:
        names = [check.name for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("trusted verification check names must be unique")
        return self


__all__ = ["TrustedVerificationCheckConfig", "TrustedVerificationRecipeConfig"]
