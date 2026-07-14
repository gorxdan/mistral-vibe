from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core._trusted_command import validate_trusted_command_argv
from vibe.core._verification_output import (
    validate_custom_runner_contract,
    validate_output_patterns,
)

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class TrustedExecutionTopologyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str = Field(min_length=1)
    packet_path: str = Field(min_length=1)
    status_path: str = "docs/design/fork-maintenance/status.yaml"
    state: Literal["active", "verification"]
    control_worktree: str = Field(min_length=1)
    control_sha: str = Field(min_length=1)
    candidate_worktree: str = Field(min_length=1)
    candidate_branch: str = Field(min_length=1)
    baseline_sha: str = Field(min_length=1)
    candidate_sha: str | None = None
    upstream_sha: str = Field(min_length=1)
    evidence_workspace: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    runner_id: str = Field(min_length=1)
    evidence_manifest_sha256: str | None = None
    max_turns: int = Field(default=80, gt=0, le=200)
    max_session_tokens: int = Field(default=2_000_000, gt=0, le=10_000_000)

    @field_validator("packet_id", "run_id", "runner_id", mode="after")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if (
            not normalized
            or normalized in {".", ".."}
            or "\0" in normalized
            or "/" in normalized
        ):
            raise ValueError(
                "execution topology identifiers must be nonempty path segments"
            )
        return normalized

    @field_validator("candidate_branch")
    @classmethod
    def _validate_branch(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\0" in normalized:
            raise ValueError("candidate branch must be nonempty")
        return normalized

    @field_validator("packet_path", "status_path")
    @classmethod
    def _validate_repository_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        path = PurePosixPath(normalized)
        if not normalized or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"invalid repository-relative topology path: {value!r}")
        return normalized

    @field_validator("control_worktree", "candidate_worktree", "evidence_workspace")
    @classmethod
    def _validate_absolute_path(cls, value: str) -> str:
        from pathlib import Path

        normalized = value.strip()
        if not normalized or "\0" in normalized or not Path(normalized).is_absolute():
            raise ValueError("execution topology paths must be absolute")
        return normalized

    @field_validator("control_sha", "baseline_sha", "upstream_sha")
    @classmethod
    def _validate_required_sha(cls, value: str) -> str:
        normalized = value.strip().lower()
        if _FULL_SHA.fullmatch(normalized) is None:
            raise ValueError("execution topology SHAs must be full 40-character hex")
        return normalized

    @field_validator("candidate_sha")
    @classmethod
    def _validate_candidate_sha(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if _FULL_SHA.fullmatch(normalized) is None:
            raise ValueError("execution topology SHAs must be full 40-character hex")
        return normalized

    @field_validator("evidence_manifest_sha256")
    @classmethod
    def _validate_evidence_manifest_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if _SHA256.fullmatch(normalized) is None:
            raise ValueError(
                "evidence_manifest_sha256 must be a full lowercase SHA-256"
            )
        return normalized

    @model_validator(mode="after")
    def _validate_state_identity(self) -> TrustedExecutionTopologyConfig:
        if self.state == "active" and self.candidate_sha is not None:
            raise ValueError("active topology must not predeclare candidate_sha")
        if self.state == "active" and self.evidence_manifest_sha256 is not None:
            raise ValueError(
                "active topology must not predeclare evidence_manifest_sha256"
            )
        if self.state == "verification" and self.candidate_sha is None:
            raise ValueError("verification topology requires candidate_sha")
        if self.state == "verification" and self.evidence_manifest_sha256 is None:
            raise ValueError("verification topology requires evidence_manifest_sha256")
        return self


class TrustedVerificationCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float = Field(default=300.0, gt=0, le=3_600)
    executable_sha256: str | None = None
    required_output_patterns: tuple[str, ...] = ()
    forbidden_output_patterns: tuple[str, ...] = ()
    test_count_pattern: str | None = None
    minimum_test_count: int | None = Field(default=None, ge=1)
    custom_runner: bool = False
    environment_attestation_path: str | None = None
    environment_attestation_sha256: str | None = None

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
        validate_trusted_command_argv(value)
        return value

    @field_validator("executable_sha256")
    @classmethod
    def _validate_executable_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _SHA256.fullmatch(value) is None:
            raise ValueError("executable_sha256 must be a full lowercase SHA-256")
        return value

    @field_validator("environment_attestation_path")
    @classmethod
    def _validate_environment_attestation_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if (
            not normalized
            or "\0" in normalized
            or not PurePosixPath(normalized).is_absolute()
        ):
            raise ValueError("environment_attestation_path must be an absolute path")
        return normalized

    @field_validator("environment_attestation_sha256")
    @classmethod
    def _validate_environment_attestation_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _SHA256.fullmatch(value) is None:
            raise ValueError(
                "environment_attestation_sha256 must be a full lowercase SHA-256"
            )
        return value

    @field_validator(
        "required_output_patterns", "forbidden_output_patterns", mode="after"
    )
    @classmethod
    def _validate_output_patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validate_output_patterns(value)
        for pattern in value:
            _compile_output_pattern(pattern)
        return value

    @model_validator(mode="after")
    def _validate_test_count_evidence(self) -> TrustedVerificationCheckConfig:
        if (self.environment_attestation_path is None) != (
            self.environment_attestation_sha256 is None
        ):
            raise ValueError(
                "environment attestation path and digest must be configured together"
            )
        if (self.test_count_pattern is None) != (self.minimum_test_count is None):
            raise ValueError(
                "test_count_pattern and minimum_test_count must be configured together"
            )
        if self.test_count_pattern is not None:
            validate_output_patterns((self.test_count_pattern,))
            pattern = _compile_output_pattern(self.test_count_pattern)
            if "count" not in pattern.groupindex:
                raise ValueError("test_count_pattern must define a named 'count' group")
        validate_custom_runner_contract(
            custom_runner=self.custom_runner,
            executable_sha256=self.executable_sha256,
            required_output_patterns=self.required_output_patterns,
            test_count_pattern=self.test_count_pattern,
            minimum_test_count=self.minimum_test_count,
        )
        return self


def _compile_output_pattern(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid verification output pattern: {exc}") from exc


class TrustedVerificationRecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe_version: str = Field(min_length=1)
    task_brief: str = Field(min_length=1)
    acceptance_contract: str = Field(min_length=1)
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    checks: tuple[TrustedVerificationCheckConfig, ...] = Field(min_length=1)
    execution_topology: TrustedExecutionTopologyConfig | None = None

    @field_validator("recipe_version", "task_brief", "acceptance_contract")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trusted verification recipe text must be nonempty")
        return value

    @field_validator("allowed_paths")
    @classmethod
    def _validate_allowed_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized_patterns: list[str] = []
        for pattern in value:
            normalized = pattern.strip().replace("\\", "/")
            path = PurePosixPath(normalized)
            if not normalized or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"invalid allowed-path pattern: {pattern!r}")
            normalized_patterns.append(normalized)
        return tuple(normalized_patterns)

    @model_validator(mode="after")
    def _validate_check_names(self) -> TrustedVerificationRecipeConfig:
        names = [check.name for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("trusted verification check names must be unique")
        if missing := [
            check.name for check in self.checks if check.executable_sha256 is None
        ]:
            raise ValueError(
                "trusted verification recipes require executable_sha256: "
                f"{', '.join(missing)}"
            )
        if unattested := [
            check.name
            for check in self.checks
            if check.environment_attestation_path is None
            or check.environment_attestation_sha256 is None
        ]:
            raise ValueError(
                "trusted verification recipes require an environment "
                f"attestation: {', '.join(unattested)}"
            )
        if bootstrap := next(
            (check for check in self.checks if _uses_offline_bootstrap(check.argv)),
            None,
        ):
            raise ValueError(
                "trusted verification recipes cannot use an offline bootstrap "
                f"command ({bootstrap.name}: {bootstrap.argv[0]}); configure an "
                "absolute pre-provisioned host executable"
            )
        return self


def _uses_offline_bootstrap(argv: tuple[str, ...]) -> bool:
    executable = PurePosixPath(argv[0]).name.casefold().replace("_", "-")
    if executable in {"uv", "pre-commit"}:
        return True
    return any(
        argument == "-m"
        and index + 1 < len(argv)
        and argv[index + 1].casefold().replace("_", "-") == "pre-commit"
        for index, argument in enumerate(argv)
    )


__all__ = [
    "TrustedExecutionTopologyConfig",
    "TrustedVerificationCheckConfig",
    "TrustedVerificationRecipeConfig",
]
