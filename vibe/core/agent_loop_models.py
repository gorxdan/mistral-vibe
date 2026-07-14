"""Data models for tool execution decisions.

Extracted from the loop module so mixins (and the loop itself) can construct
``ToolDecision`` / ``ToolExecutionResponse`` at runtime without a circular
import through ``_loop``. Pure Pydantic/enum models — no AgentLoop coupling.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel, ConfigDict

from vibe.core.tools.base import ToolAuthorizationSource, ToolPermission


class ToolExecutionResponse(StrEnum):
    SKIP = auto()
    EXECUTE = auto()


class ToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: ToolExecutionResponse
    approval_type: ToolPermission
    authorization_source: ToolAuthorizationSource = ToolAuthorizationSource.POLICY
    authorization_fingerprint: str | None = None
    feedback: str | None = None
    judge_approved: bool = False
    # When the user chose MODIFY at approval, the tool is re-validated and
    # re-dispatched with these args (user already approved the modified form,
    # so no re-prompt). None for EXECUTE/SKIP decisions.
    modified_args: dict[str, Any] | None = None
