"""Workspace-bound verification state for the merge gate.

Two write-seams own the flags and are the only sanctioned setters:

- ``record_contract_pass`` — called by the workflow runtime after
  ``verify_contract`` returns ``passed`` (in-process, authoritative).
- ``record_verifier_pass`` — called by the task tool after a ``verifier``-profile
  response parses as an evidence-backed ``VERDICT: PASS`` report.

``land_work`` accepts a recorded pass only while its repository fingerprint still
matches. The verifier flag stores the parsed report; model-authored notes cannot
satisfy the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from vibe.core.verification_contract import VerificationReport


def workspace_fingerprint() -> str | None:
    from vibe.core._workspace_verification import workspace_fingerprint as calculate

    return calculate()


@dataclass
class VerificationPass:
    source: str
    summary: str
    workspace_fingerprint: str | None
    report: VerificationReport | None = None
    recorded_at: float = field(default_factory=time.monotonic)


@dataclass
class VerificationState:
    last_contract_pass: VerificationPass | None = None
    last_verifier_pass: VerificationPass | None = None

    def record_contract_pass(self, summary: str) -> None:
        self.last_contract_pass = VerificationPass(
            source="workflow-contract",
            summary=summary,
            workspace_fingerprint=workspace_fingerprint(),
        )

    def record_verifier_pass(self, report: VerificationReport) -> None:
        if not report.passed:
            raise ValueError("only a passing verifier report can satisfy the gate")
        self.last_verifier_pass = VerificationPass(
            source="verifier-subagent",
            summary=report.summary(),
            workspace_fingerprint=workspace_fingerprint(),
            report=report,
        )

    def has_pass(self) -> bool:
        current = workspace_fingerprint()
        if current is None:
            return False
        return any(
            recorded is not None and recorded.workspace_fingerprint == current
            for recorded in (self.last_contract_pass, self.last_verifier_pass)
        )

    def latest(self) -> VerificationPass | None:
        c = self.last_contract_pass
        v = self.last_verifier_pass
        if c is None:
            return v
        if v is None:
            return c
        return c if c.recorded_at >= v.recorded_at else v

    def clear(self) -> None:
        self.last_contract_pass = None
        self.last_verifier_pass = None
