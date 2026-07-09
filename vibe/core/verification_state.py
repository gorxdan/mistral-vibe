"""Session-scoped verification state: recorded pass flags for the merge gate.

Two write-seams own the flags and are the only sanctioned setters:

- ``record_contract_pass`` — called by the workflow runtime after
  ``verify_contract`` returns ``passed`` (in-process, authoritative).
- ``record_verifier_pass`` — called by the task tool when a ``verifier``-profile
  subagent returns a response whose final verdict line is ``VERDICT: PASS``.

``land_work`` accepts a recorded pass flag (via ``InvokeContext``) as satisfying
the verification requirement, superseding the free-text ``verification_note``.
The flag path is stricter than prose: it cannot be forged by the host typing a
string, because only the two owning code paths can set it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class VerificationPass:
    source: str
    summary: str
    recorded_at: float = field(default_factory=time.monotonic)


@dataclass
class VerificationState:
    last_contract_pass: VerificationPass | None = None
    last_verifier_pass: VerificationPass | None = None

    def record_contract_pass(self, summary: str) -> None:
        self.last_contract_pass = VerificationPass(
            source="workflow-contract", summary=summary
        )

    def record_verifier_pass(self, summary: str) -> None:
        self.last_verifier_pass = VerificationPass(
            source="verifier-subagent", summary=summary
        )

    def has_pass(self) -> bool:
        return (
            self.last_contract_pass is not None or self.last_verifier_pass is not None
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
