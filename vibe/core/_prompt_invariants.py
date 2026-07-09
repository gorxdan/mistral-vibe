from __future__ import annotations

COMPACT_VERIFICATION_INVARIANT = """\
## Verification invariant

Before claiming non-trivial work complete, run the `verifier` subagent. Only an
evidence-backed `VERDICT: PASS` is success; fix `FAIL`, and report `PARTIAL` or a
missing verdict as incomplete. `land_work` requires that pass or an explicit
`trivial: <reason>` waiver that it validates against a documentation-only diff."""

COMPACT_INVESTIGATION_INVARIANT = """\
## Investigation invariant

For a reported failure, reproduce it with a test, deterministic trigger, or code
trace before proposing a fix. Features, refactors, docs, and cosmetics are exempt."""
