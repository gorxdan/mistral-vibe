from __future__ import annotations

COMPACT_VERIFICATION_INVARIANT = """\
## Verification invariant

Before claiming non-trivial work complete, run the `verifier` subagent. Only an
evidence-backed `VERDICT: PASS` is success; fix `FAIL`, and report `PARTIAL` or a
missing verdict as incomplete. With no trusted recipe configured, `land_work`
accepts the current recorded verifier/workflow pass or an explicit
`trivial: <reason>` documentation-only waiver, never pasted report prose."""

COMPACT_VERIFICATION_RECIPE_INVARIANT = """\
## Verification invariant

Before claiming non-trivial work complete, run the `verifier` subagent. Only an
evidence-backed `VERDICT: PASS` is success; fix `FAIL`, and report `PARTIAL` or a
missing verdict as incomplete. This session has a prebound trusted recipe: after
the verifier PASS, call no-argument `verify_work`. `land_work` then requires its
current durable receipt; pasted prose and trivial waivers cannot replace it."""

COMPACT_INVESTIGATION_INVARIANT = """\
## Investigation invariant

For a reported failure, reproduce it with a test, deterministic trigger, or code
trace before proposing a fix. Features, refactors, docs, and cosmetics are exempt."""
