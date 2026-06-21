---
name: verify-contract
description: Run a code task in an isolated worktree and gate delivery on a code-artifact contract — the agent's files must exist, match grep/size rules, and pass tests, or the work is not delivered. Emits a verdict packet; the contract (not a model) decides.
---

# Deliver a code change ONLY if it satisfies a declared contract. Unlike a
# schema= check (which validates the agent's JSON CLAIM), a contract validates
# the FILES it wrote: outputs exist / contain / match regex / size bounds,
# tree-wide invariants (grep present or absent), and executable tests. The
# contract is enforced inside the worktree's lifetime, before it is removed;
# on pass the work is ff-merged into the parent, on fail it is held back.
#
# The gate is reconciled in CODE from the ContractReport — no model decides
# whether the work ships. A failed-contract agent returns a falsy
# ContractFailure, so the canonical `[r for r in ... if r]` drops it.
#
# args = {
#   "task": "Implement the auth module per PLAN.md",   # the code task
#   "agent": "worker",                                  # worktree-capable profile
#   "contract": {
#     "outputs": [
#       {"path": "auth.py", "must_contain": ["JWT"],
#        "must_not_contain": ["password ="], "min_size": 200}
#     ],
#     "invariants": [
#       {"grep": "password\\s*=\\s*['\"]", "must_match": False,
#        "description": "no plaintext passwords"}
#     ],
#     "tests": [
#       {"command": "python -m pytest tests/test_auth.py -q", "expect": "passed"}
#     ]
#   }
# }


async def main():
    cfg = args if isinstance(args, dict) else {}
    task = cfg.get("task")
    contract = cfg.get("contract")
    worker_agent = cfg.get("agent", "worker")
    if not task or not contract:
        return {
            "gate": "error",
            "report": (
                "Missing task or contract. Pass args = {task, contract, agent?}. "
                "The contract declares outputs (path/contain/regex/size), "
                "invariants (grep present/absent), and tests (command/expect)."
            ),
        }

    phase("implement")
    result = await agent(
        task,
        agent=worker_agent,
        isolation="worktree",
        contract=contract,
        label="impl",
        phase="implement",
    )

    # A ContractFailure is falsy and dict-like (mirrors SchemaValidationFailure):
    # the agent's work was NOT delivered. We can't import ContractFailure (the
    # sandbox allowlists only stdlib modules), so distinguish by truthiness — a
    # passed contract returns the agent's truthy text/dict output and the work
    # is ff-merged into the parent; a failed contract returns the falsy failure.
    if result:
        gate = "delivered"
        summary = "contract passed; work ff-merged into the parent"
        delivered = True
    else:
        gate = "contract_failed"
        delivered = False
        summary = "contract failed; the worktree branch is kept for a manual fix"
    # The full ContractReport (every violation with file/category/message) is
    # carried on the failure object and surfaces in the run's agent results; the
    # synthesized verdict below narrates the gate, the contract decides it.

    phase("synthesize")
    packet = await agent(
        f"Write a concise CONTRACT VERDICT for the code task below. The gate is "
        f"**{gate}**.\n\n"
        f"TASK:\n{task}\n\n"
        f"VERDICT: {summary}.\n\n"
        "For a failed gate, list the structured violations from the run's agent "
        "results (each names its category and the file/check that failed). For a "
        "delivered gate, state that every output, invariant, and test passed.\n\n"
        "End with: this gate is machine-decided; the contract, not a model, "
        "determines whether the work shipped.",
        agent="reviewer",
        label="verdict",
        phase="synthesize",
    )

    return {
        "gate": gate,
        "delivered": delivered,
        "summary": summary,
        "report": packet,
        "note": (
            "Delivered" if delivered
            else "Not delivered — the worktree branch is kept for a manual fix."
        ),
    }
