---
name: security-fix-verify
description: Adversarially verify a security FIX branch before human review — refute-only per-finding panel, regression hunt, runtime gaps hard-block. Emits a review packet; never pushes.
---

# Pre-merge gate for security fixes. Unlike a hand-rolled "confirm it works"
# pass, this is REFUTE-ONLY (default-to-broken) and treats anything that cannot
# be proven from the repo (DB columns, runtime permissions, external event
# shapes) as BLOCKING, not advisory. It audits and reports — it never pushes or
# opens a PR; a human gates the merge after the blocking items are cleared.
#
# args = {
#   "base": "main",                      # branch/ref the fix merges into
#   "branch": "fix/...",                 # the fix branch (defaults to HEAD)
#   "findings": [
#     {"id": "C1",
#      "original": "<the original vulnerability>",
#      "must_be_true": "<what must hold for the fix to be COMPLETE>",
#      "file": "path/to/file",           # primary changed file
#      "commit": "abc1234"},             # optional: the fix commit
#     ...
#   ],
# }

# Two independent adversarial lenses per finding. Each defaults to "the fix is
# broken" and only concedes "sound" if it genuinely cannot break it.
LENSES = [
    {
        "key": "exploit-residual",
        "focus": (
            "Make the ORIGINAL exploit still work, or find a variant that reaches "
            "the same outcome. Hunt bypass paths: other callers of the changed "
            "code, missing input validation, missing-field/edge cases, and "
            "concurrency/race windows the fix did not cover."
        ),
    },
    {
        "key": "regression-and-runtime",
        "focus": (
            "Assume the fix code is present. Hunt for (a) a LEGITIMATE flow the fix "
            "now breaks (over-restrictive guard, wrong threshold, broken caller), "
            "and (b) any correctness claim provable ONLY at runtime — a DB "
            "column/permission exists, a privileged op is allowed, an external "
            "event's shape, an env var present in prod. For (b) you MUST return "
            "needs_runtime_check and name exactly what to prove; never assume it works."
        ),
    },
]

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["sound", "residual_hole", "regression", "needs_runtime_check"],
        },
        "evidence": {"type": "string"},
        "hole_or_gap": {"type": "string"},
        "runtime_check_required": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
}

REGRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["no_regressions", "regression_found", "needs_runtime_check"],
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                    },
                    "location": {"type": "string"},
                    "issue": {"type": "string"},
                },
                "required": ["severity", "issue"],
            },
        },
        "runtime_check_required": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
}

FILES_SCHEMA = {
    "type": "object",
    "properties": {"files": {"type": "array", "items": {"type": "string"}}},
    "required": ["files"],
}

# A finding is RESOLVED only if every lens returns this. Anything else blocks.
_SOUND = "sound"


def _refute_prompt(finding, base, branch, lens):
    commit = finding.get("commit", "")
    show = f"`git show {commit}` and " if commit else ""
    return (
        f"You are an adversarial security verifier on branch `{branch}` (base "
        f"`{base}`). A fix was applied for finding {finding.get('id', '?')}. Your "
        f"DEFAULT position is that the fix is BROKEN — only conclude `sound` if you "
        f"genuinely cannot break it after trying.\n\n"
        f"ORIGINAL VULNERABILITY:\n{finding.get('original', '')}\n\n"
        f"WHAT MUST BE TRUE FOR THE FIX TO BE COMPLETE:\n"
        f"{finding.get('must_be_true', '')}\n\n"
        f"PRIMARY FILE: {finding.get('file', '?')}\n\n"
        f"LENS — {lens['key']}: {lens['focus']}\n\n"
        f"METHOD:\n"
        f"1. {show}`git diff {base}..{branch} -- {finding.get('file', '')}` to read the fix.\n"
        f"2. Read the CURRENT tree file in full; check other callers and edge cases.\n"
        f"3. If the fix's correctness depends on a fact you CANNOT verify from the "
        f"repo alone, return verdict=`needs_runtime_check` and state exactly what "
        f"must be proven at runtime. Do not assume.\n\n"
        f"Budget: <=10 tool calls. Cite file:line in `evidence`."
    )


def _regress_prompt(path, base, branch):
    return (
        f"You are a regression hunter on security fix branch `{branch}` (base "
        f"`{base}`). Concern: collateral damage in `{path}`.\n\n"
        f"METHOD:\n"
        f"1. `git diff {base}..{branch} -- {path}` to see every change.\n"
        f"2. Read the CURRENT tree, callers, and tests. Hunt for: broken callers, "
        f"tests that asserted the OLD behavior and were silently weakened, "
        f"semantic regressions, non-idempotent migrations, or a new hole the change "
        f"introduced.\n"
        f"3. If a risk is only provable at runtime (schema, permissions, prod env), "
        f"return verdict=`needs_runtime_check` and name the check. Do not assume.\n\n"
        f"Budget: <=10 tool calls. Cite file:line."
    )


async def main():
    cfg = args if isinstance(args, dict) else {}
    base = cfg.get("base", "main")
    branch = cfg.get("branch", "HEAD")
    findings = cfg.get("findings") or []
    if not findings:
        return {
            "gate": "error",
            "report": (
                "No findings provided. Pass args = {base, branch, findings:[{id, "
                "original, must_be_true, file, commit?}]} — each finding names the "
                "original vulnerability and what must be true for the fix to be complete."
            ),
        }

    # PASS A — refute each fix. Each finding flows independently; its lens panel
    # runs concurrently. A finding is resolved only if EVERY lens says `sound`.
    phase("verify-fixes")

    async def verify_finding(finding):
        panel = await parallel(*[
            agent(
                _refute_prompt(finding, base, branch, lens),
                agent="security",
                label=f"{finding.get('id', '?')}:{lens['key']}",
                phase="verify-fixes",
                schema=VERIFY_SCHEMA,
            )
            for lens in LENSES
        ])
        panel = [p for p in panel if isinstance(p, dict)]
        # Default-to-broken: a lens that died (None) counts as unverified.
        if len(panel) < len(LENSES):
            status = "blocked"
        else:
            status = "resolved" if all(p.get("verdict") == _SOUND for p in panel) else "blocked"
        return {
            "id": finding.get("id", "?"),
            "file": finding.get("file", ""),
            "status": status,
            "panel": panel,
        }

    verify_results = [r for r in await pipeline(findings, verify_finding) if r]

    # PASS B — regression hunt over every file the branch actually touched
    # (catches collateral files not named in the findings).
    phase("scope")
    scope = await agent(
        f"List the source files changed in `{base}..{branch}`. Run "
        f"`git diff --name-only {base}..{branch}`. Return them in `files` "
        f"(exclude pure deletions and lockfiles).",
        agent="explore",
        label="scope",
        phase="scope",
        schema=FILES_SCHEMA,
    )
    files = (scope.get("files", []) if isinstance(scope, dict) else [])[:12]

    phase("regression-hunt")
    regress = []
    if files:
        regress = [
            r
            for r in await parallel(*[
                agent(
                    _regress_prompt(path, base, branch),
                    agent="security",
                    label=f"regress:{path}",
                    phase="regression-hunt",
                    schema=REGRESSION_SCHEMA,
                )
                for path in files
            ])
            if isinstance(r, dict)
        ]

    # Reconcile in code — do NOT let a model decide the gate. Anything not
    # provably resolved blocks; runtime-only claims block until proven.
    blocked = [r for r in verify_results if r["status"] == "blocked"]
    runtime_checks = []
    for r in verify_results:
        for p in r["panel"]:
            if p.get("verdict") == "needs_runtime_check" and p.get("runtime_check_required"):
                runtime_checks.append({"finding": r["id"], "check": p["runtime_check_required"]})
    regressions = [r for r in regress if r.get("verdict") == "regression_found"]
    for r in regress:
        if r.get("verdict") == "needs_runtime_check" and r.get("runtime_check_required"):
            runtime_checks.append({"concern": "regression", "check": r["runtime_check_required"]})

    gate = (
        "blocked"
        if (blocked or regressions or runtime_checks)
        else "ready_for_human_review"
    )

    phase("synthesize")
    blob = json.dumps(
        {
            "gate": gate,
            "verify": verify_results,
            "regressions": regress,
            "runtime_checks": runtime_checks,
        },
        indent=2,
    )
    report = await agent(
        f"Write a concise pre-merge SECURITY REVIEW PACKET for branch `{branch}` "
        f"(base `{base}`). The machine gate is **{gate}**.\n\n"
        f"Lead with the gate and a one-line rationale. Then, in priority order: "
        f"(1) BLOCKING fix verdicts (residual holes / regressions), (2) RUNTIME "
        f"CHECKS that must be performed before merge (these block — they could not "
        f"be proven statically), (3) regressions found, (4) findings that verified "
        f"clean. For each item give file:line and the concrete next action. End "
        f"with: this is an audit only — a HUMAN must review and run the runtime "
        f"checks before merge; nothing here authorizes a push.\n\n"
        f"DATA (JSON):\n{blob}",
        agent="reviewer",
        label="review-packet",
        phase="synthesize",
    )

    return {
        "gate": gate,
        "summary": {
            "findings": len(findings),
            "resolved": len([r for r in verify_results if r["status"] == "resolved"]),
            "blocked": len(blocked),
            "runtime_checks_required": len(runtime_checks),
            "regressions": len(regressions),
            "files_scanned": len(files),
        },
        "blocked_findings": [r["id"] for r in blocked],
        "runtime_checks_required": runtime_checks,
        "regressions": regressions,
        "verify_results": verify_results,
        "regression_results": regress,
        "report": report,
        "note": (
            "Audit only — no push/PR. Clear every blocking item AND perform every "
            "runtime check, then a HUMAN reviews before merge."
        ),
    }
