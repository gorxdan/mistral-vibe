---
name: adversarial-review
description: Adversarially review the current diff — diverse-lens finders, independent refute-verify, gated synthesis.
---

import json

# Each lens is an independent reviewer looking for one class of defect. Running
# them as separate subagents (rather than one "find everything" pass) is what
# makes the review adversarial and diverse — each is blind to the others.
LENSES = [
    {
        "key": "correctness",
        "focus": (
            "logic errors, wrong conditions, off-by-one, None/edge cases, "
            "broken control flow, incorrect API or library usage"
        ),
    },
    {
        "key": "security",
        "focus": (
            "injection, path traversal, SSRF, auth/permission gaps, unsafe "
            "deserialization, secrets handling, sandbox escapes"
        ),
    },
    {
        "key": "concurrency-resource",
        "focus": (
            "races, deadlocks, unawaited tasks, leaked processes/files/locks, "
            "unbounded growth, cancellation safety"
        ),
    },
    {
        "key": "error-edge",
        "focus": (
            "swallowed exceptions, missing error handling, wrong fallbacks, "
            "empty/huge inputs, partial-failure states"
        ),
    },
]

# Schemas are intentionally permissive (no additionalProperties:false) so an
# agent returning extra fields does not fail validation.
SCOPE_SCHEMA = {
    "type": "object",
    "properties": {"files": {"type": "array", "items": {"type": "string"}}},
    "required": ["files"],
}

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low", "info"],
                    },
                    "file": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["title", "severity", "file", "evidence"],
            },
        }
    },
    "required": ["findings"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["confirmed", "refuted", "uncertain"],
        },
        "reasoning": {"type": "string"},
        "corrected_severity": {
            "type": "string",
            "enum": ["critical", "high", "medium", "low", "info", "none"],
        },
    },
    "required": ["verdict", "reasoning"],
}


async def main():
    # `args` is the git range/ref to review (e.g. "HEAD", "main..HEAD",
    # "HEAD~3..HEAD"). Default to the working-tree diff against HEAD.
    target = args if args else "HEAD"

    phase("Scope")
    log(f"Scoping changes for: {target}")
    scope = await agent(
        f"You are scoping a code review of the diff for `{target}`.\n"
        f"Run `git diff --stat {target}` (and `git status --short` if {target!r} is "
        f"'HEAD', to include unstaged work). Return the list of changed source "
        f"files as relative paths in the `files` array. Exclude pure deletions and "
        f"lockfiles.",
        agent="reviewer",
        label="scope",
        phase="Scope",
        schema=SCOPE_SCHEMA,
    )
    files = scope.get("files", []) if isinstance(scope, dict) else []
    if not files:
        return {
            "target": target,
            "report": "No changed source files found to review.",
            "confirmed": 0,
            "candidates": 0,
        }

    files_str = "\n".join(f"- {f}" for f in files)
    log(f"{len(files)} changed file(s)")

    phase("Review")
    reviews = await parallel(*[
        (
            lambda lens=lens: agent(
                f"Adversarial code review through the **{lens['key']}** lens only.\n"
                f"Changed files:\n{files_str}\n\n"
                f"For each file, run `git diff {target} -- <file>` and read the file "
                f"to understand the change in context. Hunt specifically for: "
                f"{lens['focus']}.\n"
                f"Report ONLY real, defensible issues introduced or exposed by this "
                f"diff — each with a file:line and a quoted snippet as evidence. No "
                f"style nits. If the diff is clean for this lens, return an empty "
                f"findings array.",
                agent="reviewer",
                label=f"review:{lens['key']}",
                phase="Review",
                schema=FINDINGS_SCHEMA,
            )
        )
        for lens in LENSES
    ])

    findings = [
        f
        for r in reviews
        if r
        for f in r.get("findings", [])
    ]
    log(f"{len(findings)} candidate finding(s) across lenses")
    if not findings:
        return {
            "target": target,
            "report": "No issues found across correctness/security/concurrency/error lenses.",
            "confirmed": 0,
            "candidates": 0,
        }

    phase("Verify")

    async def verify(finding):
        # Independent skeptic: gets only the claim, re-reads the code itself, and
        # is told to refute. This is what kills plausible-but-wrong findings.
        verdict = await agent(
            f"Adversarially VERIFY a code-review finding. Your job is to REFUTE it.\n"
            f"Re-read the cited code yourself (open the file, run "
            f"`git diff {target} -- {finding.get('file', '')}`), and run the most "
            f"relevant existing test if one exists. Default to `refuted` unless the "
            f"code clearly exhibits the problem; use `uncertain` only if genuinely "
            f"ambiguous after looking.\n\n"
            f"FINDING\n"
            f"file: {finding.get('file', '?')}\n"
            f"severity(claimed): {finding.get('severity', '?')}\n"
            f"title: {finding.get('title', '')}\n"
            f"evidence(claimed): {finding.get('evidence', '')}\n",
            agent="reviewer",
            label=f"verify:{finding.get('file', '?')}",
            phase="Verify",
            schema=VERDICT_SCHEMA,
        )
        if verdict:
            return {
                **finding,
                "verdict": verdict.get("verdict", "uncertain"),
                "verify_reasoning": verdict.get("reasoning", ""),
                "final_severity": verdict.get(
                    "corrected_severity", finding.get("severity")
                ),
            }
        return {**finding, "verdict": "uncertain", "verify_reasoning": ""}

    verdicts = await pipeline(findings, verify)
    confirmed = [v for v in verdicts if v.get("verdict") == "confirmed"]
    uncertain = [v for v in verdicts if v.get("verdict") == "uncertain"]
    log(
        f"{len(confirmed)} confirmed, {len(uncertain)} uncertain, "
        f"{len(findings) - len(confirmed) - len(uncertain)} refuted"
    )

    phase("Synthesize")
    report = await agent(
        f"Write a concise, high-signal code-review report for the diff `{target}`.\n"
        f"Every finding below was independently adversarially verified.\n\n"
        f"CONFIRMED ({len(confirmed)}):\n{json.dumps(confirmed, indent=2)}\n\n"
        f"UNCERTAIN ({len(uncertain)}):\n{json.dumps(uncertain, indent=2)}\n\n"
        f"Group confirmed issues by severity (critical first); for each give the "
        f"location, what's wrong, and the fix. List uncertain items briefly under "
        f"'needs human check'. If nothing is confirmed, say the diff looks clean. "
        f"No padding.",
        agent="reviewer",
        label="synthesize",
        phase="Synthesize",
    )

    return {
        "target": target,
        "report": report,
        "candidates": len(findings),
        "confirmed": len(confirmed),
        "uncertain": len(uncertain),
    }
