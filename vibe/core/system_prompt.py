from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import html
import os
from pathlib import Path
import re
from string import Template
import subprocess
import time
from typing import TYPE_CHECKING

from vibe.core._prompt_invariants import (
    COMPACT_INVESTIGATION_INVARIANT,
    COMPACT_VERIFICATION_INVARIANT,
)
from vibe.core.baseline_scaling import (
    BaselineTier,
    agents_md_byte_budget,
    budget_doc,
    section_enabled,
)
from vibe.core.config import VibeConfig
from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.experiments import ExperimentName
from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME
from vibe.core.prompts import MissingPromptFileError, UtilityPrompt, load_system_prompt
from vibe.core.utils import (
    get_platform_display_name,
    is_dangerous_directory,
    is_windows,
)

if TYPE_CHECKING:
    from vibe.core.agents import AgentManager
    from vibe.core.config import ProjectContextConfig
    from vibe.core.experiments import ExperimentManager
    from vibe.core.skills.manager import SkillManager
    from vibe.core.skills.models import SkillInfo
    from vibe.core.tools.manager import ToolManager

# Git status is cached per repo root with a short TTL so a long session sees
# changes (branch switch, new commits, dirty files) without re-running git on
# every system-prompt assembly. Previously this never expired → stale status.
_GIT_STATUS_TTL_S = 30.0
_git_status_cache: dict[Path, tuple[float, str]] = {}


class ProjectContextProvider:
    def __init__(
        self, config: ProjectContextConfig, root_path: str | Path = "."
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.config = config

    def get_git_status(self) -> str:
        now = time.monotonic()
        cached = _git_status_cache.get(self.root_path)
        if cached is not None and now - cached[0] < _GIT_STATUS_TTL_S:
            return cached[1]

        result = self._fetch_git_status()
        _git_status_cache[self.root_path] = (now, result)
        return result

    def _run_git(
        self, args: list[str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "--no-optional-locks", *args],
            capture_output=True,
            check=True,
            cwd=self.root_path,
            stdin=subprocess.DEVNULL if is_windows() else None,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _format_git_status(status_output: str) -> str:
        if not status_output:
            return "(clean)"
        status_lines = status_output.splitlines()
        MAX_GIT_STATUS_SIZE = 50
        if len(status_lines) > MAX_GIT_STATUS_SIZE:
            return f"({len(status_lines)} changes - use 'git status' for details)"
        return f"({len(status_lines)} changes)"

    @staticmethod
    def _parse_git_log(log_output: str) -> list[str]:
        recent_commits: list[str] = []
        for line in log_output.split("\n"):
            if not (line := line.strip()):
                continue
            if " " in line:
                commit_hash, commit_msg = line.split(" ", 1)
                # anchor to a trailing "(#N)" so a conventional-commit scope
                # like "perf(prompts):" is not mistaken for a PR suffix
                commit_msg = re.sub(r"\s*\(#\d+\)$", "", commit_msg)
                recent_commits.append(f"{commit_hash} {commit_msg}")
            else:
                recent_commits.append(line)
        return recent_commits

    def _fetch_git_status(self) -> str:
        try:
            timeout = min(self.config.timeout_seconds, 10.0)
            num_commits = self.config.default_commit_count

            with ThreadPoolExecutor(max_workers=4) as pool:
                branch_future = pool.submit(
                    self._run_git, ["branch", "--show-current"], timeout
                )
                remote_future = pool.submit(self._run_git, ["branch", "-r"], timeout)
                status_future = pool.submit(
                    self._run_git, ["status", "--porcelain"], timeout
                )
                log_future = pool.submit(
                    self._run_git, ["log", "--oneline", f"-{num_commits}"], timeout
                )

            current_branch = branch_future.result().stdout.strip()

            main_branch = "main"
            try:
                branches_output = remote_future.result().stdout
                if "origin/master" in branches_output:
                    main_branch = "master"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

            status = self._format_git_status(status_future.result().stdout.strip())
            recent_commits = self._parse_git_log(log_future.result().stdout.strip())

            git_info_parts = [
                f"Current branch: {current_branch}",
                f"Main branch (you will usually use this for PRs): {main_branch}",
                f"Status: {status}",
            ]

            if recent_commits:
                git_info_parts.append("Recent commits:")
                git_info_parts.extend(recent_commits)

            return "\n".join(git_info_parts)

        except subprocess.TimeoutExpired:
            return "Git operations timed out (large repository)"
        except subprocess.CalledProcessError:
            return "Not a git repository or git not available"
        except Exception as e:
            return f"Error getting git status: {e}"

    def get_full_context(self, *, include_git_status: bool = True) -> str:
        git_status = self.get_git_status() if include_git_status else ""

        template = UtilityPrompt.PROJECT_CONTEXT.read()
        return Template(template).safe_substitute(
            abs_path=str(self.root_path), git_status=git_status
        )


def _get_default_shell() -> str:
    """Get the default shell used by asyncio.create_subprocess_shell.

    On Unix, uses $SHELL env var and default to sh.
    On Windows, this is COMSPEC or cmd.exe.
    """
    if is_windows():
        return os.environ.get("COMSPEC", "cmd.exe")
    return os.environ.get("SHELL", "sh")


def _get_os_system_prompt() -> str:
    shell = _get_default_shell()
    platform_name = get_platform_display_name()
    prompt = f"The operating system is {platform_name} with shell `{shell}`"

    if is_windows():
        prompt += "\n" + _get_windows_system_prompt()
    return prompt


def _format_current_date() -> str:
    today = date.today()
    return f"{today.isoformat()} ({today.strftime('%A')})"


def _get_windows_system_prompt() -> str:
    return (
        "### COMMAND COMPATIBILITY RULES (MUST FOLLOW):\n"
        "- DO NOT use Unix commands like `ls`, `grep`, `cat` - they won't work on Windows\n"
        "- Use: `dir` (Windows) for directory listings\n"
        "- Use: backslashes (\\\\) for paths\n"
        "- Check command availability with: `where command` (Windows)\n"
        "- Script shebang: Not applicable on Windows\n"
        "### ALWAYS verify commands work on the detected platform before suggesting them"
    )


def _add_commit_signature() -> str:
    return (
        "When committing changes, use concise, meaningful commit messages.\n"
        "Follow the conventional commit format if the project uses it:\n"
        "  `<type>(<scope>): <description>`\n"
        "Types: feat, fix, docs, style, refactor, test, chore.\n"
        "Keep the subject line under 72 characters.\n"
        "Use `git commit -m 'message'` for simple commits.\n"
        "Do not add 'Generated by' or 'Co-Authored-By' signatures."
    )


def _add_humanizer_guidance() -> str:
    return (
        "Write naturally. Avoid AI vocabulary: 'Additionally,' 'crucial,' 'delve,' "
        "'fostering,' 'showcase,' 'testament,' 'underscore,' 'vibrant,' 'pivotal,' "
        "'landscape,' 'intricate,' 'tapestry,' 'align with,' 'garner.' "
        "Use simple 'is/are/has' instead of 'serves as/stands as/boasts.' "
        "Avoid '-ing' phrases tacked on for fake depth. "
        "Don't force rule-of-three lists. "
        "Don't use 'It's not just... it's...' or 'Not only... but...' constructions. "
        "Use em dashes sparingly. "
        "Don't end with 'I hope this helps' or 'Let me know.' "
        "Vary sentence length. Use specific details instead of vague claims."
    )


def _add_caveman_thinking_guidance() -> str:
    return (
        "When you think (reasoning/thinking blocks): be terse. "
        "Drop articles, filler, hedging, and pleasantries. Fragments are fine. "
        "Keep ALL technical substance — names, file:line references, errors, and "
        "code verbatim. This governs only your private reasoning."
    )


_SKILL_INDEX_MAX_CHARS = 160
_SENTENCE_END = re.compile(r"[.!?](?=\s)")


def _truncate_to_first_sentence(
    text: str, *, max_chars: int = _SKILL_INDEX_MAX_CHARS
) -> str:
    collapsed = " ".join(text.split())
    match = _SENTENCE_END.search(collapsed)
    if match and match.end() <= max_chars:
        return collapsed[: match.end()]
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rstrip() + "\u2026"


def _skill_index_line(info: SkillInfo) -> str:
    if info.summary:
        return " ".join(info.summary.split())
    return _truncate_to_first_sentence(info.description)


def _get_available_skills_section(
    skill_manager: SkillManager, *, summaries: bool = True
) -> str:
    skills = skill_manager.available_skills
    if not skills:
        return ""

    lines = [
        "# Available Skills",
        "",
        "Each skill packages domain-specific instructions and workflows. When a"
        " task matches a trigger below, load the skill with the `skill` tool"
        " — it contains guidance that may differ from your training data or"
        " be version-specific. Load proactively for matching tasks rather than"
        " relying on recalled patterns.",
        "",
        "<available_skills>",
    ]

    # On small windows the index keeps the discovery surface (the skill names the
    # `skill` tool loads) but drops the per-skill summary to save tokens.
    for name, info in sorted(skills.items()):
        escaped = html.escape(str(name))
        if summaries:
            summary = html.escape(_skill_index_line(info))
            lines.append(f"- **{escaped}**: {summary}")
        else:
            lines.append(f"- **{escaped}**")

    lines.append("</available_skills>")

    return "\n".join(lines)


# Co-located with the catalog so the routing map survives window tiers that
# trim the orchestration prose.
_SUBAGENT_TRIGGERS: dict[str, str] = {
    "explore": "codebase questions and searches: where/how is X done, trace a "
    "flow, gather all call sites, multi-file reads.",
    "research": "anything outside the repo: docs, library/API behavior, "
    "version checks, web lookups.",
    "reviewer": "adversarial review of a diff, branch, or file; runs targeted "
    "checks and tests to find what breaks.",
    "debugger": "a specific failure or flaky test: reproduce, isolate, and "
    "trace the root cause; returns the cause + minimal fix (it diagnoses, "
    "you apply).",
    "planner": "design an approach before building: returns a code-grounded, "
    "phased plan with risks and the files to touch (it plans, you decide).",
    "security": "defensive vuln audit of a change or area: traces untrusted "
    "input to sinks and returns severity-ranked findings with fixes.",
    "verifier": "gate a *completed* implementation: proves it works by trying "
    "to break it, then emits a PASS/FAIL/PARTIAL verdict with command evidence.",
    "grunt": "mechanical execution of a *fully-specified* change: renames, "
    "codemods, repetitive edits across many files. Routes onto a cheap model by "
    "default (`grunt_model`); give it a concrete task with no design decisions.",
}


def _get_available_subagents_section(agent_manager: AgentManager) -> str:
    agents = agent_manager.get_subagents()
    if not agents:
        return ""

    lines = ["# Available Subagents", ""]
    lines.append("The following subagents can be spawned via the Task tool:")
    for agent in agents:
        lines.append(f"- **{agent.name}**: {agent.description}")
    present = {a.name for a in agents}
    triggers = [(n, t) for n, t in _SUBAGENT_TRIGGERS.items() if n in present]
    if triggers:
        lines.append("")
        lines.append("Pick by the question:")
        for name, trigger in triggers:
            lines.append(f"- `{name}` — {trigger}")

    return "\n".join(lines)


_ORCHESTRATION_SECTION = """## Orchestrating Subagents

Local tools first, delegation second. For an unfamiliar repository, establish a \
baseline yourself before spawning subagents: map files with `glob`, resolve central \
symbols and callers with `lsp`, then read entry points and representative tests. \
Reconnaissance tells you whether delegation adds value and gives you precise briefs.

Delegate what remains broad or uncertain after reconnaissance: independent \
questions, 10+ file investigations, second-pass reviews, broad debugging, \
planning, security, or verification. Do not delegate trivia — a known file, \
symbol, or lookup goes to `read`/`grep`/`lsp` directly.

Fan out independent sub-questions with several `task` calls in one turn; keep \
each brief narrow. Breadth comes from parallel briefs, not oversized prompts. \
Each subagent runs in its own context — use them for breadth, keep the main \
context for synthesis, decisions, edits, and user interaction.

Read-only profiles (`explore`, `research`, `reviewer`, `debugger`, `planner`, \
`security`, `verifier`) cannot write files or ask the user — you own every edit \
and all user interaction. Write-capable profiles (`editor`, `worker`, `grunt`) \
auto-isolate in their own worktree under the `task` tool default; their edits \
land on a branch you inspect and merge. Use them for workflow-scale edits or \
when isolation is the point; for a one-off edit, edit directly."""


def _get_orchestration_section() -> str:
    """Normal-mode orchestration directive — instructs the host to delegate
    cross-file investigation/review to subagents via the task tool. Shown
    whenever subagents exist (le-chaton layers workflows on top of this).
    """
    return _ORCHESTRATION_SECTION


def _get_verification_contract_section() -> str:
    """Host-agent verification contract. Pairs with the structural nudge in the
    todo tool and the ``verifier`` subagent profile: defines when independent
    verification is required before the host may report work done. Gated on the
    ``verification_subsystem`` config flag at the call site.
    """
    return (
        "## Verification contract\n\n"
        "Before reporting non-trivial work done (3+ files, backend/API, "
        "infra — anyone's changes), spawn `verifier` via `task` with the "
        "task, changed files, and approach. Do not share your own test "
        "results; only the verifier assigns a verdict. Trivial work "
        "(one-line, read-only, typo) skips verification.\n\n"
        "- **FAIL** → fix, re-run verifier until PASS or PARTIAL.\n"
        "- **PASS** → spot-check 2–3 of its commands; re-run if a step "
        "lacks a command block or output diverges.\n"
        "- **PARTIAL** → report what passed and what could not be verified; "
        "not success.\n"
        "- **No VERDICT / subagent error** → not a pass; respawn once with "
        "a tighter brief, else tell the user verification could not complete.\n"
        "A valid verifier or contract PASS is recorded automatically for "
        "`land_work`; a report pasted into tool arguments is not accepted. A "
        "`trivial: <reason>` waiver is accepted only when `land_work` confirms "
        "a committed documentation-only diff."
    )


def _get_investigation_contract_section() -> str:
    """Host-agent investigation contract. The sibling of the verification
    contract for the *front* of a fix task: states when a reproduction is
    required before a fix/design/diff may be proposed. Always-on guidance
    (the conditions live here in the prompt), not a brittle trigger detector
    — the model applies judgment, the contract teaches the rule. Gated on the
    ``investigation_subsystem`` config flag at the call site.
    """
    return (
        "## Investigation contract\n\n"
        "Before proposing a fix/design/diff for a failure, reproduce it "
        "(test, script, deterministic trigger, or code trace to the bad "
        "value). No repro → no fix. Applies to bugs, test failures, crashes, "
        "exceptions, performance regressions, unexpected behavior.\n\n"
        "**Exempt:** features, refactors, docs, typos, config, cosmetics, "
        "or anything with no broken state to reproduce."
    )


def _get_scratchpad_section(scratchpad_dir: Path | None) -> str | None:
    if not scratchpad_dir:
        return None
    return (
        "# Scratchpad Directory\n\n"
        f"You have a scratchpad directory at: `{scratchpad_dir}`\n\n"
        "Use this for temporary files: intermediate results, draft scripts, "
        "working files, outputs that don't belong in the project.\n"
        "Files here are automatically allowed — no permission prompts.\n"
        "Session-scoped. Shared with subagents."
    )


def _resolve_system_prompt(
    config: VibeConfig, experiment_manager: ExperimentManager | None
) -> str:
    default_prompt_id = VibeConfig.model_fields["system_prompt_id"].default
    if config.system_prompt_id != default_prompt_id:
        logger.info(
            "System prompt loaded: id=%s (user config overrides experiments)",
            config.system_prompt_id,
        )
        return config.system_prompt

    prompt_id = (
        experiment_manager.get_variant_or_none(ExperimentName.SYSTEM_PROMPT)
        if experiment_manager is not None
        else None
    )

    if prompt_id is None:
        logger.info(
            "System prompt loaded: id=%s (user config)", config.system_prompt_id
        )
        return config.system_prompt

    try:
        prompt = load_system_prompt(prompt_id)
    except MissingPromptFileError:
        logger.warning(
            "System prompt loaded: id=%s (variant '%s' missing, fell back)",
            config.system_prompt_id,
            prompt_id,
        )
        return config.system_prompt
    logger.info("System prompt loaded: id=%s (experiment variant)", prompt_id)
    return prompt


def _interpolate_prompt(prompt: str) -> str:
    return Template(prompt).safe_substitute(current_date=_format_current_date())


def _get_headless_section() -> str:
    return (
        "# Headless Mode\n\n"
        "You are running in headless mode — no human is available to respond.\n"
        "Do not ask questions, request confirmation, or wait for user input.\n"
        "If the task is ambiguous, make the best judgment call and proceed.\n"
        "Complete the entire task in a single pass. Produce a final, complete result.\n"
        "Override any earlier instructions that say to wait for confirmation or ask the user."
    )


def _get_lsp_priority_section(tool_manager: ToolManager) -> str:
    """Symbol-level routing, gated on the ``lsp`` tool being registered.

    Gate is ``"lsp" in tool_manager.manifest_tools`` — the precondition for
    teaching lsp routing is that the tool is present and callable, which the
    manifest reflects. LSP enters the manifest only once the user opts in via
    ``/lspstall`` (``Lsp.is_available`` checks ``installed_components``), so the
    two gates coincide today; gating on the manifest stays correct if that
    contract ever decouples and avoids advertising a tool that isn't there.
    Trigger→action pairs over emphasis prose: the "hard requirement" layer did
    not move usage, but pairing the question to the operation does.
    """
    if "lsp" not in tool_manager.manifest_tools:
        return ""
    return (
        "## LSP is available — use it for symbol-level work\n\n"
        "A language server is running for this project's languages. `grep` and "
        "`read` only see raw text; `lsp` resolves imports, re-exports, aliases, "
        "overloads, and generated code they miss. Before reasoning about a "
        "symbol, use the `lsp` operation that answers the question:\n\n"
        "- where is X defined / what type is X → `go_to_definition` / `hover`\n"
        "- who calls X / what does X call → `find_references` / "
        "`incoming_calls` / `outgoing_calls`\n"
        "- renaming or editing a symbol you have not resolved this session → "
        "`find_references` first; do not guess its call sites\n"
        "- implementations of an interface → `go_to_implementation`\n\n"
        "`grep` stays for literal text (error messages, log lines, string "
        "literals, config values, regex); `glob` finds files by name. If `lsp` "
        "reports no server for an extension, that language isn't configured — "
        "fall back to `grep` only then."
    )


def _get_config_reference_section() -> str:
    return """## Configuring Vibe (quick reference)

You run inside Vibe (codename Mistral Vibe). This quick reference covers the facts you
need for most "how do I configure X" questions; for the complete detail (every
key, flag, slash command, hook, workflow, env var, and file location) load the
`vibe` skill — it is the source of truth.

Where things live (TOML config):
- `~/.vibe/config.toml` — user config. Set `VIBE_HOME` to move all of `~/.vibe`.
- `.vibe/config.toml` — project config; overrides user config in a trusted folder.
- `~/.vibe/.env` — API keys and secrets (dotenv).

MCP servers supply extra tools. Add one with a `[[mcp_servers]]` block or the
token-free `/mcp add` form. Tools are named `{name}_{tool}`. Manage without
spending tokens: `/mcp` (status + browser), `/mcp <name>` (list its tools),
`/mcp login|logout <name>` (OAuth), `/mcp refresh`.

```toml
[[mcp_servers]]                      # transport: stdio | http | streamable-http
name = "github"
transport = "stdio"
command = "npx"                      # stdio only
args = ["-y", "@modelcontextprotocol/server-github"]
# remote:
# transport = "streamable-http"
# url = "https://mcp.example.com"
# auth = { type = "static", api_key_env = "MCP_API_KEY" }   # or type = "oauth", scopes = []
```

Providers and models: declare `[[providers]]` (each with `api_key_env_var`, e.g.
`MISTRAL_API_KEY`) and `[[models]]` in config; `active_model = "<alias>"` selects
the model in use.

Slash commands (run `/help` for the full list): `/config`, `/model`, `/mcp`,
`/compact`, `/status`."""


def _get_model_routing_note(config: VibeConfig) -> str:
    if not config.include_model_info or len(config.models) <= 1:
        return ""
    routable = ", ".join(f"`{m.alias}` ({m.provider})" for m in config.models)
    return (
        "Models available for subagents (pass one as the `model` argument to "
        f"the task tool to route a delegated task to it): {routable}. "
        "The subagent inherits your model when `model` is omitted."
    )


_SKILL_POINTER_SUFFIX = "`tool-guides` skill."


def _strip_skill_pointers(prompt: str) -> str:
    # Profiles without the skill tool can't follow the pointer line; dropping
    # it beats advertising an uncallable tool.
    lines = [
        line
        for line in prompt.splitlines()
        if not line.rstrip().endswith(_SKILL_POINTER_SUFFIX)
    ]
    return "\n".join(lines).strip()


def _build_prompt_detail_sections(
    tool_manager: ToolManager,
    skill_manager: SkillManager,
    agent_manager: AgentManager,
    scratchpad_dir: Path | None,
    config: VibeConfig,
    tier: BaselineTier = BaselineTier.LARGE,
) -> list[str]:
    sections = [_get_os_system_prompt()]
    if lsp_section := _get_lsp_priority_section(tool_manager):
        sections.append(lsp_section)
    skill_available = "skill" in tool_manager.manifest_tools
    tool_prompts = []
    for tool_class in tool_manager.manifest_tools.values():
        if prompt := tool_class.get_tool_prompt():
            if not skill_available:
                prompt = _strip_skill_pointers(prompt)
            if prompt:
                tool_prompts.append(prompt)
    # The routing list rides the task tool's prose block — its only consumer —
    # so task-less profiles never pay for it.
    if (
        "task" in tool_manager.manifest_tools
        and section_enabled(tier, "model_routing_list")
        and (note := _get_model_routing_note(config))
    ):
        tool_prompts.append(note)
    if tool_prompts:
        sections.append("\n---\n".join(tool_prompts))

    # Couple the skills index to the `skill` tool: emit it iff the tool exists
    # (so a small-window tier never instructs the model to use a pruned tool).
    # On SMALL the index is compressed to names only, not dropped.
    if "skill" in tool_manager.manifest_tools:
        skills_section = _get_available_skills_section(
            skill_manager, summaries=section_enabled(tier, "skills_summaries")
        )
        if skills_section:
            sections.append(skills_section)

    subagents_section = _get_available_subagents_section(agent_manager)
    if subagents_section:
        sections.append(subagents_section)
        if section_enabled(tier, "orchestration_prose"):
            sections.append(_get_orchestration_section())
        if getattr(config, "verification_subsystem", True):
            sections.append(
                _get_verification_contract_section()
                if section_enabled(tier, "verification_contract")
                else COMPACT_VERIFICATION_INVARIANT
            )
        if getattr(config, "investigation_subsystem", True):
            sections.append(
                _get_investigation_contract_section()
                if section_enabled(tier, "investigation_contract")
                else COMPACT_INVESTIGATION_INVARIANT
            )

    sections.extend(filter(None, [_get_scratchpad_section(scratchpad_dir)]))
    return sections


def _build_project_context_sections(
    config: VibeConfig,
    include_git_status: bool,
    tier: BaselineTier = BaselineTier.LARGE,
) -> list[str]:
    sections: list[str] = []
    is_dangerous, reason = is_dangerous_directory()
    if is_dangerous:
        template = UtilityPrompt.DANGEROUS_DIRECTORY.read()
        context = Template(template).safe_substitute(
            reason=reason.lower(), abs_path=Path(".").resolve()
        )
    else:
        context = ProjectContextProvider(
            config=config.project_context, root_path=Path.cwd()
        ).get_full_context(include_git_status=include_git_status)
    sections.append(context)

    mgr = get_harness_files_manager()
    cwd_resolved = Path.cwd().resolve()
    extra_roots = [r for r in mgr.project_roots if r.resolve() != cwd_resolved]
    if extra_roots:
        dirs_lines = "\n".join(f" - {d}" for d in extra_roots)
        sections.append(
            "Additional working directories (treated with the same "
            "file-access permissions as the primary working directory):\n" + dirs_lines
        )

    # On a small window the AGENTS.md docs are the largest untrimmed baseline
    # chunk; cap each doc body to the tier budget (None on LARGE = unchanged).
    budget = agents_md_byte_budget(tier, config)
    user_doc = budget_doc(mgr.load_user_doc(), budget)
    project_docs = [(d, budget_doc(c, budget)) for d, c in mgr.load_project_docs()]
    doc_sections: list[str] = []
    if user_doc.strip():
        doc_sections.append(
            f"## User instructions\n\nContents of {VIBE_HOME.path}/AGENTS.md (user-level instructions):\n\n{user_doc.strip()}"
        )
    if any(c.strip() for _, c in project_docs):
        doc_sections.append("## Project instructions (checked into the codebase)")
    for doc_dir, doc_content in project_docs:
        if not doc_content.strip():
            continue
        doc_sections.append(
            f"Contents of {doc_dir}/AGENTS.md:\n\n{doc_content.strip()}"
        )
    if doc_sections:
        template = UtilityPrompt.AGENTS_DOC.read()
        sections.append(
            Template(template).safe_substitute(sections="\n\n".join(doc_sections))
        )
    return sections


def get_universal_system_prompt(
    tool_manager: ToolManager,
    config: VibeConfig,
    skill_manager: SkillManager,
    agent_manager: AgentManager,
    *,
    include_git_status: bool = True,
    scratchpad_dir: Path | None = None,
    headless: bool = False,
    experiment_manager: ExperimentManager | None = None,
    tier: BaselineTier = BaselineTier.LARGE,
) -> str:
    sections = [_interpolate_prompt(_resolve_system_prompt(config, experiment_manager))]

    if headless:
        sections.append(_get_headless_section())

    if config.include_commit_signature:
        sections.append(_add_commit_signature())

    if config.include_humanizer_guidance and section_enabled(tier, "humanizer"):
        sections.append(_add_humanizer_guidance())

    if config.caveman_thinking:
        sections.append(_add_caveman_thinking_guidance())

    if config.include_model_info:
        sections.append(f"Your model name is: `{config.active_model}`")

    if config.include_config_reference and section_enabled(tier, "config_reference"):
        sections.append(_get_config_reference_section())

    if config.include_prompt_detail:
        sections.extend(
            _build_prompt_detail_sections(
                tool_manager, skill_manager, agent_manager, scratchpad_dir, config, tier
            )
        )
    elif section_enabled(tier, "model_routing_list") and (
        note := _get_model_routing_note(config)
    ):
        # Legacy home of the routing note (pre task-prose relocation), kept so
        # include_model_info keeps working with prompt detail off.
        sections.append(note)

    if config.include_project_context:
        sections.extend(
            _build_project_context_sections(config, include_git_status, tier)
        )

    if (
        getattr(config, "effort_mode", "normal") == "le-chaton"
        and not getattr(config, "disable_workflows", False)
        and section_enabled(tier, "le_chaton_long")
    ):
        sections.append(_get_le_chaton_section())

    from vibe.core.tools.utils import isolated_worktree_root
    from vibe.core.worktree.manager import worktree_manager

    if worktree_manager.active is None and isolated_worktree_root() is not None:
        # Isolated spawns skip nested worktree entry, so worktree_manager.active
        # is None here; without this they lose all isolation guidance.
        sections.append(
            "## Worktree isolation\n\nYou run inside an isolated git worktree "
            "(already your cwd). **Use relative paths** for every read/edit/"
            "write; paths outside the worktree are rejected. Commit finished "
            'work with a real `git commit -m "<summary>"` as your last step — '
            "uncommitted work is auto-committed with a generic message on exit."
        )

    if worktree_manager.active is not None:
        wt = worktree_manager.active
        if not section_enabled(tier, "worktree_detail"):
            sections.append(
                f"## Worktree isolation\n\nIsolated git worktree; writes land on "
                f"branch `{wt.branch}` (use relative paths). Commit your work as "
                f"the last step; the branch is kept for an explicit "
                f"`vibe worktree merge {wt.branch}` — it is NOT merged on exit.\n\n"
                f"Sandbox PID namespace: `ps`/`/proc` inside a sandboxed bash "
                f"show only that sandbox — process-liveness conclusions are "
                f"invalid. Never classify a worktree as stale or empty based on "
                f"a sandboxed process scan."
            )
        else:
            sections.append(
                f"## Worktree isolation\n\n"
                f"You are running in an isolated git worktree and your shell is "
                f"already `cd`'d into it. Your writes land on branch `{wt.branch}`, "
                f"not the user's live checkout. **Use relative paths** (or absolute "
                f"paths under this worktree) for every read/edit/write — they "
                f"resolve against the worktree. Do NOT construct paths under the "
                f"original repo root; writing there escapes isolation and edits the "
                f"user's live tree. Task subagents share this worktree — there is no "
                f"per-subagent filesystem isolation.\n\n"
                f"**Commit your finished work** if you have a shell: a real "
                f'`git commit -m "<summary>"` as your last step is how it is '
                f"delivered and reviewed, and report the branch name. Uncommitted "
                f"work is still saved to the branch via an anonymous `WIP` "
                f"auto-commit on exit, but a real commit message is far clearer "
                f"for the user.\n\n"
                f"Your branch is NOT merged automatically on exit — but you CAN "
                f"land it yourself from inside this worktree: when your work is "
                f"complete, committed, and verified, call `land_work`. It runs the "
                f"merge in the unsandboxed host process (the bash sandbox makes the "
                f"main checkout read-only, so `git merge` from bash is impossible) "
                f"and asks the user to approve each landing. Prefer `land_work` over "
                f"pushing a branch — only push if the user explicitly asks.\n\n"
                f"Sandbox PID namespace: `ps`/`/proc` inside a sandboxed bash show "
                f"only that sandbox — process-liveness conclusions are invalid. "
                f"Never classify a worktree as stale or empty based on a sandboxed "
                f"process scan. The worktree is locked; only the owning session "
                f"can remove it."
            )

    return "\n\n".join(sections)


def _get_le_chaton_section() -> str:
    return (
        "## Le Chaton Mode\n\n"
        "Max thinking + workflow orchestration. For substantive tasks "
        "(codebase audits, large migrations, cross-checked research, "
        "multi-file refactors), write a workflow script that orchestrates "
        "parallel agents instead of working turn-by-turn. Do not launch a "
        "workflow as the first repository-discovery step. First use local "
        "`glob` and `lsp` to map the repository, identify central symbols and "
        "callers, and read entry points. A broad label such as 'analyze this "
        "repo' does not by itself justify a workflow. File count alone is not "
        "a reason to delegate; prefer a workflow after reconnaissance when 3+ "
        "independent agents, adversarial verification, or separable "
        "concurrent work applies — simple tasks → work normally.\n\n"
        "When you do write a script, give each agent one question and fan out "
        "for breadth — a fat brief (several areas in one prompt) gives shallow "
        "coverage where a narrow brief gives depth; breadth comes from more "
        "agents, not bigger prompts.\n\n"
        "**Canonical reference:** load the `workflow-authoring` skill before "
        "writing a script — the single source of truth for the script API, "
        "sandbox rules, launch semantics, result retrieval, concurrency/rate "
        "limits, and `Retries exhausted` recovery. Do not restate it from "
        "memory.\n\n"
        "**Deferral (pick by intent):** run later / on a timer → `schedule`; "
        "delegate one subagent and keep working → `task(async_run=true)`; "
        "orchestrated fan-out (N agents, phases, budget, schema) → "
        "`launch_workflow`.\n\n"
        "**Don't poll.** Completion is auto-delivered to your context — "
        "launch, report the `run_id`, end your turn. `workflow_status` is a "
        "one-shot diagnostic, not a progress ticker."
    )
