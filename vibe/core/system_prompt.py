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
from vibe.core.workflows.runtime import DEFAULT_MAX_CONCURRENT

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
        context = Template(template).safe_substitute(
            abs_path=str(self.root_path), git_status=git_status
        )

        from vibe.core.worktree.manager import worktree_manager

        if (wt := worktree_manager.active) is not None:
            context += (
                f"\n\n## Worktree isolation active\n"
                f"Writes land on branch `{wt.branch}` (isolated worktree at "
                f"`{wt.worktree_path}`); changes are isolated until merged. "
                f"Commit finished work with a clear message so it merges back "
                f"cleanly. Original repo root: `{wt.original_repo_root}`."
            )
        return context


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


def _get_available_skills_section(skill_manager: SkillManager) -> str:
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

    for name, info in sorted(skills.items()):
        summary = _skill_index_line(info)
        lines.append(f"- **{html.escape(str(name))}**: {html.escape(summary)}")

    lines.append("</available_skills>")

    return "\n".join(lines)


def _get_available_subagents_section(agent_manager: AgentManager) -> str:
    agents = agent_manager.get_subagents()
    if not agents:
        return ""

    lines = ["# Available Subagents", ""]
    lines.append("The following subagents can be spawned via the Task tool:")
    for agent in agents:
        lines.append(f"- **{agent.name}**: {agent.description}")

    return "\n".join(lines)


_ORCHESTRATION_SECTION = """## Orchestrating Subagents

Local tools first, delegation second. For an unfamiliar repository, establish a \
baseline yourself before spawning subagents: map files with `glob`, resolve central \
symbols and callers with `lsp`, then read the entry points and representative \
tests. This reconnaissance determines whether delegation will add value and gives \
you enough context to write precise briefs.

Orchestration is a default skill for work that remains broad after reconnaissance, \
not a substitute for reconnaissance. Use read-only subagents for independent \
questions or a review that would otherwise require reading 10+ files. Keep the main \
context for synthesis, decisions, and edits.

Pick the profile by the question:
- `explore` — codebase questions and searches: where/how is X done, trace a \
flow, gather all call sites, multi-file reads.
- `research` — anything outside the repo: docs, library/API behavior, version \
checks, web lookups.
- `reviewer` — adversarial review of a diff, branch, or file; runs targeted \
checks and tests to find what breaks.
- `debugger` — a specific failure or flaky test: reproduce, isolate, and trace \
the root cause; returns the cause + minimal fix (it diagnoses, you apply).
- `planner` — design an approach before building: returns a code-grounded, \
phased plan with risks and the files to touch (it plans, you decide).
- `security` — defensive vuln audit of a change or area: traces untrusted input \
to sinks and returns severity-ranked findings with fixes.
- `verifier` — gate a *completed* implementation: proves it works by trying to \
break it, then emits a PASS/FAIL/PARTIAL verdict with command evidence.

Fan out: issue several `task` calls in one turn so independent sub-questions \
run in parallel. Give each a self-contained brief and ask for findings and \
conclusions, not raw file dumps; each runs in its own context, so a wide \
investigation costs you only the returned conclusions.

Don't delegate trivia. For a single known lookup — you already have the file \
path or symbol — just `read`/`grep` it directly. Delegate breadth and \
uncertainty; handle pinpoints yourself. The read-only profiles (explore, \
research, reviewer, debugger, planner, security, verifier) can't write files or \
ask the user — you own every edit and all user interaction. `editor` and \
`worker` are write-capable but only function inside a workflow worktree; in a \
plain `task` call their writes are approval-gated and skipped, so treat every \
edit as yours."""


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
        "When non-trivial implementation happens on your turn — 3+ file edits, "
        "backend/API changes, or infrastructure changes — independent "
        "verification must happen before you report completion, regardless of "
        "who did the implementing (you directly, a subagent, or a workflow). "
        "You are the one reporting to the user; you own the gate.\n\n"
        "Spawn the `verifier` subagent via the `task` tool. Pass the original "
        "task, every file that changed (by anyone), and the approach taken. "
        "Flag concerns you have, but do NOT share your own test results or "
        "claim things work — the verifier verifies independently. Your own "
        "checks and a subagent's self-checks do not substitute; only the "
        "verifier assigns a verdict, and you cannot self-assign done by "
        "listing caveats in your summary.\n\n"
        "- On **FAIL**: fix the issue and re-run the verifier with its findings "
        "plus your fix. Repeat until PASS.\n"
        "- On **PASS**: spot-check it. Re-run 2-3 of the commands from the "
        "verifier's report and confirm every PASS step has a command block "
        "whose output matches your re-run. If a PASS step has no command, or "
        "the output diverges, resume the verifier with the specifics.\n"
        "- On **PARTIAL**: report to the user what passed and what could not be "
        "verified (and why — missing tool, server wouldn't start, no test "
        "framework). Do not present PARTIAL as success.\n\n"
        "Trivial work (a one-line fix, a single read-only answer, a typo) does "
        "not need a verifier. Use judgment — but when in doubt on non-trivial "
        "work, verify."
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
        "Before you propose a fix, design, or diff for a failure, you must "
        "have reproduced the failure — a test, a script, a deterministic "
        "trigger, or a code trace that proves the failure exists and shows "
        "where it originates. Proposing mechanisms for an unreproduced "
        "premise reproduces symptoms, masks the root cause, and designs into "
        "a stale mental model.\n\n"
        "**Reproduce first when the task is:** fixing a bug, test failure, "
        "error, crash, exception, performance regression, or any unexpected "
        "behavior. The reproduction is the root-cause evidence; trace the bad "
        "value to its source before touching code.\n\n"
        "**Exempt (no reproduction needed):** adding a feature, refactoring, "
        "docs, typo, config change, one-line change, cosmetic edits, or "
        "anything where there is no 'broken' state to reproduce. A design for "
        "greenfield work proceeds from reading the code, not from a repro.\n\n"
        "When a fix task lands and you have NOT reproduced it, stop and "
        "reproduce before proposing the fix. A structural change made against "
        "an assumed gap — without a trace proving the gap — is the failure "
        "mode this contract exists to prevent."
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


def _get_lsp_priority_section(config: VibeConfig) -> str:
    """Availability-conditional LSP-mandatory guidance.

    Emitted only when the user has opted into LSP (``/lspstall``), so it never
    advertises a tool that isn't running. Marks LSP as a hard requirement for
    symbol-level work: every code analysis, review, and exploration task MUST
    resolve symbols through ``lsp`` rather than text search, and edits are
    gated on a prior ``lsp`` lookup. A static "prefer LSP" rule is noise when
    LSP isn't installed; this carries the emphasis only when it can be acted on.
    """
    if "lsp" not in getattr(config, "installed_components", []):
        return ""
    return (
        "## LSP is available — you MUST use it for symbol-level work\n\n"
        "A language server is running for this project's languages. This is a "
        "hard requirement, not a preference. On every code analysis, review, "
        "and exploration task you MUST resolve symbols through `lsp` before "
        "reasoning about them. `grep` and `read` only see raw text and will "
        "give you wrong call sites and stale signatures, because they miss "
        "imports, re-exports, aliases, overloads, and generated code.\n\n"
        "- Symbol question (definition, type, callers, callees, "
        "implementations)? `lsp` is mandatory — "
        "`go_to_definition`/`find_references`/`hover`/`incoming_calls`/"
        "`outgoing_calls`/`go_to_implementation`. Never grep for a "
        "function/method/class name that LSP can resolve.\n"
        "- Reviewing or analyzing code? Trace the real call graph and types "
        "with `lsp` instead of grepping identifiers and guessing how they "
        "connect.\n"
        "- Before editing a symbol you have not resolved this session, run "
        "`lsp hover` (and `find_references`) on it first. Do not guess its "
        "signature or call sites — wrong edits come from guessed shapes.\n\n"
        "`grep` stays for literal text (error messages, log lines, string "
        "literals, config values, regex); `glob` stays for finding files by "
        "name. If `lsp` reports no server for an extension, that language "
        "isn't supported there — fall back to `grep` only then."
    )


def _get_config_reference_section() -> str:
    return """## Configuring Vibe (quick reference)

You run inside Vibe (codename Chaton). This quick reference covers the facts you
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


def _build_prompt_detail_sections(
    tool_manager: ToolManager,
    skill_manager: SkillManager,
    agent_manager: AgentManager,
    scratchpad_dir: Path | None,
    config: VibeConfig,
) -> list[str]:
    sections = [_get_os_system_prompt()]
    if lsp_section := _get_lsp_priority_section(config):
        sections.append(lsp_section)
    tool_prompts = []
    for tool_class in tool_manager.available_tools.values():
        if prompt := tool_class.get_tool_prompt():
            tool_prompts.append(prompt)
    if tool_prompts:
        sections.append("\n---\n".join(tool_prompts))

    skills_section = _get_available_skills_section(skill_manager)
    if skills_section:
        sections.append(skills_section)

    subagents_section = _get_available_subagents_section(agent_manager)
    if subagents_section:
        sections.append(subagents_section)
        sections.append(_get_orchestration_section())
        if getattr(config, "verification_subsystem", True):
            sections.append(_get_verification_contract_section())
        if getattr(config, "investigation_subsystem", True):
            sections.append(_get_investigation_contract_section())

    sections.extend(filter(None, [_get_scratchpad_section(scratchpad_dir)]))
    return sections


def _build_project_context_sections(
    config: VibeConfig, include_git_status: bool
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

    user_doc = mgr.load_user_doc()
    project_docs = mgr.load_project_docs()
    doc_sections: list[str] = []
    if user_doc.strip():
        doc_sections.append(
            f"## User instructions\n\nContents of {VIBE_HOME.path}/AGENTS.md (user-level instructions):\n\n{user_doc.strip()}"
        )
    if project_docs:
        doc_sections.append("## Project instructions (checked into the codebase)")
    for doc_dir, doc_content in project_docs:
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
) -> str:
    sections = [_interpolate_prompt(_resolve_system_prompt(config, experiment_manager))]

    if headless:
        sections.append(_get_headless_section())

    if config.include_commit_signature:
        sections.append(_add_commit_signature())

    if config.include_humanizer_guidance:
        sections.append(_add_humanizer_guidance())

    if config.caveman_thinking:
        sections.append(_add_caveman_thinking_guidance())

    if config.include_model_info:
        sections.append(f"Your model name is: `{config.active_model}`")
        if len(config.models) > 1:
            routable = ", ".join(f"`{m.alias}` ({m.provider})" for m in config.models)
            sections.append(
                "Models available for subagents (pass one as the `model` argument "
                f"to the task tool to route a delegated task to it): {routable}. "
                "The subagent inherits your model when `model` is omitted."
            )

    if config.include_config_reference:
        sections.append(_get_config_reference_section())

    if config.include_prompt_detail:
        sections.extend(
            _build_prompt_detail_sections(
                tool_manager, skill_manager, agent_manager, scratchpad_dir, config
            )
        )

    if config.include_project_context:
        sections.extend(_build_project_context_sections(config, include_git_status))

    if getattr(config, "effort_mode", "normal") == "le-chaton" and not getattr(
        config, "disable_workflows", False
    ):
        sections.append(_get_le_chaton_section())

    from vibe.core.worktree.manager import worktree_manager

    if worktree_manager.active is not None:
        wt = worktree_manager.active
        sections.append(
            f"## Worktree isolation\n\n"
            f"You are running in an isolated git worktree. Your writes land on "
            f"branch `{wt.branch}`, not the user's live checkout. Task subagents "
            f"share this worktree — there is no per-subagent filesystem "
            f"isolation.\n\n"
            f"**Commit your finished work** if you have a shell: a real "
            f'`git commit -m "<summary>"` as your last step is how it is '
            f"delivered and reviewed, and report the branch name. Uncommitted "
            f"work still merges back via an anonymous `WIP` auto-save, but a "
            f"real commit message is far clearer for the user.\n\n"
            f"On exit your branch is merged back into the original HEAD "
            f"automatically — rebased onto the latest HEAD first (so concurrent "
            f"sessions don't strand it), then fast-forwarded, including when the "
            f"original tree was dirty at start. The branch is kept for recovery "
            f"(`chaton worktree merge {wt.branch}`) only if it genuinely conflicts "
            f"with another session's changes.\n\n"
            f"Original repo root: `{wt.original_repo_root}`"
        )

    return "\n\n".join(sections)


def _get_le_chaton_section() -> str:
    return (
        "## Le Chaton Mode\n\n"
        "Max thinking + workflow orchestration. For substantive tasks "
        "(codebase audits, large migrations, cross-checked research, multi-file "
        "refactors), write a workflow script that orchestrates parallel agents "
        "instead of working turn-by-turn. Do not launch a workflow as the first "
        "repository-discovery step. First use local `glob` and `lsp` to map the "
        "repository, identify central symbols and callers, and read entry points. "
        "A broad label such as 'analyze this repo' does not by itself justify a "
        "workflow.\n\n"
        "**Canonical reference:** load the `workflow-authoring` skill — it is "
        "the single source of truth for the script API (`agent`/`parallel`/"
        "`pipeline`/`phase`/`log`/`budget`/`workflow`/`args` + synthesis "
        "helpers), sandbox rules (safelisted imports, no `asyncio`, "
        "`str.format()` forbidden), launch semantics (pass source inline), and "
        "result retrieval (`workflow_results(run_id)`, per-agent "
        "`schema_errors`). Load it before writing a script; do not restate it "
        "from memory.\n\n"
        "**Deferral (pick by intent):**\n"
        "| Intent | Mechanism |\n"
        "|---|---|\n"
        "| Run later / on a timer | `schedule` (a timer — executes nothing itself) |\n"
        "| Delegate one subagent, keep working here | `task(async_run=true)` |\n"
        "| Orchestrated fan-out (N agents, phases, budget, schema) | `launch_workflow` |\n\n"
        "**Concurrency & rate limits:** Up to "
        f"{DEFAULT_MAX_CONCURRENT} agents run concurrently per workflow (the "
        "runtime's global cap). Pass `max_concurrency=N` on "
        "`parallel`/`pipeline` only to throttle *below* that — prefer it over "
        "hand-rolling chunked waves. Some providers throttle at 1–3 concurrent "
        "requests, and retry is per-request and uncoordinated across agents, "
        "so a saturated provider can fail several agents at once with "
        "`Retries exhausted`.\n\n"
        "**Recovery (agent died of `Retries exhausted`):** do not re-launch the "
        "same fan-out. Re-run that phase with `max_concurrency=1`, or serialize "
        "via `pipeline`, or `schedule` a retry after the provider's "
        "`Retry-After` (honored up to 60s).\n\n"
        "**Prefer workflows after reconnaissance when:** 3+ independent agents "
        "| adversarial verification adds value | separable work can proceed "
        "concurrently. File count alone is not a reason to delegate. Simple "
        "tasks → work normally.\n\n"
        "**Don't poll workflows.** Completion is auto-delivered to your "
        "context as a user message — launching a workflow and then calling "
        "`workflow_status` repeatedly to watch it is a turn-wasting anti-"
        "pattern. Launch it, report the `run_id`, end your turn, and resume "
        "when the result arrives. If you must revisit a long run, arm a "
        '`schedule` timer (`schedule create interval=2m prompt="..."`) '
        "instead of polling. `workflow_status` is a one-shot diagnostic for a "
        "run you suspect is stuck or runaway, not a progress ticker.\n\n"
        "**Monitor (TUI only):** `/workflows` — x (stop), p (pause/resume), "
        "s (save script as `/<name>` command), o (view script), Enter (drill "
        "into run/agent). In-flight agents show live token totals. This is for "
        "the human watching the terminal, not a prompt for you to poll."
    )
