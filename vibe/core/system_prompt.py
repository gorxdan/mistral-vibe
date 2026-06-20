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
                if (
                    "(" in commit_msg
                    and ")" in commit_msg
                    and (paren_index := commit_msg.rfind("(")) > 0
                ):
                    commit_msg = commit_msg[:paren_index].strip()
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
                    self._run_git,
                    ["log", "--oneline", f"-{num_commits}", "--decorate"],
                    timeout,
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
                f"`{wt.worktree_path}`). Changes are isolated until merged. "
                f"Original repo root: `{wt.original_repo_root}`."
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
        "You have access to these skills. Each entry shows only a short trigger"
        " line; call the `skill` tool with the skill's `name` to load its full"
        " instructions, workflows, and bundled resources (the `description`"
        " frontmatter is intentionally not loaded here to save context).",
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

Orchestration is a default skill, not a last resort. For non-trivial \
investigation or review, you are the lead: spin up read-only subagents via the \
`task` tool and keep the main context for synthesis, decisions, and edits. If \
answering means reading 10+ files or reviewing a branch, send a subagent and \
get back a conclusion — don't grind it all through here.

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

Fan out: issue several `task` calls in one turn so independent sub-questions \
run in parallel. Give each a self-contained brief and ask for findings and \
conclusions, not raw file dumps; each runs in its own context, so a wide \
investigation costs you only the returned conclusions.

Don't delegate trivia. For a single known lookup — you already have the file \
path or symbol — just `read`/`grep` it directly. Delegate breadth and \
uncertainty; handle pinpoints yourself. Subagents are read-only investigators \
that can't write files or ask the user — you own every edit and all user \
interaction."""


def _get_orchestration_section() -> str:
    """Normal-mode orchestration directive — instructs the host to delegate
    cross-file investigation/review to subagents via the task tool. Shown
    whenever subagents exist (le-chaton layers workflows on top of this).
    """
    return _ORCHESTRATION_SECTION


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


def get_universal_system_prompt(  # noqa: PLR0912, PLR0914, PLR0915
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

    if config.include_model_info:
        sections.append(f"Your model name is: `{config.active_model}`")

    if config.include_prompt_detail:
        sections.append(_get_os_system_prompt())
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

        sections.extend(filter(None, [_get_scratchpad_section(scratchpad_dir)]))

    if config.include_project_context:
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
                "file-access permissions as the primary working directory):\n"
                + dirs_lines
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
            f"branch `{wt.branch}`, not the user's live checkout. Commit "
            f"logically and report the branch name. Task subagents share this "
            f"worktree — there is no per-subagent filesystem isolation.\n\n"
            f"Original repo root: `{wt.original_repo_root}`"
        )

    return "\n\n".join(sections)


def _get_le_chaton_section() -> str:
    return (
        "## Le Chaton Mode\n\n"
        "You are in le chaton effort mode: max thinking combined with automatic "
        "workflow orchestration. For substantive tasks — codebase audits, large "
        "migrations, cross-checked research, multi-file refactors — write a "
        "workflow script that orchestrates parallel agents instead of working "
        "through the task turn-by-turn.\n\n"
        "A workflow script is a Python file with an `async def main()` function. "
        "The runtime injects these functions:\n"
        "- `agent(prompt, *, agent='explore', label=None, phase=None, schema=None, "
        "isolation=None)` — spawn a subagent; isolation='worktree' runs it in a "
        "fresh git worktree (isolates file edits for parallel agents). Profiles: "
        "explore (grep/read), research (+web), reviewer (+bash), worker (full "
        "tools incl. MCP; requires isolation='worktree')\n"
        "- `parallel(*thunks)` — run thunks concurrently, results in order; a "
        "thunk that raises yields None (filter the results)\n"
        "- `pipeline(items, *stages)` — run each item through all stages with no "
        "barrier between stages (item A can be in stage 3 while B is in stage 1); "
        "each stage gets (prev, item, index); one stage acts as a concurrent map\n"
        "- `phase(name)` — declare a phase for progress tracking\n"
        "- `log(msg)` — log a progress message\n"
        "- `budget` — token budget object with `.total` and `.remaining()`\n"
        "- `workflow(name, args=None)` — run another workflow inline as a "
        "sub-step (shares budget/agents; one level deep)\n"
        "- `args` — structured input from the invocation command\n\n"
        "Write the script to a file, then tell the user to run it with the "
        "workflow tool or save it as a command. You can also launch it directly "
        "using the `launch_workflow` tool, which validates and runs the script "
        "in the background and previews the planned phases at the approval "
        "prompt. For simple tasks (single-file edits, quick questions), work "
        "normally without a workflow.\n\n"
        "Prefer workflows when: the task needs 3+ independent agents, adversarial "
        "verification adds value, or the work spans many files. Use `parallel` "
        "for independent same-stage work and `pipeline` for multi-stage "
        "per-item flows (e.g. find -> verify).\n\n"
        "Once launched, the run is monitored in `/workflows`: keys are x (stop), "
        "p (pause/resume — in-flight agents finish, new agents block), s (save "
        "the run's script as a reusable /<name> command), o (view the full "
        "script), and Enter drills into a run and then into each agent. "
        "In-flight agents show live token totals in the drill-down."
    )
