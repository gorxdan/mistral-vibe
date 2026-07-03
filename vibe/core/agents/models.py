from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING, Any

from vibe.core.paths import PLANS_DIR
from vibe.core.utils import name_matches

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class AgentSafety(StrEnum):
    SAFE = auto()
    NEUTRAL = auto()
    DESTRUCTIVE = auto()
    YOLO = auto()


class AgentType(StrEnum):
    AGENT = auto()
    SUBAGENT = auto()


class BuiltinAgentName(StrEnum):
    DEFAULT = "default"
    CHAT = "chat"
    PLAN = "plan"
    ACCEPT_EDITS = "accept-edits"
    AUTO_APPROVE = "auto-approve"
    EXPLORE = "explore"
    RESEARCH = "research"
    REVIEWER = "reviewer"
    DEBUGGER = "debugger"
    PLANNER = "planner"
    SECURITY = "security"
    VERIFIER = "verifier"
    EDITOR = "editor"
    WORKER = "worker"
    GRUNT = "grunt"
    LEAN = "lean"
    COORDINATOR = "coordinator"


@dataclass(frozen=True)
class AgentProfile:
    name: str
    display_name: str
    description: str
    safety: AgentSafety
    agent_type: AgentType = AgentType.AGENT
    overrides: dict[str, Any] = field(default_factory=dict)
    install_required: bool = False

    def apply_to_config(self, base: VibeConfig) -> VibeConfig:
        from vibe.core.config import VibeConfig as VC

        merged = _deep_merge(
            base.model_dump(),
            {k: v for k, v in self.overrides.items() if k != "base_disabled"},
        )
        base_disabled = self.overrides.get("base_disabled")
        if isinstance(base_disabled, list):
            merged["disabled_tools"] = list({
                *base_disabled,
                *merged.get("disabled_tools", []),
            })

        # Environment-level disables (set by ACP/programmatic mode) must take
        # precedence over an agent's enabled_tools allowlist
        if base.disabled_tools and merged.get("enabled_tools"):
            merged["enabled_tools"] = [
                t
                for t in merged["enabled_tools"]
                if not name_matches(t, base.disabled_tools)
            ]

        from vibe.core.config._settings import skip_api_key_check

        with skip_api_key_check():
            return VC.model_validate(merged)

    @classmethod
    def from_toml(cls, path: Path) -> AgentProfile:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls(
            name=path.stem,
            display_name=data.pop("display_name", path.stem.replace("-", " ").title()),
            description=data.pop("description", ""),
            safety=AgentSafety(data.pop("safety", AgentSafety.NEUTRAL)),
            agent_type=AgentType(data.pop("agent_type", AgentType.AGENT)),
            overrides=data,
        )


# Tools whose presence in an enabled_tools allowlist means the profile can write
# destructively and must run in its own worktree.
_WRITE_TOOLS = {"write_file", "edit"}


def profile_requires_isolation(profile: AgentProfile) -> bool:
    """True if *profile* can mutate files or run unrestricted shell, and so
    must run in an isolated worktree to avoid racing other agents or the live
    checkout.

    - No ``enabled_tools`` allowlist -> full tool set (incl. write) -> isolate.
    - Allowlist contains a write tool (write_file/edit) -> isolate.
    - Allowlist contains ``bash`` *without* a denylist jail -> isolate. The
      reviewer/debugger/security profiles ship a read-only bash jail
      (``overrides['tools']['bash']['denylist']`` hard-denies rm/git reset/etc.),
      so they are safe in-process; an un-jailed bash can rm -rf, so isolate.
    """
    overrides = profile.overrides or {}
    tools = overrides.get("enabled_tools")
    if not tools:
        return True
    if any(t in _WRITE_TOOLS for t in tools):
        return True
    if "bash" in tools:
        bash_cfg = (overrides.get("tools") or {}).get("bash") or {}
        if not bash_cfg.get("denylist"):
            return True
    return False


CHAT_AGENT_TOOLS = ["grep", "read", "ask_user_question", "task"]


def _plan_overrides() -> dict[str, Any]:
    plans_pattern = str(PLANS_DIR.path / "*")
    return {
        "base_disabled": ["enter_plan_mode"],
        "tools": {
            "write_file": {"permission": "never", "allowlist": [plans_pattern]},
            "edit": {"permission": "never", "allowlist": [plans_pattern]},
        },
    }


# Read-only review agents (reviewer / debugger / security) get `bash` so they
# can run git inspection and tests — but they must NOT mutate the user's code.
# This is a hard policy jail on bash, enforced by the tool's AST-split
# allow/deny engine (every command node in a compound command is checked):
#   - ALLOW (auto-run, works headless): git inspection + file-reading + named
#     test/lint runners → the agent can do its job without an approval prompt.
#   - DENY (hard NEVER, precedence over allow): anything that mutates code/git
#     state, touches the network, installs packages, or escalates privilege.
#   - Anything else falls through to the default ASK (skipped headless; the
#     user can approve "when necessary" in an interactive session).
# General-purpose write-capable tools (sed/awk/find/python/perl/env/xargs/sort
# -o, shell redirects) are deliberately NOT allowlisted — they fall to ASK
# rather than auto-run, since a prefix policy can't prove they won't write.
_REVIEW_BASH_ALLOWLIST = [
    "cd",
    "echo",
    "pwd",
    "true",
    "false",
    # git inspection (read-only)
    "git diff",
    "git log",
    "git status",
    "git show",
    "git blame",
    "git rev-parse",
    "git diff-tree",
    "git ls-files",
    "git ls-tree",
    "git cat-file",
    "git shortlog",
    "git describe",
    "git for-each-ref",
    "git rev-list",
    "git name-rev",
    "git merge-base",
    "git grep",
    "git branch --list",
    "git branch -a",
    "git branch -v",
    "git tag --list",
    # file reading / inspection
    "cat",
    "head",
    "tail",
    "wc",
    "file",
    "stat",
    "ls",
    "tree",
    "diff",
    "comm",
    "nl",
    "column",
    "jq",
    "grep",
    "rg",
    "ag",
    "whoami",
    "date",
    "which",
    "type",
    "uname",
    "basename",
    "dirname",
    # test / lint / typecheck runners (run the repo's own checks)
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "tox",
    "ruff",
    "mypy",
    "pyright",
    "flake8",
    "bandit",
    "npm test",
    "npm run test",
    "yarn test",
    "pnpm test",
    "jest",
    "vitest",
    "eslint",
    "tsc",
    "cargo test",
    "cargo check",
    "cargo clippy",
    "go test",
    "go vet",
    "make test",
    "make check",
    "make lint",
]
_REVIEW_BASH_DENYLIST = [
    # git mutation / network
    "git commit",
    "git push",
    "git pull",
    "git fetch",
    "git reset",
    "git checkout",
    "git switch",
    "git restore",
    "git clean",
    "git add",
    "git rm",
    "git mv",
    "git stash",
    "git merge",
    "git rebase",
    "git cherry-pick",
    "git revert",
    "git apply",
    "git am",
    "git update-ref",
    "git gc",
    "git filter-branch",
    "git config",
    "git remote",
    "git tag -d",
    "git branch -d",
    "git branch -D",
    "git worktree",
    # filesystem mutation
    "rm",
    "rmdir",
    "mv",
    "cp",
    "dd",
    "shred",
    "truncate",
    "ln",
    "chmod",
    "chown",
    "chgrp",
    "sed -i",
    "perl -i",
    "tee",
    "install",
    # network / exfil
    "curl",
    "wget",
    "nc",
    "ncat",
    "netcat",
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "telnet",
    # package installs / privilege / system control
    "sudo",
    "su",
    "pip install",
    "pip3 install",
    "pip uninstall",
    "npm install",
    "npm i",
    "npm uninstall",
    "yarn add",
    "pnpm add",
    "apt",
    "apt-get",
    "brew",
    "cargo install",
    "go install",
    "gem install",
    "kill",
    "killall",
    "pkill",
    "systemctl",
    "service",
    "mount",
    "umount",
    "crontab",
    "reboot",
    "shutdown",
    # interactive / write-capable editors
    "vim",
    "vi",
    "nano",
    "emacs",
]


def _review_bash_overrides() -> dict[str, Any]:
    """Per-agent bash policy for read-only review agents — see the comment on
    _REVIEW_BASH_ALLOWLIST. Permission stays the default ASK so unknown
    commands are skipped headless / approvable interactively.
    """
    return {
        "tools": {
            "bash": {
                "allowlist": list(_REVIEW_BASH_ALLOWLIST),
                "denylist": list(_REVIEW_BASH_DENYLIST),
            }
        }
    }


DEFAULT = AgentProfile(
    BuiltinAgentName.DEFAULT,
    "Default",
    "Requires approval for tool executions",
    AgentSafety.NEUTRAL,
    overrides={"base_disabled": ["exit_plan_mode"]},
)
PLAN = AgentProfile(
    BuiltinAgentName.PLAN,
    "Plan",
    "Read-only agent for exploration and planning",
    AgentSafety.SAFE,
    overrides=_plan_overrides(),
)
CHAT = AgentProfile(
    BuiltinAgentName.CHAT,
    "Chat",
    "Read-only conversational mode for questions and discussions",
    AgentSafety.SAFE,
    overrides={"bypass_tool_permissions": True, "enabled_tools": CHAT_AGENT_TOOLS},
)
ACCEPT_EDITS = AgentProfile(
    BuiltinAgentName.ACCEPT_EDITS,
    "Accept Edits",
    "Auto-approves file edits only",
    AgentSafety.DESTRUCTIVE,
    overrides={
        "base_disabled": ["exit_plan_mode"],
        "tools": {
            "write_file": {"permission": "always"},
            "edit": {"permission": "always"},
        },
    },
)
AUTO_APPROVE = AgentProfile(
    BuiltinAgentName.AUTO_APPROVE,
    "Auto Approve",
    "Auto-approves all tool executions",
    AgentSafety.YOLO,
    overrides={"bypass_tool_permissions": True, "base_disabled": ["exit_plan_mode"]},
)

EXPLORE = AgentProfile(
    name=BuiltinAgentName.EXPLORE,
    display_name="Explore",
    description="Read-only subagent for codebase exploration",
    safety=AgentSafety.SAFE,
    agent_type=AgentType.SUBAGENT,
    overrides={"enabled_tools": ["grep", "read", "lsp"], "system_prompt_id": "explore"},
)

RESEARCH = AgentProfile(
    name=BuiltinAgentName.RESEARCH,
    display_name="Research",
    description="Read-only subagent for web research with search and fetch tools",
    safety=AgentSafety.SAFE,
    agent_type=AgentType.SUBAGENT,
    overrides={
        "enabled_tools": ["grep", "read", "lsp", "web_search", "web_fetch"],
        "system_prompt_id": "explore",
    },
)

REVIEWER = AgentProfile(
    name=BuiltinAgentName.REVIEWER,
    display_name="Reviewer",
    description=(
        "Subagent for adversarial code review: reads code, inspects diffs via "
        "git, and runs targeted checks/tests. bash is jailed read-only — "
        "inspection + tests auto-run, code/git mutations are denied."
    ),
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={
        "enabled_tools": ["read", "grep", "lsp", "bash"],
        "system_prompt_id": "explore",
        **_review_bash_overrides(),
    },
)

DEBUGGER = AgentProfile(
    name=BuiltinAgentName.DEBUGGER,
    display_name="Debugger",
    description=(
        "Subagent for systematic debugging: reproduces a failure, isolates it, "
        "and traces the root cause via read/grep + targeted bash (tests, git "
        "inspection). Returns root cause + minimal fix; read-only, so it "
        "diagnoses rather than edits. bash is jailed read-only — inspection + "
        "tests auto-run, code/git mutations are denied."
    ),
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={
        "enabled_tools": ["read", "grep", "lsp", "bash"],
        "system_prompt_id": "debugger",
        **_review_bash_overrides(),
    },
)

PLANNER = AgentProfile(
    name=BuiltinAgentName.PLANNER,
    display_name="Planner",
    description=(
        "Read-only subagent for planning: investigates the code and returns a "
        "code-grounded, phased plan with risks, critical files, and tradeoffs. "
        "Designs the approach; the lead decides and implements."
    ),
    safety=AgentSafety.SAFE,
    agent_type=AgentType.SUBAGENT,
    overrides={"enabled_tools": ["read", "grep", "lsp"], "system_prompt_id": "planner"},
)

SECURITY = AgentProfile(
    name=BuiltinAgentName.SECURITY,
    display_name="Security",
    description=(
        "Subagent for defensive security audit: traces untrusted input to "
        "sinks, checks the vulnerability classes (injection, path-escape, "
        "authz, secrets, deserialization), and reports severity-ranked findings "
        "with fixes. Read + grep + bash jailed read-only — inspection probes "
        "auto-run, repo mutations and network are denied."
    ),
    # Has bash, so not SAFE — bash invocations still route through the normal
    # approval flow (no bypass_tool_permissions).
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={
        "enabled_tools": ["read", "grep", "lsp", "bash"],
        "system_prompt_id": "security",
        **_review_bash_overrides(),
    },
)

VERIFIER = AgentProfile(
    name=BuiltinAgentName.VERIFIER,
    display_name="Verifier",
    description=(
        "Verdict-oriented verification subagent: proves a *completed* "
        "implementation works by trying to break it, then emits a strict "
        "PASS/FAIL/PARTIAL verdict with command-level evidence. Distinct from "
        "'reviewer' (which surveys a diff for issues across diverse lenses). "
        "Read-only: read/grep/lsp + a jailed read-only bash that auto-runs "
        "tests/lint/git-inspection but denies mutations, network, and package "
        "installs. The gate, not the surveyor."
    ),
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={
        "enabled_tools": ["read", "grep", "lsp", "bash"],
        "system_prompt_id": "verifier",
        **_review_bash_overrides(),
    },
)

EDITOR = AgentProfile(
    name=BuiltinAgentName.EDITOR,
    display_name="Editor",
    description=(
        "Workflow-only subagent for surgical file edits (renames, codemods, "
        "targeted changes): write/edit + read/grep, no bash/MCP — lower blast "
        "radius than worker. MUST run with isolation='worktree' in a workflow; "
        "in a plain task call it also runs isolated, with write/edit/read "
        "auto-approved and confined to the worktree (no parent-tree races)."
    ),
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={
        "enabled_tools": ["read", "grep", "lsp", "write_file", "edit"],
        "system_prompt_id": "editor",
    },
)

WORKER = AgentProfile(
    name=BuiltinAgentName.WORKER,
    display_name="Worker",
    description=(
        "Full-capability subagent for workflows: all builtin tools plus any "
        "discovered MCP tools (no enabled_tools allowlist). For workflow agents "
        "that need to act, not just read. MUST run with isolation='worktree' in "
        "a workflow — it then executes as a `vibe -p` subprocess in its own git "
        "worktree. write/edit/read are auto-approved and confined to the "
        "worktree (no headless skip, no parent-tree races); bash is auto-confined "
        "to the worktree by the OS sandbox (bwrap) when one is available. In a "
        "plain `task` call it routes to the same isolated worktree under the "
        "tool's default, with write/edit/read auto-approved there too."
    ),
    # No enabled_tools override -> the full tool set (incl. MCP in the subprocess).
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={"system_prompt_id": "explore"},
)

GRUNT = AgentProfile(
    name=BuiltinAgentName.GRUNT,
    display_name="Grunt",
    description=(
        "Write-capable subagent for grunt work: renames, codemods, "
        "boilerplate generation, repetitive edits across many files — concrete, "
        "fully-specified tasks with no design decisions. Routes onto a cheap "
        "model by default (the `grunt_model` config key, falling back to "
        "subagent_model then the host). Writes run isolated in its own "
        "worktree, like worker (same plain-task auto-isolation under the "
        "tool's default); the difference is intent and prompt — give it "
        "the grunt work and keep thinkers (planner/verifier) for the reasoning "
        "around it. Composes in workflows: thinkers plan and verify, the grunt "
        "executes."
    ),
    # Same tool surface as worker — the prompt and model routing, not a tool
    # restriction, make this a cheap-work profile.
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={"system_prompt_id": "grunt"},
)

LEAN = AgentProfile(
    name=BuiltinAgentName.LEAN,
    display_name="Lean",
    description="Specialized mode for Lean 4 code analysis, proof assistance, and theorem proving",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.AGENT,
    install_required=True,
    overrides={
        "system_prompt_id": "lean",
        "active_model": "leanstral",
        "providers": [
            {
                "name": "mistral-testing",
                "api_base": "https://api.mistral.ai/v1",
                "api_key_env_var": "MISTRAL_API_KEY",
                "backend": "mistral",
            }
        ],
        "models": [
            {
                "name": "labs-leanstral-1-5",
                "provider": "mistral-testing",
                "alias": "leanstral",
                "thinking": "high",
                "temperature": 1.0,
                "auto_compact_threshold": 168_000,
            }
        ],
        "compaction_model": {
            "name": "mistral-small-latest",
            "provider": "mistral-testing",
            "alias": "devstral-compact",
            "temperature": 0.2,
            "thinking": "off",
        },
        "tools": {"bash": {"default_timeout": 1200}},
        "base_disabled": ["exit_plan_mode"],
    },
)

# Read-only orchestration tools — no bash, no write/edit. The coordinator
# investigates with read/grep/glob, delegates every concrete action to a
# subagent via task/launch_workflow, and coordinates teammates. Write-capable
# subagents it spawns still isolate in their own worktree under task.isolation.
_COORDINATOR_TOOLS = [
    "task",
    "launch_workflow",
    "workflow_status",
    "workflow_results",
    "workflow_stop",
    "team",
    "team_spawn",
    "team_message",
    "read",
    "grep",
    "glob",
    "todo",
    "ask_user_question",
    "manage_memory",
    "skill",
]

COORDINATOR = AgentProfile(
    name=BuiltinAgentName.COORDINATOR,
    display_name="Coordinator",
    description=(
        "Orchestration-only lead: delegates to subagents via task/launch_workflow "
        "and coordinates teammates, but cannot write files or run bash directly. "
        "Use for fan-out-and-synthesize workflows where the lead should stay "
        "above the implementation."
    ),
    safety=AgentSafety.SAFE,
    agent_type=AgentType.AGENT,
    overrides={
        "enabled_tools": list(_COORDINATOR_TOOLS),
        "system_prompt_id": "coordinator",
    },
)

BUILTIN_AGENTS: dict[str, AgentProfile] = {
    BuiltinAgentName.DEFAULT: DEFAULT,
    BuiltinAgentName.PLAN: PLAN,
    BuiltinAgentName.ACCEPT_EDITS: ACCEPT_EDITS,
    BuiltinAgentName.AUTO_APPROVE: AUTO_APPROVE,
    BuiltinAgentName.EXPLORE: EXPLORE,
    BuiltinAgentName.RESEARCH: RESEARCH,
    BuiltinAgentName.REVIEWER: REVIEWER,
    BuiltinAgentName.DEBUGGER: DEBUGGER,
    BuiltinAgentName.PLANNER: PLANNER,
    BuiltinAgentName.SECURITY: SECURITY,
    BuiltinAgentName.VERIFIER: VERIFIER,
    BuiltinAgentName.EDITOR: EDITOR,
    BuiltinAgentName.WORKER: WORKER,
    BuiltinAgentName.GRUNT: GRUNT,
    BuiltinAgentName.LEAN: LEAN,
    BuiltinAgentName.COORDINATOR: COORDINATOR,
}
