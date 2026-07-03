from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
import contextlib
import os
from pathlib import Path
import sys

from vibe import __version__

# Module scope must stay argparse+stdlib: `vibe --help`/`--version` exit inside
# parse_arguments(), so the config stack only loads after parsing (in main).


def _rprint(*objects: object) -> None:
    from rich import print as rich_print

    rich_print(*objects)


_SPLASH_GRADIENT = (
    "#ff6b00",
    "#ff7b00",
    "#ff8c00",
    "#ff9d00",
    "#ffae00",
    "#ffbf00",
    "#ffae00",
    "#ff9d00",
    "#ff8c00",
    "#ff7b00",
)

_SPLASH_MASCOT = "  /\\_/\\\n ( o.o )\n  > ^ <"


def _gradient_banner(text: str) -> str:
    return "".join(
        f"[bold {_SPLASH_GRADIENT[i % len(_SPLASH_GRADIENT)]}]{c}[/]"
        for i, c in enumerate(text)
    )


@contextlib.contextmanager
def _interactive_splash(enabled: bool) -> Iterator[None]:
    if not enabled or not sys.stdout.isatty():
        yield
        return
    from rich.console import Console

    console = Console()
    console.print(_gradient_banner("Mistral Vibe"), style=None, highlight=False)
    console.print(f"[dim]{_SPLASH_MASCOT}[/]")
    with console.status("[dim]Loading harness…[/]", spinner="dots"):
        yield


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Mistral Vibe interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  VIBE_HOME       Override the Vibe home directory (default: ~/.vibe)\n"
            "  LOG_LEVEL       Logging level: DEBUG, INFO, WARNING (default), ERROR, CRITICAL.\n"
            "                  Logs are written to $VIBE_HOME/logs/vibe.log.\n"
            "  LOG_MAX_BYTES   Max size of vibe.log before rotation (default: 10485760).\n"
            "  VIBE_*          Override any config field (e.g. VIBE_ACTIVE_MODEL=local)."
        ),
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "initial_prompt",
        nargs="?",
        metavar="PROMPT",
        help="Initial prompt to start the interactive session with.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        nargs="?",
        const="",
        metavar="TEXT",
        help="Run in programmatic mode: send prompt, output response, and exit. "
        "Tool approval follows the selected --agent (or 'default_agent' config); "
        "pass --auto-approve or --yolo to allow all tool calls.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        metavar="N",
        help="Maximum number of assistant turns "
        "(only applies in programmatic mode with -p).",
    )
    parser.add_argument(
        "--keep-alive",
        type=int,
        metavar="SECONDS",
        help="After the response, keep firing scheduled-loop turns (from the "
        "schedule tool) for up to SECONDS before exiting (programmatic -p only). "
        "Without it, -p exits immediately and scheduled loops only persist.",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        metavar="DOLLARS",
        help="Maximum cost in dollars (only applies in programmatic mode with -p). "
        "Session will be interrupted if cost exceeds this limit.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        help="Maximum total prompt + completion tokens across the session "
        "(only applies in programmatic mode with -p). "
        "Session will be interrupted if usage exceeds this limit.",
    )
    parser.add_argument(
        "--enabled-tools",
        action="append",
        metavar="TOOL",
        help="Enable specific tools. In programmatic mode (-p), this disables "
        "all other tools. "
        "Can use exact names, glob patterns (e.g., 'bash*'), or "
        "regex with 're:' prefix. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["text", "json", "streaming"],
        default="text",
        help="Output format for programmatic mode (-p): 'text' "
        "for human-readable (default), 'json' for all messages at end, "
        "'streaming' for newline-delimited JSON per message.",
    )
    parser.add_argument(
        "--agent",
        metavar="NAME",
        default=None,
        help="Agent to use (builtin: default, plan, accept-edits, auto-approve, "
        "or custom from ~/.vibe/agents/NAME.toml). Defaults to the "
        "'default_agent' config setting in both interactive and programmatic "
        "(-p/--prompt) mode.",
    )
    parser.add_argument(
        "--auto-approve",
        "--yolo",
        action="store_true",
        help="Approves all tool calls without prompting for the selected agent.",
    )
    parser.add_argument(
        "--model",
        metavar="ALIAS",
        default=None,
        help="Model alias to use as the active model for this session (must match "
        "a configured [[models]] alias). Overrides the 'active_model' config "
        "setting. Also the flag the task tool threads into isolated subagents.",
    )
    parser.add_argument("--setup", action="store_true", help="Setup API key and exit")
    parser.add_argument("--zai-callback", metavar="URI", help=argparse.SUPPRESS)
    parser.add_argument(
        "--check-upgrade",
        action="store_true",
        help="Check for a Vibe update now, prompt to install it, and exit",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        metavar="DIR",
        help="Change to this directory before running",
    )
    parser.add_argument(
        "--add-dir",
        action="append",
        metavar="DIR",
        default=[],
        help="Additional working directory for file access and context. "
        "Implicitly trusted for the session (same semantics as --trust). "
        "Can be specified multiple times.",
    )
    parser.add_argument(
        "--trust",
        action="store_true",
        help="Trust the working directory for this invocation only (not "
        "persisted to trusted_folders.toml). Skips the trust prompt. "
        "Use this for non-interactive automation.",
    )
    worktree_group = parser.add_mutually_exclusive_group()
    worktree_group.add_argument(
        "--worktree",
        action="store_true",
        help="Force worktree isolation on (overrides config mode='off'). "
        "Writes land on a throwaway branch, not your live checkout.",
    )
    worktree_group.add_argument(
        "--no-worktree",
        action="store_true",
        help="Force worktree isolation off for this invocation (overrides the "
        "default). Writes land in your live checkout.",
    )

    # Feature flag for teleport, not exposed to the user yet
    parser.add_argument("--teleport", action="store_true", help=argparse.SUPPRESS)

    continuation_group = parser.add_mutually_exclusive_group()
    continuation_group.add_argument(
        "-c",
        "--continue",
        action="store_true",
        dest="continue_session",
        help="Continue from the most recent saved session",
    )
    continuation_group.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="SESSION_ID",
        help="Resume a session. Without SESSION_ID, shows an interactive picker.",
    )
    return parser.parse_args()


def _handle_zai_callback(uri: str) -> None:
    from vibe.setup.auth.zai_callback import write_zai_callback
    from vibe.setup.auth.zai_sign_in import ZaiSignInError

    try:
        write_zai_callback(uri)
    except ZaiSignInError as err:
        _rprint(f"[red]Could not capture Z.ai callback: {err}[/]")
        sys.exit(1)
    _rprint("[green]Z.ai callback captured. Return to Mistral Vibe setup.[/]")
    sys.exit(0)


def check_and_resolve_trusted_folder(cwd: Path) -> None:
    from vibe.core.trusted_folders import (
        apply_workspace_trust_decision,
        maybe_build_workspace_trust_prompt,
    )

    prompt = maybe_build_workspace_trust_prompt(cwd)
    if prompt is None:
        return

    from vibe.setup.trusted_folders.trust_folder_dialog import (
        TrustDialogQuitException,
        ask_trust_folder,
    )

    try:
        decision = ask_trust_folder(
            prompt.cwd,
            prompt.repo_root,
            prompt.detected_files,
            repo_detected_files=prompt.repo_detected_files,
            offer_repo_trust=prompt.offer_repo_trust,
            repo_explicitly_untrusted=prompt.repo_explicitly_untrusted,
        )
    except (KeyboardInterrupt, EOFError, TrustDialogQuitException):
        sys.exit(0)
    except Exception as e:
        _rprint(f"[yellow]Error showing trust dialog: {e}[/]")
        return

    if decision is not None:
        apply_workspace_trust_decision(prompt, decision)


def main() -> None:
    if sys.argv[1:2] and sys.argv[1].startswith("zcode://zai-auth/callback"):
        _handle_zai_callback(sys.argv[1])

    # Pre-dispatch the `worktree` maintenance subcommand before the main parser,
    # whose positional `initial_prompt` would otherwise swallow it. Runs without
    # config/trust/harness setup — it only touches git.
    if sys.argv[1:2] == ["worktree"]:
        from vibe.cli.worktree_cmd import run_worktree_command

        sys.exit(run_worktree_command(sys.argv[2:]))

    args = parse_arguments()

    if (zai_callback := getattr(args, "zai_callback", None)) is not None:
        _handle_zai_callback(zai_callback)

    if args.workdir:
        workdir = args.workdir.expanduser().resolve()
        if not workdir.is_dir():
            _rprint(
                f"[red]Error: --workdir does not exist or is not a directory: {workdir}[/]"
            )
            sys.exit(1)
        os.chdir(workdir)

    try:
        cwd = Path.cwd()
    except FileNotFoundError:
        _rprint(
            "[red]Error: Current working directory no longer exists.[/]\n"
            "[yellow]The directory you started vibe from has been deleted. "
            "Please change to an existing directory and try again, "
            "or use --workdir to specify a working directory.[/]"
        )
        sys.exit(1)

    from vibe.core.config.harness_files import init_harness_files_manager
    from vibe.core.trusted_folders import trusted_folders_manager
    from vibe.core.utils.paths import is_dangerous_directory

    if args.trust:
        trusted_folders_manager.trust_for_session(cwd)

    additional_dirs: list[Path] = []
    for d in args.add_dir:
        resolved = Path(d).expanduser().resolve()
        if not resolved.is_dir():
            _rprint(
                f"[red]Error: --add-dir path does not exist "
                f"or is not a directory: {d}[/]"
            )
            sys.exit(1)
        is_dangerous, reason = is_dangerous_directory(resolved)
        if is_dangerous:
            _rprint(
                f"[red]Error: --add-dir path is not allowed: {resolved} ({reason})[/]"
            )
            sys.exit(1)
        additional_dirs.append(resolved)
        trusted_folders_manager.trust_for_session(resolved)

    init_harness_files_manager("user", "project", additional_dirs=additional_dirs)

    interactive = args.prompt is None and not args.check_upgrade
    with _interactive_splash(interactive):
        from vibe.cli.cli import run_cli

    resolve_trusted_folder: Callable[[], None] | None = None
    if interactive:

        def _resolve_trusted_folder() -> None:
            check_and_resolve_trusted_folder(cwd)

        resolve_trusted_folder = _resolve_trusted_folder

    run_cli(args, resolve_trusted_folder=resolve_trusted_folder)


if __name__ == "__main__":
    main()
