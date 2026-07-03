"""Structural-divergence guard for the mistralai/mistral-vibe fork.

A fork stays cheap to maintain only while its edits *overlap* upstream edits as
little as possible. The most expensive kind of divergence is structural: when we
delete, rename, or split a file that upstream still ships. Git merges by path, so
every future upstream change to that path lands as a ``modify/delete`` conflict
that must be re-applied by hand — it can never 3-way merge.

This module enumerates the files present at the last upstream sync point that no
longer exist at their original path in HEAD, and flags any that are not on the
explicitly-accepted allowlist. New structural divergence should be a conscious,
reviewed decision, not something that slips in.

Baseline: ``VIBE_UPSTREAM_BASE`` env var if set (the upstream-sync workflow can
pass the freshly-merged tag), else ``_MERGE_BASE`` below. Bump ``_MERGE_BASE`` on
each upstream sync so the comparison tracks the current sync point.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

# Last upstream sync point (mistralai/mistral-vibe). Bump on each upstream merge.
_MERGE_BASE = "ac8f1a0"  # v2.18.4

# Structural divergences we have consciously accepted. Path -> why it is tolerated.
# Removing a path from here means "we expect it back at its upstream location";
# the guard then fails until it is restored. agent_loop.py is the active target of
# the un-split work (restores 3-way merge on upstream's 2nd-most-churned file).
_ACCEPTED_DIVERGENCE: dict[str, str] = {
    "vibe/core/llm/backend/base.py": (
        "Renamed to adapter_port.py. Low upstream churn (3 commits); revert candidate."
    ),
    "vibe/acp/user_display_content.py": "Removed/inlined; 1 upstream commit.",
    "vibe/cli/profiler.py": "Removed; 2 upstream commits.",
}

_TRACKED_PREFIX = "vibe/"
_TRACKED_SUFFIX = ".py"


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def baseline_ref() -> str:
    return os.environ.get("VIBE_UPSTREAM_BASE", _MERGE_BASE)


def baseline_available(root: Path | None = None) -> bool:
    """True when the baseline commit is present (full clone). Shallow CI checkouts
    lack it, in which case the caller should skip the git-based check.
    """
    root = root or repo_root()
    try:
        _git(["cat-file", "-e", f"{baseline_ref()}^{{commit}}"], root)
    except subprocess.CalledProcessError:
        return False
    return True


def upstream_unsynced_count(root: Path | None = None) -> int | None:
    """Commits in the upstream ref not yet in HEAD — a stale-baseline signal.

    None when the upstream ref is unavailable (not fetched). After syncing
    upstream, bump ``_MERGE_BASE`` so the divergence diff tracks the new point.
    """
    root = root or repo_root()
    ref = os.environ.get("VIBE_UPSTREAM_REF", "upstream/main")
    try:
        out = _git(["rev-list", "--count", f"HEAD..{ref}"], root)
    except subprocess.CalledProcessError:
        return None
    return int(out.strip() or "0")


def deleted_upstream_files(root: Path | None = None) -> list[str]:
    """Files under vibe/ that existed at the baseline but not at HEAD's original
    path — i.e. our structural deletions/renames/splits relative to upstream.
    """
    root = root or repo_root()
    out = _git(
        [
            "diff",
            "--diff-filter=D",
            "--name-only",
            baseline_ref(),
            "HEAD",
            "--",
            "vibe/**/*.py",
        ],
        root,
    )
    return sorted(
        line
        for line in out.splitlines()
        if line.startswith(_TRACKED_PREFIX) and line.endswith(_TRACKED_SUFFIX)
    )


def unexpected_divergences(root: Path | None = None) -> list[str]:
    return [f for f in deleted_upstream_files(root) if f not in _ACCEPTED_DIVERGENCE]


@dataclass
class Report:
    accepted: list[str]
    unexpected: list[str]


def build_report(root: Path | None = None) -> Report:
    deleted = deleted_upstream_files(root)
    return Report(
        accepted=[f for f in deleted if f in _ACCEPTED_DIVERGENCE],
        unexpected=[f for f in deleted if f not in _ACCEPTED_DIVERGENCE],
    )


def main() -> int:
    root = repo_root()
    if not baseline_available(root):
        print(
            "SKIP: baseline commit "
            f"{baseline_ref()[:12]} not in history (shallow clone). "
            "Add `fetch-depth: 0` to the checkout to enable this guard."
        )
        return 0
    report = build_report(root)
    for f in report.accepted:
        print(f"accepted divergence: {f}")
    unsynced = upstream_unsynced_count(root)
    if unsynced:
        print(
            f"\nNOTE: {unsynced} upstream commit(s) not in HEAD — the divergence "
            "baseline (_MERGE_BASE) is stale. Sync upstream, then bump _MERGE_BASE."
        )
    if report.unexpected:
        print("\nNEW structural divergence (upstream ships these; HEAD deleted them):")
        for f in report.unexpected:
            print(f"  - {f}")
        print(
            "\nEach becomes a modify/delete conflict on every upstream merge. "
            "Prefer the sidecar-file pattern (new files upstream never touches). "
            "If the deletion is intentional, add it to _ACCEPTED_DIVERGENCE with a reason."
        )
        return 1
    print("\nOK: no new structural divergence beyond the accepted allowlist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
