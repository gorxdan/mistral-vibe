from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import time
import tomllib

import tomli_w

from vibe.core.logger import logger
from vibe.core.paths import (
    AGENTS_MD_FILENAME,
    TRUSTED_FOLDERS_FILE,
    find_local_config_dirs,
)
from vibe.core.utils.io import read_safe


class WorkspaceTrustDecision(StrEnum):
    TRUST_REPO = "trust_repo"
    TRUST_CWD = "trust_cwd"
    TRUST_SESSION = "trust_session"
    DECLINE = "decline"


class WorkspaceTrustStatus(StrEnum):
    TRUSTED = "trusted"
    SESSION = "session"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class WorkspaceTrustPrompt:
    cwd: Path
    repo_root: Path | None
    detected_files: list[str]
    repo_detected_files: list[str]
    offer_repo_trust: bool
    repo_explicitly_untrusted: bool


def has_agents_md_file(path: Path) -> bool:
    agents_md = path / AGENTS_MD_FILENAME
    try:
        return agents_md.is_file()
    except OSError as e:
        logger.warning("Skipping unreadable path=%s: %s", agents_md, e)
        return False


def _is_git_repo_root(path: Path) -> bool:
    git = path / ".git"
    try:
        if git.is_dir():
            return (git / "HEAD").is_file()
        # In a worktree, .git is a file pointing to the main repo's
        # .git/worktrees/<name> directory. Validate the gitdir pointer so a
        # stray plain file named .git is not mistaken for a repo root.
        if git.is_file():
            return read_safe(git).text.lstrip().startswith("gitdir:")
        return False
    except OSError as e:
        logger.warning("Skipping unreadable git path=%s: %s", git, e)
        return False


def find_git_repo_ancestor(path: Path) -> Path | None:
    """Closest ancestor (or *path*) with a ``.git`` dir or worktree file.

    Handles both normal repos (``.git/`` directory with ``HEAD``) and
    worktrees (``.git`` file pointing to the main repo's worktree admin).

    Excludes the home directory and the filesystem root.
    """
    resolved = path.expanduser().resolve()
    home = Path.home().resolve()
    current = resolved
    while current not in {home, current.parent}:
        if _is_git_repo_root(current):
            return current
        current = current.parent
    return None


def find_trustable_files(path: Path) -> list[str]:
    """Relative paths of files/dirs under *path* that would modify agent behavior."""
    resolved = path.resolve()
    found: list[str] = []

    if has_agents_md_file(path):
        found.append(AGENTS_MD_FILENAME)

    for config_dir in find_local_config_dirs(path).config_dirs:
        label = f"{config_dir.relative_to(resolved)}/"
        if label not in found:
            found.append(label)

    return sorted(found)


def find_repo_trustable_files_for_cwd(cwd: Path, repo_root: Path | None) -> list[str]:
    """Repo-context files that influence *cwd* when inside a git repository.

    Includes:
    - all trustable files at ``repo_root``
    - all ``AGENTS.md`` files on ancestors between ``cwd`` and ``repo_root``
    """
    if repo_root is None:
        return []

    resolved_cwd = cwd.resolve()
    resolved_repo_root = repo_root.resolve()
    if resolved_repo_root not in resolved_cwd.parents:
        return []

    found = set(find_trustable_files(resolved_repo_root))

    current = resolved_cwd.parent
    while current != resolved_repo_root:
        if has_agents_md_file(current):
            relative_path = (current / AGENTS_MD_FILENAME).relative_to(
                resolved_repo_root
            )
            found.add(relative_path.as_posix())
        current = current.parent

    return sorted(found)


def maybe_build_workspace_trust_prompt(
    cwd: Path, *, include_explicitly_untrusted: bool = False
) -> WorkspaceTrustPrompt | None:
    resolved_cwd = cwd.resolve()
    if resolved_cwd == Path.home().resolve():
        return None

    if trusted_folders_manager.is_trusted(cwd) is True:
        return None
    if (
        not include_explicitly_untrusted
        and trusted_folders_manager.is_explicitly_untrusted(cwd)
    ):
        return None

    repo_root = find_git_repo_ancestor(cwd)
    detected_files = find_trustable_files(cwd)
    repo_detected_files = find_repo_trustable_files_for_cwd(cwd, repo_root)
    if not detected_files and not repo_detected_files:
        return None

    resolved_repo_root = repo_root.resolve() if repo_root else None
    offer_repo_trust = (
        resolved_repo_root is not None
        and resolved_repo_root in resolved_cwd.parents
        and trusted_folders_manager.is_trusted(resolved_repo_root) is not True
        and (
            include_explicitly_untrusted
            or not trusted_folders_manager.is_explicitly_untrusted(resolved_repo_root)
        )
    )
    repo_explicitly_untrusted = (
        resolved_repo_root is not None
        and trusted_folders_manager.is_explicitly_untrusted(resolved_repo_root)
    )

    return WorkspaceTrustPrompt(
        cwd=cwd,
        repo_root=resolved_repo_root,
        detected_files=detected_files,
        repo_detected_files=repo_detected_files,
        offer_repo_trust=offer_repo_trust,
        repo_explicitly_untrusted=repo_explicitly_untrusted,
    )


def available_workspace_trust_decisions(
    prompt: WorkspaceTrustPrompt, *, include_session: bool = False
) -> list[WorkspaceTrustDecision]:
    decisions = [WorkspaceTrustDecision.TRUST_CWD, WorkspaceTrustDecision.DECLINE]
    if include_session:
        decisions.insert(1, WorkspaceTrustDecision.TRUST_SESSION)
    if prompt.offer_repo_trust:
        decisions.insert(0, WorkspaceTrustDecision.TRUST_REPO)
    return decisions


def apply_workspace_trust_decision(
    prompt: WorkspaceTrustPrompt, decision: WorkspaceTrustDecision
) -> None:
    match decision:
        case WorkspaceTrustDecision.TRUST_REPO if (
            prompt.offer_repo_trust and prompt.repo_root is not None
        ):
            trusted_folders_manager.add_trusted(prompt.repo_root)
        case WorkspaceTrustDecision.TRUST_CWD:
            trusted_folders_manager.add_trusted(prompt.cwd)
        case WorkspaceTrustDecision.TRUST_SESSION:
            trusted_folders_manager.trust_for_session(prompt.cwd)
        case WorkspaceTrustDecision.DECLINE:
            trusted_folders_manager.add_untrusted(prompt.cwd)
        case _:
            raise ValueError(f"Unsupported trust decision: {decision}")


class TrustedFoldersManager:
    def __init__(self) -> None:
        self._file_path = TRUSTED_FOLDERS_FILE.path
        self._trusted: list[str] = []
        self._untrusted: list[str] = []
        self._session_trusted: list[str] = []
        self._load()

    def trust_for_session(self, path: Path) -> None:
        self._session_trusted.append(self._normalize_path(path))

    def _normalize_path(self, path: Path) -> str:
        return str(path.expanduser().resolve())

    def _load(self) -> None:
        if not self._file_path.is_file():
            self._trusted = []
            self._untrusted = []
            self._save()
            return

        try:
            with self._file_path.open("rb") as f:
                data = tomllib.load(f)
            self._trusted = list(data.get("trusted", []))
            self._untrusted = list(data.get("untrusted", []))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            # Fail closed without destroying data: start from an empty DB in
            # memory (so the user is re-prompted for each path) but preserve the
            # corrupt/unreadable file as a backup rather than overwriting it.
            # Overwriting would silently erase prior trust/distrust decisions.
            logger.warning(
                "Trust DB at %s is unreadable (%s); starting from empty. "
                "The original file was preserved as a backup.",
                self._file_path,
                exc,
            )
            self._trusted = []
            self._untrusted = []
            self._preserve_corrupt_backup()

    def _preserve_corrupt_backup(self) -> None:
        """Rename a corrupt trust DB aside instead of clobbering it.

        Best-effort: a failure to move the file is logged but not raised (the
        in-memory state is already empty and safe). A timestamped suffix avoids
        overwriting an earlier backup.
        """
        backup = self._file_path.with_suffix(
            self._file_path.suffix + f".corrupt-{time.time_ns()}"
        )
        try:
            self._file_path.replace(backup)
        except OSError as exc:
            logger.warning(
                "Could not preserve corrupt trust DB at %s as %s: %s",
                self._file_path,
                backup,
                exc,
            )

    def _save(self) -> None:
        """Persist atomically: write a temp file then os.replace into place.

        Atomic replace avoids the truncate-then-write corruption loop the old
        code created (an interrupted write produced exactly the malformed state
        that _load then discarded). Persistence failures are logged — never
        silently swallowed — so a dropped decision is diagnosable. mkdir errors
        propagate (parent unwritable is a real environment problem, not a
        best-effort case).
        """
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"trusted": self._trusted, "untrusted": self._untrusted}
        tmp = self._file_path.with_suffix(self._file_path.suffix + ".tmp")
        try:
            with tmp.open("wb") as f:
                tomli_w.dump(data, f)
            os.replace(tmp, self._file_path)
        except OSError as exc:
            logger.error(
                "Failed to persist trust DB at %s: %s. Decisions are held only "
                "in memory and will be lost on restart.",
                self._file_path,
                exc,
            )
            try:
                tmp.unlink()
            except OSError:
                pass

    def _closest_decision(self, path: Path) -> tuple[bool, Path] | None:
        """``(trusted, ancestor)`` for the closest decision, ``None`` if undecided."""
        current = Path(self._normalize_path(path))
        while True:
            s = str(current)
            if s in self._trusted or s in self._session_trusted:
                return True, current
            if s in self._untrusted:
                return False, current
            if current.parent == current:
                return None
            current = current.parent

    def is_trusted(self, path: Path) -> bool | None:
        """Tri-state closest decision; ``None`` when no ancestor has one."""
        match self._closest_decision(path):
            case (trusted, _):
                return trusted
            case None:
                return None

    def trust_status(self, path: Path) -> WorkspaceTrustStatus:
        current = Path(self._normalize_path(path))
        while True:
            s = str(current)
            if s in self._session_trusted:
                return WorkspaceTrustStatus.SESSION
            if s in self._trusted:
                return WorkspaceTrustStatus.TRUSTED
            if s in self._untrusted:
                return WorkspaceTrustStatus.UNTRUSTED
            if current.parent == current:
                return WorkspaceTrustStatus.UNTRUSTED
            current = current.parent

    def is_explicitly_untrusted(self, path: Path) -> bool:
        """*path* literally in the untrusted list (no ancestor walk)."""
        return self._normalize_path(path) in self._untrusted

    def find_trust_root(self, path: Path) -> Path | None:
        """Closest explicitly trusted ancestor; ``None`` if a closer untrust blocks."""
        match self._closest_decision(path):
            case (True, root):
                return root
            case _:
                return None

    def add_trusted(self, path: Path) -> None:
        normalized = self._normalize_path(path)
        if normalized not in self._trusted:
            self._trusted.append(normalized)
        if normalized in self._untrusted:
            self._untrusted.remove(normalized)
        self._save()

    def add_untrusted(self, path: Path) -> None:
        normalized = self._normalize_path(path)
        if normalized not in self._untrusted:
            self._untrusted.append(normalized)
        if normalized in self._trusted:
            self._trusted.remove(normalized)
        self._save()


trusted_folders_manager = TrustedFoldersManager()
