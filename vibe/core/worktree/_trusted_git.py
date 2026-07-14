from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess

from vibe.core._trusted_command import (
    TRUSTED_GIT_CONFIG_ARGS,
    TrustedCommandError,
    minimal_trusted_git_environment,
    resolve_trusted_system_executable,
)
from vibe.core.utils.io import decode_safe, read_safe

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_MAX_CONFIG_BYTES = 1024 * 1024
_UNSAFE_CONFIG = (
    re.compile(r"filter\..+\.(?:clean|process|smudge)"),
    re.compile(r"include(?:if)?\..+"),
    re.compile(r"merge\..+\.driver"),
)


class TrustedGitError(RuntimeError):
    pass


def _resolve_git_directory(work_tree: Path) -> Path:
    dot_git = work_tree / ".git"
    if dot_git.is_symlink():
        raise TrustedGitError("Git metadata entry cannot be a symlink")
    if dot_git.is_dir():
        return dot_git.resolve(strict=True)
    if not dot_git.is_file():
        raise TrustedGitError(f"not a physical Git worktree: {work_tree}")
    line = read_safe(dot_git, raise_on_error=True).text.strip()
    if not line.startswith("gitdir: "):
        raise TrustedGitError("linked worktree metadata is malformed")
    raw_git_dir = Path(line.removeprefix("gitdir: "))
    if not raw_git_dir.is_absolute():
        raw_git_dir = work_tree / raw_git_dir
    try:
        return raw_git_dir.resolve(strict=True)
    except OSError as exc:
        raise TrustedGitError("linked worktree Git directory is missing") from exc


def _resolve_common_directory(git_dir: Path) -> Path:
    common_file = git_dir / "commondir"
    if common_file.is_symlink():
        raise TrustedGitError("Git common-directory pointer cannot be a symlink")
    if not common_file.is_file():
        return git_dir
    raw_common = Path(read_safe(common_file, raise_on_error=True).text.strip())
    if not raw_common.is_absolute():
        raw_common = git_dir / raw_common
    try:
        common_dir = raw_common.resolve(strict=True)
    except OSError as exc:
        raise TrustedGitError("Git common directory is missing") from exc
    if not common_dir.is_dir():
        raise TrustedGitError("Git common directory is not a directory")
    return common_dir


@dataclass(frozen=True, slots=True)
class TrustedGitWorktree:
    work_tree: Path
    git_dir: Path
    common_dir: Path
    index_file: Path

    @classmethod
    def open(cls, path: Path) -> TrustedGitWorktree:
        try:
            work_tree = path.resolve(strict=True)
        except OSError as exc:
            raise TrustedGitError(f"Git worktree is unavailable: {path}") from exc
        git_dir = _resolve_git_directory(work_tree)
        common_dir = _resolve_common_directory(git_dir)
        index_file = git_dir / "index"
        protected_paths = (
            git_dir / "HEAD",
            index_file,
            common_dir / "objects",
            common_dir / "refs",
        )
        if any(path.is_symlink() for path in protected_paths):
            raise TrustedGitError("Git transaction metadata cannot contain symlinks")
        if not (common_dir / "objects").is_dir():
            raise TrustedGitError("Git object directory is missing")
        return cls(
            work_tree=work_tree,
            git_dir=git_dir,
            common_dir=common_dir,
            index_file=index_file,
        )

    def text(self, *arguments: str, input_bytes: bytes | None = None) -> str:
        return decode_safe(
            self.bytes(*arguments, input_bytes=input_bytes), from_subprocess=True
        ).text.strip()

    def bytes(self, *arguments: str, input_bytes: bytes | None = None) -> bytes:
        self._require_safe_local_config()
        return self._invoke(*arguments, input_bytes=input_bytes)

    def resolve_commit(self, revision: str, label: str) -> str:
        sha = self.text("rev-parse", "--verify", f"{revision}^{{commit}}")
        if _FULL_SHA.fullmatch(sha) is None:
            raise TrustedGitError(f"could not resolve {label} exactly")
        return sha

    def head_sha(self) -> str:
        return self.resolve_commit("HEAD", "checked-out HEAD")

    def head_ref(self) -> str:
        target = self.text("symbolic-ref", "--quiet", "HEAD")
        if not target.startswith("refs/heads/") or any(
            character in target for character in "\r\n\0"
        ):
            raise TrustedGitError("exact delivery requires a checked-out branch")
        return target

    def branch_sha(self, branch: str) -> str:
        if not branch or any(character in branch for character in "\r\n\0"):
            raise TrustedGitError("candidate branch name is invalid")
        self.text("check-ref-format", "--branch", branch)
        return self.resolve_commit(f"refs/heads/{branch}", "candidate branch")

    def tree_sha(self, commit_sha: str) -> str:
        tree_sha = self.text("rev-parse", "--verify", f"{commit_sha}^{{tree}}")
        if _FULL_SHA.fullmatch(tree_sha) is None:
            raise TrustedGitError("could not resolve commit tree exactly")
        return tree_sha

    def index_tree(self) -> str:
        tree_sha = self.text("write-tree")
        if _FULL_SHA.fullmatch(tree_sha) is None:
            raise TrustedGitError("could not resolve index tree exactly")
        return tree_sha

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = self._result("merge-base", "--is-ancestor", ancestor, descendant)
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise self._command_error(("merge-base", "--is-ancestor"), result)

    def clean_against(self, commit_sha: str, *, include_untracked: bool) -> bool:
        if self._result("update-index", "--refresh", "--").returncode != 0:
            return False
        if self._result("diff-index", "--quiet", commit_sha, "--").returncode != 0:
            return False
        if not include_untracked:
            return True
        return not self.bytes("ls-files", "--others", "--exclude-standard", "-z")

    def read_tree(self, commit_sha: str) -> None:
        self.bytes("read-tree", "--reset", "-u", commit_sha)

    def update_ref(self, target_ref: str, new_sha: str, old_sha: str) -> None:
        if not target_ref.startswith("refs/heads/") or any(
            character in target_ref for character in "\r\n\0"
        ):
            raise TrustedGitError("checked-out branch ref is invalid")
        payload = (
            f"start\nupdate {target_ref} {new_sha} {old_sha}\nprepare\ncommit\n"
        ).encode()
        self.bytes("update-ref", "--stdin", input_bytes=payload)

    def changed_paths(self, base_sha: str, candidate_sha: str) -> list[str]:
        output = self.bytes(
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--name-only",
            "-z",
            f"{base_sha}...{candidate_sha}",
            "--",
        )
        return [os.fsdecode(path) for path in output.split(b"\0") if path]

    def merge_tree(self, base_sha: str, candidate_sha: str) -> str:
        output = self.text(
            "merge-tree", "--write-tree", "--no-messages", base_sha, candidate_sha
        )
        tree_sha = output.splitlines()[0].strip() if output else ""
        if (
            _FULL_SHA.fullmatch(tree_sha) is None
            or self.text("cat-file", "-t", tree_sha) != "tree"
        ):
            raise TrustedGitError("Git did not produce an exact merge tree")
        return tree_sha

    def commit_tree(
        self, tree_sha: str, first_parent: str, second_parent: str, message: str
    ) -> str:
        if "\0" in message:
            raise TrustedGitError("merge commit message cannot contain NUL")
        try:
            encoded_message = message.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TrustedGitError("merge commit message is not valid UTF-8") from exc
        environment = {
            "GIT_AUTHOR_EMAIL": "vibe@localhost",
            "GIT_AUTHOR_NAME": "Vibe",
            "GIT_COMMITTER_EMAIL": "vibe@localhost",
            "GIT_COMMITTER_NAME": "Vibe",
        }
        self._require_safe_local_config()
        commit_sha = decode_safe(
            self._invoke(
                "commit-tree",
                tree_sha,
                "-p",
                first_parent,
                "-p",
                second_parent,
                "-F",
                "-",
                input_bytes=encoded_message,
                extra_environment=environment,
            ),
            from_subprocess=True,
        ).text.strip()
        if _FULL_SHA.fullmatch(commit_sha) is None:
            raise TrustedGitError("Git did not produce an exact merge commit")
        return commit_sha

    def commit_parents(self, commit_sha: str) -> list[str]:
        fields = self.text("rev-list", "--parents", "-n", "1", commit_sha).split()
        if not fields or fields[0] != commit_sha:
            raise TrustedGitError("could not inspect exact merge parents")
        return fields[1:]

    def ahead_count(self, branch: str) -> int:
        value = self.text("rev-list", "--count", f"HEAD..refs/heads/{branch}")
        try:
            return int(value or "0")
        except ValueError as exc:
            raise TrustedGitError("Git returned an invalid ahead count") from exc

    def fingerprint(self) -> str:
        head = self.head_sha()
        working = self.text(
            "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--"
        )
        staged = self.text(
            "diff",
            "--binary",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            "--",
        )
        names = sorted(
            os.fsdecode(path)
            for path in self.bytes(
                "ls-files", "--others", "--exclude-standard", "-z"
            ).split(b"\0")
            if path
        )
        untracked = [
            (name, self.text("hash-object", "--no-filters", "--", name))
            for name in names
        ]
        digest = hashlib.sha256()
        for value in (head, staged, working):
            digest.update(value.encode("utf-8", errors="replace"))
            digest.update(b"\0")
        for name, blob_hash in untracked:
            digest.update(name.encode("utf-8", errors="replace"))
            digest.update(b"\0")
            digest.update(blob_hash.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _require_safe_local_config(self) -> None:
        for path in (self.common_dir / "config", self.git_dir / "config.worktree"):
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise TrustedGitError("Git executable configuration is not trusted")
            if path.stat().st_size > _MAX_CONFIG_BYTES:
                raise TrustedGitError("Git executable configuration is too large")
            result = self._invoke(
                "config",
                "--file",
                str(path),
                "--no-includes",
                "--null",
                "--name-only",
                "--list",
            )
            keys = decode_safe(result, from_subprocess=True).text.split("\0")
            unsafe = next(
                (
                    key
                    for key in keys
                    if key
                    and any(
                        pattern.fullmatch(key.casefold()) for pattern in _UNSAFE_CONFIG
                    )
                ),
                None,
            )
            if unsafe is not None:
                raise TrustedGitError(
                    f"unsafe local executable Git configuration is set: {unsafe}"
                )

    def _result(
        self, *arguments: str, input_bytes: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        self._require_safe_local_config()
        return self._invoke_result(*arguments, input_bytes=input_bytes)

    def _invoke(
        self,
        *arguments: str,
        input_bytes: bytes | None = None,
        extra_environment: dict[str, str] | None = None,
    ) -> bytes:
        result = self._invoke_result(
            *arguments, input_bytes=input_bytes, extra_environment=extra_environment
        )
        if result.returncode != 0:
            raise self._command_error(arguments, result)
        return result.stdout

    def _invoke_result(
        self,
        *arguments: str,
        input_bytes: bytes | None = None,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            git = resolve_trusted_system_executable("git")
            environment = minimal_trusted_git_environment(Path("/"))
            environment.update({
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_INDEX_FILE": str(self.index_file),
                "GIT_OBJECT_DIRECTORY": str(self.common_dir / "objects"),
            })
            if extra_environment is not None:
                environment.update(extra_environment)
            return subprocess.run(
                [
                    str(git),
                    f"--git-dir={self.git_dir}",
                    f"--work-tree={self.work_tree}",
                    *TRUSTED_GIT_CONFIG_ARGS,
                    *arguments,
                ],
                input=input_bytes,
                check=False,
                capture_output=True,
                cwd=self.work_tree,
                env=environment,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError, TrustedCommandError) as exc:
            raise TrustedGitError(f"trusted Git command failed: {exc}") from exc

    @staticmethod
    def _command_error(
        arguments: tuple[str, ...], result: subprocess.CompletedProcess[bytes]
    ) -> TrustedGitError:
        raw = result.stderr.strip() or result.stdout.strip()
        diagnostic = decode_safe(raw, from_subprocess=True).text if raw else "no output"
        return TrustedGitError(
            f"trusted Git command failed ({' '.join(arguments[:2])}): {diagnostic}"
        )


__all__ = ["TrustedGitError", "TrustedGitWorktree"]
