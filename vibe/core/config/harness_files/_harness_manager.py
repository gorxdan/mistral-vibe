from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vibe.core.config.harness_files._paths import (
    GLOBAL_AGENTS_DIR,
    GLOBAL_AGENTS_SKILLS_DIR,
    GLOBAL_PROMPTS_DIR,
    GLOBAL_SKILLS_DIR,
    GLOBAL_TOOLS_DIR,
    GLOBAL_WORKFLOWS_DIR,
)
from vibe.core.paths import (
    AGENTS_MD_FILENAME,
    VIBE_HOME,
    LocalConfigDirs,
    dedup_paths,
    find_local_config_dirs,
    resolved_within,
)
from vibe.core.trusted_folders import trusted_folders_manager
from vibe.core.utils.io import read_safe

FileSource = Literal["user", "project"]


def _normalize_sources(sources: tuple[FileSource, ...]) -> tuple[FileSource, ...]:
    return tuple(source for source in ("user", "project") if source in sources)


@dataclass(frozen=True)
class HarnessFilesManager:
    sources: tuple[FileSource, ...] = ("user",)
    cwd: Path | None = field(default=None)
    _additional_dirs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", _normalize_sources(self.sources))
        object.__setattr__(
            self, "_additional_dirs", tuple(dedup_paths(self._additional_dirs))
        )

    @property
    def _effective_cwd(self) -> Path:
        return (self.cwd or Path.cwd()).resolve()

    @property
    def additional_dirs(self) -> tuple[Path, ...]:
        return self._additional_dirs

    @property
    def trusted_additional_dirs(self) -> list[Path]:
        """Add-dirs that pass the trust gate (DB-trusted or session-trusted).

        Entry points session-trust accepted add-dirs before they reach the
        manager; this re-validates them so a declined or untrusted dir cannot
        feed discovery even if it lands in ``_additional_dirs``. Symmetric with
        the cwd gate in ``trusted_workdir``.
        """
        return [
            d
            for d in dedup_paths(self._additional_dirs)
            if trusted_folders_manager.is_trusted(d) is True
        ]

    @property
    def trusted_workdir(self) -> Path | None:
        """Return cwd if project source is enabled and trusted, else None."""
        if "project" not in self.sources:
            return None
        cwd = self._effective_cwd
        if trusted_folders_manager.is_trusted(cwd) is not True:
            return None
        return cwd

    @property
    def config_file(self) -> Path | None:
        return self.config_file_with_root[0]

    @property
    def config_file_with_root(self) -> tuple[Path | None, Path | None]:
        project_configs = self.project_config_files_with_roots
        if project_configs:
            return project_configs[0]
        if "user" in self.sources:
            return VIBE_HOME.path / "config.toml", None
        return None, None

    @property
    def project_config_files_with_roots(self) -> list[tuple[Path, Path]]:
        configs: list[tuple[Path, Path]] = []
        for root in self.project_roots:
            root_resolved = root.resolve()
            candidate = root_resolved / ".vibe" / "config.toml"
            if candidate.is_file() and resolved_within(candidate, root_resolved):
                configs.append((candidate, root_resolved))
        return configs

    @property
    def project_roots(self) -> list[Path]:
        """Open project directories resolved to their trust roots, plus ``--add-dir`` paths.

        The trusted cwd is resolved to its closest explicitly-trusted ancestor
        (the trust root) so that repo-root config/hooks/skills/workflows are
        discovered even when vibe is launched from a subdirectory. When launched
        at the trust root itself this is a no-op. ``--add-dir`` entries are
        resolved and deduplicated; nested paths are preserved because project
        config discovery is root-level only. Add-dirs equal to the trust root
        are dropped (redundant). Add-dirs are trust-gated via
        ``trusted_additional_dirs``.
        """
        add_dirs = self.trusted_additional_dirs
        workdir = self.trusted_workdir
        if workdir is None:
            return add_dirs
        root = (trusted_folders_manager.find_trust_root(workdir) or workdir).resolve()
        return [root, *(p for p in add_dirs if p != root)]

    @property
    def hook_files(self) -> list[Path]:
        files: list[Path] = []
        for root in self.project_roots:
            root_resolved = root.resolve()
            candidate = root_resolved / ".vibe" / "hooks.toml"
            if resolved_within(candidate, root_resolved):
                files.append(candidate)
        if "user" in self.sources:
            files.append(VIBE_HOME.path / "hooks.toml")
        return files

    @property
    def plugin_dirs(self) -> list[Path]:
        """Plugin parent dirs to scan: ``<root>/.vibe/plugins`` for each
        (trust-gated) project root + ``~/.vibe/plugins``. project_roots already
        excludes untrusted cwds, so plugin discovery inherits that gating.
        """
        dirs: list[Path] = []
        for root in self.project_roots:
            root_resolved = root.resolve()
            candidate = root_resolved / ".vibe" / "plugins"
            if candidate.is_dir() and resolved_within(candidate, root_resolved):
                dirs.append(candidate)
        if "user" in self.sources:
            user_plugins = VIBE_HOME.path / "plugins"
            if user_plugins.is_dir():
                dirs.append(user_plugins)
        return dirs

    @property
    def persist_allowed(self) -> bool:
        return "user" in self.sources

    @property
    def user_tools_dirs(self) -> list[Path]:
        if "user" not in self.sources:
            return []
        d = GLOBAL_TOOLS_DIR.path
        return [d] if d.is_dir() else []

    @property
    def user_skills_dirs(self) -> list[Path]:
        if "user" not in self.sources:
            return []
        return [
            d
            for d in (GLOBAL_SKILLS_DIR.path, GLOBAL_AGENTS_SKILLS_DIR.path)
            if d.is_dir()
        ]

    @property
    def user_agents_dirs(self) -> list[Path]:
        if "user" not in self.sources:
            return []
        d = GLOBAL_AGENTS_DIR.path
        return [d] if d.is_dir() else []

    @property
    def user_workflows_dirs(self) -> list[Path]:
        if "user" not in self.sources:
            return []
        d = GLOBAL_WORKFLOWS_DIR.path
        return [d] if d.is_dir() else []

    def _collect_project_roots(self) -> LocalConfigDirs:
        result = LocalConfigDirs()
        for root in self.project_roots:
            result |= find_local_config_dirs(root)
        return result

    @property
    def project_tools_dirs(self) -> list[Path]:
        return list(self._collect_project_roots().tools)

    @property
    def project_skills_dirs(self) -> list[Path]:
        return list(self._collect_project_roots().skills)

    @property
    def project_agents_dirs(self) -> list[Path]:
        return list(self._collect_project_roots().agents)

    @property
    def project_workflows_dirs(self) -> list[Path]:
        return list(self._collect_project_roots().workflows)

    @property
    def user_config_file(self) -> Path:
        return VIBE_HOME.path / "config.toml"

    @property
    def project_prompts_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        for root in self.project_roots:
            root_resolved = root.resolve()
            candidate = root_resolved / ".vibe" / "prompts"
            if candidate.is_dir() and resolved_within(candidate, root_resolved):
                dirs.append(candidate)
        return dirs

    @property
    def user_prompts_dirs(self) -> list[Path]:
        if "user" not in self.sources:
            return []
        d = GLOBAL_PROMPTS_DIR.path
        return [d] if d.is_dir() else []

    def load_user_doc(self) -> str:
        if "user" not in self.sources:
            return ""
        path = VIBE_HOME.path / AGENTS_MD_FILENAME
        try:
            stripped = read_safe(path).text.strip()
            return stripped if stripped else ""
        except (FileNotFoundError, OSError):
            return ""

    def _collect_agents_md(
        self, start: Path, stop: Path, *, stop_inclusive: bool
    ) -> list[tuple[Path, str]]:
        """Walk up from start toward stop, collecting non-empty AGENTS.md files.

        Returns ``(directory, content)`` pairs ordered outermost-first.
        When ``stop_inclusive`` is True the stop directory is included in the
        walk; when False the walk stops before reaching it.
        """
        try:
            start_resolved = start.resolve()
            stop_resolved = stop.resolve()
        except (ValueError, OSError):
            return []
        if not start_resolved.is_relative_to(stop_resolved):
            return []

        docs: list[tuple[Path, str]] = []
        current = start_resolved
        while True:
            if current == stop_resolved and not stop_inclusive:
                break
            path = current / AGENTS_MD_FILENAME
            # Resolve-once-read-resolved: validate the symlink target stays within
            # the trust root, then read that exact resolved target (not the link).
            # A symlink swapped between a separate check and read can no longer
            # pull external content into the system prompt; an unresolvable link
            # fails closed.
            try:
                resolved = path.resolve()
            except (ValueError, OSError):
                resolved = None
            if resolved is not None and (
                resolved == stop_resolved or resolved.is_relative_to(stop_resolved)
            ):
                try:
                    stripped = read_safe(resolved).text.strip()
                    if stripped:
                        docs.append((current, stripped))
                except (FileNotFoundError, OSError):
                    pass
            if current == stop_resolved:
                break
            parent = current.parent
            if parent == current:  # fs-root safety
                break
            current = parent
        docs.reverse()  # outermost first
        return docs

    def find_subdirectory_agents_md(self, file_path: Path) -> list[tuple[Path, str]]:
        """Find AGENTS.md files between file_path's parent and its containing
        open dir (exclusive of the open dir itself).

        For lazy injection when reading files below any open project root.
        Does not overlap with load_project_docs() which covers the open dir
        and above.
        """
        try:
            resolved = file_path.resolve()
        except (ValueError, OSError):
            return []
        boundaries: list[Path] = []
        workdir = self.trusted_workdir
        if workdir is not None:
            boundaries.append(workdir)
        boundaries.extend(self.trusted_additional_dirs)
        containing = [
            root.resolve() for root in boundaries if resolved.is_relative_to(root)
        ]
        if not containing:
            return []
        stop = max(containing, key=lambda root: len(root.parts))
        start = resolved if resolved.is_dir() else resolved.parent
        return self._collect_agents_md(start, stop, stop_inclusive=False)

    def load_project_docs(self) -> list[tuple[Path, str]]:
        """Collect AGENTS.md files from each open project root up to its trust
        root.

        For the trusted cwd entry the trust root is found via
        ``trusted_folders_manager`` (and may sit above cwd). ``--add-dir``
        entries that aren't registered there fall back to the root itself.

        Returns ``(directory, content)`` pairs ordered outermost-first; later
        entries take priority. The same resolved directory is only emitted
        once across all roots.

        The walk starts at the actual cwd (not the trust-root-resolved
        project_roots) so AGENTS.md files between cwd and the repo root are
        collected on the way up.
        """
        by_dir: dict[Path, tuple[Path, str]] = {}
        walk_starts: list[Path] = []
        workdir = self.trusted_workdir
        if workdir is not None:
            walk_starts.append(workdir)
        walk_starts.extend(self.trusted_additional_dirs)
        for start in walk_starts:
            stop = trusted_folders_manager.find_trust_root(start) or start
            for d, content in self._collect_agents_md(start, stop, stop_inclusive=True):
                by_dir.setdefault(d.resolve(), (d, content))
        return list(by_dir.values())

    def _agents_md_has_content(self, path: Path) -> bool:
        try:
            return bool(read_safe(path).text.strip())
        except (FileNotFoundError, OSError):
            return False

    def agents_md_file_paths(self) -> list[Path]:
        """Return file paths of non-empty AGENTS.md docs.

        Lightweight change-detection helper. Walks from the actual cwd and
        add-dirs up to their trust roots (not from the trust-root-resolved
        project_roots).
        """
        paths: list[Path] = []
        if "user" in self.sources:
            user_path = VIBE_HOME.path / AGENTS_MD_FILENAME
            if self._agents_md_has_content(user_path):
                paths.append(user_path)
        seen: set[Path] = set()
        walk_starts: list[Path] = []
        workdir = self.trusted_workdir
        if workdir is not None:
            walk_starts.append(workdir)
        walk_starts.extend(self.trusted_additional_dirs)
        for start in walk_starts:
            stop = trusted_folders_manager.find_trust_root(start) or start
            try:
                start_resolved = start.resolve()
                stop_resolved = stop.resolve()
            except (ValueError, OSError):
                continue
            if not start_resolved.is_relative_to(stop_resolved):
                continue
            current = start_resolved
            while True:
                path = current / AGENTS_MD_FILENAME
                try:
                    resolved = path.resolve()
                except (ValueError, OSError):
                    resolved = None
                # Resolve-once-read-resolved (see _collect_agents_md): validate
                # and read the resolved target, not the link, so a swap between
                # check and read cannot change which file is watched.
                contained = resolved is not None and (
                    resolved == stop_resolved or resolved.is_relative_to(stop_resolved)
                )
                if (
                    resolved is not None
                    and contained
                    and resolved not in seen
                    and self._agents_md_has_content(resolved)
                ):
                    seen.add(resolved)
                    paths.append(resolved)
                if current == stop_resolved:
                    break
                parent = current.parent
                if parent == current:
                    break
                current = parent
        return paths


_manager: HarnessFilesManager | None = None


def init_harness_files_manager(
    *sources: FileSource,
    additional_dirs: list[Path] | None = None,
    cwd: Path | None = None,
) -> None:
    """Initialize the global HarnessFilesManager singleton.

    *additional_dirs* are extra working directories supplied via ``--add-dir`` /
    ACP ``additional_directories``. Entry points session-trust the accepted
    subset before passing them here; discovery re-validates each dir against
    the trust DB (see ``trusted_additional_dirs``), so a declined or untrusted
    dir never feeds project-root discovery.
    """
    global _manager
    candidate = HarnessFilesManager(
        sources=sources, cwd=cwd, _additional_dirs=tuple(additional_dirs or [])
    )
    if _manager is not None:
        if _manager == candidate:
            return
        raise RuntimeError(
            "HarnessFilesManager already initialized with different configuration"
        )
    _manager = candidate


def get_harness_files_manager() -> HarnessFilesManager:
    if _manager is None:
        raise RuntimeError(
            "HarnessFilesManager not initialized — call init_harness_files_manager() first"
        )
    return _manager


def reset_harness_files_manager() -> None:
    """Reset the singleton. Only intended for use in tests."""
    global _manager
    _manager = None


def add_session_dirs(dirs: list[Path]) -> None:
    """Set the session's extra working directories on the singleton (replace).

    Mirrors the CLI's ``--add-dir`` flow for ACP's ``additional_directories``
    parameter, which arrives per-session after the singleton is created at
    startup. The dirs feed project-root discovery (skills/tools/hooks/workflows/
    config) and file-tool in-bounds checks — without this, ACP silently provides
    less capability than the CLI for the same dirs.

    The list is REPLACED, not merged, so a long-lived ACP server's session N
    does not inherit session N-1's dirs (cross-session isolation). Each session
    declares its complete set; the startup singleton carries no add-dirs in ACP.
    """
    global _manager
    if _manager is None:
        raise RuntimeError(
            "HarnessFilesManager not initialized — call init_harness_files_manager() first"
        )
    if not dirs:
        return
    resolved = tuple(dedup_paths(dirs))
    if resolved == _manager.additional_dirs:
        return
    _manager = HarnessFilesManager(
        sources=_manager.sources, cwd=_manager.cwd, _additional_dirs=resolved
    )
