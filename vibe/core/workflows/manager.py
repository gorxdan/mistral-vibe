from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
import yaml

from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.logger import logger
from vibe.core.paths import dedup_paths
from vibe.core.utils.io import read_safe

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


class WorkflowMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    args_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkflowInfo:
    name: str
    description: str
    source: str
    path: Path
    is_bundled: bool = False
    args_schema: dict[str, Any] | None = None


class WorkflowManager:
    def __init__(self, config_getter: Callable[[], VibeConfig]) -> None:
        self._config_getter = config_getter
        self._search_paths = self._compute_search_paths(self._config)
        self._discovered: dict[str, WorkflowInfo] = self._discover_workflows()

        if custom_names := [
            n for n in self._discovered if not self._discovered[n].is_bundled
        ]:
            logger.info(
                "Discovered custom workflows %s in %s",
                " ".join(custom_names),
                " ".join(str(p) for p in self._search_paths),
            )

    @property
    def _config(self) -> VibeConfig:
        return self._config_getter()

    @property
    def workflows(self) -> dict[str, WorkflowInfo]:
        return dict(self._discovered)

    def get_workflow(self, name: str) -> WorkflowInfo | None:
        return self._discovered.get(name)

    def get_workflow_names(self) -> list[str]:
        return list(self._discovered.keys())

    def reload(self) -> None:
        """Re-run discovery.

        Call after writing/removing a workflow file so the new command is
        picked up without restarting the session.
        """
        self._search_paths = self._compute_search_paths(self._config)
        self._discovered = self._discover_workflows()

    def save_workflow_source(
        self,
        name: str,
        source: str,
        location: str = "auto",
    ) -> Path:
        """Persist a workflow script and return its path.

        ``location`` selects the destination:

        - ``"project"``: the closest project ``.vibe/workflows/`` (created at
          the closest project root if none exists yet).
        - ``"user"``: the user-global ``~/.vibe/workflows/``.
        - ``"auto"`` (default): project if a project root exists, else user.
          Mirrors Claude Code's "closest workflows dir" save rule.
        """
        slug = _SLUG_RE.sub("-", name.lower()).strip("-") or "workflow"
        mgr = get_harness_files_manager()
        target_dir = self._resolve_save_dir(mgr, location)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{slug}.py"
        path.write_text(source, encoding="utf-8")
        return path

    @staticmethod
    def _resolve_save_dir(mgr: Any, location: str) -> Path:
        if location == "user":
            from vibe.core.config.harness_files._paths import GLOBAL_WORKFLOWS_DIR

            return GLOBAL_WORKFLOWS_DIR.path
        # "project" or "auto": prefer the closest existing project workflows
        # dir, else create one at the closest project root.
        project_dirs = mgr.project_workflows_dirs
        if project_dirs:
            return project_dirs[0]
        roots = mgr.project_roots
        if roots:
            return roots[0] / ".vibe" / "workflows"
        if location == "project":
            # Explicit project requested but none available — fall back to user
            # rather than failing, and let the caller's notify surface the path.
            from vibe.core.config.harness_files._paths import GLOBAL_WORKFLOWS_DIR

            return GLOBAL_WORKFLOWS_DIR.path
        from vibe.core.config.harness_files._paths import GLOBAL_WORKFLOWS_DIR

        return GLOBAL_WORKFLOWS_DIR.path

    @staticmethod
    def _compute_search_paths(config: VibeConfig) -> list[Path]:
        mgr = get_harness_files_manager()
        return dedup_paths([
            *(p for p in config.workflow_paths if p.is_dir()),
            *mgr.project_workflows_dirs,
            *mgr.user_workflows_dirs,
        ])

    def _discover_workflows(self) -> dict[str, WorkflowInfo]:
        workflows: dict[str, WorkflowInfo] = {}
        workflows.update(self._discover_bundled())

        for base in self._search_paths:
            if not base.is_dir():
                continue
            for wf_file in sorted(base.glob("*.py")):
                if not wf_file.is_file():
                    continue
                if info := self._try_load_workflow(wf_file):
                    if info.name in workflows and not workflows[info.name].is_bundled:
                        logger.debug(
                            "Skipping duplicate workflow '%s' at %s", info.name, wf_file
                        )
                        continue
                    workflows[info.name] = info

        return workflows

    def _discover_bundled(self) -> dict[str, WorkflowInfo]:
        bundled_dir = Path(__file__).parent / "bundled"
        workflows: dict[str, WorkflowInfo] = {}
        if not bundled_dir.is_dir():
            return workflows
        for wf_file in sorted(bundled_dir.glob("*.py")):
            if info := self._try_load_workflow(wf_file, is_bundled=True):
                workflows[info.name] = info
        return workflows

    def _try_load_workflow(
        self, path: Path, *, is_bundled: bool = False
    ) -> WorkflowInfo | None:
        try:
            content = read_safe(path).text
            metadata, source = self._parse_workflow_file(content, path)
            return WorkflowInfo(
                name=metadata.name,
                description=metadata.description,
                source=source,
                path=path,
                is_bundled=is_bundled,
                args_schema=metadata.args_schema,
            )
        except Exception as e:
            logger.warning("Failed to load workflow at %s: %s", path, e)
            return None

    @staticmethod
    def _parse_workflow_file(
        content: str, path: Path
    ) -> tuple[WorkflowMetadata, str]:
        match = _FRONTMATTER_RE.match(content)
        if match is None:
            return WorkflowMetadata(name=path.stem), content

        source = content[match.end() :]

        try:
            data = yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            logger.warning("Invalid YAML frontmatter in %s: %s", path, e)
            data = None
        if not isinstance(data, dict):
            data = {}
        # Default the name to the filename stem if absent or null.
        if not data.get("name"):
            data["name"] = path.stem

        try:
            metadata = WorkflowMetadata.model_validate(data)
        except Exception as e:
            logger.warning("Invalid workflow metadata in %s: %s", path, e)
            metadata = WorkflowMetadata(name=path.stem)
        return metadata, source
