from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.logger import logger
from vibe.core.paths import dedup_paths
from vibe.core.utils.io import read_safe

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)


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
            name, description, source = self._parse_workflow_file(content, path)
            return WorkflowInfo(
                name=name,
                description=description,
                source=source,
                path=path,
                is_bundled=is_bundled,
            )
        except Exception as e:
            logger.warning("Failed to load workflow at %s: %s", path, e)
            return None

    @staticmethod
    def _parse_workflow_file(content: str, path: Path) -> tuple[str, str, str]:
        match = _FRONTMATTER_RE.match(content)
        if match is None:
            name = path.stem
            return name, "", content

        frontmatter_text = match.group(1)
        source = content[match.end() :]

        name = path.stem
        description = ""

        for line in frontmatter_text.strip().splitlines():
            if line.strip().startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.strip().startswith("description:"):
                description = line.split(":", 1)[1].strip()

        return name, description, source
