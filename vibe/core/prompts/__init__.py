from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum, auto
from pathlib import Path

from vibe import VIBE_ROOT
from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.utils.io import read_safe

PROMPTS_DIR = VIBE_ROOT / "core" / "prompts"

_VERBOSE_SUBDIR = "_verbose"
_prompt_variant = "compressed"


def set_prompt_variant(variant: str) -> None:
    """Select the prompt-compression A/B arm: "compressed" (default, the shipped
    files) or "verbose" (the pre-compression copies under each prompt dir's
    _verbose/). Resolved once from ExperimentName.PROMPT_COMPRESSION at session
    bootstrap; refreshes the caches and baked prompts that captured the prior arm.
    """
    global _prompt_variant
    normalized = "verbose" if variant == "verbose" else "compressed"
    if normalized == _prompt_variant:
        return
    _prompt_variant = normalized
    from vibe.core.tools.base import clear_tool_prompt_cache

    clear_tool_prompt_cache()
    from vibe.core.skills.builtins.workflow import refresh_prompt

    refresh_prompt()


def prompt_variant() -> str:
    return _prompt_variant


def verbose_override(path: Path) -> Path | None:
    """The pre-compression copy under a sibling _verbose/ dir when the verbose arm
    is active and that copy exists; else None (use the shipped file).
    """
    if _prompt_variant != "verbose":
        return None
    candidate = path.parent / _VERBOSE_SUBDIR / path.name
    return candidate if candidate.is_file() else None


class Prompt(StrEnum):
    @property
    def path(self) -> Path:
        return (PROMPTS_DIR / self.value).with_suffix(".md")

    def read(self) -> str:
        path = self.path
        return read_safe(verbose_override(path) or path).text.strip()


class SystemPrompt(Prompt):
    CLI = auto()
    EXPLORE = auto()
    DEBUGGER = auto()
    PLANNER = auto()
    SECURITY = auto()
    VERIFIER = auto()
    EDITOR = auto()
    TESTS = auto()
    LEAN = auto()
    MINIMAL = auto()
    COORDINATOR = auto()


class UtilityPrompt(Prompt):
    AGENTS_DOC = auto()
    COMPACT = auto()
    COMPACT_SUMMARY_PREFIX = auto()
    DANGEROUS_DIRECTORY = auto()
    PROJECT_CONTEXT = auto()
    TURN_SUMMARY = auto()


class MissingPromptFileError(ValueError):
    def __init__(
        self,
        setting_name: str,
        prompt_id: str,
        builtin_ids: Iterable[str],
        custom_dirs: Iterable[Path],
        custom_ids: Iterable[str],
    ) -> None:
        builtin_hint = ", ".join('"' + i + '"' for i in builtin_ids)
        dirs_hint = " or ".join(str(d) for d in custom_dirs) or "<no prompt dirs>"
        custom_hint = ", ".join('"' + i + '"' for i in custom_ids) or "<none>"
        super().__init__(
            f"Invalid {setting_name} value: '{prompt_id}'. "
            f"Must be one of the available prompts ({builtin_hint}), "
            f"or correspond to a .md file in {dirs_hint} (available: {custom_hint})"
        )
        self.setting_name = setting_name
        self.prompt_id = prompt_id


def _validate_prompt_id(prompt_id: str, setting_name: str) -> None:
    if (
        not prompt_id
        or prompt_id in {".", ".."}
        or "/" in prompt_id
        or "\\" in prompt_id
    ):
        raise ValueError(
            f"Invalid {setting_name} value: '{prompt_id}' must be a bare filename "
            "without path separators"
        )


def load_prompt(
    prompt_id: str,
    *,
    setting_name: str,
    builtins: Mapping[str, Path],
    extra_dirs: Iterable[Path] = (),
) -> str:
    _validate_prompt_id(prompt_id, setting_name)
    mgr = get_harness_files_manager()
    # extra_dirs (e.g. prompt_paths from config, including plugin-supplied dirs)
    # take precedence over the harness-managed dirs, which in turn take
    # precedence over builtins — so a user/plugin can override a builtin prompt
    # by stem without forking the package.
    custom_dirs: list[Path] = [Path(d) for d in extra_dirs]
    custom_dirs += mgr.project_prompts_dirs + mgr.user_prompts_dirs
    for d in custom_dirs:
        path = (d / prompt_id).with_suffix(".md")
        if path.is_file():
            return read_safe(path).text.strip()

    builtin_path = builtins.get(prompt_id.lower())
    if builtin_path is not None:
        chosen = verbose_override(builtin_path) or builtin_path
        if chosen.is_file():
            return read_safe(chosen).text.strip()

    custom_ids = sorted({p.stem for d in custom_dirs for p in d.glob("*.md")})
    raise MissingPromptFileError(
        setting_name, prompt_id, tuple(builtins), custom_dirs, custom_ids
    )


def load_system_prompt(prompt_id: str, *, extra_dirs: Iterable[Path] = ()) -> str:
    builtins: dict[str, Path] = {p.name.lower(): p.path for p in SystemPrompt}
    # Experiment variants may reference bundled .md files not in the enum.
    fallback = (PROMPTS_DIR / prompt_id).with_suffix(".md")
    if fallback.is_file():
        builtins.setdefault(prompt_id.lower(), fallback)
    return load_prompt(
        prompt_id,
        setting_name="system_prompt_id",
        builtins=builtins,
        extra_dirs=extra_dirs,
    )


__all__ = [
    "PROMPTS_DIR",
    "MissingPromptFileError",
    "Prompt",
    "SystemPrompt",
    "UtilityPrompt",
    "load_prompt",
    "load_system_prompt",
    "prompt_variant",
    "set_prompt_variant",
    "verbose_override",
]
