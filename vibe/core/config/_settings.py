from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import re
import shlex
import tomllib
from typing import Any, ClassVar, Literal, get_args
from urllib.parse import urljoin

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
import tomli_w

from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config._defaults import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_AUTO_COMPACT_THRESHOLD,
    DEFAULT_CONSOLE_BASE_URL,
    DEFAULT_MISTRAL_API_ENV_KEY,
    DEFAULT_MISTRAL_BROWSER_AUTH_API_BASE_URL,
    DEFAULT_MISTRAL_BROWSER_AUTH_BASE_URL,
    DEFAULT_MISTRAL_SERVER_URL,
    DEFAULT_THEME,
    DEFAULT_VIBE_BASE_URL,
)
from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.config.models import (
    THINKING_LEVELS as THINKING_LEVELS,
    ConnectorConfig,
    ExperimentsConfig,
    MCPServer,
    MissingAPIKeyError,
    ModelConfig,
    OtelSpanExporterConfig,
    ProjectContextConfig,
    ProviderConfig,
    SessionLoggingConfig,
    ThinkingLevel,
    TranscribeModelConfig,
    TranscribeProviderConfig,
    TTSModelConfig,
    TTSProviderConfig,
)
from vibe.core.logger import logger
from vibe.core.paths import GLOBAL_ENV_FILE, VIBE_HOME
from vibe.core.prompts import (
    SystemPrompt,
    UtilityPrompt,
    load_prompt,
    load_system_prompt,
)
from vibe.core.types import Backend
from vibe.core.utils import configure_ssl_context


def _strip_bash_pattern_wildcard(pattern: str) -> str:
    if pattern.endswith(" *"):
        return pattern[:-2]
    return pattern


def deep_update(
    mapping: dict[str, Any], updating_mapping: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(mapping)
    for key, value in updating_mapping.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_dotenv_values(
    env_path: Path = GLOBAL_ENV_FILE.path,
    environ: MutableMapping[str, str] = os.environ,
) -> None:
    # We allow FIFO path to support some environment management solutions (e.g. https://developer.1password.com/docs/environments/local-env-file/)
    if not env_path.is_file() and not env_path.is_fifo():
        return

    env_vars = dotenv_values(env_path)
    for key, value in env_vars.items():
        if not value:
            continue
        existing = environ.get(key)
        if existing:
            # An explicit non-empty process/shell value wins over the .env file.
            # When it DIFFERS from a saved credential, the saved one (e.g. from a
            # browser sign-in) is silently shadowed — warn so the stale-key trap
            # is visible rather than surfacing later as auth failures.
            if existing != value and key.endswith("_API_KEY"):
                logger.warning(
                    "%s in your shell environment shadows the value saved in %s "
                    "(e.g. a browser sign-in); the shell value is used. Remove the "
                    "shell export to use the saved key.",
                    key,
                    env_path,
                )
            continue
        environ[key] = value


# API-key presence is environmental, not structural. Internal re-derivations of
# an already-loaded config (AgentProfile.apply_to_config) run inside this guard
# so a missing key never raises from lazy, unguarded call sites (UI banner
# refresh, shutdown). A pydantic `context=` cannot do this: BaseSettings
# validates twice and the inner sources-build pass ignores the context.
_skip_api_key_check: ContextVar[bool] = ContextVar("_skip_api_key_check", default=False)


@contextmanager
def skip_api_key_check() -> Iterator[None]:
    token = _skip_api_key_check.set(True)
    try:
        yield
    finally:
        _skip_api_key_check.reset(token)


class TomlFileSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self.toml_data = self._load_toml()

    @staticmethod
    def _read_toml(file: Path) -> dict[str, Any]:
        try:
            with file.open("rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            return {}
        except tomllib.TOMLDecodeError as e:
            raise RuntimeError(f"Invalid TOML in {file}: {e}") from e
        except OSError as e:
            raise RuntimeError(f"Cannot read {file}: {e}") from e

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = TomlFileSettingsSource._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _confine_plugin_paths(
        data: dict[str, Any], project_root: Path
    ) -> dict[str, Any]:
        if not isinstance(data.get("plugin_paths"), list):
            return data
        root_resolved = project_root.resolve()
        kept: list[Any] = []
        for entry in data["plugin_paths"]:
            resolved: Path | None = None
            try:
                entry_path = Path(str(entry)).expanduser()
                resolved = (
                    entry_path.resolve()
                    if entry_path.is_absolute()
                    else (root_resolved / entry_path).resolve()
                )
                contained = resolved == root_resolved or resolved.is_relative_to(
                    root_resolved
                )
            except (OSError, ValueError):
                contained = False
            if contained and resolved is not None:
                kept.append(resolved)
            else:
                logger.warning(
                    "plugin_paths entry %r from project config escapes the "
                    "trusted project root %s; dropping",
                    entry,
                    root_resolved,
                )
        return {**data, "plugin_paths": kept}

    def _load_toml(self) -> dict[str, Any]:
        mgr = get_harness_files_manager()
        user_file = mgr.user_config_file
        data = (
            self._read_toml(user_file)
            if "user" in mgr.sources and user_file.is_file()
            else {}
        )
        project_configs = mgr.project_config_files_with_roots
        if not project_configs:
            return data
        for file, project_root in reversed(project_configs):
            project_data = self._confine_plugin_paths(
                self._read_toml(file), project_root
            )
            data = self._deep_merge(data, project_data)
        return data

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        return self.toml_data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self.toml_data


def _remove_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned_value
            for key, item in value.items()
            if (cleaned_value := _remove_none_values(item)) is not None
        }
    if isinstance(value, list):
        return [
            cleaned_item
            for item in value
            if (cleaned_item := _remove_none_values(item)) is not None
        ]
    return value


class SafetyJudgeConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    model: str | None = None
    max_tokens: int = 512
    # None → use the judge model's own configured temperature (some providers
    # reject anything but their fixed value, e.g. Kimi requires 1).
    temperature: float | None = None
    # Generous default: reasoning judge models (e.g. GLM with thinking=max) can
    # take >15s on large tool args before emitting the verdict. On timeout the
    # judge fails closed (the user is prompted).
    timeout: float = 30.0
    extra_body: dict[str, Any] = Field(default_factory=dict)
    verdict_cache_size: int = 256


class MaxOutputEscalationConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    # Floor for the first escalation step when no override is active yet. The
    # first retry uses min(base*factor, cap) = 16384, the >=16k that thinking
    # models (Moonshot k2.7-code) need for reasoning+output.
    base: int = 8192
    factor: float = 2.0
    cap: int = 65536
    max_attempts: int = 3


class SnipConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    # Watermarks are fractions of the active model's auto_compact_threshold.
    high_watermark: float = 0.6
    target: float = 0.5
    keep_recent_turns: int = 8
    min_message_tokens: int = 300


class MicrocompactConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    # Engages just above snip's 0.6 so snip (lossless, recoverable) gets first
    # crack, then microcompact gists the non-recoverable remainder. The old 0.8
    # left a ~51k unshaped band (snip abstains on non-recoverable, microcompact
    # not yet triggered) — live glm session climbed 153.6k->204.8k unshaped.
    # INVARIANT: target < high_watermark, else the loop's `est <= target` break
    # fires the instant the gate opens and nothing is gisted (target 0.7 > the
    # 0.65 gate made the watermark inert — micro effectively engaged at 0.7, not
    # 0.65, so the band never actually closed). target 0.6 drives the
    # non-recoverable floor to just above snip's engage point — gentler than
    # snip's 0.5 since gisting is lossy and snip's pointer is not.
    high_watermark: float = 0.65
    target: float = 0.6
    # Each gisted block costs this much; the floor is ~(block count) x this, so a
    # long session's floor rises with length. 2000 was too high — new blocks
    # barely exceeded it (tiny sheds) and a live session climbed to 250k. 1000
    # halves the floor and makes more blocks eligible per pass.
    per_message_cap_tokens: int = 1000
    # Now that microcompact is the sole shaper for non-recoverable content, 1/turn
    # treads water: a live session sat at ~204k for ~10 turns shedding 280-900
    # tokens/turn while growth replaced them. Several blocks/turn lets it actually
    # reduce. Cache stays ~95% even under heavy per-turn shaping, so the
    # prefix-churn cost is small. Tune up if deep sessions still stall.
    max_blocks_per_turn: int = 4


class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    select_mode: Literal["per-turn", "per-session", "always"] = "per-turn"
    # When true, deep-recall selection races the LLM loop instead of blocking
    # it: the always-on index is injected at turn start, and full bodies are
    # folded in only if the selector settles before the first LLM call. A slow
    # selector is abandoned to index-only rather than stalling the turn for the
    # full timeout. Set false to restore the blocking pre-loop selection.
    prefetch: bool = True
    model: str | None = None
    max_selected: int = 5
    max_inject_chars: int = 8000
    # Where the volatile recall block is placed. "system" embeds it in the
    # system prompt (replaced each turn): a selection change then mutates the
    # prefix root and busts the cached history behind it. "late" (default) keeps
    # the system prompt byte-stable and rides recall on an ephemeral message
    # just before the latest user turn, so only the small tail is reprocessed.
    # Live glm-5.2 A/B on a selection flip: 8.7% cached (system) -> 96.3% (late).
    inject_mode: Literal["system", "late"] = "late"
    max_entries_scanned: int = 200
    timeout: float = 20.0
    extra_body: dict[str, Any] = Field(default_factory=dict)
    auto_extract: bool = False
    auto_extract_model: str | None = None
    auto_extract_max_writes: int = 3
    auto_extract_min_messages: int = 4
    auto_extract_timeout: float = 30.0
    consolidate: bool = False
    consolidate_model: str | None = None
    consolidate_min_age_days: int = 14
    consolidate_min_candidates: int = 6
    consolidate_interval_days: int = 7
    consolidate_max_actions: int = 5
    consolidate_timeout: float = 45.0
    verify: bool = False
    verify_model: str | None = None
    verify_min_age_days: int = 7
    verify_min_candidates: int = 3
    verify_interval_days: int = 14
    verify_max_memories: int = 5
    verify_timeout: float = 45.0
    # Trash retention: how long deleted/merged memory files stay recoverable in
    # the per-directory .trash/ tree before a session-start sweep unlinks them.
    # Aged by the timestamp encoded in the trash filename. 0 disables sweeping
    # (trash accumulates indefinitely, the historical behavior). The sweep runs
    # once per session when the memory store is first built; the ledger is
    # compacted to drop entries whose files are gone.
    trash_max_age_days: int = 30


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    write_dirs: list[str] = Field(default_factory=list)
    allow_network: bool = True
    scrub_env: bool = True
    env_passthrough: list[str] = Field(default_factory=list)
    require_backend: bool = False
    backend: str = "auto"  # auto | bwrap | unshare | sandbox-exec | none
    extra_args: list[str] = Field(default_factory=list)


class ContextShapingConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    snip: SnipConfig = Field(default_factory=SnipConfig)
    microcompact: MicrocompactConfig = Field(default_factory=MicrocompactConfig)
    # Per-injected-message backend cap. The persisted transcript stays verbatim;
    # only the model-facing copy is middle-truncated. 0 disables this pass.
    max_injected_message_tokens: int = 20_000
    # Never edit messages within the first N estimated tokens after the system
    # prompt, to keep the provider's auto-cached prefix stable across edits.
    cache_prefix_guard_tokens: int = 4000


class BaselineScalingConfig(BaseSettings):
    """Window-tiered shrinking of the irreducible baseline (system prompt + tool
    schemas + project context). Opt-in per model via ModelConfig.context_window:
    a model with no declared window is always tier LARGE and unchanged.
    """

    model_config = SettingsConfigDict(extra="ignore")

    # Master switch. False => every model is LARGE even with context_window set.
    enabled: bool = True
    # effective_context_window < small_max => SMALL; < medium_max => MEDIUM;
    # otherwise LARGE. Below small_max the baseline is aggressively trimmed.
    small_max: int = 48_000
    medium_max: int = 200_000
    # Window-scaled cache_prefix_guard_tokens for tiered (non-LARGE) models, so
    # the fixed 4000 guard isn't a large fraction of a small window.
    guard_window_fraction: float = 0.05
    guard_floor: int = 512
    # auto_compact_threshold = window * derive_threshold_fraction when a window is
    # declared and no explicit threshold is set; an explicit threshold is clamped
    # to <= window * safety_cap_fraction so a declared window always governs.
    derive_threshold_fraction: float = 0.85
    safety_cap_fraction: float = 0.95
    # SMALL-tier builtin tool-schema description trimming.
    trim_tool_descriptions_small: bool = True
    tool_description_max_chars: int = 220
    # SMALL-tier byte budget for each injected AGENTS.md doc (user + per-project).
    # The project context is the largest untrimmed baseline chunk on a small
    # window; each doc body is truncated to this many bytes (0 keeps full text).
    small_agents_md_bytes: int = 2000


class WorktreeConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    mode: Literal["off", "on", "auto-by-entrypoint"] = "on"
    base_dir: str = ""
    branch_prefix: str = "vibe/"
    merge: Literal["manual", "auto-ff"] = "auto-ff"
    cleanup: Literal["remove", "keep"] = "remove"
    carry_dirty: bool = True
    carry_ignored: list[str] = Field(
        default_factory=lambda: ["node_modules", ".venv", "venv", ".env"]
    )
    report_on_startup: bool = True
    # Garbage-collect abandoned worktrees/branches older than this many days,
    # but only when the branch is already merged into HEAD or empty (never when
    # it holds unmerged work). 0 disables age-based GC.
    gc_age_days: int = 7

    @field_validator("base_dir", mode="before")
    @classmethod
    def _default_base_dir(cls, v: str) -> str:
        return v or str(VIBE_HOME.path / "worktrees")


class LSPServer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Short alias identifying this language server.")
    command: str | list[str] = Field(
        description="Executable and its arguments to launch the server (stdio transport)."
    )
    languages: dict[str, str] = Field(
        description=(
            "Mapping of file extension (with dot, e.g. '.py') to LSP language id "
            "(e.g. 'python'). The server claims ownership of these extensions."
        )
    )
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set for the language server process.",
    )
    cwd: str | None = Field(
        default=None, description="Working directory for the language server process."
    )
    initialization_options: dict[str, Any] | None = Field(default=None)
    root_uri: str | None = Field(
        default=None,
        description=(
            "Workspace root URI passed during initialize. Defaults to the current "
            "project directory when omitted."
        ),
    )
    manifest_markers: tuple[str, ...] = Field(
        default=(),
        description=(
            "Filenames (e.g. Cargo.toml, go.mod) used to discover the workspace "
            "root for a project by walking up from the opened file. Empty for "
            "servers that accept the session root as-is."
        ),
    )
    startup_timeout_sec: float = Field(default=20.0, gt=0)
    request_timeout_sec: float = Field(default=10.0, gt=0)

    @field_validator("name", mode="after")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", v).strip("_-")
        if not normalized:
            raise ValueError(
                "LSP server name must contain at least one identifier char"
            )
        return normalized[:256]

    @field_validator("languages", mode="after")
    @classmethod
    def _must_have_languages(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError(
                "an LSP server must declare at least one language/extension"
            )
        return v

    def argv(self) -> list[str]:
        base = (
            shlex.split(self.command)
            if isinstance(self.command, str)
            else list(self.command or [])
        )
        return [*base, *self.args] if self.args else base

    def to_server_config(self) -> Any:
        from vibe.core.lsp._server import ServerConfig

        return ServerConfig(
            name=self.name,
            command=self.argv(),
            languages=dict(self.languages),
            env=dict(self.env),
            cwd=self.cwd,
            root_uri=self.root_uri,
            initialization_options=self.initialization_options,
            manifest_markers=tuple(self.manifest_markers),
            startup_timeout=self.startup_timeout_sec,
            request_timeout=self.request_timeout_sec,
        )


class ToolManifestConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dynamic_subset_enabled: bool = Field(
        default=True,
        description=(
            "When true, large remote MCP/connector tool catalogs are withheld "
            "from the model-facing manifest and exposed through tool_search."
        ),
    )
    dynamic_subset_threshold: int = Field(
        default=80,
        ge=1,
        description="Total available-tool count above which remote tools are gated.",
    )
    dynamic_pinned_tool_limit: int = Field(
        default=8,
        ge=1,
        description="Maximum number of tool_search-selected remote tools kept visible.",
    )
    dynamic_search_results: int = Field(
        default=8, ge=1, description="Default maximum matches returned by tool_search."
    )


EffortLevel = Literal["normal", "le-chaton"]
EFFORT_LEVELS: list[str] = list(get_args(EffortLevel))

DEFAULT_VIBE_CODE_WORKFLOW_ID = "__shared-nuage-workflow"
DEFAULT_VIBE_CODE_TASK_QUEUE = "shared-vibe-nuage"


DEFAULT_PROVIDERS = [
    ProviderConfig(
        name="mistral",
        api_base=f"{DEFAULT_MISTRAL_SERVER_URL}/v1",
        api_key_env_var=DEFAULT_MISTRAL_API_ENV_KEY,
        browser_auth_base_url=DEFAULT_MISTRAL_BROWSER_AUTH_BASE_URL,
        browser_auth_api_base_url=DEFAULT_MISTRAL_BROWSER_AUTH_API_BASE_URL,
        backend=Backend.MISTRAL,
    ),
    ProviderConfig(
        name="llamacpp",
        api_base="http://127.0.0.1:8080/v1",
        api_key_env_var="",  # NOTE: if you wish to use --api-key in llama-server, change this value
    ),
]

DEFAULT_ACTIVE_MODEL_CONFIG = ModelConfig(
    name="mistral-vibe-cli-latest",
    provider="mistral",
    alias="mistral-medium-3.5",
    temperature=1.0,
    input_price=1.5,
    output_price=7.5,
    thinking="high",
    supports_images=True,
)

DEFAULT_MODELS = [
    DEFAULT_ACTIVE_MODEL_CONFIG,
    ModelConfig(
        name="devstral-small-latest",
        provider="mistral",
        alias="devstral-small",
        input_price=0.1,
        output_price=0.3,
    ),
    ModelConfig(
        name="devstral",
        provider="llamacpp",
        alias="local",
        input_price=0.0,
        output_price=0.0,
    ),
]

DEFAULT_TRANSCRIBE_PROVIDERS = [
    TranscribeProviderConfig(
        name="mistral",
        api_base="wss://api.mistral.ai",
        api_key_env_var=DEFAULT_MISTRAL_API_ENV_KEY,
    )
]

DEFAULT_ACTIVE_TRANSCRIBE_MODEL_CONFIG = TranscribeModelConfig(
    name="voxtral-mini-transcribe-realtime-2602",
    provider="mistral",
    alias="voxtral-realtime",
)

DEFAULT_TRANSCRIBE_MODELS = [DEFAULT_ACTIVE_TRANSCRIBE_MODEL_CONFIG]

DEFAULT_TTS_PROVIDERS = [
    TTSProviderConfig(
        name="mistral",
        api_base="https://api.mistral.ai",
        api_key_env_var=DEFAULT_MISTRAL_API_ENV_KEY,
    )
]

DEFAULT_ACTIVE_TTS_MODEL_CONFIG = TTSModelConfig(
    name="voxtral-mini-tts-latest", provider="mistral", alias="voxtral-tts"
)

DEFAULT_TTS_MODELS = [DEFAULT_ACTIVE_TTS_MODEL_CONFIG]


def resolve_api_key(env_key: str) -> str | None:
    if not env_key:
        return None
    value = os.environ.get(env_key)
    if value:
        from vibe.core.utils.keyring import get_api_key_from_keyring

        try:
            keyring_value = get_api_key_from_keyring(env_key)
        except Exception:
            keyring_value = None
        if keyring_value and keyring_value != value:
            logger.warning(
                "%s in your environment shadows the keyring value; the env var is used. "
                "If auth fails, unset the env var to fall back to the keyring key.",
                env_key,
            )
        return value
    from vibe.core.utils.keyring import get_api_key_from_keyring

    return get_api_key_from_keyring(env_key)


def resolve_theme_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return DEFAULT_THEME
    if value == DEFAULT_THEME:
        return value
    from textual.theme import BUILTIN_THEMES

    if value not in BUILTIN_THEMES:
        logger.warning("Unknown theme=%s; falling back to %s", value, DEFAULT_THEME)
        return DEFAULT_THEME
    return value


class VibeConfig(BaseSettings):
    active_model: str = DEFAULT_ACTIVE_MODEL_CONFIG.alias
    vim_keybindings: bool = False
    theme: str = DEFAULT_THEME
    disable_welcome_banner_animation: bool = False
    autocopy_to_clipboard: bool = True
    file_watcher_for_autocomplete: bool = True
    ask_confirmation_on_exit: bool = True
    displayed_workdir: str = ""
    context_warnings: bool = True
    voice_mode_enabled: bool = False
    narrator_enabled: bool = False
    active_transcribe_model: str = DEFAULT_ACTIVE_TRANSCRIBE_MODEL_CONFIG.alias
    active_tts_model: str = DEFAULT_ACTIVE_TTS_MODEL_CONFIG.alias
    bypass_tool_permissions: bool = False
    raise_on_compaction_failure: bool = False
    enable_telemetry: bool = True
    experiment_overrides: dict[str, str] = Field(default_factory=dict)
    applied_migrations: list[str] = Field(default_factory=list, exclude=True)
    system_prompt_id: str = SystemPrompt.CLI
    compaction_prompt_id: str = UtilityPrompt.COMPACT
    include_commit_signature: bool = True
    include_humanizer_guidance: bool = True
    caveman_thinking: bool = True
    include_model_info: bool = True
    include_project_context: bool = True
    include_prompt_detail: bool = True
    include_config_reference: bool = True
    enable_update_checks: bool = False
    enable_notifications: bool = True
    enable_system_trust_store: bool = False
    api_timeout: float = DEFAULT_API_TIMEOUT
    auto_compact_threshold: int = DEFAULT_AUTO_COMPACT_THRESHOLD

    vibe_code_enabled: bool = Field(default=True, exclude=True)
    vibe_code_base_url: str = Field(default=DEFAULT_MISTRAL_SERVER_URL, exclude=True)
    vibe_code_sessions_base_url: str = Field(
        default="https://chat.mistral.ai", exclude=True
    )
    vibe_code_workflow_id: str = Field(
        default=DEFAULT_VIBE_CODE_WORKFLOW_ID, exclude=True
    )
    vibe_code_api_key_env_var: str = Field(
        default=DEFAULT_MISTRAL_API_ENV_KEY, exclude=True
    )
    vibe_code_project_name: str | None = Field(default=None, exclude=True)

    enable_otel: bool = Field(default=True, exclude=True)
    otel_endpoint: str = Field(default="", exclude=True)
    otel_local_export: bool = Field(default=True, exclude=True)
    # Opt-in: attach prompt/response prose to chat spans as events (recall /
    # forgetfulness debugging). Off by default — content capture balloons local
    # trace files and records user/source bytes.
    otel_capture_content: bool = Field(default=False, exclude=True)

    console_base_url: str = Field(default=DEFAULT_CONSOLE_BASE_URL, exclude=True)
    vibe_base_url: str = Field(default=DEFAULT_VIBE_BASE_URL, exclude=True)

    enable_experimental_hooks: bool = Field(default=False, exclude=True)

    providers: list[ProviderConfig] = Field(
        default_factory=lambda: list(DEFAULT_PROVIDERS)
    )
    models: list[ModelConfig] = Field(default_factory=lambda: list(DEFAULT_MODELS))
    compaction_model: ModelConfig | None = None
    fallback_models: list[str] = Field(default_factory=list)
    # Empty = inherit the host session's model.
    subagent_model: str = ""
    # Default model for the `mechanic` subagent (bulk/mechanical work). Empty =
    # fall back to subagent_model, then the host session's model.
    mechanical_model: str = ""
    max_output_escalation: MaxOutputEscalationConfig = Field(
        default_factory=MaxOutputEscalationConfig
    )
    context_shaping: ContextShapingConfig = Field(default_factory=ContextShapingConfig)
    baseline_scaling: BaselineScalingConfig = Field(
        default_factory=BaselineScalingConfig
    )
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    transcribe_providers: list[TranscribeProviderConfig] = Field(
        default_factory=lambda: list(DEFAULT_TRANSCRIBE_PROVIDERS)
    )
    transcribe_models: list[TranscribeModelConfig] = Field(
        default_factory=lambda: list(DEFAULT_TRANSCRIBE_MODELS)
    )

    tts_providers: list[TTSProviderConfig] = Field(
        default_factory=lambda: list(DEFAULT_TTS_PROVIDERS)
    )
    tts_models: list[TTSModelConfig] = Field(
        default_factory=lambda: list(DEFAULT_TTS_MODELS)
    )

    project_context: ProjectContextConfig = Field(default_factory=ProjectContextConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    session_logging: SessionLoggingConfig = Field(default_factory=SessionLoggingConfig)
    worktree: WorktreeConfig = Field(default_factory=WorktreeConfig)
    safety_judge: SafetyJudgeConfig = Field(default_factory=SafetyJudgeConfig)
    tools: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional directories or files to explore for custom tools. "
            "Paths may be absolute or relative to the current working directory. "
            "Directories are shallow-searched for tool definition files, "
            "while files are loaded directly if valid."
        ),
    )

    mcp_servers: list[MCPServer] = Field(
        default_factory=list, description="Preferred MCP server configuration entries."
    )
    lsp_servers: list[LSPServer] = Field(
        default_factory=list,
        description=(
            "Language Server Protocol servers. Each entry owns one or more file "
            "extensions; declare one entry per language. The feature is opt-in: "
            "install it with /lspstall before configuring servers."
        ),
    )
    lsp_auto_discover: bool = Field(
        default=True,
        description=(
            "When True (default), auto-discover installed language servers and "
            "filter by project manifest markers. When False, only explicitly-"
            "declared lsp_servers entries are used."
        ),
    )
    enable_connectors: bool = Field(
        default=True,
        description=(
            "Master switch for Mistral connectors. When False, no connector "
            "tools are discovered or registered, regardless of provider/API key."
        ),
    )
    tool_manifest: ToolManifestConfig = Field(default_factory=ToolManifestConfig)
    connectors: list[ConnectorConfig] = Field(
        default_factory=list,
        description="Per-connector settings (disable, disabled_tools).",
    )

    enabled_tools: list[str] = Field(
        default_factory=list,
        description=(
            "An explicit list of tool names/patterns to enable. If set, only these"
            " tools will be active. Supports glob patterns (e.g., 'serena_*') and"
            " regex with 're:' prefix (e.g., 're:^serena_.*')."
        ),
    )
    disabled_tools: list[str] = Field(
        default_factory=list,
        description=(
            "A list of tool names/patterns to disable. Ignored if 'enabled_tools'"
            " is set. Supports glob patterns and regex with 're:' prefix."
        ),
    )
    agent_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for custom agent profiles. "
            "Each path may be absolute or relative to the current working directory."
        ),
    )
    prompt_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for custom prompt (.md) files. "
            "Searched before the builtin prompts, so an id here overrides a "
            "builtin of the same stem. Each path may be absolute or relative "
            "to the current working directory."
        ),
    )
    plugin_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Plugin root directories (each containing a plugin.toml) whose "
            "components are layered into the matching *_paths / mcp_servers / hooks."
        ),
    )
    enabled_plugins: list[str] = Field(
        default_factory=list,
        description="Glob/regex allowlist of plugin names; if set, only these load.",
    )
    disabled_plugins: list[str] = Field(
        default_factory=list,
        description="Glob/regex denylist of plugin names (ignored if enabled set).",
    )
    enabled_agents: list[str] = Field(
        default_factory=list,
        description=(
            "An explicit list of agent names/patterns to enable. If set, only these"
            " agents will be available. Supports glob patterns (e.g., 'custom-*')"
            " and regex with 're:' prefix."
        ),
    )
    disabled_agents: list[str] = Field(
        default_factory=list,
        description=(
            "A list of agent names/patterns to disable. Ignored if 'enabled_agents'"
            " is set. Supports glob patterns and regex with 're:' prefix."
        ),
    )
    installed_agents: list[str] = Field(
        default_factory=list,
        description=(
            "A list of opt-in builtin agent names that have been explicitly installed."
        ),
    )
    installed_components: list[str] = Field(
        default_factory=list,
        description=(
            "Opt-in feature components explicitly installed via their setup command "
            "(e.g. /lspstall adds 'lsp'). Components stay dormant until listed here."
        ),
    )
    default_agent: str = Field(
        default=BuiltinAgentName.DEFAULT,
        description=(
            "Agent profile to use when no --agent flag is passed. "
            "Builtin: default, plan, accept-edits, auto-approve. "
            "Applies in both interactive and programmatic (-p/--prompt) mode."
        ),
    )
    skill_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for skills. "
            "Each path may be absolute or relative to the current working directory."
        ),
    )
    enabled_skills: list[str] = Field(
        default_factory=list,
        description=(
            "An explicit list of skill names/patterns to enable. If set, only these"
            " skills will be active. Supports glob patterns (e.g., 'search-*') and"
            " regex with 're:' prefix."
        ),
    )
    disabled_skills: list[str] = Field(
        default_factory=list,
        description=(
            "A list of skill names/patterns to disable. Ignored if 'enabled_skills'"
            " is set. Supports glob patterns and regex with 're:' prefix."
        ),
    )
    workflow_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for workflow scripts. "
            "Each path may be absolute or relative to the current working directory."
        ),
    )
    disable_workflows: bool = Field(
        default=False,
        description=(
            "Disable workflow features entirely. When true, /workflows is "
            "unavailable, workflow commands are not registered, and le chaton "
            "effort mode cannot be activated."
        ),
    )
    effort_mode: str = Field(
        default="normal",
        description=(
            "Effort mode: 'normal' (default) or 'le-chaton' (max thinking + "
            "automatic workflow planning)."
        ),
    )
    verification_subsystem: bool = Field(
        default=True,
        description=(
            "Enable the host-agent verification layer: a completion nudge in "
            "the todo tool and a verification contract in the system prompt "
            "that require independent verification (the 'verifier' subagent) "
            "before non-trivial work is reported done. The verifier profile "
            "itself is always available; this gates the host-side nudge and "
            "contract section."
        ),
    )
    investigation_subsystem: bool = Field(
        default=True,
        description=(
            "Enable the host-agent investigation layer: a contract section in "
            "the system prompt stating when a reproduction is required before "
            "a fix/design/diff may be proposed for a failure. Always-on "
            "guidance (the conditions live in the prompt), not a trigger "
            "detector — the model applies judgment. Gates the contract section."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="VIBE_", case_sensitive=False, extra="ignore"
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    @property
    def vibe_code_api_key(self) -> str:
        return resolve_api_key(self.vibe_code_api_key_env_var) or ""

    @property
    def otel_span_exporter_config(self) -> OtelSpanExporterConfig | None:
        # Remote OTLP export is opt-in: configured only when the user points
        # otel_endpoint at their own collector / OTLP receiver. Auth is the
        # user's responsibility (via OTEL_EXPORTER_OTLP_* env vars), so no
        # headers are attached. Local JSONL export (otel_local_export) is
        # separate and on by default — traces are local-only out of the box.
        # Lazy: OTLP exporter pulls a heavy chain; _settings loads every startup.
        if not self.otel_endpoint:
            return None

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            DEFAULT_TRACES_EXPORT_PATH,
        )

        traces_export_path = DEFAULT_TRACES_EXPORT_PATH.lstrip("/")
        return OtelSpanExporterConfig(
            endpoint=urljoin(f"{self.otel_endpoint.rstrip('/')}/", traces_export_path)
        )

    @property
    def system_prompt(self) -> str:
        return load_system_prompt(self.system_prompt_id, extra_dirs=self.prompt_paths)

    @property
    def compaction_prompt(self) -> str:
        return load_prompt(
            self.compaction_prompt_id,
            setting_name="compaction_prompt_id",
            builtins={"compact": UtilityPrompt.COMPACT.path},
            extra_dirs=self.prompt_paths,
        )

    def get_active_model(self) -> ModelConfig:
        for model in self.models:
            if model.alias == self.active_model:
                return model
        raise ValueError(
            f"Active model '{self.active_model}' not found in configuration."
        )

    def get_compaction_model(self) -> ModelConfig:
        if self.compaction_model is not None:
            return self.compaction_model
        return self.get_active_model()

    def connectors_by_name(self) -> dict[str, ConnectorConfig]:
        return {c.name: c for c in self.connectors}

    def get_mistral_provider(self) -> ProviderConfig | None:
        try:
            active_provider = self.get_active_provider()
            if active_provider.backend == Backend.MISTRAL:
                return active_provider
        except ValueError:
            pass
        return next((p for p in self.providers if p.backend == Backend.MISTRAL), None)

    def get_provider_for_model(self, model: ModelConfig) -> ProviderConfig:
        for provider in self.providers:
            if provider.name == model.provider:
                return provider
        raise ValueError(
            f"Provider '{model.provider}' for model '{model.name}' not found in configuration."
        )

    def is_provider_available(self, provider: ProviderConfig) -> bool:
        if not provider.api_key_env_var:
            return True
        return bool(os.getenv(provider.api_key_env_var))

    def is_model_available(self, model: ModelConfig) -> bool:
        try:
            provider = self.get_provider_for_model(model)
        except ValueError:
            return False
        return self.is_provider_available(provider)

    @property
    def available_models(self) -> list[ModelConfig]:
        return [m for m in self.models if self.is_model_available(m)]

    def get_active_provider(self) -> ProviderConfig:
        return self.get_provider_for_model(self.get_active_model())

    def is_active_model_mistral(self) -> bool:
        try:
            return self.get_active_provider().backend == Backend.MISTRAL
        except ValueError:
            return False

    def get_active_transcribe_model(self) -> TranscribeModelConfig:
        for model in self.transcribe_models:
            if model.alias == self.active_transcribe_model:
                return model
        raise ValueError(
            f"Active transcribe model '{self.active_transcribe_model}' not found in configuration."
        )

    def get_transcribe_provider_for_model(
        self, model: TranscribeModelConfig
    ) -> TranscribeProviderConfig:
        for provider in self.transcribe_providers:
            if provider.name == model.provider:
                return provider
        raise ValueError(
            f"Transcribe provider '{model.provider}' for transcribe model '{model.name}' not found in configuration."
        )

    def get_active_tts_model(self) -> TTSModelConfig:
        for model in self.tts_models:
            if model.alias == self.active_tts_model:
                return model
        raise ValueError(
            f"Active TTS model '{self.active_tts_model}' not found in configuration."
        )

    def get_tts_provider_for_model(self, model: TTSModelConfig) -> TTSProviderConfig:
        for provider in self.tts_providers:
            if provider.name == model.provider:
                return provider
        raise ValueError(
            f"TTS provider '{model.provider}' for TTS model '{model.name}' not found in configuration."
        )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlFileSettingsSource(settings_cls),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _apply_global_auto_compact_threshold(self) -> VibeConfig:
        # Per model, derive the one runtime overflow number (auto_compact_threshold)
        # so every consumer reads a window-correct value with no signature change:
        #  - explicit threshold + declared window -> keep, clamped to <= 95% window
        #  - declared window, no explicit threshold -> 85% of the window
        #  - no declared window -> legacy global-threshold propagation (unchanged)
        bs = self.baseline_scaling
        derive = bs.derive_threshold_fraction
        cap = bs.safety_cap_fraction
        updated: list[ModelConfig] = []
        for model in self.models:
            explicit = "auto_compact_threshold" in model.model_fields_set
            window = model.context_window
            if window and window > 0:
                if explicit:
                    threshold = min(model.auto_compact_threshold, int(window * cap))
                else:
                    threshold = int(window * derive)
            elif explicit:
                threshold = model.auto_compact_threshold
            else:
                threshold = self.auto_compact_threshold
            if threshold != model.auto_compact_threshold:
                model = model.model_copy(update={"auto_compact_threshold": threshold})
            updated.append(model)
        self.models = updated
        return self

    @model_validator(mode="after")
    def _check_compaction_model_provider(self) -> VibeConfig:
        if self.compaction_model is None:
            return self

        compaction_provider = self.get_provider_for_model(self.compaction_model)
        try:
            active_provider = self.get_active_provider()
        except ValueError:
            return self
        if active_provider.name != compaction_provider.name:
            raise ValueError(
                f"Compaction model '{self.compaction_model.alias}' uses provider "
                f"'{compaction_provider.name}' but active model uses provider "
                f"'{active_provider.name}'. They must share the same provider."
            )
        return self

    @model_validator(mode="after")
    def _check_api_key(self) -> VibeConfig:
        if _skip_api_key_check.get():
            return self
        try:
            provider = self.get_active_provider()
            api_key_env = provider.api_key_env_var
            if api_key_env and not resolve_api_key(api_key_env):
                raise MissingAPIKeyError(api_key_env, provider.name)
        except ValueError:
            pass
        return self

    @field_validator("theme", mode="before")
    @classmethod
    def _validate_theme(cls, v: Any) -> str:
        if not isinstance(v, str) or not v:
            return DEFAULT_THEME
        if v == DEFAULT_THEME:
            return v
        from textual.theme import BUILTIN_THEMES

        if v not in BUILTIN_THEMES:
            logger.warning(
                "Unknown theme=%s in config; falling back to %s", v, DEFAULT_THEME
            )
            return DEFAULT_THEME
        return v

    @field_validator("tool_paths", mode="before")
    @classmethod
    def _expand_tool_paths(cls, v: Any) -> list[Path]:
        if not v:
            return []
        return [Path(p).expanduser().resolve() for p in v]

    @field_validator("skill_paths", mode="before")
    @classmethod
    def _expand_skill_paths(cls, v: Any) -> list[Path]:
        if not v:
            return []
        return [Path(p).expanduser().resolve() for p in v]

    @field_validator("workflow_paths", mode="before")
    @classmethod
    def _expand_workflow_paths(cls, v: Any) -> list[Path]:
        if not v:
            return []
        return [Path(p).expanduser().resolve() for p in v]

    @field_validator("prompt_paths", mode="before")
    @classmethod
    def _expand_prompt_paths(cls, v: Any) -> list[Path]:
        if not v:
            return []
        return [Path(p).expanduser().resolve() for p in v]

    @field_validator("plugin_paths", mode="before")
    @classmethod
    def _expand_plugin_paths(cls, v: Any) -> list[Path]:
        if not v:
            return []
        return [Path(p).expanduser().resolve() for p in v]

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tool_configs(cls, v: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(v, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for tool_name, tool_config in v.items():
            if isinstance(tool_config, dict):
                normalized[tool_name] = tool_config
            else:
                normalized[tool_name] = {}

        return normalized

    @model_validator(mode="after")
    def _validate_model_uniqueness(self) -> VibeConfig:
        seen_aliases: set[str] = set()
        for model in self.models:
            if model.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate model alias found: '{model.alias}'. Aliases must be unique."
                )
            seen_aliases.add(model.alias)
        return self

    @model_validator(mode="after")
    def _validate_transcribe_model_uniqueness(self) -> VibeConfig:
        seen_aliases: set[str] = set()
        for model in self.transcribe_models:
            if model.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate transcribe model alias found: '{model.alias}'. Aliases must be unique."
                )
            seen_aliases.add(model.alias)
        return self

    @model_validator(mode="after")
    def _validate_tts_model_uniqueness(self) -> VibeConfig:
        seen_aliases: set[str] = set()
        for model in self.tts_models:
            if model.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate TTS model alias found: '{model.alias}'. Aliases must be unique."
                )
            seen_aliases.add(model.alias)
        return self

    @model_validator(mode="after")
    def _check_system_prompt(self) -> VibeConfig:
        _ = self.system_prompt
        return self

    @model_validator(mode="after")
    def _check_compaction_prompt(self) -> VibeConfig:
        _ = self.compaction_prompt
        return self

    def set_thinking(self, level: ThinkingLevel) -> None:
        model = self.get_active_model()

        for i, m in enumerate(self.models):
            if m.alias == model.alias:
                self.models[i] = m.model_copy(update={"thinking": level})
                break

        current_config = TomlFileSettingsSource(type(self)).toml_data
        models = current_config.get("models", [])
        for entry in models:
            if entry.get("alias", entry.get("name")) == model.alias:
                entry["thinking"] = level
                break
        else:
            # Model comes from defaults; materialize the identities so we
            # don't lose the other models.
            models = [
                {
                    "name": m.name,
                    "provider": m.provider,
                    "alias": m.alias,
                    "thinking": level if m.alias == model.alias else m.thinking,
                    **({"supports_images": True} if m.supports_images else {}),
                }
                for m in self.models
            ]
        type(self).save_updates({"models": models})

    def is_le_chaton(self) -> bool:
        return self.effort_mode == "le-chaton"

    def set_effort_mode(self, mode: str) -> None:
        self.effort_mode = mode
        type(self).save_updates({"effort_mode": mode})
        if mode == "le-chaton":
            self.set_thinking("max")

    def add_tool_allowlist_patterns(self, tool_name: str, patterns: list[str]) -> None:
        if tool_name == "bash":
            patterns = [_strip_bash_pattern_wildcard(p) for p in patterns]
        current_allowlist: list[str] = list(
            self.tools.get(tool_name, {}).get("allowlist", [])
        )
        new_patterns = [p for p in patterns if p not in current_allowlist]
        if not new_patterns:
            return
        merged = sorted(current_allowlist + new_patterns)
        self.save_updates({"tools": {tool_name: {"allowlist": merged}}})
        if tool_name not in self.tools:
            self.tools[tool_name] = {}
        self.tools[tool_name]["allowlist"] = merged

    @classmethod
    def get_persisted_config(cls) -> dict[str, Any]:
        return TomlFileSettingsSource(cls).toml_data

    @classmethod
    def save_updates(cls, updates: dict[str, Any]) -> None:
        if not get_harness_files_manager().persist_allowed:
            return
        current_config = TomlFileSettingsSource(cls).toml_data
        merged_config = deep_update(current_config, updates)
        cls.dump_config(merged_config)

    @classmethod
    def dump_config(cls, config: dict[str, Any]) -> None:
        mgr = get_harness_files_manager()
        if not mgr.persist_allowed:
            return
        target = mgr.config_file or mgr.user_config_file
        target.parent.mkdir(parents=True, exist_ok=True)
        jsonable = to_jsonable_python(config, fallback=str)
        if not isinstance(jsonable, dict):
            toml_document = {}
        else:
            toml_document = _remove_none_values(jsonable)
        cls.model_validate(toml_document)
        with target.open("wb") as f:
            tomli_w.dump(toml_document, f)

    @classmethod
    def _migrate(cls) -> None:
        mgr = get_harness_files_manager()
        if not mgr.persist_allowed:
            return
        file = mgr.config_file
        if file is None:
            return
        try:
            with file.open("rb") as f:
                data = tomllib.load(f)
        except (FileNotFoundError, tomllib.TOMLDecodeError, OSError):
            return

        changed = False

        bash_tools = data.get("tools", {}).get("bash", {})
        allowlist = bash_tools.get("allowlist")
        if allowlist is not None and "find" not in allowlist:
            allowlist.append("find")
            allowlist.sort()
            changed = True

        if allowlist is not None and any(p.endswith(" *") for p in allowlist):
            stripped = [_strip_bash_pattern_wildcard(p) for p in allowlist]
            deduped = sorted(set(stripped))
            bash_tools["allowlist"] = deduped
            allowlist = deduped
            changed = True

        applied: list[str] = data.get("applied_migrations", [])
        if allowlist is not None and cls._BASH_READ_ONLY_MIGRATION not in applied:
            from vibe.core.tools.builtins.bash import default_read_only_commands

            bash_tools["allowlist"] = sorted(
                set(allowlist) | set(default_read_only_commands())
            )
            data["applied_migrations"] = [*applied, cls._BASH_READ_ONLY_MIGRATION]
            changed = True

        for model in data.get("models", []):
            if (
                model.get("name") == "mistral-vibe-cli-latest"
                and model.get("alias") == "devstral-2"
            ):
                model["alias"] = "mistral-medium-3.5"
                model["temperature"] = 1.0
                model["input_price"] = 1.5
                model["output_price"] = 7.5
                model["thinking"] = "high"
                changed = True

            if (
                model.get("name") == "mistral-vibe-cli-latest"
                and model.get("alias") == "mistral-medium-3.5"
                and "supports_images" not in model
            ):
                model["supports_images"] = True
                changed = True

        if data.get("active_model") == "devstral-2":
            data["active_model"] = "mistral-medium-3.5"
            changed = True

        if cls._migrate_renamed_tools(data):
            changed = True

        if cls._migrate_kimi_glm_reasoning(data):
            changed = True

        if changed:
            cls.dump_config(data)

    # GLM temperature left at the generic ModelConfig default by old presets.
    _GLM_LEGACY_TEMPERATURE: ClassVar[float] = 0.2

    @classmethod
    def _migrate_kimi_glm_reasoning(cls, data: dict[str, Any]) -> bool:
        # Backfill preserve_reasoning (Kimi/GLM Preserved Thinking) and lift GLM
        # off the legacy 0.2 temperature on pre-fix configs. One-shot.
        applied = data.get("applied_migrations", [])
        if cls._KIMI_GLM_REASONING_MIGRATION in applied:
            return False
        migrated = False
        for model in data.get("models", []):
            provider = model.get("provider")
            if provider in {"kimi", "zai"} and "preserve_reasoning" not in model:
                model["preserve_reasoning"] = True
                migrated = True
            if (
                provider == "zai"
                and model.get("temperature") == cls._GLM_LEGACY_TEMPERATURE
            ):
                model["temperature"] = 1.0
                migrated = True
        if not migrated:
            return False
        data["applied_migrations"] = [*applied, cls._KIMI_GLM_REASONING_MIGRATION]
        return True

    # One-shot id: syncs an existing bash allowlist up to the current default
    # read-only commands once, so users keep the ability to remove any of them.
    _BASH_READ_ONLY_MIGRATION: ClassVar[str] = "bash_read_only_defaults_v1"

    # One-shot id: backfills preserve_reasoning + GLM temperature on configs
    # written before the Kimi/GLM Preserved-Thinking fix.
    _KIMI_GLM_REASONING_MIGRATION: ClassVar[str] = "kimi_glm_preserve_reasoning_v1"

    # Old tool name -> new tool name. The new tools replaced these in-place, so
    # existing user configs keyed by the old names need their settings moved over.
    _RENAMED_TOOLS: ClassVar[dict[str, str]] = {
        "read_file": "read",
        "search_replace": "edit",
    }
    _DROPPED_TOOL_OPTIONS: ClassVar[dict[str, tuple[str, ...]]] = {
        "edit": ("max_content_size", "create_backup")
    }

    @classmethod
    def _migrate_renamed_tools(cls, data: dict[str, Any]) -> bool:
        changed = False

        tools = data.get("tools")
        if isinstance(tools, dict):
            for old, new in cls._RENAMED_TOOLS.items():
                if old not in tools:
                    continue
                old_config = tools.pop(old)
                changed = True
                # Prefer an already-present new key; don't clobber it.
                if new not in tools:
                    if isinstance(old_config, dict):
                        for dropped in cls._DROPPED_TOOL_OPTIONS.get(new, ()):
                            old_config.pop(dropped, None)
                    tools[new] = old_config

        for list_key in ("enabled_tools", "disabled_tools"):
            names = data.get(list_key)
            if not isinstance(names, list):
                continue
            renamed = [cls._RENAMED_TOOLS.get(name, name) for name in names]
            if renamed != names:
                data[list_key] = renamed
                changed = True

        return changed

    @classmethod
    def load(cls, **overrides: Any) -> VibeConfig:
        cls._migrate()
        config = cls(**(overrides or {}))
        configure_ssl_context(
            enable_system_trust_store=config.enable_system_trust_store
        )
        return config

    @classmethod
    def create_default(cls) -> dict[str, Any]:
        config = cls.model_construct()
        config_dict = config.model_dump(mode="json")

        from vibe.core.tools.manager import ToolManager

        tool_defaults = ToolManager.discover_tool_defaults()
        if tool_defaults:
            config_dict["tools"] = tool_defaults

        return config_dict
