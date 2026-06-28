from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.events import DescendantBlur
from textual.message import Message
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option
from textual.worker import Worker

from vibe.cli.clipboard import copy_text_to_clipboard
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput
from vibe.core.auth.openai_oauth import OpenAIOAuthError
from vibe.core.config import DEFAULT_PROVIDERS, ModelConfig, ProviderConfig, VibeConfig
from vibe.core.logger import logger
from vibe.core.types import Backend
from vibe.setup.auth import (
    BrowserSignInAttemptStarted,
    BrowserSignInError,
    BrowserSignInEvent,
    BrowserSignInService,
    BrowserSignInStatus,
    BrowserSignInStatusChanged,
    HttpBrowserSignInGateway,
)
from vibe.setup.auth.api_key_persistence import (
    persist_api_key,
    resolve_api_key_provider,
)
from vibe.setup.auth.openai_sign_in import OpenAISignInService
from vibe.setup.auth.zai_callback import wait_for_zai_callback
from vibe.setup.auth.zai_protocol_handler import (
    ZaiProtocolHandlerInstallResult,
    install_zai_protocol_handler,
)
from vibe.setup.auth.zai_sign_in import ZaiSignInError, ZaiSignInService
from vibe.setup.onboarding.provider_presets import (
    PRESETS,
    ProviderPreset,
    apply_provider_config,
)

if TYPE_CHECKING:
    from vibe.core.telemetry.types import EntrypointMetadata

_OPTION_PADDING = "  "
_HELP_SELECT = "Enter Select  Esc Close"
_HELP_METHOD = "Enter Select  Backspace Providers  Esc Close"
_HELP_INPUT = "Enter Save  Backspace Methods  Esc Close"
_HELP_BROWSER = "C Copy URL  S Show URL  R Retry  Esc Close"
_HELP_ZAI_PASTE = "Paste callback URL or code, then Enter  C Copy URL  Esc Close"

_PROVIDER_ALIASES = {
    "chatgpt": "openai-chatgpt",
    "glm": "zai",
    "glm-5.2": "zai",
    "z.ai": "zai",
    "zai": "zai",
    "zhipu": "zai",
}
_UNSUPPORTED_PRESETS = frozenset({"custom", "ollama"})


class _OptionId(StrEnum):
    PROVIDER = auto()
    BROWSER = auto()
    API_KEY = auto()
    COPY_URL = auto()
    SHOW_URL = auto()


@dataclass(frozen=True)
class _ProviderLoginTarget:
    key: str
    label: str
    provider: ProviderConfig
    model: ModelConfig | None
    help_url: str | None
    supports_browser: bool
    supports_api_key: bool
    source: str


@dataclass(frozen=True)
class _LoginResult:
    authenticated: bool
    provider_name: str
    error: str | None = None


BrowserSignInServiceFactory = Callable[[ProviderConfig], BrowserSignInService]
OpenAISignInServiceFactory = Callable[[], OpenAISignInService]
ZaiSignInServiceFactory = Callable[[], ZaiSignInService]
ZaiProtocolHandlerInstaller = Callable[[], ZaiProtocolHandlerInstallResult]


def normalize_login_provider_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    return _PROVIDER_ALIASES.get(normalized, normalized)


class ProviderLoginApp(Container):
    can_focus = True
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding("backspace", "back", "Back", show=False),
        Binding("c", "copy_url", "Copy URL", show=False),
        Binding("s", "show_url", "Show URL", show=False),
        Binding("r", "retry", "Retry", show=False),
    ]

    class ProviderLoginClosed(Message):
        def __init__(
            self,
            *,
            authenticated: bool = False,
            provider_name: str = "",
            error: str | None = None,
        ) -> None:
            super().__init__()
            self.authenticated = authenticated
            self.provider_name = provider_name
            self.error = error

    def __init__(
        self,
        config: VibeConfig,
        provider_name: str | None = None,
        *,
        browser_sign_in_service_factory: BrowserSignInServiceFactory | None = None,
        openai_sign_in_service_factory: OpenAISignInServiceFactory | None = None,
        zai_sign_in_service_factory: ZaiSignInServiceFactory | None = None,
        zai_protocol_handler_installer: ZaiProtocolHandlerInstaller | None = None,
        entrypoint_metadata: EntrypointMetadata | None = None,
    ) -> None:
        super().__init__(id="providerlogin-app")
        self._config = config
        self._initial_provider_name = (
            normalize_login_provider_name(provider_name) if provider_name else None
        )
        self._browser_sign_in_service_factory = (
            browser_sign_in_service_factory or self._build_browser_sign_in_service
        )
        self._openai_sign_in_service_factory = openai_sign_in_service_factory or (
            lambda: OpenAISignInService()
        )
        self._zai_sign_in_service_factory = zai_sign_in_service_factory or (
            lambda: ZaiSignInService()
        )
        self._zai_protocol_handler_installer = (
            zai_protocol_handler_installer or install_zai_protocol_handler
        )
        self._entrypoint_metadata = entrypoint_metadata
        self._targets = self._build_targets()
        self._target: _ProviderLoginTarget | None = None
        self._mode = "providers"
        self._auth_url: str | None = None
        self._auth_url_visible = False
        self._paste_future: asyncio.Future[str] | None = None
        self._zai_callback_task: asyncio.Task[str] | None = None
        self._worker: Worker[_LoginResult] | None = None
        self._title_widget: NoMarkupStatic
        self._options_widget: OptionList
        self._detail_widget: NoMarkupStatic
        self._input_widget: Input
        self._help_widget: NoMarkupStatic

    def compose(self) -> ComposeResult:
        with Vertical(id="providerlogin-content"):
            self._title_widget = NoMarkupStatic(
                "Provider Login", classes="settings-title"
            )
            yield self._title_widget
            self._options_widget = OptionList(id="providerlogin-options")
            yield self._options_widget
            self._detail_widget = NoMarkupStatic("", id="providerlogin-detail")
            yield self._detail_widget
            self._input_widget = VscodeCompatInput(
                password=True,
                placeholder="Paste API key",
                id="providerlogin-input",
                classes="providerlogin-input",
            )
            self._input_widget.display = False
            yield self._input_widget
            self._help_widget = NoMarkupStatic(
                "", id="providerlogin-help", classes="settings-help"
            )
            yield self._help_widget

    def on_mount(self) -> None:
        if self._initial_provider_name:
            if not self._select_provider(self._initial_provider_name):
                self._show_unknown_provider(self._initial_provider_name)
            return
        self._show_provider_options()

    def on_unmount(self) -> None:
        self._cancel_running_login()

    def on_descendant_blur(self, _event: DescendantBlur) -> None:
        if self._mode in {"api_key", "zai_paste"}:
            self._input_widget.focus()
            return
        self._options_widget.focus()

    def focus(self, scroll_visible: bool = True) -> ProviderLoginApp:
        if self._mode in {"api_key", "zai_paste"}:
            self._input_widget.focus(scroll_visible=scroll_visible)
            return self
        self._options_widget.focus(scroll_visible=scroll_visible)
        return self

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = str(event.option.id or "")
        if option_id.startswith(f"{_OptionId.PROVIDER}:"):
            self._select_provider(option_id.split(":", maxsplit=1)[1])
            return
        match option_id:
            case _OptionId.BROWSER:
                self._start_browser_login()
            case _OptionId.API_KEY:
                self._show_api_key_input()
            case _OptionId.COPY_URL:
                self.action_copy_url()
            case _OptionId.SHOW_URL:
                self.action_show_url()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self._input_widget:
            return
        value = event.value.strip()
        if not value:
            return
        if self._mode == "zai_paste":
            if self._paste_future is not None and not self._paste_future.done():
                self._paste_future.set_result(value)
                self._input_widget.disabled = True
                self._set_detail("Exchanging the code with Z.ai...")
            return
        if self._mode == "api_key":
            self._save_api_key(value)

    def action_close(self) -> None:
        self._cancel_running_login()
        self.post_message(self.ProviderLoginClosed())

    def action_back(self) -> None:
        self._cancel_running_login()
        self._input_widget.value = ""
        self._input_widget.display = False
        self._input_widget.disabled = False
        if self._mode == "providers":
            self.action_close()
            return
        if self._mode in {"api_key", "browser", "zai_paste"} and self._target:
            self._show_method_options(self._target)
            return
        self._show_provider_options()

    def action_retry(self) -> None:
        if self._mode != "browser" or self._worker is not None:
            return
        self._start_browser_login()

    def action_copy_url(self) -> None:
        if self._auth_url is None:
            return
        copy_text_to_clipboard(
            self.app, self._auth_url, success_message="Sign-in URL copied to clipboard"
        )

    def action_show_url(self) -> None:
        if self._auth_url is None:
            return
        self._auth_url_visible = not self._auth_url_visible
        self._update_browser_detail()

    def _build_targets(self) -> list[_ProviderLoginTarget]:
        targets: list[_ProviderLoginTarget] = []
        seen: set[str] = set()
        for preset in PRESETS:
            if preset.key in _UNSUPPORTED_PRESETS:
                continue
            target = self._target_from_preset(preset)
            if target is None or target.key in seen:
                continue
            targets.append(target)
            seen.add(target.key)

        for provider in self._config.providers:
            if provider.name in seen or not provider.api_key_env_var:
                continue
            model = next(
                (
                    model
                    for model in self._config.models
                    if model.provider == provider.name
                ),
                None,
            )
            targets.append(
                _ProviderLoginTarget(
                    key=provider.name,
                    label=provider.name,
                    provider=provider,
                    model=model,
                    help_url=None,
                    supports_browser=False,
                    supports_api_key=True,
                    source="config",
                )
            )
            seen.add(provider.name)
        return targets

    def _target_from_preset(
        self, preset: ProviderPreset
    ) -> _ProviderLoginTarget | None:
        if preset.key == "mistral":
            provider = self._mistral_provider()
            return _ProviderLoginTarget(
                key=preset.key,
                label=preset.label,
                provider=provider,
                model=None,
                help_url=None,
                supports_browser=provider.supports_browser_sign_in,
                supports_api_key=bool(provider.api_key_env_var),
                source="preset",
            )
        if preset.provider is None:
            return None
        return _ProviderLoginTarget(
            key=preset.key,
            label=preset.label,
            provider=preset.provider,
            model=preset.model,
            help_url=preset.help_url,
            supports_browser=preset.key in {"openai-chatgpt", "zai"},
            supports_api_key=preset.requires_api_key,
            source="preset",
        )

    def _mistral_provider(self) -> ProviderConfig:
        for provider in self._config.providers:
            if provider.name == "mistral" or provider.backend == Backend.MISTRAL:
                return provider
        return next(
            provider for provider in DEFAULT_PROVIDERS if provider.name == "mistral"
        )

    def _show_provider_options(self) -> None:
        self._mode = "providers"
        self._target = None
        self._title_widget.update("Provider Login")
        self._input_widget.display = False
        self._options_widget.display = True
        self._options_widget.clear_options()
        self._detail_widget.update("")
        for target in self._targets:
            self._options_widget.add_option(
                Option(
                    Text(f"{_OPTION_PADDING}{target.label}", no_wrap=True),
                    id=f"{_OptionId.PROVIDER}:{target.key}",
                )
            )
        self._help_widget.update(_HELP_SELECT)
        self._options_widget.focus()

    def _show_unknown_provider(self, provider_name: str) -> None:
        self._mode = "providers"
        known = ", ".join(target.key for target in self._targets)
        self._title_widget.update("Provider Login")
        self._options_widget.display = True
        self._options_widget.clear_options()
        self._options_widget.add_option(
            Option(f"Unknown provider: {provider_name}", disabled=True)
        )
        self._detail_widget.update(f"Known providers: {known}")
        self._help_widget.update(_HELP_SELECT)
        self._options_widget.focus()

    def _select_provider(self, provider_name: str) -> bool:
        key = normalize_login_provider_name(provider_name)
        target = next((target for target in self._targets if target.key == key), None)
        if target is None:
            target = next(
                (target for target in self._targets if target.provider.name == key),
                None,
            )
        if target is None:
            return False
        self._target = target
        if target.supports_browser and target.supports_api_key:
            self._show_method_options(target)
        elif target.supports_browser:
            self._start_browser_login()
        elif target.supports_api_key:
            self._show_api_key_input()
        else:
            self._show_error("This provider does not have a login flow.")
        return True

    def _show_method_options(self, target: _ProviderLoginTarget) -> None:
        self._mode = "method"
        self._title_widget.update(f"Login: {target.label}")
        self._input_widget.display = False
        self._options_widget.display = True
        self._options_widget.clear_options()
        self._options_widget.add_option(
            Option(
                Text(f"{_OPTION_PADDING}Continue in browser", no_wrap=True),
                id=_OptionId.BROWSER,
            )
        )
        self._options_widget.add_option(
            Option(
                Text(f"{_OPTION_PADDING}Use an API key", no_wrap=True),
                id=_OptionId.API_KEY,
            )
        )
        self._detail_widget.update("")
        self._help_widget.update(_HELP_METHOD)
        self._options_widget.focus()

    def _show_api_key_input(self) -> None:
        target = self._require_target()
        self._mode = "api_key"
        self._title_widget.update(f"API Key: {target.label}")
        self._options_widget.display = False
        self._input_widget.display = True
        self._input_widget.disabled = False
        self._input_widget.password = True
        self._input_widget.placeholder = f"Paste {target.provider.api_key_env_var}"
        self._input_widget.value = ""
        details = [f"Paste a key for {target.provider.api_key_env_var}."]
        if target.help_url:
            details.append(target.help_url)
        self._detail_widget.update("\n".join(details))
        self._help_widget.update(_HELP_INPUT)
        self._input_widget.focus()

    def _save_api_key(self, api_key: str) -> None:
        target = self._require_target()
        result = persist_api_key(
            resolve_api_key_provider(target.provider),
            api_key,
            entrypoint_metadata=self._entrypoint_metadata,
        )
        if result != "completed":
            self._show_error(f"Could not save API key: {result}")
            return
        try:
            self._apply_provider_config_if_available(target)
        except (OSError, ValueError) as exc:
            self._show_error(f"Saved the key, but could not save the provider: {exc}")
            return
        self.post_message(
            self.ProviderLoginClosed(
                authenticated=True, provider_name=target.provider.name
            )
        )

    def _start_browser_login(self) -> None:
        target = self._require_target()
        self._cancel_running_login()
        self._mode = "browser"
        self._auth_url = None
        self._auth_url_visible = False
        self._title_widget.update(f"Login: {target.label}")
        self._input_widget.display = False
        self._options_widget.display = True
        self._options_widget.clear_options()
        self._options_widget.add_option(
            Option("Preparing browser sign-in...", disabled=True)
        )
        self._detail_widget.update("")
        self._help_widget.update(_HELP_BROWSER)
        self._worker = self.run_worker(
            self._run_browser_login(target), exclusive=True, group="provider_login"
        )
        self._options_widget.focus()

    async def _run_browser_login(self, target: _ProviderLoginTarget) -> _LoginResult:
        try:
            match target.key:
                case "zai":
                    return await self._run_zai_login(target)
                case "openai-chatgpt":
                    return await self._run_chatgpt_login(target)
                case "mistral":
                    return await self._run_browser_api_key_login(target)
                case _:
                    return _LoginResult(
                        authenticated=False,
                        provider_name=target.provider.name,
                        error="This provider only supports API-key login.",
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Provider login failed for provider=%s", target.provider.name
            )
            return _LoginResult(
                authenticated=False,
                provider_name=target.provider.name,
                error=str(exc) or "Provider login failed.",
            )

    async def _run_zai_login(self, target: _ProviderLoginTarget) -> _LoginResult:
        handler_result = self._zai_protocol_handler_installer()
        if handler_result.status in {"failed", "existing_handler"}:
            logger.debug("Z.ai protocol handler not registered: %s", handler_result)
        service = self._zai_sign_in_service_factory()
        service.receive_code = self._await_zai_code
        try:
            api_key = await service.authenticate(on_url=self._on_auth_url)
        except ZaiSignInError as exc:
            return _LoginResult(False, target.provider.name, str(exc))
        result = persist_api_key(
            target.provider, api_key, entrypoint_metadata=self._entrypoint_metadata
        )
        if result != "completed":
            return _LoginResult(
                False,
                target.provider.name,
                "Signed in, but could not save the API key.",
            )
        self._apply_provider_config_if_available(target)
        return _LoginResult(True, target.provider.name)

    async def _run_chatgpt_login(self, target: _ProviderLoginTarget) -> _LoginResult:
        service = self._openai_sign_in_service_factory()
        try:
            await service.authenticate(on_url=self._on_auth_url)
        except OpenAIOAuthError as exc:
            return _LoginResult(False, target.provider.name, str(exc))
        self._apply_provider_config_if_available(target)
        return _LoginResult(True, target.provider.name)

    async def _run_browser_api_key_login(
        self, target: _ProviderLoginTarget
    ) -> _LoginResult:
        service = self._browser_sign_in_service_factory(target.provider)
        try:
            api_key = await service.authenticate(
                lambda event: self._on_browser_sign_in_event(event)
            )
        except BrowserSignInError as exc:
            return _LoginResult(False, target.provider.name, str(exc))
        finally:
            await service.aclose()
        result = persist_api_key(
            resolve_api_key_provider(target.provider),
            api_key,
            entrypoint_metadata=self._entrypoint_metadata,
        )
        if result != "completed":
            return _LoginResult(
                False,
                target.provider.name,
                "Signed in, but could not save the API key.",
            )
        self._apply_provider_config_if_available(target)
        return _LoginResult(True, target.provider.name)

    async def _await_zai_code(self, authorize_url: str) -> str:
        self._mode = "zai_paste"
        self._input_widget.display = True
        self._input_widget.disabled = False
        self._input_widget.password = False
        self._input_widget.placeholder = "Paste zcode:// callback URL or raw code"
        self._input_widget.value = ""
        self._options_widget.clear_options()
        self._options_widget.add_option(
            Option("Waiting for Z.ai callback...", disabled=True)
        )
        self._help_widget.update(_HELP_ZAI_PASTE)
        self._input_widget.focus()
        paste_future = asyncio.get_running_loop().create_future()
        callback_task = asyncio.create_task(wait_for_zai_callback(authorize_url))
        self._paste_future = paste_future
        self._zai_callback_task = callback_task
        try:
            done, _ = await asyncio.wait(
                {paste_future, callback_task}, return_when=asyncio.FIRST_COMPLETED
            )
            return next(iter(done)).result()
        finally:
            self._cancel_zai_waiters()

    def _on_auth_url(self, url: str) -> None:
        self._auth_url = url
        self._update_browser_detail()
        self._options_widget.clear_options()
        self._options_widget.add_option(
            Option("Waiting for browser sign-in...", disabled=True)
        )
        self._options_widget.add_option(
            Option(
                Text(f"{_OPTION_PADDING}Copy sign-in URL", no_wrap=True),
                id=_OptionId.COPY_URL,
            )
        )
        self._options_widget.add_option(
            Option(
                Text(f"{_OPTION_PADDING}Show sign-in URL", no_wrap=True),
                id=_OptionId.SHOW_URL,
            )
        )
        self._help_widget.update(_HELP_BROWSER)

    def _on_browser_sign_in_event(self, event: BrowserSignInEvent) -> None:
        if isinstance(event, BrowserSignInAttemptStarted):
            self._on_auth_url(event.sign_in_url)
            return
        if not isinstance(event, BrowserSignInStatusChanged):
            return
        match event.status:
            case BrowserSignInStatus.OPENING_BROWSER:
                self._set_detail("Opening your browser...")
            case BrowserSignInStatus.WAITING_FOR_BROWSER_SIGN_IN:
                self._set_detail("Waiting for browser sign-in...")
            case BrowserSignInStatus.EXCHANGING:
                self._set_detail("Exchanging sign-in code...")
            case BrowserSignInStatus.COMPLETED:
                self._set_detail("Sign-in complete.")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.group != "provider_login" or not event.worker.is_finished:
            return
        self._worker = None
        result = event.worker.result
        if not isinstance(result, _LoginResult):
            self._show_error("Provider login failed.")
            return
        if result.authenticated:
            self.post_message(
                self.ProviderLoginClosed(
                    authenticated=True, provider_name=result.provider_name
                )
            )
            return
        self._show_error(result.error or "Provider login failed.")

    def _show_error(self, message: str) -> None:
        self._cancel_zai_waiters()
        self._mode = (
            "browser" if self._target and self._target.supports_browser else "providers"
        )
        self._input_widget.display = False
        self._options_widget.display = True
        self._options_widget.clear_options()
        if self._target and self._target.supports_browser:
            self._options_widget.add_option(
                Option(
                    Text(f"{_OPTION_PADDING}Retry browser sign-in", no_wrap=True),
                    id=_OptionId.BROWSER,
                )
            )
        if self._target and self._target.supports_api_key:
            self._options_widget.add_option(
                Option(
                    Text(f"{_OPTION_PADDING}Use an API key", no_wrap=True),
                    id=_OptionId.API_KEY,
                )
            )
        self._detail_widget.update(message)
        self._help_widget.update(_HELP_BROWSER)
        self._options_widget.focus()

    def _set_detail(self, message: str) -> None:
        self._detail_widget.update(message)

    def _update_browser_detail(self) -> None:
        parts = ["Complete sign-in in your browser, then return to Chaton."]
        if self._auth_url_visible and self._auth_url:
            parts.extend(["", self._auth_url])
        self._detail_widget.update("\n".join(parts))

    def _apply_provider_config_if_available(self, target: _ProviderLoginTarget) -> None:
        if target.model is not None:
            apply_provider_config(target.provider, target.model)

    def _build_browser_sign_in_service(
        self, provider: ProviderConfig
    ) -> BrowserSignInService:
        browser_base_url = provider.browser_auth_base_url
        api_base_url = provider.browser_auth_api_base_url
        if not browser_base_url or not api_base_url:
            raise ValueError("This provider does not define browser sign-in URLs.")
        return BrowserSignInService(
            HttpBrowserSignInGateway(
                browser_base_url=browser_base_url, api_base_url=api_base_url
            )
        )

    def _require_target(self) -> _ProviderLoginTarget:
        if self._target is None:
            raise AssertionError("Provider login target is not selected.")
        return self._target

    def _cancel_running_login(self) -> None:
        self._cancel_zai_waiters()
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

    def _cancel_zai_waiters(self) -> None:
        if self._paste_future is not None and not self._paste_future.done():
            self._paste_future.cancel()
        self._paste_future = None
        if self._zai_callback_task is not None and not self._zai_callback_task.done():
            self._zai_callback_task.cancel()
        self._zai_callback_task = None

    def on_blur(self, _event: events.Blur) -> None:
        self.call_after_refresh(self.focus)
