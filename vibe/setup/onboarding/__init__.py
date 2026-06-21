from __future__ import annotations

from collections.abc import Callable
import sys
from typing import Any

from rich import print as rprint
from textual.app import App
from textual.screen import Screen

from vibe.cli.clipboard import try_copy_text_to_clipboard
from vibe.core.config import DEFAULT_PROVIDERS, ModelConfig, ProviderConfig, VibeConfig
from vibe.core.paths import GLOBAL_ENV_FILE
from vibe.core.telemetry.types import EntrypointMetadata
from vibe.core.types import Backend
from vibe.setup.auth import BrowserSignInService, HttpBrowserSignInGateway
from vibe.setup.onboarding.context import OnboardingContext
from vibe.setup.onboarding.screens import (
    ApiKeyScreen,
    AuthMethodScreen,
    BrowserSignInScreen,
    CustomProviderScreen,
    ProviderSelectionScreen,
    ThemeSelectionScreen,
    WebSearchScreen,
    WelcomeScreen,
)
from vibe.setup.onboarding.screens.browser_sign_in import (
    SIGN_IN_URL_HELP_DELAY_SECONDS,
    SUCCESS_EXIT_DELAY_SECONDS,
    CopySignInUrl,
)


class OnboardingApp(App[str | None]):
    CSS_PATH = "onboarding.tcss"

    def __init__(
        self,
        config: OnboardingContext | VibeConfig | None = None,
        browser_sign_in_service_factory: Callable[[], BrowserSignInService]
        | None = None,
        entrypoint_metadata: EntrypointMetadata | None = None,
        browser_sign_in_success_delay: float = SUCCESS_EXIT_DELAY_SECONDS,
        browser_sign_in_url_help_delay: float = SIGN_IN_URL_HELP_DELAY_SECONDS,
        copy_sign_in_url: CopySignInUrl | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            config = OnboardingContext.load()
        elif isinstance(config, VibeConfig):
            config = OnboardingContext.from_config(config)

        self._config = config
        self._provider = config.provider
        self._vibe_base_url = config.vibe_base_url
        self._entrypoint_metadata = entrypoint_metadata
        self._browser_sign_in_success_delay = browser_sign_in_success_delay
        self._browser_sign_in_url_help_delay = browser_sign_in_url_help_delay
        self._copy_sign_in_url = copy_sign_in_url or self._copy_sign_in_url_to_clipboard
        self._browser_sign_in_service_factory = self._resolve_browser_sign_in_factory(
            browser_sign_in_service_factory
        )
        self._installed_dynamic_screens: set[str] = set()

    def on_mount(self) -> None:
        self.theme = "ansi-dark"

        self.install_screen(WelcomeScreen(next_screen="theme_selection"), "welcome")
        self.install_screen(
            ThemeSelectionScreen(next_screen="web_search"), "theme_selection"
        )
        self.install_screen(
            WebSearchScreen(next_screen="provider_selection"), "web_search"
        )
        self.install_screen(
            ProviderSelectionScreen(resolved_provider_name=self._provider.name),
            "provider_selection",
        )
        self.install_screen(CustomProviderScreen(), "custom_provider")
        self.push_screen("welcome")

    def _install_screen_once(self, name: str, screen: Screen[str | None]) -> None:
        if name in self._installed_dynamic_screens:
            self.uninstall_screen(name)
            self._installed_dynamic_screens.discard(name)
        self.install_screen(screen, name)
        self._installed_dynamic_screens.add(name)

    def install_api_key_screen(
        self,
        provider: ProviderConfig,
        *,
        help_url: str | None = None,
        pending_model: ModelConfig | None = None,
    ) -> None:
        self._provider = provider
        self._install_screen_once(
            "api_key",
            ApiKeyScreen(
                provider,
                vibe_base_url=self._vibe_base_url,
                entrypoint_metadata=self._entrypoint_metadata,
                help_url=help_url,
                pending_model=pending_model,
            ),
        )

    def _mistral_provider(self) -> ProviderConfig:
        if (
            self._provider.name == "mistral"
            or self._provider.backend == Backend.MISTRAL
        ):
            return self._provider
        return DEFAULT_PROVIDERS[0]

    def install_mistral_screens(self) -> None:
        provider = self._mistral_provider()
        self._provider = provider
        if (
            self._browser_sign_in_service_factory is None
            and provider.supports_browser_sign_in
        ):
            self._browser_sign_in_service_factory = (
                self._build_browser_sign_in_service_factory()
            )
        self.install_api_key_screen(provider)
        if self._browser_sign_in_service_factory is not None:
            self._install_screen_once("auth_method", AuthMethodScreen(provider))
            self._install_screen_once(
                "browser_sign_in",
                BrowserSignInScreen(
                    provider,
                    self._browser_sign_in_service_factory,
                    copy_sign_in_url=self._copy_sign_in_url,
                    entrypoint_metadata=self._entrypoint_metadata,
                    success_exit_delay=self._browser_sign_in_success_delay,
                    sign_in_url_help_delay=self._browser_sign_in_url_help_delay,
                ),
            )

    @property
    def supports_browser_sign_in(self) -> bool:
        return self._browser_sign_in_service_factory is not None

    def _build_browser_sign_in_service_factory(
        self,
    ) -> Callable[[], BrowserSignInService]:
        browser_base_url = self._provider.browser_auth_base_url
        api_base_url = self._provider.browser_auth_api_base_url
        if not browser_base_url or not api_base_url:
            msg = "Browser sign-in requires both browser auth URLs."
            raise AssertionError(msg)

        return lambda: BrowserSignInService(
            HttpBrowserSignInGateway(
                browser_base_url=browser_base_url, api_base_url=api_base_url
            )
        )

    def _resolve_browser_sign_in_factory(
        self, browser_sign_in_service_factory: Callable[[], BrowserSignInService] | None
    ) -> Callable[[], BrowserSignInService] | None:
        if not self._config.supports_browser_sign_in:
            return None

        return (
            browser_sign_in_service_factory
            or self._build_browser_sign_in_service_factory()
        )

    def _copy_sign_in_url_to_clipboard(self, text: str) -> bool:
        return try_copy_text_to_clipboard(text)


def run_onboarding(
    app: App | None = None, *, entrypoint_metadata: EntrypointMetadata | None = None
) -> None:
    result = (app or OnboardingApp(entrypoint_metadata=entrypoint_metadata)).run()
    match result:
        case None:
            rprint("\n[yellow]Setup cancelled. See you next time![/]")
            sys.exit(0)
        case str() as s if s.startswith("env_var_error:"):
            env_key = s.removeprefix("env_var_error:")
            rprint(
                "\n[yellow]Could not save the API key because this provider is "
                f"configured with an invalid environment variable name: {env_key}.[/]"
                "\n[dim]The API key was not saved for this session. "
                "Update the provider's `api_key_env_var` setting in your config and try again.[/]\n"
            )
            sys.exit(1)
        case str() as s if s.startswith("save_error:"):
            err = s.removeprefix("save_error:")
            rprint(
                f"\n[yellow]Warning: Could not save API key to .env file: {err}[/]"
                "\n[dim]The API key is set for this session only. "
                f"You may need to set it manually in {GLOBAL_ENV_FILE.path}[/]\n"
            )
        case "completed":
            rprint('\nSetup complete 🎉. Run "chaton" to start using the Chaton CLI.\n')
