from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Vertical
from textual.worker import Worker

from vibe.cli.textual_ui.widgets.banner.petit_chat import PetitChat
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.auth.openai_oauth import OpenAIOAuthError
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.logger import logger
from vibe.setup.auth.openai_sign_in import OpenAISignInService
from vibe.setup.onboarding.base import OnboardingScreen
from vibe.setup.onboarding.provider_presets import apply_provider_config

SignInServiceFactory = Callable[[], OpenAISignInService]
CopySignInUrl = Callable[[str], bool]

SUCCESS_EXIT_DELAY_SECONDS: float = 2.0

_OPENING_MESSAGE = "Opening your browser to sign in to ChatGPT..."
_WAITING_MESSAGE = "Waiting for you to authorize in the browser..."
_SUCCESS_MESSAGE = "Signed in to ChatGPT. Finishing setup..."
_UNEXPECTED_ERROR = "Something went wrong during ChatGPT sign-in. Please try again."

PENDING_HINT = "Press C to copy the sign-in URL - Esc to cancel"
ERROR_HINT = "Press R to retry - Esc to cancel"
SUCCESS_HINT = "Finishing setup..."
COPY_URL_SUCCESS_MESSAGE = "Sign-in URL copied to clipboard"


class ChatGPTSignInScreen(OnboardingScreen):
    """Drives the loopback PKCE "Sign in with ChatGPT" flow during onboarding."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("r", "retry", "Retry", show=False),
        Binding("c", "copy_url", "Copy URL", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        service_factory: SignInServiceFactory,
        *,
        copy_sign_in_url: CopySignInUrl,
        success_exit_delay: float = SUCCESS_EXIT_DELAY_SECONDS,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model = model
        self._service_factory = service_factory
        self._copy_sign_in_url = copy_sign_in_url
        self._success_exit_delay = success_exit_delay
        self._sign_in_url: str | None = None
        self._running = False
        self._variant = "pending"
        self._worker: Worker[None] | None = None
        self._status_widget: NoMarkupStatic
        self._url_widget: NoMarkupStatic
        self._hint_widget: NoMarkupStatic

    def compose(self) -> ComposeResult:
        with Vertical(id="chatgpt-sign-in-content", classes="onboarding-content"):
            with Center():
                with Vertical(id="chatgpt-sign-in-panel", classes="onboarding-panel"):
                    yield PetitChat(
                        id="chatgpt-sign-in-chat", classes="onboarding-chat"
                    )
                    yield NoMarkupStatic(
                        "Sign in with ChatGPT",
                        id="chatgpt-sign-in-title",
                        classes="onboarding-heading",
                    )
                    yield NoMarkupStatic(
                        "Use your ChatGPT Plus/Pro subscription (no API key).",
                        id="chatgpt-sign-in-subtitle",
                    )
                    self._status_widget = NoMarkupStatic(
                        _OPENING_MESSAGE, id="chatgpt-sign-in-status"
                    )
                    yield self._status_widget
                    self._url_widget = NoMarkupStatic("", id="chatgpt-sign-in-url")
                    yield self._url_widget
                    self._hint_widget = NoMarkupStatic(
                        PENDING_HINT,
                        id="chatgpt-sign-in-hint",
                        classes="onboarding-hint-row",
                    )
                    yield self._hint_widget

    def on_mount(self) -> None:
        self.call_after_refresh(self._start)

    def on_unmount(self) -> None:
        self._cancel_worker()

    def action_retry(self) -> None:
        if not self._running:
            self._start()

    def action_copy_url(self) -> None:
        if self._variant == "success" or self._sign_in_url is None:
            return
        if self._copy_sign_in_url(self._sign_in_url):
            self.app.notify(
                COPY_URL_SUCCESS_MESSAGE,
                severity="information",
                timeout=2,
                markup=False,
            )
            return
        self._url_widget.update(
            f"Copy failed. Open this URL manually:\n{self._sign_in_url}"
        )

    def action_cancel(self) -> None:
        if self._variant == "success":
            return
        self._cancel_worker()
        super().action_cancel()

    def _start(self) -> None:
        self._running = True
        self._variant = "pending"
        self._set_status(_OPENING_MESSAGE)
        self._hint_widget.update(PENDING_HINT)
        self._worker = self.run_worker(
            self._sign_in(), group="chatgpt-sign-in", exclusive=True
        )

    async def _sign_in(self) -> None:
        try:
            service = self._service_factory()
            await service.authenticate(on_url=self._on_url)
        except asyncio.CancelledError:
            return
        except OpenAIOAuthError as err:
            self._show_error(str(err))
            return
        except Exception:
            logger.exception("Unexpected ChatGPT sign-in failure")
            self._show_error(_UNEXPECTED_ERROR)
            return

        try:
            apply_provider_config(self.provider, self.model)
        except (OSError, ValueError) as err:
            self._show_error(f"Signed in, but could not save the provider: {err}")
            return

        self._variant = "success"
        self._running = False
        self._set_status(_SUCCESS_MESSAGE)
        self._url_widget.update("")
        self._hint_widget.update(SUCCESS_HINT)
        if self._success_exit_delay > 0:
            await asyncio.sleep(self._success_exit_delay)
        self.app.exit("completed")

    def _on_url(self, url: str) -> None:
        self._sign_in_url = url
        self._set_status(_WAITING_MESSAGE)
        self._url_widget.update(url)

    def _set_status(self, message: str) -> None:
        self._status_widget.remove_class("error")
        if self._variant == "error":
            self._status_widget.add_class("error")
        self._status_widget.update(message)

    def _show_error(self, message: str) -> None:
        self._running = False
        self._variant = "error"
        self._worker = None
        self._set_status(message)
        self._hint_widget.update(ERROR_HINT)

    def _cancel_worker(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
