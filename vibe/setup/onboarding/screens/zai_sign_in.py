from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Vertical
from textual.widgets import Input
from textual.worker import Worker

from vibe.cli.textual_ui.widgets.banner.petit_chat import PetitChat
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.logger import logger
from vibe.core.telemetry.types import EntrypointMetadata
from vibe.setup.auth.api_key_persistence import persist_api_key
from vibe.setup.auth.zai_callback import wait_for_zai_callback
from vibe.setup.auth.zai_protocol_handler import (
    ZaiProtocolHandlerInstallResult,
    install_zai_protocol_handler,
)
from vibe.setup.auth.zai_sign_in import ZaiSignInError, ZaiSignInService
from vibe.setup.onboarding.base import OnboardingScreen
from vibe.setup.onboarding.provider_presets import apply_provider_config

SignInServiceFactory = Callable[[], ZaiSignInService]
CopySignInUrl = Callable[[str], bool]
ProtocolHandlerInstaller = Callable[[], ZaiProtocolHandlerInstallResult]

SUCCESS_EXIT_DELAY_SECONDS: float = 2.0

_OPENING_MESSAGE = "Opening your browser to sign in to Z.ai..."
_PASTE_MESSAGE = (
    "Approve in Z.ai, then paste the zcode:// callback URL containing code= here:"
)
_SUCCESS_MESSAGE = "Signed in to Z.ai. Finishing setup..."
_UNEXPECTED_ERROR = "Something went wrong during Z.ai sign-in. Please try again."

PENDING_HINT = "Press C to copy the sign-in URL - Esc to cancel"
PASTE_HINT = "Press Enter to submit the pasted code - Esc to cancel"
ERROR_HINT = "Press R to retry - Esc to cancel"
SUCCESS_HINT = "Finishing setup..."
COPY_URL_SUCCESS_MESSAGE = "Sign-in URL copied to clipboard"


class ZaiSignInScreen(OnboardingScreen):
    """Drives the "Continue with Z.ai" account login during onboarding.

    Unlike the ChatGPT flow, the Z.ai login mints a durable coding-plan API key;
    on success we persist it like a manually-pasted ``ZAI_API_KEY``.

    Z.ai redirects to a ``zcode://`` custom scheme; setup accepts the hidden
    handler callback when registered and keeps manual paste as the fallback.
    """

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
        entrypoint_metadata: EntrypointMetadata | None = None,
        protocol_handler_installer: ProtocolHandlerInstaller | None = None,
        success_exit_delay: float = SUCCESS_EXIT_DELAY_SECONDS,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model = model
        self._service_factory = service_factory
        self._copy_sign_in_url = copy_sign_in_url
        self._protocol_handler_installer = (
            protocol_handler_installer or install_zai_protocol_handler
        )
        self._entrypoint_metadata = entrypoint_metadata
        self._success_exit_delay = success_exit_delay
        self._sign_in_url: str | None = None
        self._running = False
        self._variant = "pending"
        self._worker: Worker[None] | None = None
        self._paste_future: asyncio.Future[str] | None = None
        self._callback_task: asyncio.Task[str] | None = None
        self._status_widget: NoMarkupStatic
        self._url_widget: NoMarkupStatic
        self._hint_widget: NoMarkupStatic
        self._paste_input: Input

    def compose(self) -> ComposeResult:
        with Vertical(id="zai-sign-in-content", classes="onboarding-content"):
            with Center():
                with Vertical(id="zai-sign-in-panel", classes="onboarding-panel"):
                    yield PetitChat(id="zai-sign-in-chat", classes="onboarding-chat")
                    yield NoMarkupStatic(
                        "Continue with Z.ai",
                        id="zai-sign-in-title",
                        classes="onboarding-heading",
                    )
                    yield NoMarkupStatic(
                        "Sign in to your Z.ai account to set up GLM automatically.",
                        id="zai-sign-in-subtitle",
                    )
                    self._status_widget = NoMarkupStatic(
                        _OPENING_MESSAGE, id="zai-sign-in-status"
                    )
                    yield self._status_widget
                    self._url_widget = NoMarkupStatic("", id="zai-sign-in-url")
                    yield self._url_widget
                    self._paste_input = Input(
                        placeholder="Paste zcode://...?code=... or raw code, then Enter",
                        id="zai-sign-in-paste-input",
                        disabled=True,
                    )
                    yield self._paste_input
                    self._hint_widget = NoMarkupStatic(
                        PENDING_HINT,
                        id="zai-sign-in-hint",
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
        self._paste_future = None
        self._callback_task = None
        self._paste_input.value = ""
        self._paste_input.disabled = True
        self._set_status(_OPENING_MESSAGE)
        self._hint_widget.update(PENDING_HINT)
        self._worker = self.run_worker(
            self._sign_in(), group="zai-sign-in", exclusive=True
        )

    async def _sign_in(self) -> None:
        try:
            handler_result = self._protocol_handler_installer()
            if handler_result.status in {"failed", "existing_handler"}:
                logger.debug("Z.ai protocol handler not registered: %s", handler_result)
            service = self._service_factory()
            service.receive_code = self._await_pasted_code
            api_key = await service.authenticate(on_url=self._on_url)
        except asyncio.CancelledError:
            return
        except ZaiSignInError as err:
            self._show_error(str(err))
            return
        except Exception:
            logger.exception("Unexpected Z.ai sign-in failure")
            self._show_error(_UNEXPECTED_ERROR)
            return

        result = persist_api_key(
            self.provider, api_key, entrypoint_metadata=self._entrypoint_metadata
        )
        if result != "completed":
            self._show_error("Signed in, but could not save the API key.")
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
        self._paste_input.disabled = True
        self._hint_widget.update(SUCCESS_HINT)
        if self._success_exit_delay > 0:
            await asyncio.sleep(self._success_exit_delay)
        self.app.exit("completed")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self._paste_input:
            return
        value = event.value.strip()
        if not value or self._paste_future is None or self._paste_future.done():
            return
        self._paste_future.set_result(value)
        self._paste_input.disabled = True
        self._set_status("Exchanging the code with Z.ai...")

    def _on_url(self, url: str) -> None:
        self._sign_in_url = url
        self._url_widget.update(url)

    async def _await_pasted_code(self, authorize_url: str) -> str:
        self._variant = "paste"
        self._set_status(_PASTE_MESSAGE)
        self._hint_widget.update(PASTE_HINT)
        paste_future = asyncio.get_running_loop().create_future()
        callback_task = asyncio.create_task(wait_for_zai_callback(authorize_url))
        self._paste_future = paste_future
        self._callback_task = callback_task
        self._paste_input.disabled = False
        self._paste_input.value = ""
        self._paste_input.focus()
        try:
            done, _ = await asyncio.wait(
                {paste_future, callback_task}, return_when=asyncio.FIRST_COMPLETED
            )
            result = next(iter(done)).result()
        finally:
            self._cancel_paste_waiters()
        self._paste_input.disabled = True
        self._set_status("Exchanging the code with Z.ai...")
        return result

    def _set_status(self, message: str) -> None:
        self._status_widget.remove_class("error")
        if self._variant == "error":
            self._status_widget.add_class("error")
        self._status_widget.update(message)

    def _show_error(self, message: str) -> None:
        self._running = False
        self._variant = "error"
        self._worker = None
        self._paste_input.disabled = True
        self._cancel_paste_waiters()
        self._set_status(message)
        self._hint_widget.update(ERROR_HINT)

    def _cancel_worker(self) -> None:
        self._running = False
        self._cancel_paste_waiters()
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

    def _cancel_paste_waiters(self) -> None:
        if self._paste_future is not None and not self._paste_future.done():
            self._paste_future.cancel()
        self._paste_future = None
        if self._callback_task is not None and not self._callback_task.done():
            self._callback_task.cancel()
        self._callback_task = None
