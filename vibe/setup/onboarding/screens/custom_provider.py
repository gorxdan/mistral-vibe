from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, cast

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Vertical
from textual.validation import Function, Length
from textual.widgets import Input

from vibe.cli.textual_ui.widgets.banner.petit_chat import PetitChat
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.setup.onboarding.base import OnboardingScreen

if TYPE_CHECKING:
    from vibe.setup.onboarding import OnboardingApp

CUSTOM_PROVIDER_NAME = "custom"
URL_PREFIXES = ("http://", "https://")
ENV_VAR_PATTERN = r"^[A-Z][A-Z0-9_]*$"


def _is_http_url(value: str) -> bool:
    return value.startswith(URL_PREFIXES) and "." in value


def _is_env_var_name(value: str) -> bool:
    return re.match(ENV_VAR_PATTERN, value) is not None


class CustomProviderScreen(OnboardingScreen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "cancel", "Cancel", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    NEXT_SCREEN = None

    def __init__(self) -> None:
        super().__init__()
        self._base_url_input: Input
        self._model_input: Input
        self._env_var_input: Input
        self._feedback: NoMarkupStatic

    def compose(self) -> ComposeResult:
        self._base_url_input = Input(
            placeholder="https://api.example.com/v1",
            id="custom-base-url",
            validators=[
                Function(
                    _is_http_url,
                    failure_description="Enter a http(s) URL ending with the API version, e.g. https://api.example.com/v1",
                )
            ],
        )
        self._model_input = Input(
            placeholder="model-name",
            id="custom-model",
            validators=[Length(minimum=1, failure_description="Enter a model name.")],
        )
        self._env_var_input = Input(
            placeholder="PROVIDER_API_KEY",
            id="custom-env-var",
            validators=[
                Function(
                    _is_env_var_name,
                    failure_description=(
                        "Use an uppercase env var name (letters, digits, underscore),"
                        " e.g. PROVIDER_API_KEY"
                    ),
                )
            ],
        )
        with Vertical(id="custom-provider-content", classes="onboarding-content"):
            with Center():
                with Vertical(id="custom-provider-panel", classes="onboarding-panel"):
                    yield PetitChat(
                        id="custom-provider-chat", classes="onboarding-chat"
                    )
                    yield NoMarkupStatic(
                        "Custom OpenAI-compatible provider",
                        id="custom-provider-title",
                        classes="onboarding-heading",
                    )
                    yield NoMarkupStatic(
                        "Point Mistral Vibe at any OpenAI-compatible /chat/completions"
                        " endpoint.",
                        id="custom-provider-subtitle",
                    )
                    yield self._labeled_input("Base URL", self._base_url_input)
                    yield self._labeled_input("Model name", self._model_input)
                    yield self._labeled_input(
                        "API key environment variable", self._env_var_input
                    )
                    self._feedback = NoMarkupStatic("", id="custom-provider-feedback")
                    yield self._feedback

    def _labeled_input(self, label: str, widget: Input) -> Vertical:
        box = Vertical(widget, classes="onboarding-card custom-input-box")
        box.border_title = label
        return box

    def on_mount(self) -> None:
        self._base_url_input.focus()

    def _input_is_valid(self, inp: Input) -> bool:
        result = inp.validate(inp.value)
        return result is not None and result.is_valid

    def _all_valid(self) -> bool:
        return all(
            self._input_is_valid(inp)
            for inp in (self._base_url_input, self._model_input, self._env_var_input)
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input not in {
            self._base_url_input,
            self._model_input,
            self._env_var_input,
        }:
            return
        if self._all_valid():
            self._feedback.update("Press Enter to continue \u21a9")
        else:
            self._feedback.update("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        for inp in (self._base_url_input, self._model_input, self._env_var_input):
            if not self._input_is_valid(inp):
                inp.focus()
                return
        self._proceed()

    def _proceed(self) -> None:
        base_url = self._base_url_input.value.strip().rstrip("/")
        model_name = self._model_input.value.strip()
        env_var = self._env_var_input.value.strip()

        provider_name = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
        if not provider_name:
            provider_name = CUSTOM_PROVIDER_NAME

        provider = ProviderConfig(
            name=provider_name, api_base=base_url, api_key_env_var=env_var
        )
        model = ModelConfig(name=model_name, provider=provider_name, alias=model_name)
        cast("OnboardingApp", self.app).install_api_key_screen(
            provider, pending_model=model
        )
        self.app.switch_screen("api_key")
