from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.content import Content
from textual.events import DescendantBlur
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from vibe.cli.textual_ui.shortcut_hints import shortcut, shortcut_hint
from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


class ConfigOptionKind(StrEnum):
    ACTION_MODEL = auto()
    ACTION_THINKING = auto()
    ACTION_JUDGE_MODEL = auto()
    ACTION_SUBAGENT_MODEL = auto()

    @staticmethod
    def toggle(key: str) -> str:
        return f"toggle:{key}"

    @staticmethod
    def is_toggle(option_id: str) -> bool:
        return option_id.startswith("toggle:")

    @staticmethod
    def toggle_key(option_id: str) -> str:
        return option_id.removeprefix("toggle:")


class ConfigApp(Container):
    """Settings panel with navigatable option picker."""

    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False)
    ]

    class SettingChanged(Message):
        def __init__(self, key: str, value: str) -> None:
            super().__init__()
            self.key = key
            self.value = value

    class ConfigClosed(Message):
        def __init__(self, changes: dict[str, Any]) -> None:
            super().__init__()
            self.changes = changes

    class OpenModelPicker(Message):
        pass

    class OpenThinkingPicker(Message):
        pass

    class OpenJudgeModelPicker(Message):
        pass

    class OpenSubagentModelPicker(Message):
        pass

    def __init__(self, config: VibeConfig) -> None:
        super().__init__(id="config-app")
        self.config = config
        self.changes: dict[str, str] = {}
        self._toggle_settings: list[tuple[str, str]] = [
            ("autocopy_to_clipboard", "Auto-copy"),
            (
                "file_watcher_for_autocomplete",
                "Autocomplete watcher (may delay first autocompletion)",
            ),
            (
                "safety_judge.enabled",
                "Safety judge (let an LLM auto-approve safe tool calls)",
            ),
            ("ask_confirmation_on_exit", "Confirm quit on Ctrl+D"),
        ]

    def _get_current_model(self) -> str:
        return str(getattr(self.config, "active_model", ""))

    def _nested_get(self, key: str) -> Any:
        """Read a possibly dotted config key (e.g. ``safety_judge.enabled``)."""
        obj: Any = self.config
        for part in key.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def _get_toggle_value(self, key: str) -> str:
        if key in self.changes:
            return self.changes[key]
        raw = self._nested_get(key)
        if isinstance(raw, bool):
            return "On" if raw else "Off"
        return str(raw) if raw is not None else "Off"

    def _model_prompt(self) -> Text:
        text = Text(no_wrap=True)
        text.append("Model: ")
        text.append(self._get_current_model(), style="bold")
        return text

    def _get_current_thinking(self) -> str:
        try:
            return str(self.config.get_active_model().thinking)
        except ValueError:
            return "off"

    def _thinking_prompt(self) -> Text:
        text = Text(no_wrap=True)
        text.append("Thinking: ")
        text.append(self._get_current_thinking().capitalize(), style="bold")
        return text

    def _judge_model_prompt(self) -> Text:
        text = Text(no_wrap=True)
        text.append("  Judge model: ")
        model = self._nested_get("safety_judge.model")
        if model:
            text.append(str(model), style="bold")
        else:
            text.append("(none — select one)", style="dim")
        return text

    def _subagent_model_prompt(self) -> Text:
        text = Text(no_wrap=True)
        text.append("  Subagent model: ")
        model = str(getattr(self.config, "subagent_model", "") or "")
        if model:
            text.append(model, style="bold")
        else:
            text.append("(inherit host)", style="dim")
        return text

    def _toggle_prompt(self, key: str, label: str) -> Content:
        value = self._get_toggle_value(key)
        if value == "On":
            return Content.assemble(f"{label}: ", ("On", "bold $success"))
        return Content.assemble(f"{label}: ", ("Off", "dim"))

    def compose(self) -> ComposeResult:
        options: list[Option] = [
            Option(self._model_prompt(), id=ConfigOptionKind.ACTION_MODEL),
            Option(self._thinking_prompt(), id=ConfigOptionKind.ACTION_THINKING),
        ]
        for key, label in self._toggle_settings:
            options.append(
                Option(self._toggle_prompt(key, label), id=ConfigOptionKind.toggle(key))
            )
            if key == "safety_judge.enabled":
                options.append(
                    Option(
                        self._judge_model_prompt(),
                        id=ConfigOptionKind.ACTION_JUDGE_MODEL,
                    )
                )
        options.append(
            Option(
                self._subagent_model_prompt(), id=ConfigOptionKind.ACTION_SUBAGENT_MODEL
            )
        )

        with Vertical(id="config-content"):
            yield NoMarkupStatic("Settings", classes="settings-title")
            yield NoMarkupStatic("")
            yield NavigableOptionList(*options, id="config-options")
            yield NoMarkupStatic("")
            yield NoMarkupStatic(
                shortcut_hint(
                    f"{shortcut('↑↓/jk')} Navigate  {shortcut('Enter')} Select/Toggle  "
                    f"{shortcut('Esc')} Exit"
                ),
                classes="settings-help",
            )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_descendant_blur(self, _event: DescendantBlur) -> None:
        self.query_one(OptionList).focus()

    def _refresh_options(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.replace_option_prompt(
            ConfigOptionKind.ACTION_MODEL, self._model_prompt()
        )
        option_list.replace_option_prompt(
            ConfigOptionKind.ACTION_THINKING, self._thinking_prompt()
        )
        for key, label in self._toggle_settings:
            option_list.replace_option_prompt(
                ConfigOptionKind.toggle(key), self._toggle_prompt(key, label)
            )
        option_list.replace_option_prompt(
            ConfigOptionKind.ACTION_JUDGE_MODEL, self._judge_model_prompt()
        )
        option_list.replace_option_prompt(
            ConfigOptionKind.ACTION_SUBAGENT_MODEL, self._subagent_model_prompt()
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
        if not option_id:
            return

        if option_id == ConfigOptionKind.ACTION_MODEL:
            self.post_message(self.OpenModelPicker())
            return

        if option_id == ConfigOptionKind.ACTION_THINKING:
            self.post_message(self.OpenThinkingPicker())
            return

        if option_id == ConfigOptionKind.ACTION_JUDGE_MODEL:
            self.post_message(self.OpenJudgeModelPicker())
            return

        if option_id == ConfigOptionKind.ACTION_SUBAGENT_MODEL:
            self.post_message(self.OpenSubagentModelPicker())
            return

        if ConfigOptionKind.is_toggle(option_id):
            key = ConfigOptionKind.toggle_key(option_id)
            current = self._get_toggle_value(key)
            new_value = "Off" if current == "On" else "On"
            self.changes[key] = new_value
            self.post_message(self.SettingChanged(key=key, value=new_value))
            self._refresh_options()

    def convert_changes_for_save(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.changes.items():
            val: str | bool = value == "On" if value in {"On", "Off"} else value
            # Expand dotted keys (e.g. "safety_judge.enabled") into nested dicts
            # so save_updates deep-merges them into the right TOML table.
            parts = key.split(".")
            cursor = result
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = val
        return result

    def action_close(self) -> None:
        self.post_message(self.ConfigClosed(changes=self.convert_changes_for_save()))
