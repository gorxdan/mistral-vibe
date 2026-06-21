from __future__ import annotations

from typing import NamedTuple

from textual import events

from vibe.cli.autocompletion.base import CompletionResult, CompletionView
from vibe.cli.autocompletion.slash_command import SlashCommandController
from vibe.core.autocompletion.completers import CommandCompleter
from vibe.core.autocompletion.menu import MenuEntry, MenuGroup, MenuRow, RowKind


class Suggestion(NamedTuple):
    alias: str
    description: str


class SuggestionEvent(NamedTuple):
    suggestions: list[Suggestion]
    selected_index: int


class MenuEvent(NamedTuple):
    rows: list[MenuRow]
    selected_index: int


class Replacement(NamedTuple):
    start: int
    end: int
    replacement: str


class StubView(CompletionView):
    def __init__(self) -> None:
        self.suggestion_events: list[SuggestionEvent] = []
        self.menu_events: list[MenuEvent] = []
        self.reset_count = 0
        self.replacements: list[Replacement] = []

    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        typed = [Suggestion(alias, description) for alias, description in suggestions]
        self.suggestion_events.append(SuggestionEvent(typed, selected_index))

    def render_slash_menu(self, rows: list[MenuRow], selected_index: int) -> None:
        self.menu_events.append(MenuEvent(list(rows), selected_index))
        items = [(index, row) for index, row in enumerate(rows) if row.selectable]
        typed = [Suggestion(row.text, row.description) for _, row in items]
        item_selected = next(
            (pos for pos, (index, _) in enumerate(items) if index == selected_index), 0
        )
        self.suggestion_events.append(SuggestionEvent(typed, item_selected))

    def clear_completion_suggestions(self) -> None:
        self.reset_count += 1

    def replace_completion_range(self, start: int, end: int, replacement: str) -> None:
        self.replacements.append(Replacement(start, end, replacement))


def key_event(key: str) -> events.Key:
    return events.Key(key, character=None)


def make_controller(
    *, prefix: str | None = None
) -> tuple[SlashCommandController, StubView]:
    commands = [
        ("/config", "Show current configuration"),
        ("/compact", "Compact history"),
        ("/help", "Display help"),
        ("/config", "Override description"),
        ("/summarize", "Summarize history"),
        ("/logpath", "Show log path"),
        ("/exit", "Exit application"),
        ("/vim", "Toggle vim keybindings"),
    ]
    completer = CommandCompleter(lambda: commands)
    view = StubView()
    controller = SlashCommandController(completer, view)

    if prefix is not None:
        controller.on_text_changed(prefix, cursor_index=len(prefix))
        view.suggestion_events.clear()

    return controller, view


def test_on_text_change_emits_matching_suggestions_in_insertion_order_and_ignores_duplicates() -> (
    None
):
    controller, view = make_controller(prefix="/c")

    controller.on_text_changed("/c", cursor_index=2)

    suggestions, selected = view.suggestion_events[-1]
    assert suggestions == [
        Suggestion("/config", "Override description"),
        Suggestion("/compact", "Compact history"),
    ]
    assert selected == 0


def test_on_text_change_filters_suggestions_case_insensitively() -> None:
    controller, view = make_controller(prefix="/c")

    controller.on_text_changed("/CO", cursor_index=3)

    suggestions, _ = view.suggestion_events[-1]
    assert [suggestion.alias for suggestion in suggestions] == ["/config", "/compact"]


def test_on_text_change_clears_suggestions_when_no_matches() -> None:
    controller, view = make_controller(prefix="/c")

    controller.on_text_changed("/c", cursor_index=2)
    controller.on_text_changed("config", cursor_index=6)

    assert view.reset_count >= 1


def test_on_text_change_limits_the_number_of_results_and_preserves_insertion_order() -> (
    None
):
    controller, view = make_controller(prefix="/")

    controller.on_text_changed("/", cursor_index=1)

    suggestions, selected_index = view.suggestion_events[-1]
    assert len(suggestions) == 7
    assert [suggestion.alias for suggestion in suggestions] == [
        "/help",
        "/config",
        "/compact",
        "/summarize",
        "/logpath",
        "/exit",
        "/vim",
    ]


def test_on_key_tab_applies_selected_completion() -> None:
    controller, view = make_controller(prefix="/c")

    result = controller.on_key(key_event("tab"), text="/c", cursor_index=2)

    assert result is CompletionResult.HANDLED
    assert view.replacements == [Replacement(0, 2, "/config")]
    assert view.reset_count == 1


def test_on_key_down_and_up_cycle_selection() -> None:
    controller, view = make_controller(prefix="/c")

    controller.on_key(key_event("down"), text="/c", cursor_index=2)
    suggestions, selected_index = view.suggestion_events[-1]
    assert selected_index == 1

    controller.on_key(key_event("down"), text="/c", cursor_index=2)
    suggestions, selected_index = view.suggestion_events[-1]
    assert selected_index == 0

    controller.on_key(key_event("up"), text="/c", cursor_index=2)
    suggestions, selected_index = view.suggestion_events[-1]
    assert selected_index == 1
    assert [suggestion.alias for suggestion in suggestions] == ["/config", "/compact"]


def test_on_key_enter_submits_selected_completion() -> None:
    controller, view = make_controller(prefix="/c")

    controller.on_key(key_event("down"), text="/c", cursor_index=2)

    result = controller.on_key(key_event("enter"), text="/c", cursor_index=2)

    assert result is CompletionResult.SUBMIT
    assert view.replacements == [Replacement(0, 2, "/compact")]
    assert view.reset_count == 1


def test_callable_entries_updates_completions_dynamically() -> None:
    """Test that CommandCompleter with a callable updates entries when the callable returns different values.

    This simulates config reload where available skills change.
    """
    available_skills: list[tuple[str, str]] = []

    def get_entries() -> list[tuple[str, str]]:
        base_commands = [("/help", "Display help"), ("/config", "Show configuration")]
        return base_commands + available_skills

    completer = CommandCompleter(get_entries)
    view = StubView()
    controller = SlashCommandController(completer, view)

    # Initially, only base commands are available
    controller.on_text_changed("/", cursor_index=1)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/help", "/config"]

    # Simulate config reload: add a skill
    available_skills.append(("/summarize", "Summarize the conversation"))

    # Now completions should include the new skill
    controller.on_text_changed("/", cursor_index=1)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/help", "/config", "/summarize"]

    # And searching for "/s" should find the new skill
    controller.on_text_changed("/s", cursor_index=2)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/summarize"]
    assert suggestions[0].description == "Summarize the conversation"


def test_tab_on_slash_command_with_args_replaces_only_head() -> None:
    controller, view = make_controller()
    text = "/compact some args"
    controller.on_text_changed(text, cursor_index=len(text))

    result = controller.on_key(key_event("tab"), text=text, cursor_index=len(text))

    assert result is CompletionResult.HANDLED
    assert view.replacements == [Replacement(0, 8, "/compact")]


def test_enter_on_slash_command_with_args_submits_with_head_only_replacement() -> None:
    controller, view = make_controller()
    text = "/compact some args"
    controller.on_text_changed(text, cursor_index=len(text))

    result = controller.on_key(key_event("enter"), text=text, cursor_index=len(text))

    assert result is CompletionResult.SUBMIT
    assert view.replacements == [Replacement(0, 8, "/compact")]


def test_on_text_change_matches_substring_not_just_prefix() -> None:
    controller, view = make_controller()

    controller.on_text_changed("/path", cursor_index=5)

    suggestions, _ = view.suggestion_events[-1]
    assert any(s.alias == "/logpath" for s in suggestions)


def test_on_text_change_matches_middle_segment_of_hyphenated_command() -> None:
    commands = [
        ("/foo-bar-skill", "A skill"),
        ("/baz-bar-other", "Another skill"),
        ("/unrelated", "No match"),
    ]
    completer = CommandCompleter(lambda: commands)
    view = StubView()
    controller = SlashCommandController(completer, view)

    controller.on_text_changed("/bar", cursor_index=4)

    suggestions, _ = view.suggestion_events[-1]
    aliases = [s.alias for s in suggestions]
    assert "/foo-bar-skill" in aliases
    assert "/baz-bar-other" in aliases
    assert "/unrelated" not in aliases


def test_on_text_change_fuzzy_matches_scattered_characters() -> None:
    controller, view = make_controller()

    controller.on_text_changed("/sm", cursor_index=3)

    suggestions, _ = view.suggestion_events[-1]
    assert any(s.alias == "/summarize" for s in suggestions)


def test_on_text_change_fuzzy_ranks_prefix_matches_higher() -> None:
    commands = [("/zoo-config", "Zoo config"), ("/config", "Main config")]
    completer = CommandCompleter(lambda: commands)
    view = StubView()
    controller = SlashCommandController(completer, view)

    controller.on_text_changed("/config", cursor_index=7)

    suggestions, _ = view.suggestion_events[-1]
    aliases = [s.alias for s in suggestions]
    assert aliases.index("/config") < aliases.index("/zoo-config")


def test_callable_entries_reflects_enabled_disabled_skills() -> None:
    """Test that skill enable/disable changes are reflected in completions.

    This simulates the scenario where a user changes enabled_skills in config
    and runs /reload.
    """
    enabled_skills: set[str] = {"commit", "review"}

    all_skills = [
        ("/commit", "Create a git commit"),
        ("/review", "Review code changes"),
        ("/deploy", "Deploy to production"),
    ]

    def get_entries() -> list[tuple[str, str]]:
        return [(name, desc) for name, desc in all_skills if name[1:] in enabled_skills]

    completer = CommandCompleter(get_entries)
    view = StubView()
    controller = SlashCommandController(completer, view)

    # Initially only commit and review are enabled
    controller.on_text_changed("/", cursor_index=1)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/commit", "/review"]

    # Simulate config reload: enable deploy, disable commit
    enabled_skills.discard("commit")
    enabled_skills.add("deploy")

    # Now completions should reflect the change
    controller.on_text_changed("/", cursor_index=1)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/review", "/deploy"]


def make_grouped_controller(
    *, skill_cap: int = 8
) -> tuple[SlashCommandController, StubView]:
    entries = [
        MenuEntry("/help", "Show help", MenuGroup.COMMAND),
        MenuEntry("/config", "Edit config", MenuGroup.COMMAND),
        MenuEntry("/review", "Review code", MenuGroup.SKILL),
        MenuEntry("/deep-research", "Research", MenuGroup.SKILL),
        MenuEntry("/verify", "Verify", MenuGroup.SKILL),
    ]
    completer = CommandCompleter(lambda: entries)
    view = StubView()
    controller = SlashCommandController(completer, view, skill_cap=skill_cap)
    return controller, view


def _skills_header_index(rows: list[MenuRow]) -> int:
    return next(
        index
        for index, row in enumerate(rows)
        if row.kind is RowKind.HEADER and row.text.startswith("SKILLS")
    )


def test_bare_slash_groups_commands_then_skills_under_headers() -> None:
    controller, view = make_grouped_controller()

    controller.on_text_changed("/", cursor_index=1)

    rows = view.menu_events[-1].rows
    assert rows[0].kind is RowKind.HEADER
    assert rows[0].text == "COMMANDS"

    skills_at = _skills_header_index(rows)
    command_items = [r.text for r in rows[1:skills_at] if r.kind is RowKind.ITEM]
    skill_items = [r.text for r in rows[skills_at + 1 :] if r.kind is RowKind.ITEM]

    assert command_items == ["/help", "/config"]
    assert skill_items == ["/review", "/deep-research", "/verify"]
    assert "(3 — type to filter)" in rows[skills_at].text


def test_no_headers_when_only_commands_present() -> None:
    completer = CommandCompleter(lambda: [("/help", "h"), ("/config", "c")])
    view = StubView()
    controller = SlashCommandController(completer, view)

    controller.on_text_changed("/", cursor_index=1)

    rows = view.menu_events[-1].rows
    assert all(row.kind is RowKind.ITEM for row in rows)


def test_arrow_navigation_skips_headers_and_lands_only_on_items() -> None:
    controller, view = make_grouped_controller()

    controller.on_text_changed("/", cursor_index=1)
    selected_rows = [view.menu_events[-1].selected_index]
    for _ in range(5):
        controller.on_key(key_event("down"), text="/", cursor_index=1)
        selected_rows.append(view.menu_events[-1].selected_index)

    rows = view.menu_events[-1].rows
    assert all(rows[index].kind is RowKind.ITEM for index in selected_rows)
    # five items, so the sixth "down" wraps back to the first item
    assert selected_rows[0] == selected_rows[-1]


def test_enter_on_a_skill_inserts_the_skill_alias() -> None:
    controller, view = make_grouped_controller()

    controller.on_text_changed("/", cursor_index=1)
    controller.on_key(key_event("down"), text="/", cursor_index=1)  # /help -> /config
    controller.on_key(key_event("down"), text="/", cursor_index=1)  # -> /review

    result = controller.on_key(key_event("enter"), text="/", cursor_index=1)

    assert result is CompletionResult.SUBMIT
    assert view.replacements[-1] == Replacement(0, 1, "/review")


def test_bare_slash_caps_skills_with_hint_and_typing_reveals_all() -> None:
    entries = [MenuEntry("/cmd", "c", MenuGroup.COMMAND)]
    entries += [MenuEntry(f"/skill-{i}", f"s{i}", MenuGroup.SKILL) for i in range(5)]
    completer = CommandCompleter(lambda: entries)
    view = StubView()
    controller = SlashCommandController(completer, view, skill_cap=2)

    controller.on_text_changed("/", cursor_index=1)
    rows = view.menu_events[-1].rows
    shown_skills = [
        r for r in rows if r.kind is RowKind.ITEM and r.text.startswith("/skill")
    ]
    assert len(shown_skills) == 2
    assert any(r.kind is RowKind.HINT and "+3 more" in r.text for r in rows)

    controller.on_text_changed("/skill", cursor_index=6)
    rows = view.menu_events[-1].rows
    shown_skills = [r for r in rows if r.kind is RowKind.ITEM]
    assert len(shown_skills) == 5
    assert all(row.kind is not RowKind.HINT for row in rows)
