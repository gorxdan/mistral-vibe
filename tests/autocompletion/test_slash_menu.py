from __future__ import annotations

from vibe.core.autocompletion.menu import (
    MenuEntry,
    MenuGroup,
    RowKind,
    build_menu_rows,
    first_selectable_index,
)


def _commands(*names: str) -> list[MenuEntry]:
    return [MenuEntry(name, name, MenuGroup.COMMAND) for name in names]


def _skills(*names: str) -> list[MenuEntry]:
    return [MenuEntry(name, name, MenuGroup.SKILL) for name in names]


def test_commands_only_has_no_headers() -> None:
    rows = build_menu_rows(_commands("/help", "/config"), query_empty=True)

    assert [row.kind for row in rows] == [RowKind.ITEM, RowKind.ITEM]
    assert [row.text for row in rows] == ["/help", "/config"]


def test_both_groups_emit_headers_with_commands_first() -> None:
    rows = build_menu_rows([*_commands("/help"), *_skills("/review")], query_empty=True)

    assert rows[0].kind is RowKind.HEADER and rows[0].text == "COMMANDS"
    assert rows[1].kind is RowKind.ITEM and rows[1].text == "/help"
    assert rows[2].kind is RowKind.HEADER and rows[2].text.startswith("SKILLS")
    assert rows[3].kind is RowKind.ITEM and rows[3].text == "/review"


def test_empty_query_caps_skills_and_appends_hint() -> None:
    skills = _skills(*[f"/s{i}" for i in range(10)])
    rows = build_menu_rows(
        [*_commands("/help"), *skills], query_empty=True, skill_cap=3
    )

    skill_items = [
        r for r in rows if r.kind is RowKind.ITEM and r.text.startswith("/s")
    ]
    hints = [r for r in rows if r.kind is RowKind.HINT]
    assert len(skill_items) == 3
    assert len(hints) == 1
    assert hints[0].text == "+7 more — type to filter"


def test_non_empty_query_does_not_cap_or_hint() -> None:
    skills = _skills(*[f"/s{i}" for i in range(10)])
    rows = build_menu_rows(
        [*_commands("/help"), *skills], query_empty=False, skill_cap=3
    )

    skill_items = [
        r for r in rows if r.kind is RowKind.ITEM and r.text.startswith("/s")
    ]
    assert len(skill_items) == 10
    assert all(row.kind is not RowKind.HINT for row in rows)


def test_empty_query_skill_count_in_header() -> None:
    skills = _skills(*[f"/s{i}" for i in range(12)])
    rows = build_menu_rows(
        [*_commands("/help"), *skills], query_empty=True, skill_cap=5
    )

    header = next(
        r for r in rows if r.kind is RowKind.HEADER and r.text.startswith("SKILLS")
    )
    assert header.text == "SKILLS  (12 — type to filter)"


def test_first_selectable_index_skips_leading_header() -> None:
    rows = build_menu_rows([*_commands("/help"), *_skills("/review")], query_empty=True)

    index = first_selectable_index(rows)
    assert index == 1
    assert rows[index].text == "/help"


def test_empty_input_yields_no_rows() -> None:
    assert build_menu_rows([], query_empty=True) == []
