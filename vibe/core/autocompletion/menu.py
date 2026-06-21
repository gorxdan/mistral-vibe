from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

# Builtin commands are the small curated core surface and are never capped.
# Skills are unbounded (users install many), so a bare "/" shows a scannable
# preview and lets typing reveal the rest.
DEFAULT_SKILL_PREVIEW_CAP = 8


class MenuGroup(StrEnum):
    COMMAND = auto()
    SKILL = auto()


class RowKind(StrEnum):
    HEADER = auto()
    ITEM = auto()
    HINT = auto()


@dataclass(frozen=True)
class MenuEntry:
    alias: str
    description: str
    group: MenuGroup = MenuGroup.COMMAND

    @classmethod
    def coerce(cls, value: MenuEntry | tuple[str, str]) -> MenuEntry:
        if isinstance(value, MenuEntry):
            return value
        alias, description = value
        return cls(alias, description)


@dataclass(frozen=True)
class MenuRow:
    # For ITEM rows, `text` is both the displayed alias and the value inserted on
    # accept. HEADER and HINT rows are decoration: they render but cannot be
    # selected, and navigation skips over them.
    text: str
    description: str = ""
    kind: RowKind = RowKind.ITEM

    @property
    def selectable(self) -> bool:
        return self.kind is RowKind.ITEM


def build_menu_rows(
    entries: list[MenuEntry],
    *,
    query_empty: bool,
    skill_cap: int = DEFAULT_SKILL_PREVIEW_CAP,
) -> list[MenuRow]:
    # `entries` must already be filtered and ranked; this only groups for
    # display. Commands always come first. Section headers appear only when both
    # groups are present, so a commands-only menu stays a flat list. On a bare
    # "/" the skill group is capped with a trailing hint.
    commands = [e for e in entries if e.group is MenuGroup.COMMAND]
    skills = [e for e in entries if e.group is MenuGroup.SKILL]
    show_headers = bool(commands) and bool(skills)

    rows: list[MenuRow] = []

    if commands:
        if show_headers:
            rows.append(MenuRow("COMMANDS", kind=RowKind.HEADER))
        rows.extend(_item(e) for e in commands)

    if skills:
        shown = skills
        hidden = 0
        if query_empty and len(skills) > skill_cap:
            shown = skills[:skill_cap]
            hidden = len(skills) - skill_cap

        if show_headers:
            label = (
                f"SKILLS  ({len(skills)} — type to filter)" if query_empty else "SKILLS"
            )
            rows.append(MenuRow(label, kind=RowKind.HEADER))
        rows.extend(_item(e) for e in shown)
        if hidden:
            rows.append(MenuRow(f"+{hidden} more — type to filter", kind=RowKind.HINT))

    return rows


def first_selectable_index(rows: list[MenuRow]) -> int | None:
    for index, row in enumerate(rows):
        if row.selectable:
            return index
    return None


def _item(entry: MenuEntry) -> MenuRow:
    return MenuRow(entry.alias, entry.description, RowKind.ITEM)
