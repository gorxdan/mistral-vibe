from __future__ import annotations

from collections.abc import AsyncGenerator
from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel, Field

from vibe.core.tools._schema import strip_titles
from vibe.core.tools.base import BaseTool, BaseToolConfig, BaseToolState, InvokeContext
from vibe.core.tools.builtins.manage_memory import ManageMemory
from vibe.core.tools.builtins.team_message import TeamMessage
from vibe.core.types import ToolStreamEvent


class _Color(StrEnum):
    RED = auto()
    GREEN = auto()


class _Inner(BaseModel):
    shade: _Color = Field(default=_Color.RED, description="inner shade")


class _DemoArgs(BaseModel):
    # Enum field with a Field(description=...) is the exact shape that makes
    # Pydantic emit {"$ref": "#/$defs/_Color", "description": "..."} — a $ref
    # with sibling keywords. Strict backends (Moonshot/kimi) reject this.
    color: _Color = Field(default=_Color.RED, description="pick a color")
    # Nested model forces recursive dereferencing (_Inner -> _Color).
    inner: _Inner = Field(default_factory=_Inner)


class _DemoResult(BaseModel):
    message: str = ""


class _DemoTool(BaseTool[_DemoArgs, _DemoResult, BaseToolConfig, BaseToolState]):
    async def run(
        self, args: _DemoArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | _DemoResult, None]:
        yield _DemoResult(message=str(args.color))


def _collect_refs(node: Any) -> list[tuple[str, set[str]]]:
    """Return (path, sibling_keys) for every dict containing a $ref."""
    found: list[tuple[str, set[str]]] = []

    def walk(n: Any, path: str) -> None:
        if isinstance(n, dict):
            if "$ref" in n:
                found.append((path, set(n.keys()) - {"$ref"}))
            for k, v in n.items():
                walk(v, f"{path}.{k}")
        elif isinstance(n, list):
            for i, v in enumerate(n):
                walk(v, f"{path}[{i}]")

    walk(node, "$")
    return found


def test_team_message_schema_has_no_ref_with_siblings() -> None:
    # Regression: kimi/moonshot rejects `tools.function.parameters` whose
    # `kind` property is {"$ref": "#/$defs/MessageKind", "description": ...}
    # ("conflicting keywords found after $ref expansion"). The schema must be
    # fully dereferenced so no $ref — with or without siblings — reaches the
    # wire.
    params = TeamMessage.get_parameters()

    refs = _collect_refs(params)
    assert not refs, f"unexpected $ref in TeamMessage schema: {refs}"

    kind = params["properties"]["kind"]
    assert "enum" in kind, f"kind not dereferenced: {kind}"
    assert kind["enum"] == [
        "text",
        "permission_request",
        "permission_response",
        "plan_approval",
        "shutdown",
    ]


def test_generic_enum_with_description_is_dereferenced() -> None:
    params = _DemoTool.get_parameters()

    refs = _collect_refs(params)
    assert not refs, f"unexpected $ref in schema: {refs}"

    color = params["properties"]["color"]
    assert "enum" in color
    # Field-level description survives dereferencing (sibling wins).
    assert color["description"] == "pick a color"


def test_nested_enum_refs_are_recursively_dereferenced() -> None:
    # _DemoArgs.inner.shade references _Color via _Inner via $defs; the
    # dereferencer must recurse so the inner $ref is also inlined.
    params = _DemoTool.get_parameters()

    refs = _collect_refs(params)
    assert not refs, f"unexpected $ref in nested schema: {refs}"

    inner = params["properties"]["inner"]
    # After dereferencing _Inner, its `shade` property holds the _Color enum
    # inline rather than a $ref.
    assert "properties" in inner
    assert "shade" in inner["properties"]
    assert "enum" in inner["properties"]["shade"]


def test_strip_titles_keeps_property_named_title() -> None:
    # Regression: strip_titles removed Pydantic's auto-generated "title"
    # metadata key, but its blanket node.pop("title") also deleted any tool arg
    # field literally named "title". Field-name keys inside a `properties`
    # object are NOT metadata — they must survive — while metadata titles on the
    # root and on sub-schemas are still stripped.
    schema = {
        "title": "Args",
        "type": "object",
        "properties": {
            "title": {"type": "string", "title": "Title"},
            "inner": {
                "title": "Inner",
                "type": "object",
                "properties": {"shade": {"type": "string", "title": "Shade"}},
            },
        },
    }
    strip_titles(schema)
    # Field-named "title" key under properties survives...
    assert "title" in schema["properties"]
    # ...but the metadata title nested inside that field is stripped.
    assert "title" not in schema["properties"]["title"]
    # Root metadata stripped.
    assert "title" not in schema
    # Nested object: field key survives, metadata stripped at every level.
    assert "shade" in schema["properties"]["inner"]["properties"]
    assert "title" not in schema["properties"]["inner"]
    assert "title" not in schema["properties"]["inner"]["properties"]["shade"]


def test_manage_memory_exposes_title_arg() -> None:
    # Regression: ManageMemory.add hard-requires "title" (manage_memory.py), but
    # strip_titles had deleted it from the model-facing schema, so every agent
    # add call failed with "add requires 'title'" — the field was invisible.
    params = ManageMemory.get_parameters()
    assert "title" in params["properties"], (
        "manage_memory 'title' arg missing from schema; the model cannot "
        "supply it and add always fails with 'add requires title'."
    )
