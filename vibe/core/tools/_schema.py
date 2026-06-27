from __future__ import annotations

from typing import Any


def dereference_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline every ``$defs`` reference so the schema is flat.

    Pydantic (and some MCP servers) emit ``{"$ref": "#/$defs/X", ...siblings}``
    for referenced sub-schemas; strict OpenAI-compatible backends
    (Moonshot/kimi) reject a ``$ref`` with sibling keywords ("conflicting
    keywords found after $ref expansion"). Each reference is expanded by
    deep-merging its target and letting the sibling keys
    (description/default) override. Genuinely recursive definitions (a model
    that contains itself) cannot be inlined and keep their ``$ref``; their
    ``$defs`` entry is retained so it resolves.
    """
    defs = schema.get("$defs", {})
    cycled: set[str] = set()

    def expand(node: Any, resolving: frozenset[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref[len("#/$defs/") :]
                target = defs.get(name)
                if target is not None:
                    if name in resolving:
                        cycled.add(name)
                        return node
                    merged: dict[str, Any] = {}
                    nested = resolving | {name}
                    for k, v in target.items():
                        merged[k] = expand(v, nested)
                    for k, v in node.items():
                        if k != "$ref":
                            merged[k] = expand(v, nested)
                    # The target may itself be a $ref (A -> B -> concrete);
                    # re-resolve so multi-level chains fully inline rather
                    # than leaving a dangling reference after $defs is dropped.
                    if "$ref" in merged:
                        return expand(merged, nested)
                    return merged
            return {k: expand(v, resolving) for k, v in node.items()}
        if isinstance(node, list):
            return [expand(v, resolving) for v in node]
        return node

    expanded = expand(schema, frozenset())
    if "$defs" in expanded:
        if cycled:
            expanded["$defs"] = {
                n: v for n, v in expanded["$defs"].items() if n in cycled
            }
        else:
            del expanded["$defs"]
    return expanded


def strip_titles(node: Any) -> None:
    """Recursively remove auto-generated ``title`` metadata from a JSON schema.

    Pydantic emits a ``title`` (the field/model class name) on every node; that
    is noise for tool args and never sent to the LLM. Recurses into dicts and
    lists. Safe on the dereferenced output of :func:`dereference_refs`.

    The keys inside a ``properties`` object are field *names*, not metadata: a
    field literally named ``title`` (e.g. ``ManageMemoryArgs.title``) must
    survive. Stripping happens only on schema nodes whose parent is not a
    ``properties`` mapping; nested object schemas re-enter stripping under their
    own ``properties``. See ``test_strip_titles_keeps_property_named_title``.
    """

    def _strip(n: Any, in_properties: bool) -> None:
        if isinstance(n, dict):
            if not in_properties:
                n.pop("title", None)
            for k, v in n.items():
                _strip(v, k == "properties")
        elif isinstance(n, list):
            for v in n:
                _strip(v, False)

    _strip(node, False)
