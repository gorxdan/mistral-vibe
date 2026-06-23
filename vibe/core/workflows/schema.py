from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


class SchemaValidationError(Exception):
    pass


@dataclass
class ValidationError:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _validate_object(value: Any, schema: dict, path: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not isinstance(value, dict):
        errors.append(
            ValidationError(path, f"expected object, got {type(value).__name__}")
        )
        return errors
    for prop_name in schema.get("required", []):
        if prop_name not in value:
            errors.append(
                ValidationError(f"{path}.{prop_name}", "required property missing")
            )
    for prop_name, prop_schema in schema.get("properties", {}).items():
        if prop_name in value:
            errors.extend(
                _validate_value(value[prop_name], prop_schema, f"{path}.{prop_name}")
            )
    return errors


def _validate_array(value: Any, schema: dict, path: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not isinstance(value, list):
        errors.append(
            ValidationError(path, f"expected array, got {type(value).__name__}")
        )
        return errors
    item_schema = schema.get("items")
    if item_schema:
        for i, item in enumerate(value):
            errors.extend(_validate_value(item, item_schema, f"{path}[{i}]"))
    return errors


def _validate_string(value: Any, schema: dict, path: str) -> list[ValidationError]:
    if not isinstance(value, str):
        return [ValidationError(path, f"expected string, got {type(value).__name__}")]
    if "enum" in schema and value not in schema["enum"]:
        return [ValidationError(path, f"'{value}' not in enum {schema['enum']}")]
    return []


def _validate_number(value: Any, path: str) -> list[ValidationError]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return [ValidationError(path, f"expected number, got {type(value).__name__}")]
    return []


def _validate_boolean(value: Any, path: str) -> list[ValidationError]:
    if not isinstance(value, bool):
        return [ValidationError(path, f"expected boolean, got {type(value).__name__}")]
    return []


def _validate_integer(value: Any, path: str) -> list[ValidationError]:
    if not isinstance(value, int) or isinstance(value, bool):
        return [ValidationError(path, f"expected integer, got {type(value).__name__}")]
    return []


_VALIDATORS: dict[str, Any] = {
    "object": _validate_object,
    "array": _validate_array,
    "string": _validate_string,
    "number": _validate_number,
    "boolean": _validate_boolean,
    "integer": _validate_integer,
}


def _validate_value(value: Any, schema: dict, path: str) -> list[ValidationError]:
    if value is None and "type" not in schema:
        return []

    schema_type = schema.get("type")
    validator = _VALIDATORS.get(schema_type or "")
    if validator is None:
        return []

    if schema_type in {"number", "boolean", "integer"}:
        return validator(value, path)
    return validator(value, schema, path)


def validate_against_schema(value: Any, schema: dict) -> list[ValidationError]:
    return _validate_value(value, schema, "$")


def _strip_object_unknown(value: Any, schema: dict) -> Any:
    """Recursively remove object properties not declared in the schema's
    ``properties``. Used as a lenient pre-step before validation so an agent
    that emits extra fields (e.g. a free-form ``confidence`` note alongside
    structured findings) still validates and yields a clean, schema-shaped
    result, instead of either failing or passing the extras through to the host.

    Strips at every object node that has a ``properties`` map; arrays are
    recursed element-wise; non-object values are returned unchanged. Unknown
    properties on objects with no ``properties`` key are preserved (the schema
    makes no claim about shape there).
    """
    if isinstance(value, list):
        item_schema = schema.get("items") if isinstance(schema, dict) else None
        if isinstance(item_schema, dict):
            return [_strip_object_unknown(v, item_schema) for v in value]
        return list(value)
    if not isinstance(value, dict) or not isinstance(schema, dict):
        return value
    props = schema.get("properties")
    if not isinstance(props, dict):
        return value
    out: dict[str, Any] = {}
    for k, v in value.items():
        if k not in props:
            continue
        out[k] = _strip_object_unknown(v, props[k])
    return out


def strip_unknown_properties(value: Any, schema: dict) -> Any:
    """Public entry point for the lenient pre-validation strip. Returns a new
    value with unknown properties removed; the input is not mutated.
    """
    return _strip_object_unknown(value, schema)


def build_response_format(schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"schema": schema, "name": "workflow_output"},
    }


def build_prompt_fallback(schema: dict) -> str:
    return (
        "\n\nYou MUST respond with a single valid JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Respond with ONLY the JSON object, no markdown fences, no explanation."
    )
