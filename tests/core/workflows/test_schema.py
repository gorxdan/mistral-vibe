from __future__ import annotations

from vibe.core.workflows.schema import (
    SchemaValidationError,
    ValidationError,
    build_prompt_fallback,
    build_response_format,
    validate_against_schema,
)

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {"type": "number"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "med", "high", "crit"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["title", "file", "evidence"],
            },
        }
    },
    "required": ["findings"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"refuted": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["refuted", "reason"],
}


def test_valid_object_passes() -> None:
    value = {
        "findings": [
            {
                "title": "Null deref",
                "file": "main.py",
                "line": 42,
                "severity": "high",
                "evidence": "ptr used without null check",
            }
        ]
    }
    assert not validate_against_schema(value, FINDINGS_SCHEMA)


def test_missing_required_property() -> None:
    value = {"findings": [{"file": "main.py", "evidence": "x"}]}
    errors = validate_against_schema(value, FINDINGS_SCHEMA)
    assert len(errors) == 1
    assert "title" in errors[0].path


def test_invalid_enum_value() -> None:
    value = {
        "findings": [
            {
                "title": "Bug",
                "file": "main.py",
                "line": 1,
                "severity": "critical",
                "evidence": "x",
            }
        ]
    }
    errors = validate_against_schema(value, FINDINGS_SCHEMA)
    assert len(errors) == 1
    assert "enum" in errors[0].message


def test_wrong_type() -> None:
    value = {"findings": "not a list"}
    errors = validate_against_schema(value, FINDINGS_SCHEMA)
    assert errors and "expected array" in errors[0].message


def test_boolean_not_integer() -> None:
    value = {"refuted": True, "reason": "ok"}
    assert not validate_against_schema(value, VERDICT_SCHEMA)

    value = {"refuted": 1, "reason": "ok"}
    errors = validate_against_schema(value, VERDICT_SCHEMA)
    assert errors and "expected boolean" in errors[0].message


def test_nested_array_errors() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["id"],
                },
            }
        },
        "required": ["items"],
    }
    value = {"items": [{"id": "a", "count": 1}, {"count": 2}]}
    errors = validate_against_schema(value, schema)
    assert len(errors) == 1
    assert "id" in errors[0].path


def test_build_response_format() -> None:
    rf = build_response_format(VERDICT_SCHEMA)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == VERDICT_SCHEMA
    assert "name" in rf["json_schema"]


def test_build_prompt_fallback() -> None:
    fb = build_prompt_fallback(VERDICT_SCHEMA)
    assert "JSON" in fb
    assert "refuted" in fb
    assert "no markdown fences" in fb


def test_schema_validation_error_is_exception() -> None:
    assert issubclass(SchemaValidationError, Exception)


def test_validation_error_str() -> None:
    e = ValidationError("$.findings[0].title", "required property missing")
    assert "title" in str(e)
    assert "required" in str(e)
