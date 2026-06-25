from __future__ import annotations

from vibe.core.workflows.security import (
    INJECTED_NAMES,
    check_script,
    lint_script,
    validate_script,
)

# --------------------------------------------------------------------------- #
# validate_script — safety gate                                               #
# --------------------------------------------------------------------------- #


def test_validate_clean_script_no_violations() -> None:
    src = "import json\nx = json.dumps([1, 2])\n"
    assert validate_script(src) == []


def test_validate_unsafe_import_flagged() -> None:
    violations = validate_script("import os\n")
    assert any(v.rule == "unsafe-import" and "os" in v.detail for v in violations)


def test_validate_unsafe_from_import_flagged() -> None:
    violations = validate_script("from pathlib import Path\n")
    assert any(v.rule == "unsafe-import" for v in violations)


def test_validate_relative_import_flagged() -> None:
    violations = validate_script("from . import foo\n")
    assert any(v.rule == "unsafe-import" and "relative" in v.detail for v in violations)


def test_validate_dangerous_call_name() -> None:
    violations = validate_script("exec('print(1)')\n")
    assert any(v.rule == "dangerous-call" and "exec" in v.detail for v in violations)


def test_validate_dangerous_call_attribute() -> None:
    violations = validate_script("obj.eval('x')\n")
    assert any(v.rule == "dangerous-call" and "eval" in v.detail for v in violations)


def test_validate_dunder_attribute_access_flagged() -> None:
    violations = validate_script("x.__class__\n")
    assert any(v.rule == "dunder-access" for v in violations)


def test_validate_forbidden_format_attr_flagged() -> None:
    violations = validate_script("x.format('y')\n")
    assert any(v.rule == "forbidden-attr" and "format" in v.detail for v in violations)


def test_validate_dunder_subscript_flagged() -> None:
    violations = validate_script('x["__class__"]\n')
    assert any(v.rule == "dunder-subscript" for v in violations)


def test_validate_dunder_name_reference_flagged() -> None:
    violations = validate_script("__globals__\n")
    assert any(v.rule == "dunder-name" for v in violations)


def test_validate_name_dunder_allowed_for_name() -> None:
    # __name__ is explicitly allowed
    assert validate_script("x = __name__\n") == []


def test_validate_syntax_error_returns_violation() -> None:
    violations = validate_script("def (\n")
    assert len(violations) == 1
    assert violations[0].rule == "syntax-error"


def test_validate_safe_builtin_allowed() -> None:
    assert validate_script("x = len([1, 2])\n") == []


def test_violation_str_format() -> None:
    violations = validate_script("import os\n")
    assert len(violations) >= 1
    s = str(violations[0])
    assert "line" in s and "unsafe-import" in s


# --------------------------------------------------------------------------- #
# lint_script — correctness gate                                              #
# --------------------------------------------------------------------------- #


def test_lint_undefined_name_flagged() -> None:
    violations = lint_script("x = undefined_thing\n")
    assert any(v.rule == "undefined-name" for v in violations)


def test_lint_defined_name_not_flagged() -> None:
    assert lint_script("x = 1\ny = x\n") == []


def test_lint_injected_name_not_flagged() -> None:
    # INJECTED_NAMES are allowed without import
    src = f"result = {next(iter(INJECTED_NAMES))}\n"
    assert lint_script(src) == []


def test_lint_safe_module_allowed_without_import() -> None:
    # SAFE_MODULES are allowed (auto-bound by runtime)
    assert lint_script("x = json\n") == []


def test_lint_pipeline_thunk_misuse_flagged() -> None:
    src = "pipeline([1, 2], agent('prompt'))\n"
    violations = lint_script(src)
    assert any(v.rule == "non-thunk-arg" for v in violations)


def test_lint_pipeline_correct_thunk_not_flagged() -> None:
    src = "pipeline([1, 2], lambda x: agent('prompt'))\n"
    violations = lint_script(src)
    assert not any(v.rule == "non-thunk-arg" for v in violations)


def test_lint_pipeline_parallel_not_flagged() -> None:
    # parallel accepts bare coroutines (no thunk needed)
    src = "parallel(agent('a'), agent('b'))\n"
    violations = lint_script(src)
    assert not any(v.rule == "non-thunk-arg" for v in violations)


def test_lint_late_binding_lambda_flagged() -> None:
    src = "[lambda: agent(x) for x in items]\n"
    violations = lint_script(src)
    assert any(v.rule == "late-binding-closure" for v in violations)


def test_lint_late_binding_with_default_not_flagged() -> None:
    src = "[lambda x=x: agent(x) for x in items]\n"
    violations = lint_script(src)
    assert not any(v.rule == "late-binding-closure" for v in violations)


def test_lint_syntax_error_returns_violation() -> None:
    violations = lint_script("def (\n")
    assert violations[0].rule == "syntax-error"


def test_lint_clean_script_no_violations() -> None:
    src = "import json\nx = json.dumps([1])\n"
    assert lint_script(src) == []


def test_lint_results_sorted_by_line_col() -> None:
    src = "y = undefined_b\nx = undefined_a\n"
    violations = lint_script(src)
    lines = [v.line for v in violations]
    assert lines == sorted(lines)


# --------------------------------------------------------------------------- #
# check_script — combined gate                                                #
# --------------------------------------------------------------------------- #


def test_check_script_combines_safety_and_correctness() -> None:
    src = "import os\nx = undefined_thing\n"
    violations = check_script(src)
    rules = {v.rule for v in violations}
    assert "unsafe-import" in rules
    assert "undefined-name" in rules


def test_check_script_clean_passes() -> None:
    src = "import json\nresult = json.dumps({'a': 1})\n"
    assert check_script(src) == []


def test_check_script_results_sorted() -> None:
    src = "import os\nx = undefined_b\ny = undefined_a\n"
    violations = check_script(src)
    keys = [(v.line, v.col) for v in violations]
    assert keys == sorted(keys)
