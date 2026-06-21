from __future__ import annotations

import pytest

from vibe.core.workflows.security import (
    build_namespace,
    check_script,
    lint_script,
    restricted_import,
    validate_script,
)

CLEAN_SCRIPT = """
import json

LENSES = ["correctness", "security"]
FINDINGS = {"type": "object", "properties": {"findings": {"type": "array"}}}

def key(f):
    return f"{f['file']}:{f.get('line', '?')}:{f['title']}".lower()

async def main():
    seen = set()
    results = await parallel(
        lambda: agent("audit through correctness lens", schema=FINDINGS),
        lambda: agent("audit through security lens", schema=FINDINGS),
    )
    raw = [f for r in results if r for f in r["findings"]]
    fresh = [f for f in raw if key(f) not in seen]
    for f in fresh:
        seen.add(key(f))
    return {"findings": fresh}
"""

UNSAFE_IMPORTS = [
    "import os",
    "import subprocess",
    "import sys",
    "import asyncio",
    "from os import system",
    "from subprocess import Popen",
    "import socket",
]

SAFE_IMPORTS = [
    "import json",
    "import re",
    "import math",
    "import statistics",
    "import collections",
    "import itertools",
    "import functools",
    "import datetime",
    "from json import dumps, loads",
    "from collections import defaultdict",
]

DANGEROUS_CALL_SOURCES = [
    "exec('print(1)')",
    "eval('1+1')",
    "compile('x', '', 'exec')",
    "open('/etc/passwd')",
    "globals()",
    "locals()",
    "vars()",
    "getattr(obj, 'x')",
    "__import__('os')",
]

DUNDER_ACCESS_SOURCES = [
    "obj.__class__",
    "obj.__subclasses__()",
    "obj.__globals__",
    "obj.__builtins__",
    "obj.__dict__",
    "obj.__mro__",
    "''.__class__",
    '"".__class__',
    "b''.__class__",
    "''.__class__.__mro__",
]

DUNDER_SUBSCRIPT_SOURCES = ["d['__class__']", "d['__globals__']", "d['__builtins__']"]


# Escapes reproduced end-to-end during the phases 0-4 audit. Each must now be
# rejected by validate_script (the gate that runs before exec).
SANDBOX_ESCAPE_SOURCES = [
    # pathlib filesystem access (no open() call)
    ('import pathlib\nx = pathlib.Path("/etc/hostname").read_text()', "unsafe-import"),
    # operator.attrgetter reaches dunders via string args
    ("import operator\nf = operator.attrgetter('__globals__')", "unsafe-import"),
    ("import operator\ng = operator.itemgetter('__builtins__')", "unsafe-import"),
    # string.Formatter().get_field() attribute traversal
    ("import string\nfm = string.Formatter()", "unsafe-import"),
    # str.format mini-language attribute traversal (dunder hidden in a string)
    ('y = "{0.__class__.__base__.__subclasses__}".format(())', "forbidden-attr"),
    ('z = "{0}".format_map({})', "forbidden-attr"),
    # aliasing the bound method to dodge the call-site check
    ('h = "{0.__class__}".format\nh(())', "forbidden-attr"),
]


def test_clean_script_passes() -> None:
    violations = validate_script(CLEAN_SCRIPT)
    assert not violations, f"expected no violations, got: {violations}"


@pytest.mark.parametrize(("src", "rule"), SANDBOX_ESCAPE_SOURCES)
def test_sandbox_escapes_rejected(src: str, rule: str) -> None:
    v = validate_script(src)
    assert v, f"expected escape to be blocked, but validate_script passed: {src!r}"
    assert any(viol.rule == rule for viol in v), (
        f"expected a '{rule}' violation for {src!r}, got {[str(x) for x in v]}"
    )


@pytest.mark.parametrize("mod", ["pathlib", "operator", "string"])
def test_removed_modules_not_importable(mod: str) -> None:
    with pytest.raises(ImportError):
        restricted_import(mod)


def test_format_builtin_removed_from_namespace() -> None:
    ns = build_namespace({})
    with pytest.raises(NameError):
        exec("v = format(123)", ns)


@pytest.mark.parametrize("src", UNSAFE_IMPORTS)
def test_unsafe_import_rejected(src: str) -> None:
    v = validate_script(src)
    assert v and v[0].rule == "unsafe-import"


@pytest.mark.parametrize("src", SAFE_IMPORTS)
def test_safe_imports_allowed(src: str) -> None:
    v = validate_script(src)
    assert not v, f"expected no violations for: {src}, got: {v}"


@pytest.mark.parametrize("src", DANGEROUS_CALL_SOURCES)
def test_dangerous_calls_rejected(src: str) -> None:
    v = validate_script(src)
    assert v and v[0].rule in ("dangerous-call", "dunder-name")


@pytest.mark.parametrize("src", DUNDER_ACCESS_SOURCES)
def test_dunder_access_rejected(src: str) -> None:
    v = validate_script(src)
    assert v and v[0].rule == "dunder-access"


@pytest.mark.parametrize("src", DUNDER_SUBSCRIPT_SOURCES)
def test_dunder_subscript_rejected(src: str) -> None:
    v = validate_script(src)
    assert v and v[0].rule == "dunder-subscript"


def test_restricted_import_allows_safe() -> None:
    json_mod = restricted_import("json")
    assert json_mod.dumps({"a": 1}) == '{"a": 1}'


def test_restricted_import_blocks_unsafe() -> None:
    with pytest.raises(ImportError):
        restricted_import("os")
    with pytest.raises(ImportError):
        restricted_import("subprocess")


def test_build_namespace_exec_safe_code() -> None:
    ns = build_namespace({"agent": lambda: "mock"})
    exec("x = [i**2 for i in range(5)]", ns)
    assert ns["x"] == [0, 1, 4, 9, 16]


def test_build_namespace_exec_json_import() -> None:
    ns = build_namespace({})
    exec("import json; y = json.dumps({'k': 1})", ns)
    assert ns["y"] == '{"k": 1}'


def test_build_namespace_autoinjects_safe_modules() -> None:
    # Safelisted modules are pre-bound, so a script can use them with NO import.
    ns = build_namespace({})
    exec("y = json.dumps({'k': 1}); z = re.findall(r'a', 'aa')", ns)
    assert ns["y"] == '{"k": 1}'
    assert ns["z"] == ["a", "a"]


def test_build_namespace_blocks_open() -> None:
    ns = build_namespace({})
    with pytest.raises(NameError):
        exec("open('/etc/passwd')", ns)


def test_build_namespace_blocks_exec() -> None:
    ns = build_namespace({})
    with pytest.raises(NameError):
        exec("exec('x=1')", ns)


def test_build_namespace_blocks_unsafe_import() -> None:
    ns = build_namespace({})
    with pytest.raises(ImportError):
        exec("__import__('os')", ns)


def test_syntax_error_reported() -> None:
    v = validate_script("def main(\n")
    assert v and v[0].rule == "syntax-error"


# --- correctness lint (lint_script / check_script) ----------------------------
# Catches classes that PASS the safety AST check but crash at exec time:
# genuinely undefined names, and a coroutine used where a per-item pipeline stage
# is required. Safelisted modules are auto-injected (no import needed) and
# `parallel` accepts coroutines directly, so neither of those is flagged.


def test_lint_passes_clean_script() -> None:
    assert not lint_script(CLEAN_SCRIPT)


def test_lint_allows_safe_module_without_import() -> None:
    # Safelisted modules are auto-injected by build_namespace, so using `json`
    # without `import json` is valid — this used to be the #1 run-killer.
    src = (
        "async def main():\n"
        "    return json.dumps([1, 2, 3])\n"
    )
    assert not lint_script(src)


def test_lint_allows_safe_module_with_explicit_import() -> None:
    # An explicit import still works (rebinds the same module) — no double-flag.
    src = (
        "import json\n"
        "async def main():\n"
        "    return json.dumps([1, 2, 3])\n"
    )
    assert not lint_script(src)


def test_lint_flags_undefined_helper_name() -> None:
    src = "async def main():\n    return frobnicate(3)\n"
    v = lint_script(src)
    assert v and v[0].rule == "undefined-name"


@pytest.mark.parametrize(
    "name",
    ["agent", "parallel", "pipeline", "phase", "log", "workflow", "budget", "args"],
)
def test_lint_allows_injected_names(name: str) -> None:
    src = f"async def main():\n    return {name}\n"
    assert not lint_script(src), f"{name} is injected and must not be flagged"


def test_lint_allows_coroutine_passed_to_parallel() -> None:
    # parallel accepts coroutines directly — the natural form is valid now.
    src = (
        "async def main():\n"
        '    return await parallel(agent("a"), agent("b"))\n'
    )
    assert not lint_script(src)


def test_lint_allows_lambda_thunks_in_parallel() -> None:
    src = (
        "async def main():\n"
        '    return await parallel(lambda: agent("a"), lambda: agent("b"))\n'
    )
    assert not lint_script(src)


def test_lint_allows_pipeline_items_first_arg() -> None:
    # pipeline's first positional is data (items), not a stage — must not flag it.
    src = (
        "async def main():\n"
        "    async def stage(x):\n"
        "        return x\n"
        "    return await pipeline([1, 2, 3], stage)\n"
    )
    assert not lint_script(src)


def test_lint_flags_coroutine_as_pipeline_stage() -> None:
    # A pipeline STAGE is invoked per item, so a bare coroutine cannot serve as
    # one — flag `pipeline(items, agent(...))` (use `lambda x: agent(...)`).
    src = (
        "async def main():\n"
        '    return await pipeline([1, 2], agent("x"))\n'
    )
    v = lint_script(src)
    assert v and v[0].rule == "non-thunk-arg"


def test_lint_flags_late_binding_lambda_over_loop() -> None:
    # Classic footgun: lambda reads the loop var at call time -> all collapse to
    # the last item (silent: wrong labels/profiles).
    src = (
        "async def main():\n"
        '    return await parallel(*[lambda: agent(a["p"], label=a["k"]) '
        "for a in areas])\n"
    )
    v = [x for x in lint_script(src) if x.rule == "late-binding-closure"]
    assert v and "a" in v[0].detail


def test_lint_allows_coroutine_comprehension() -> None:
    # The new canonical fan-out: coroutines, no lambda — binds correctly.
    src = (
        "async def main():\n"
        '    return await parallel(*[agent(a["p"], label=a["k"]) for a in areas])\n'
    )
    assert not [x for x in lint_script(src) if x.rule == "late-binding-closure"]


def test_lint_allows_default_bound_lambda_over_loop() -> None:
    # Capturing the loop var as a default arg is the other valid fix.
    src = (
        "async def main():\n"
        '    return await parallel(*[lambda a=a: agent(a["p"]) for a in areas])\n'
    )
    assert not [x for x in lint_script(src) if x.rule == "late-binding-closure"]


def test_lint_allows_lambda_rebinding_loop_var_in_inner_comprehension() -> None:
    # The loop var is REBOUND by an inner comprehension inside the lambda body,
    # so there is no late-binding bug — must NOT be flagged (was a false positive
    # that hard-rejected valid scripts).
    src = (
        "async def main():\n"
        "    return await parallel(*[lambda: [a * 2 for a in inner] for a in outer])\n"
    )
    assert not [x for x in lint_script(src) if x.rule == "late-binding-closure"]


def test_lint_allows_lambda_rebinding_loop_var_in_nested_lambda() -> None:
    src = (
        "async def main():\n"
        "    return await parallel(*[lambda: (lambda a: a)(1) for a in xs])\n"
    )
    assert not [x for x in lint_script(src) if x.rule == "late-binding-closure"]


def test_lint_still_flags_genuine_capture_alongside_inner_rebind() -> None:
    # `n` is genuinely late-bound; `a` is rebound by the inner generator. Only n.
    src = (
        "async def main():\n"
        "    return await parallel(*[lambda: sum(a for a in range(n)) for n in nums])\n"
    )
    v = [x for x in lint_script(src) if x.rule == "late-binding-closure"]
    assert v and "{n}" in v[0].detail


def test_check_script_combines_safety_and_correctness() -> None:
    # Unsafe import (safety) + genuinely undefined name (correctness) together.
    src = (
        "import os\n"
        "async def main():\n"
        "    return frobnicate(os.listdir())\n"
    )
    rules = {v.rule for v in check_script(src)}
    assert "unsafe-import" in rules
    assert "undefined-name" in rules
