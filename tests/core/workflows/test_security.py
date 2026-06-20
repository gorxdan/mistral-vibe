from __future__ import annotations

import pytest

from vibe.core.workflows.security import (
    build_namespace,
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
