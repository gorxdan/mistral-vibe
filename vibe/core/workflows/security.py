from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import importlib
from typing import Any

# NOTE: this is a best-effort, in-process AST allowlist. It is NOT a hard
# security boundary: a sufficiently determined script could still find a gap,
# because the script ultimately runs in this interpreter. Untrusted scripts
# must be gated behind explicit user approval (the launch_workflow tool uses
# ToolPermission.ASK) — do not auto-run scripts from untrusted sources. The
# rules below close every escape vector reproduced in the phases 0-4 audit
# (pathlib filesystem access, operator.attrgetter / string.Formatter attribute
# traversal via string-encoded dunders, and the str.format mini-language).

SAFE_MODULES = frozenset({
    "json",
    "re",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "decimal",
    "copy",
    "hashlib",
    "base64",
    "textwrap",
    "unicodedata",
    # Deliberately excluded (audit-confirmed escape primitives):
    #   pathlib  -> Path.read_text/write_text/unlink = arbitrary filesystem I/O
    #   operator -> attrgetter/itemgetter reach dunders via string args,
    #               bypassing the AST dunder checks
    #   string   -> string.Formatter().get_field() does attribute traversal
})

SAFE_BUILTINS = frozenset({
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "True",
    "False",
    "None",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "RuntimeError",
    "StopIteration",
    "NotImplementedError",
    "AttributeError",
})

DANGEROUS_CALLS = frozenset({
    "exec",
    "eval",
    "compile",
    "open",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "__import__",
    "help",
})

# Attribute names that are forbidden anywhere they appear (load or call). The
# str.format / str.format_map mini-language performs attribute and index access
# from inside a format string ("{0.__class__}"), which the AST dunder checks
# cannot see because the dunder lives in a string literal. Blocking the methods
# themselves closes that vector (including aliasing: `f = x.format; f(...)`).
# The Formatter.* names are listed defensively even though `string` is no longer
# importable.
FORBIDDEN_ATTRS = frozenset({
    "format",
    "format_map",
    "format_field",
    "vformat",
    "get_field",
})


@dataclass
class Violation:
    line: int
    col: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"line {self.line}:{self.col} [{self.rule}] {self.detail}"


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def _v(self, node: ast.expr | ast.stmt, rule: str, detail: str) -> None:
        self.violations.append(Violation(node.lineno, node.col_offset, rule, detail))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in SAFE_MODULES:
                self._v(
                    node, "unsafe-import", f"import of '{alias.name}' not in safelist"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self._v(node, "unsafe-import", "relative import not allowed")
        else:
            root = node.module.split(".")[0]
            if root not in SAFE_MODULES:
                self._v(node, "unsafe-import", f"from '{node.module}' not in safelist")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in DANGEROUS_CALLS:
            self._v(node, "dangerous-call", f"call to '{func.id}' is forbidden")
        if isinstance(func, ast.Attribute) and func.attr in DANGEROUS_CALLS:
            self._v(node, "dangerous-call", f"call to '.{func.attr}' is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr.startswith("__") and attr.endswith("__"):
            self._v(node, "dunder-access", f"access to dunder '.{attr}' is forbidden")
        elif attr in FORBIDDEN_ATTRS:
            self._v(node, "forbidden-attr", f"access to '.{attr}' is forbidden")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            if node.slice.value.startswith("__") and node.slice.value.endswith("__"):
                self._v(
                    node,
                    "dunder-subscript",
                    f"subscript with dunder key '{node.slice.value}' is forbidden",
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id.endswith("__"):
            if node.id != "__name__":
                self._v(node, "dunder-name", f"reference to '{node.id}' is forbidden")
        self.generic_visit(node)


def validate_script(source: str) -> list[Violation]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [Violation(e.lineno or 0, e.offset or 0, "syntax-error", str(e))]

    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.violations


def restricted_import(
    name: str,
    globals: dict | None = None,
    locals: dict | None = None,
    fromlist: tuple = (),
    level: int = 0,
) -> Any:
    root = name.split(".")[0]
    if root not in SAFE_MODULES:
        raise ImportError(f"import of '{name}' is not allowed in workflow scripts")
    return importlib.import_module(name)


def build_namespace(injected: dict[str, Any]) -> dict[str, Any]:
    safe_builtins = {
        name: getattr(builtins, name)
        for name in SAFE_BUILTINS
        if hasattr(builtins, name)
    }
    safe_builtins["__import__"] = restricted_import

    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    namespace.update(injected)
    return namespace
