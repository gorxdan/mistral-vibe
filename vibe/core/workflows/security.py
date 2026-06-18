from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import importlib
from typing import Any

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
    "string",
    "textwrap",
    "unicodedata",
    "operator",
    "pathlib",
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
    "format",
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
