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

# Names the runtime injects into every workflow namespace. Kept in sync with
# WorkflowRuntime.build_script_namespace() in runtime.py. Used by the correctness
# lint (lint_script) to tell a genuinely-undefined name (e.g. `json` used without
# `import json`) from a legitimately-available injected helper.
INJECTED_NAMES = frozenset({
    "agent",
    "parallel",
    "pipeline",
    "phase",
    "log",
    "workflow",
    "budget",
    "post_message",
    "fetch_messages",
    "flatten",
    "dedup_by",
    "merge_by",
    "args",
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


def _bound_names(tree: ast.AST) -> set[str]:
    """Over-approximate every name BOUND anywhere in the script: imports,
    assignments, def/lambda params, function/class names, loop/comprehension/
    with/except targets, and walrus.

    Over-approximation (collecting bindings from every scope, ignoring scope
    boundaries) is deliberate: the goal is ZERO false positives on valid scripts
    at the cost of possibly missing some genuinely-undefined uses. A name bound
    anywhere is treated as defined everywhere.
    """
    bound: set[str] = set()

    def add_target(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                add_target(elt)
        elif isinstance(t, ast.Starred):
            add_target(t.value)

    def add_args(a: ast.arguments) -> None:
        for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
            bound.add(arg.arg)
        if a.vararg:
            bound.add(a.vararg.arg)
        if a.kwarg:
            bound.add(a.kwarg.arg)

    # Nodes that expose a single `.target` to bind (assignment-like, loop, and
    # comprehension targets all share the attribute), grouped to keep the
    # dispatch flat.
    target_nodes = (
        ast.AnnAssign,
        ast.AugAssign,
        ast.For,
        ast.AsyncFor,
        ast.comprehension,
        ast.NamedExpr,
    )

    for node in ast.walk(tree):
        # def/class introduce a name; def/lambda also bind their parameters.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            add_args(node.args)

        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                add_target(t)
        elif isinstance(node, target_nodes):
            add_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    add_target(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)

    return bound


def _undefined_names(tree: ast.AST, bound: set[str]) -> list[Violation]:
    """Flag names that are LOADead but never bound, injected, a pre-bound safe
    module, or a safelisted builtin — the exec-time `name 'X' is not defined`
    class, caught pre-flight at no cost. SAFE_MODULES are allowed because the
    runtime auto-binds them (build_namespace), and DANGEROUS_CALLS names are
    allowed because they are reported by the more specific `dangerous-call`
    rule, not double-flagged.
    """
    allowed = (
        bound
        | INJECTED_NAMES
        | SAFE_BUILTINS
        | SAFE_MODULES
        | DANGEROUS_CALLS
        | {"__name__"}
    )
    violations: list[Violation] = []
    reported: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        if node.id in allowed:
            continue
        key = (node.id, node.lineno)
        if key in reported:
            continue
        reported.add(key)
        detail = (
            f"name '{node.id}' is not defined "
            "(not assigned, an injected helper, a safelisted module, or a builtin)"
        )
        violations.append(
            Violation(node.lineno, node.col_offset, "undefined-name", detail)
        )
    return violations


def _thunk_misuse(tree: ast.AST) -> list[Violation]:
    """`pipeline` STAGES must be callables of the item (`lambda x: agent(...)` or
    a def), not a bare coroutine: each stage is invoked per item, so a single
    `agent(...)` coroutine cannot serve as one. `parallel` is NOT flagged — it
    accepts coroutines directly (`parallel(agent(...))`), so the natural form is
    valid. Catch the common `pipeline(items, agent(...))` slip pre-flight.
    """
    violations: list[Violation] = []

    def is_spawn_call(n: ast.AST) -> bool:
        return (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in {"agent", "workflow"}
        )

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "pipeline":
            continue
        positional = [a for a in node.args if not isinstance(a, ast.Starred)]
        # pipeline's first positional is `items` (data); the rest are stages.
        for a in positional[1:]:
            if (
                isinstance(a, ast.Call)
                and is_spawn_call(a)
                and isinstance(a.func, ast.Name)
            ):
                callee = a.func.id
                violations.append(
                    Violation(
                        a.lineno,
                        a.col_offset,
                        "non-thunk-arg",
                        f"pipeline() stages must be callables of the item; pass "
                        f"`lambda x: {callee}(...)` or a def, not `{callee}(...)` "
                        f"(a bare {callee}(...) is one coroutine, not a per-item stage)",
                    )
                )
    return violations


def _late_binding_lambda(tree: ast.AST) -> list[Violation]:
    """Flag the classic Python late-binding footgun: a `lambda` inside a
    comprehension that reads the comprehension's loop variable in its body
    WITHOUT capturing it as a default arg. The body resolves the name at CALL
    time, so every lambda collapses to the loop's LAST value — e.g.
    `parallel(*[lambda: agent(a["prompt"], label=a["key"]) for a in areas])`
    runs every agent with the last area's label/profile. Silent (no crash), just
    wrong. Fix: drop the lambda and pass the coroutine directly
    (`parallel(*[agent(...) for a in areas])`), or bind a default (`lambda a=a:`).
    """
    violations: list[Violation] = []
    comp_types = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)

    def target_names(generators: list[ast.comprehension]) -> set[str]:
        names: set[str] = set()
        for gen in generators:
            for n in ast.walk(gen.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        return names

    def param_names(lam: ast.Lambda) -> set[str]:
        a = lam.args
        names = {arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
        if a.vararg:
            names.add(a.vararg.arg)
        if a.kwarg:
            names.add(a.kwarg.arg)
        return names

    reported: set[tuple[int, int]] = set()
    for comp in ast.walk(tree):
        if not isinstance(comp, comp_types):
            continue
        loop_vars = target_names(comp.generators)
        if not loop_vars:
            continue
        elts = [comp.key, comp.value] if isinstance(comp, ast.DictComp) else [comp.elt]
        for elt in elts:
            for lam in ast.walk(elt):
                if not isinstance(lam, ast.Lambda):
                    continue
                loads = {
                    n.id
                    for n in ast.walk(lam.body)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
                # Subtract names rebound INSIDE the body too (inner-comprehension
                # targets, nested-lambda params, walrus). `loads` descends nested
                # scopes, so without this a loop var that is shadowed/rebound in a
                # nested scope (e.g. `[lambda: [a for a in inner] for a in outer]`)
                # is a false positive — and the lint hard-rejects valid scripts.
                shadowed = param_names(lam) | _bound_names(lam.body)
                risky = (loop_vars & loads) - shadowed
                if not risky or (lam.lineno, lam.col_offset) in reported:
                    continue
                reported.add((lam.lineno, lam.col_offset))
                names = ", ".join(sorted(risky))
                pick = sorted(risky)[0]
                violations.append(
                    Violation(
                        lam.lineno,
                        lam.col_offset,
                        "late-binding-closure",
                        f"lambda reads loop var(s) {{{names}}} at call time inside a "
                        f"comprehension — they collapse to the LAST value. Drop the "
                        f"lambda and pass the coroutine directly "
                        f"(parallel(*[agent(...) for ...])), or bind a default "
                        f"(lambda {pick}={pick}: ...).",
                    )
                )
    return violations


def lint_script(source: str) -> list[Violation]:
    """Correctness lint (distinct from validate_script's safety gate): catches
    classes that PASS the AST safety check but crash or silently misbehave once
    running — undefined names, a coroutine used as a pipeline stage, and the
    late-binding-closure footgun in lambda thunks over a comprehension.

    Returns violations sorted by (line, col). Empty list means clean.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [Violation(e.lineno or 0, e.offset or 0, "syntax-error", str(e))]

    violations = _undefined_names(tree, _bound_names(tree))
    violations.extend(_thunk_misuse(tree))
    violations.extend(_late_binding_lambda(tree))
    violations.sort(key=lambda v: (v.line, v.col))
    return violations


def check_script(source: str) -> list[Violation]:
    """Full pre-flight gate: safety (validate_script) + correctness (lint_script).
    Use at the agent-authored launch boundary so the common authoring mistakes
    fail before any agent spawns instead of at exec time.
    """
    violations = list(validate_script(source))
    violations.extend(lint_script(source))
    violations.sort(key=lambda v: (v.line, v.col))
    return violations


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
    # Pre-bind the safelisted modules so scripts can use `json`/`re`/... WITHOUT
    # an explicit `import` line. The import was pure boilerplate (these modules
    # are already allowlisted), and forgetting it (`json.dumps` with no
    # `import json`) was the single most common run-killer. A script may still
    # import them explicitly — that just rebinds the same module object.
    for mod in SAFE_MODULES:
        try:
            namespace[mod] = importlib.import_module(mod)
        except ImportError:
            pass
    namespace.update(injected)
    return namespace
