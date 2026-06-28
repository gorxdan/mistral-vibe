#!/usr/bin/env python3
"""Fail a commit that adds a multi-line comment block (>=3 consecutive `#`
lines) or a test docstring, enforcing the AGENTS.md comments rule.
"""

from __future__ import annotations

import subprocess
import sys

_MIN_BLOCK = 3


def _added(path: str) -> list[tuple[int, str]]:
    diff = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--", path],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    out: list[tuple[int, str]] = []
    lineno: int | None = None
    for line in diff.splitlines():
        if line.startswith("@@"):
            try:
                lineno = int(line.split("+", 1)[1].split()[0].split(",")[0])
            except (IndexError, ValueError):
                lineno = None
        elif lineno is not None and line.startswith("+") and not line.startswith("+++"):
            out.append((lineno, line[1:]))
            lineno += 1
    return out


def _violations(path: str) -> list[str]:
    added = _added(path)
    is_test = "test" in path.rsplit("/", 1)[-1] or "/tests/" in path
    found: list[str] = []
    run: list[int] = []
    for lineno, text in added:
        stripped = text.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            run.append(lineno)
            continue
        if len(run) >= _MIN_BLOCK:
            found.append(f"{path}:{run[0]}: {len(run)}-line comment block")
        run = []
    if len(run) >= _MIN_BLOCK:
        found.append(f"{path}:{run[0]}: {len(run)}-line comment block")
    if is_test:
        for lineno, text in added:
            if '"""' in text:
                found.append(f"{path}:{lineno}: test docstring")
                break
    return found


def main(paths: list[str]) -> int:
    found: list[str] = []
    for path in paths:
        if path.endswith(".py"):
            found.extend(_violations(path))
    if not found:
        return 0
    sys.stderr.write("AGENTS.md comments rule (terse hard-corners only) — fix:\n")
    for violation in found:
        sys.stderr.write(f"  {violation}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
