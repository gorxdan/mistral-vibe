Recursively search file contents by regex (ripgrep-backed).

- Fast; auto-ignores files you shouldn't read (.pyc, .venv, etc.).
- Use for text-level patterns: error messages, log lines, string literals, config values, TODO markers.
- `output_mode`: `content` (default; `file:line:text`) / `files_with_matches` (filenames only) / `count` (per-file counts).
- Narrow with `glob` (e.g. `*.py`), `type` (ripgrep type, e.g. `py`/`rust`), or `case_insensitive`.
- Content mode: add surrounding lines with `context` / `context_before` / `context_after`.
- `multiline` spans lines across newlines; `head_limit` caps output.
- `type` and `multiline` require ripgrep — they error on the GNU grep fallback; prefer `glob` for portability.
- Routing: filenames -> `glob`; symbols (definitions, references, types) -> `lsp`.
