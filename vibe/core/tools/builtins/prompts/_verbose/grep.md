Use `grep` to recursively search for a regular expression pattern in file contents (ripgrep-backed).

- It's very fast and automatically ignores files that you should not read like .pyc files, .venv directories, etc.
- Use this for text-level patterns: error messages, log lines, string literals, config values, TODO markers.
- `output_mode`: `content` (default, `file:line:text`), `files_with_matches` (just the filenames — fast "which files mention X"), or `count` (per-file match counts).
- Narrow the search with `glob` (e.g. `*.py`), `type` (ripgrep type like `py`/`rust`), or `case_insensitive`.
- In content mode, add surrounding lines with `context`, `context_before`, or `context_after`.
- `multiline` lets a pattern span lines; `head_limit` caps the output. `type` and `multiline` require ripgrep (they error on the GNU grep fallback — prefer `glob` for portability).
- For files by name use `glob`; for symbols (definitions, references, types) use `lsp` (the short form lives on each tool's own description; this file is the parameter-level detail).
