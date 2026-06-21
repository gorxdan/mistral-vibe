Use `glob` to find files by name or path pattern. Results are sorted most-recently-modified first.

- Patterns: `**/*.py` (all Python files), `src/**/*.ts`, `*.md` (matches at any depth), `**/test_*.py`.
- A pattern without `/` matches the filename at any depth; include `/` (or `**/`) to match by path.
- Respects `.gitignore` and `.vibeignore` by default. Set `use_default_ignore=false` to include ignored files.
- Scope with `path` (defaults to the working directory); cap output with `max_results`.
- Use `glob` to discover files by name; use `grep` to search file *contents*. Prefer it over bash `find`/`ls`.
