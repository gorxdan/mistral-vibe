# LSP Tool

Semantic code intelligence: `lsp` resolves what a symbol *is* (imports, overloads, generated code a name-`grep` misses); `grep` finds literal text/regex; `glob` finds files by name.

`pos` = `file_path, line, character` — **1-based Unicode code-point columns**,
cursor at the symbol's *start* (for `foo(bar)` with `foo` at column 5, pass
`character=5`). Vibe converts these columns to the server's UTF-16 coordinates.

## Operations

| Operation | Needs | Use it to |
|---|---|---|
| `status` | optional file_path | report live readiness, selected server, extensions, and advertised operations without starting a server |
| `go_to_definition` | pos | find where the symbol is defined |
| `go_to_implementation` | pos | find concrete impls of an interface/abstract |
| `find_references` | pos | list paged call/usage sites |
| `hover` | pos | get type signature and docs |
| `prepare_call_hierarchy` | pos | get call targets at position |
| `incoming_calls` | pos | trace who calls this |
| `outgoing_calls` | pos | trace what this calls |
| `document_symbol` | file_path | outline symbols in one file |
| `workspace_symbol` | query | find a symbol by name across the project |
| `grep` | — | search for a literal string or regex |
| `glob` | — | find files by name |

Under a path-scoped TaskBrief, `workspace_symbol` is unavailable because it can
search beyond the assignment. Start from an in-scope file and use
`document_symbol` or a position-based operation instead.

## Architecture analysis
1. `glob` → map packages and entry points.
2. `workspace_symbol`/`document_symbol` → identify central symbols (use only
   `document_symbol` under a path-scoped TaskBrief).
3. `find_references` or call hierarchy → verify dependency direction.
4. Read only the files needed to confirm that semantic map.

Use `status` when readiness is uncertain. A tool manifest entry means LSP is
enabled; `status.ready=true` proves that the matching server is running. No
matching server for an extension → fall back to `grep`.

Location, symbol, and call-hierarchy results are paged. If
`continuation_token` is present, repeat the exact operation/path/position/query
in the same session with that token until `has_more=false` and the result returns
no token. Tokens expire after two minutes and are bound to the current session,
task scope, workspace, and LSP manager generation. If a token is invalid or
expired, rerun the original query from its first page. Do not claim complete
coverage before consuming every page.
