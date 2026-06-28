# LSP Tool

Semantic code intelligence: `lsp` resolves what a symbol *is* (imports, overloads, generated code a name-`grep` misses); `grep` finds literal text/regex; `glob` finds files by name.

`pos` = `file_path, line, character` — **1-based**, cursor at the symbol's *start* (for `foo(bar)` with `foo` at column 5, pass `character=5`).

## Operations

| Operation | Needs | Use it to |
|---|---|---|
| `go_to_definition` | pos | find where the symbol is defined |
| `go_to_implementation` | pos | find concrete impls of an interface/abstract |
| `find_references` | pos | list all call/usage sites |
| `hover` | pos | get type signature and docs |
| `prepare_call_hierarchy` | pos | get call targets at position |
| `incoming_calls` | pos | trace who calls this |
| `outgoing_calls` | pos | trace what this calls |
| `document_symbol` | file_path | outline symbols in one file |
| `workspace_symbol` | query | find a symbol by name across the project |
| `grep` | — | search for a literal string or regex |
| `glob` | — | find files by name |

## Architecture analysis
1. `glob` → map packages and entry points.
2. `workspace_symbol`/`document_symbol` → identify central symbols.
3. `find_references` or call hierarchy → verify dependency direction.
4. Read only the files needed to confirm that semantic map.

No server for an extension → no language server is configured for that language; fall back to `grep`.
