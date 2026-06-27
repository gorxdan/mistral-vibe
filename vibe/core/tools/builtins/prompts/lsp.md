# LSP Tool

Semantic code intelligence via a language server: `lsp` resolves what a symbol *is* (imports, overloads, generated code that a name-`grep` misses); `grep` finds literal text/regex; `glob` finds files by name.

`pos` below = `file_path, line, character`. `line`/`character` are **1-based**; place the cursor at the *start* of the symbol (e.g. for `foo(bar)` where `foo` begins at column 5, pass `character=5`).

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

## Architecture analysis workflow

1. `glob` to map packages and entry points.
2. `workspace_symbol`/`document_symbol` to identify central symbols.
3. `find_references` or call hierarchy to verify dependency direction.
4. Read only the files needed to confirm that semantic map.

If the tool reports no server for an extension, no language server is configured for that language — fall back to `grep`.
