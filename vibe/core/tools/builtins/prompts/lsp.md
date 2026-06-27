# LSP Tool

Semantic code intelligence via a language server. Use this **instead of grep**
whenever you need to understand what a symbol *is*, not just where a string
appears.

## When to use LSP vs grep

| Task | Use |
|---|---|
| Where is this function/class/variable defined? | `lsp go_to_definition` |
| Who calls this? Where is this referenced? | `lsp find_references` |
| What is the type / signature / docstring here? | `lsp hover` |
| Outline the structure of a file | `lsp document_symbol` |
| Find a symbol by name across the project | `lsp workspace_symbol` |
| Trace the call graph (callers / callees) | `lsp incoming_calls` / `outgoing_calls` |
| Find concrete impls of an interface/abstract | `lsp go_to_implementation` |
| Search for a literal string or regex pattern | `grep` |
| Find files by name | `glob` |

**Rule of thumb:** `lsp` for symbols, `grep` for text. The short form lives on each tool's own description; this table is the detailed map. The payoff is semantic resolution — imports, overloads, generated code — that a plain `grep` for a method name misses.

## Operations

| Operation | Needs | Returns |
|---|---|---|
| `go_to_definition` | file_path, line, character | where the symbol is defined |
| `go_to_implementation` | file_path, line, character | concrete implementations of an interface |
| `find_references` | file_path, line, character | all call/usage sites |
| `hover` | file_path, line, character | type signature and docs |
| `document_symbol` | file_path | symbols in this file (outline) |
| `workspace_symbol` | query | symbols across the workspace |
| `prepare_call_hierarchy` | file_path, line, character | call targets at position |
| `incoming_calls` | file_path, line, character | who calls this |
| `outgoing_calls` | file_path, line, character | what this calls |

## Positions

`line` and `character` are **1-based**. Place the cursor at the *start* of the
symbol. Example: for `foo(bar)` where `foo` begins at column 5, pass
`character=5`.

For repository architecture analysis, start with `glob` to map packages and
entry points, then use `workspace_symbol`/`document_symbol` to identify central
symbols, and `find_references` or call hierarchy to verify dependency direction.
Read only the files needed to confirm that semantic map.

If the tool reports no server for an extension, no language server is
configured for that language — fall back to `grep`.
