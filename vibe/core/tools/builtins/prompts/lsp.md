# LSP Tool

Query language servers for code intelligence. Use this instead of grepping when you
need semantic understanding: definitions, references, type info, call graphs.

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

`line` and `character` are **1-based**. The cursor sits at the start of the
symbol. Example: for `foo(bar)` where `foo` begins at column 5, pass
`character=5`.

## When to use

Prefer this tool over `grep` for: finding definitions, resolving references,
inspecting type signatures, or building a call graph. It is faster and more
accurate than textual search for code with overloaded names, imports, or
generated code.

If the tool reports no server for an extension, the project has not configured
a language server for that language — fall back to `grep`.
