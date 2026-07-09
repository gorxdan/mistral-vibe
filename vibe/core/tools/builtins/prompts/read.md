Use `read` for file contents with line numbers (encoding-safe). Reading registers the file for `edit`.

- Default: first 2000 lines. Output >50KB errors — use `offset` (1-based) + `limit`.
- Line format: `     N→content` (1-indexed). Prefer `grep` over sequential chunk reads.
- Cap: do not call `read` more than 3 times on the same file without responding to the user.
- Do not read: model weights (`.bin`/`.safetensors`/`.pt`/`.gguf`), binaries, or entire training-run trees unless the user names a specific file.
