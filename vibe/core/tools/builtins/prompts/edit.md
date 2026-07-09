Use `edit` for exact string replacements in existing files.

**Hard rules:**
- **Read first** — runtime-enforced; unread files refuse edit. Re-read after a failed match; do not guess variants.
- `old_string` must match on-disk text exactly (whitespace/indent). From `read` output, strip the `     N→` line-number prefix — it is not in the file.
- Unique match required unless `replace_all=true`. Prefer more context over blind replace-all.
- Empty `old_string` is invalid — use `write_file` for new files.
- Prefer edit over rewrite. No emojis in files unless the user asks.
