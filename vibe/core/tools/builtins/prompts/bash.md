Use `bash` for one-off shell commands. Each call is a fresh, stateless subprocess.

**Prefer dedicated tools:** `cat`/`head`/`tail` → `read` | content search → `grep` | find-by-name → `glob` | create file → `write_file` | in-place edit → `edit`.

**Bash is for:** system info, `ls`, git, process/network/package/env checks, file metadata (`stat`/`wc`).

**Timeout:** set `timeout` for slow commands. **Never `sleep` to poll** — use `schedule`; short `sleep 1`/`2` only to let a service start.
