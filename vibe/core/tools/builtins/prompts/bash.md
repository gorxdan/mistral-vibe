Use the `bash` tool to run one-off shell commands. Each command runs independently in a fresh, stateless environment.

**Timeout:** the `timeout` argument caps how long a command runs before it's killed; omit it (or `None`) to use the config default. If a command times out and is legitimately slow, raise `timeout`.

**Use the dedicated tool instead of bash for these:**

| Instead of | Use |
| --- | --- |
| `cat` / `head` / `tail` / `sed -n` / `less` / `more` | `read` (with `offset`/`limit` for ranges) |
| `grep` / `rg` / `ag` / `ack` / `find` / `locate` | `grep` |
| `echo > file` / new file | `write_file` |
| `echo >> file` / `sed -i` / `awk` / any in-place edit | `edit` (read first) |

**Appropriate bash uses:** system info (`pwd`, `whoami`, `uname -a`), directory listings (`ls -la`), git (`git status`, `git log --oneline -10`, `git diff`), process/network/package/env checks, file metadata (`stat`, `file`, `wc -l`).

**Never `sleep` to wait, poll, or track an interval.** A long `sleep` blocks the turn, hits the timeout, and is denied. To run something later or repeatedly, use the `schedule` tool. A short `sleep 1`/`sleep 2` to let a service start is fine.
