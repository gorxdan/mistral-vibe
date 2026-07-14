Use `bash` for one-off shell commands. Each call is a fresh, stateless Bash
subprocess; the user's login shell does not select the command language.
Fish in a proven executable position or recognized literal execution carrier is
rejected because Fish syntax cannot be validated by the Bash parser. Opaque or
dynamic executable positions require explicit user approval. Exact standalone
`fish -v` and `fish --version` queries are approval-only diagnostic exceptions.
Nested shell or language-interpreter processes, leading environment assignments,
and package/Git forms that can launch child commands also require explicit user
approval, even when Bash is configured `permission = "always"`.

Core-managed automated execution uses a trusted system PATH and minimal
noninteractive environment. On Linux, the executable must be a root-owned regular
file without set-ID or group/world-write bits and without write access for the
current non-root user; its lexical and resolved ancestry has the same root-control
and write restrictions. Project, user, or unprovable executables require explicit
user approval. Exact standalone `git log`, `show`, `blame`, and `grep` calls use a
hardened profile that disables hooks, pagers, external diff/text conversion, and
signature helpers. `git diff`, `status`, and wrapped, composed, redirected, or
other Git forms require explicit user approval.

**Prefer dedicated tools:** `cat`/`head`/`tail` → `read` | content search → `grep` | find-by-name → `glob` | create file → `write_file` | in-place edit → `edit`.

**Bash is for:** system info, `ls`, git, process/network/package/env checks, file metadata (`stat`/`wc`).

Package acquisition, dependency-graph changes, and recognized test/build commands
require explicit user approval. A Safety Judge verdict, auto-approve mode,
`permission = "always"`, or a stored/session wildcard cannot substitute for that
approval. `dotnet test --no-restore` still executes project code and still needs
approval; use `--no-restore` only after an approved restore succeeds.

Run verification commands directly. Do not hide their status with `|`, `|&`,
`||`, `;`, `&`, `!`, command substitution, embedded newlines, or nested-shell
equivalents. Bash already caps and persists output, so never pipe a check through
`head` or `tail`.

**Timeout:** set `timeout` for slow commands. **Never `sleep` to poll** — use `schedule`; short `sleep 1`/`2` only to let a service start.
