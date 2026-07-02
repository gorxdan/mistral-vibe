# web_search — extended guide

**Query:** resolve relative-time terms ("latest", "today", "this week") to actual dates; be specific, use concrete terms.

**When to use** | **When NOT to use**
--- | ---
Recent events, or user explicitly asks to search | General programming concepts/patterns (use training knowledge)
Docs/APIs/libraries possibly updated since training cutoff | Searching the local codebase (use `grep`/file search)
Verifying outdatable facts (versions, deprecations, breaking changes) | Static reference unlikely to change (math, algorithms, syntax)
Specific error messages with known solutions | Info you're confident about and unlikely to have changed
A library/framework/version you're not familiar with |

**Using results:** stay critical — web content may be outdated, wrong, or misleading; cross-reference multiple sources; prefer official documentation. Always cite your sources; never present a fact recalled from memory as if it were web-sourced — if you did not fetch it, say so.
