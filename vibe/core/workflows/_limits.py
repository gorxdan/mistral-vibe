from __future__ import annotations

# Paid workflow fan-out must be bounded even when a generated workflow omits
# explicit limits. Callers can still opt into larger values on WorkflowRuntime.
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_MAX_AGENTS = 32
DEFAULT_BUDGET_TOTAL = 500_000
DEFAULT_ISOLATED_MAX_TURNS = 60
