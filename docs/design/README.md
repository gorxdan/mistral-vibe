# Architecture Gap-Closure Design Specs

Design specs for closing the remaining gaps between vibe/chaton and the
Claude Code reference architecture. Each spec was produced by a deep design
pass over the real codebase and then adversarially verified against it
(feasibility, integration points, wrong assumptions). All six came back
**`sound_with_fixes`** — feasible, with corrections folded into each doc's
*Adversarial verification* section.

These are the gaps **not** already closed in the resilience/safety work landed
this cycle (read-parallel/write-serial tools, tool-result budget, context-overflow
recovery, model failover, Retry-After, the LLM safety judge, `extra_body` thinking
control, `cached_tokens` visibility).

## The six specs

| # | Spec | Closes (vs Claude Code) | Effort | Verdict |
|---|------|--------------------------|--------|---------|
| 1 | [output-escalation](output-escalation.md) | Max-output-token escalation (3× retry on response-too-long) | **M** | sound_with_fixes |
| 2 | [compaction](compaction.md) | Multi-stage shaper pipeline (snip + microcompact) — the #1 gap | **L** | sound_with_fixes |
| 3 | [sandbox](sandbox.md) | Shell sandboxing (safety layer 5) — the #1 security gap | **L** | sound_with_fixes |
| 4 | [prompt-caching](prompt-caching.md) | Managed prompt caching + cache telemetry | **M** | sound_with_fixes |
| 5 | [memory](memory.md) | File-based auto-memory (LLM-selected, no embeddings) | **L** | sound_with_fixes |
| 6 | [hooks](hooks.md) | Hook breadth (6 new events) + plugin/bundle model | **XL** | sound_with_fixes |
| 7 | [subagent-isolation](subagent-isolation.md) | Default worktree isolation for write-capable agents (`task()` + workflow) | **M** | sound_with_fixes |

All six are **independent** (no cross-spec dependencies) — they can be built in
any order or in parallel.

## Recommended build order (value × effort × risk)

1. **output-escalation (M, low risk).** Smallest, mirrors the existing
   `ContextTooLongError`/`RateLimitError` self-heal arms in `_conversation_loop`.
   Completes the resilience set already started this cycle. Build first.
2. **compaction-pipeline (L).** The single biggest architectural distance — vibe
   has 1 shaper (full LLM summary) vs Claude's 5. Pure-local (no LLM), runs in the
   existing `MiddlewarePipeline` before AutoCompact. Highest architectural payoff.
3. **shell-sandbox (L).** The biggest *security* gap. Opt-in, default-off →
   byte-for-byte current behavior until enabled, so low blast radius. Wraps the
   bash subprocess as a layer after the permission gate + judge.
4. **prompt-caching (M).** Mostly telemetry + a thin opt-in escape hatch; the spec
   is honest that generic-path auto-caching already works — value is *protecting*
   the cacheable prefix + measuring hit rate. Good once the shaper pipeline (2)
   lands, since shaping edits history and busts cache.
5. **auto-memory (L).** Net-new product surface; build a standalone judge-style
   backend, never reuse `self.backend`.
6. **hooks-extensibility (XL).** Largest. Part A (6 new events) is a set of
   independent S-sized increments (SessionStart, SessionEnd, UserPromptSubmit,
   PreCompact, Stop, Notification); Part B (plugin/bundle manifest) is separable.
   Ship incrementally.

## Cross-cutting corrections the verification surfaced

These recurred across specs and should be treated as house rules:

- **Config sub-models** must extend `BaseSettings` and mirror existing nested
  blocks (`SafetyJudgeConfig`/`WorktreeConfig` at `_settings.py:211/241`), with
  per-leaf merge annotations in `vibe_schema.py` (`WithReplaceMerge()` like the
  siblings) — do **not** slap one `WithShallowMerge` on a nested model.
- **Feature backends** (memory selector, judge) build a **standalone** backend
  (`backend_cls(provider=..., timeout=...)`), never `self.backend`, so their
  failures can't trigger `_switch_to_fallback_model` or emergency compaction.
- **Compaction interactions:** anything that overrides per-turn state
  (`_max_output_override`, injected memories) must not leak into the
  `compact()` LLM call (`agent_loop.py:2168`), and must be re-applied after
  `compact()` rebuilds the message list / `_reset_session()` — or explicitly
  documented to vanish.
- **Tool auto-discovery:** new builtin tools go in `vibe/core/tools/builtins/`
  and are auto-discovered by `ToolManager` (rglob); gate availability via a
  classmethod, don't hand-register.

Each spec's own *Adversarial verification* section has the gap-specific fixes.
