from __future__ import annotations

import time

from vibe.core.config import ProjectContextConfig
from vibe.core.system_prompt import (
    _GIT_STATUS_TTL_S,
    ProjectContextProvider,
    _git_status_cache,
)


def _provider() -> tuple[ProjectContextProvider, dict[str, int]]:
    provider = ProjectContextProvider(ProjectContextConfig(), ".")
    calls = {"n": 0}

    def fake_fetch() -> str:
        calls["n"] += 1
        return f"status-{calls['n']}"

    provider._fetch_git_status = fake_fetch  # type: ignore[method-assign]
    _git_status_cache.clear()
    return provider, calls


def test_status_cached_within_ttl() -> None:
    provider, calls = _provider()
    first = provider.get_git_status()
    second = provider.get_git_status()
    assert first == second
    assert calls["n"] == 1, "second call served from cache"


def test_status_refetched_after_ttl_expires() -> None:
    provider, calls = _provider()
    provider.get_git_status()
    key = provider.root_path
    stamp, value = _git_status_cache[key]
    _git_status_cache[key] = (stamp - _GIT_STATUS_TTL_S - 1, value)  # force expiry

    refreshed = provider.get_git_status()
    assert calls["n"] == 2, "expired entry triggers refetch"
    assert refreshed == "status-2"


def test_cache_keyed_per_root() -> None:
    _git_status_cache.clear()
    p1 = ProjectContextProvider(ProjectContextConfig(), ".")
    p1._fetch_git_status = lambda: "root1"  # type: ignore[method-assign]
    p2 = ProjectContextProvider(ProjectContextConfig(), "/")
    p2._fetch_git_status = lambda: "root2"  # type: ignore[method-assign]
    assert p1.get_git_status() == "root1"
    assert p2.get_git_status() == "root2"
    assert time.monotonic() > 0  # sanity


def test_parse_git_log_preserves_conventional_commit_scope() -> None:
    log = (
        "e9b9920 perf(prompts): Vercel-style compression of 14 surfaces\n"
        "ee4c21e fix(memory): derive manage_memory title from body"
    )
    parsed = ProjectContextProvider._parse_git_log(log)
    assert parsed == [
        "e9b9920 perf(prompts): Vercel-style compression of 14 surfaces",
        "ee4c21e fix(memory): derive manage_memory title from body",
    ]


def test_parse_git_log_strips_trailing_pr_number_only() -> None:
    parsed = ProjectContextProvider._parse_git_log(
        "6bedf27 v2.18.0 (#843)\n725d3a5 fix(x): handle null (edge case)"
    )
    assert parsed == ["6bedf27 v2.18.0", "725d3a5 fix(x): handle null (edge case)"]
