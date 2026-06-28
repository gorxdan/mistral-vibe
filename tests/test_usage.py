from __future__ import annotations

import time

from vibe.core.types import LLMUsage
from vibe.core.usage import UsageRecord, UsageRecorder, summarize


def _rec(
    *,
    provider: str,
    model: str,
    prompt: int,
    completion: int,
    cached: int = 0,
    ts: float,
    session: str = "s1",
    cost: float = 0.0,
) -> UsageRecord:
    return UsageRecord.from_usage(
        timestamp=ts,
        provider=provider,
        model=model,
        usage=LLMUsage(
            prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached
        ),
        cost_usd=cost,
        duration_s=1.0,
        session_id=session,
    )


class TestUsageRecorder:
    def test_append_and_read_roundtrip(self, tmp_path):
        path = tmp_path / "usage.jsonl"
        rec = UsageRecorder(path=path)
        now = time.time()
        rec.record(
            _rec(provider="mistral", model="m1", prompt=100, completion=50, ts=now)
        )
        rec.record(
            _rec(provider="openai", model="gpt", prompt=200, completion=10, ts=now)
        )
        records = rec.read_all()
        assert len(records) == 2
        assert records[0].provider == "mistral"
        assert records[1].provider == "openai"

    def test_read_missing_file_returns_empty(self, tmp_path):
        rec = UsageRecorder(path=tmp_path / "absent.jsonl")
        assert rec.read_all() == []

    def test_read_skips_unparsable_lines(self, tmp_path):
        path = tmp_path / "usage.jsonl"
        path.write_text("not json\n\n", encoding="utf-8")
        rec = UsageRecorder(path=path)
        rec.record(
            _rec(provider="mistral", model="m1", prompt=1, completion=1, ts=time.time())
        )
        records = rec.read_all()
        assert len(records) == 1
        assert records[0].provider == "mistral"

    def test_trim_drops_old_records_when_over_size(self, tmp_path):
        import vibe.core.usage._recorder as recmod

        path = tmp_path / "usage.jsonl"
        rec = UsageRecorder(path=path)
        # Seed two records: one stale (older than retention), one fresh.
        now = time.time()
        rec.record(
            _rec(
                provider="mistral", model="m1", prompt=1, completion=1, ts=now - 100_000
            )
        )
        rec.record(_rec(provider="mistral", model="m1", prompt=1, completion=1, ts=now))
        assert len(rec.read_all()) == 2

        # Force the trim thresholds and invoke the compactor directly.
        old_threshold = recmod._TRIM_BYTES
        old_retention = recmod._RETENTION_DAYS
        recmod._TRIM_BYTES = 1
        recmod._RETENTION_DAYS = 1 / 24  # one hour
        try:
            rec._maybe_trim_locked()
            surviving = rec.read_all()
            # The 100_000s-old record (~1.1 days) is dropped; the fresh one stays.
            assert len(surviving) == 1
            assert surviving[0].timestamp >= now - 1
        finally:
            recmod._TRIM_BYTES = old_threshold
            recmod._RETENTION_DAYS = old_retention


class TestSummarize:
    def test_breakdown_groups_by_provider_then_model(self):
        now = 1000.0
        records = [
            _rec(
                provider="mistral", model="large", prompt=1000, completion=500, ts=now
            ),
            _rec(provider="mistral", model="large", prompt=500, completion=100, ts=now),
            _rec(provider="openai", model="gpt-5", prompt=2000, completion=300, ts=now),
        ]
        summary = summarize(records, now=now)
        # mistral: 1500 prompt + 600 completion; openai: 2000 + 300
        assert summary.grand_total_tokens == 4400
        # openai has more total tokens -> listed first
        assert summary.providers[0].provider == "openai"
        assert summary.providers[1].provider == "mistral"
        mistral = summary.providers[1]
        assert len(mistral.models) == 1
        assert mistral.models[0].prompt_tokens == 1500
        assert mistral.calls == 2

    def test_windows_partition_by_time(self):
        now = 100_000.0
        records = [
            _rec(
                provider="p",
                model="m",
                prompt=10,
                completion=5,
                ts=now - 30,
                session="a",
            ),
            _rec(
                provider="p",
                model="m",
                prompt=10,
                completion=5,
                ts=now - 6000,
                session="b",
            ),
            _rec(
                provider="p",
                model="m",
                prompt=10,
                completion=5,
                ts=now - 200_000,
                session="c",
            ),
        ]
        summary = summarize(records, now=now)
        by_label = {w.label: w for w in summary.windows}
        assert by_label["Last hour"].calls == 1
        assert by_label["Last hour"].sessions == 1
        assert by_label["Last 24h"].calls == 2
        assert by_label["Last 7 days"].calls == 3
        assert by_label["Last 7 days"].sessions == 3

    def test_empty_records(self):
        summary = summarize([], now=1.0)
        assert summary.providers == []
        assert summary.grand_total_tokens == 0
        assert all(w.calls == 0 for w in summary.windows)
