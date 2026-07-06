from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.workflows.citations import (
    CitationFailure,
    CitationSpec,
    apply_citation_report,
    verify_citations,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "auth.py").write_text("def login():\n    return token\n")
    (tmp_path / "models.py").write_text(
        "class User:\n    pass\n\nclass Admin:\n    pass\n"
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "util.py").write_text("def helper():\n    return 1\n")
    return tmp_path


def _spec(**kwargs: object) -> CitationSpec:
    base: dict[str, object] = {"items_path": "findings", "path_field": "file"}
    base.update(kwargs)
    return CitationSpec.model_validate(base)


def test_all_citations_valid(repo: Path) -> None:
    output = {
        "findings": [
            {"file": "auth.py", "line": 1, "snippet": "def login()"},
            {"file": "models.py", "line": "2-3", "snippet": "class Admin"},
        ]
    }
    report = verify_citations(
        repo, output, _spec(line_field="line", snippet_field="snippet")
    )
    assert report.passed
    assert report.items_checked == 2
    assert report.items_verified == 2
    assert report.dropped_indices == []


def test_missing_path_dropped(repo: Path) -> None:
    output = {"findings": [{"file": "auth.py"}, {"file": "nonexistent.py"}]}
    report = verify_citations(repo, output, _spec())
    assert not report.passed
    assert report.items_verified == 1
    assert report.dropped_indices == [1]
    assert report.violations[0].category == "path"


def test_line_out_of_range_dropped(repo: Path) -> None:
    output = {"findings": [{"file": "auth.py", "line": 999}]}
    report = verify_citations(repo, output, _spec(line_field="line"))
    assert not report.passed
    assert report.dropped_indices == [0]
    assert report.violations[0].category == "line"


def test_snippet_absent_dropped(repo: Path) -> None:
    output = {"findings": [{"file": "auth.py", "quote": "not in the file"}]}
    report = verify_citations(repo, output, _spec(snippet_field="quote"))
    assert not report.passed
    assert report.dropped_indices == [0]
    assert report.violations[0].category == "snippet"


def test_require_all_false_skips_missing_fields(repo: Path) -> None:
    output = {"findings": [{"file": "auth.py"}, {"title": "no file field at all"}]}
    spec = _spec(require_all=False)
    report = verify_citations(repo, output, spec)
    assert report.passed
    assert report.items_verified == 1


def test_path_escape_blocked(repo: Path) -> None:
    output = {"findings": [{"file": "../../../etc/passwd"}]}
    report = verify_citations(repo, output, _spec())
    assert not report.passed
    assert report.dropped_indices == [0]


def test_items_path_missing_returns_empty_pass(repo: Path) -> None:
    output = {"results": []}
    report = verify_citations(repo, output, _spec())
    assert report.passed
    assert report.items_checked == 0


def test_line_range_parsed(repo: Path) -> None:
    output = {"findings": [{"file": "models.py", "line": "1-4"}]}
    report = verify_citations(repo, output, _spec(line_field="line"))
    assert report.passed
    assert report.items_verified == 1


def test_dotted_items_path(repo: Path) -> None:
    output = {"results": {"critical": [{"file": "auth.py"}]}}
    spec = CitationSpec.model_validate({
        "items_path": "results.critical",
        "path_field": "file",
    })
    report = verify_citations(repo, output, spec)
    assert report.passed
    assert report.items_verified == 1


def test_apply_drops_bad_keeps_good(repo: Path) -> None:
    output = {
        "findings": [
            {"file": "auth.py", "line": 1},
            {"file": "nope.py", "line": 1},
            {"file": "models.py", "line": 1},
        ]
    }
    spec = _spec(line_field="line")
    report = verify_citations(repo, output, spec)
    result = apply_citation_report(output, report, spec)
    assert isinstance(result, dict)
    assert len(result["findings"]) == 2
    assert result["findings"][0]["file"] == "auth.py"
    assert result["findings"][1]["file"] == "models.py"
    assert result["citation_report"]["items_verified"] == 2


def test_apply_strict_returns_failure(repo: Path) -> None:
    output = {"findings": [{"file": "nope.py"}]}
    spec = _spec(strict=True)
    report = verify_citations(repo, output, spec)
    result = apply_citation_report(output, report, spec)
    assert isinstance(result, CitationFailure)
    assert not result
    assert result["report"]["items_checked"] == 1


def test_apply_all_passed_attaches_report(repo: Path) -> None:
    output = {"findings": [{"file": "auth.py"}]}
    spec = _spec()
    report = verify_citations(repo, output, spec)
    result = apply_citation_report(output, report, spec)
    assert isinstance(result, dict)
    assert result["findings"] == [{"file": "auth.py"}]
    assert result["citation_report"]["passed"] is True


def test_citation_failure_falsy_in_filter(repo: Path) -> None:
    output = {"findings": [{"file": "nope.py"}]}
    spec = _spec(strict=True)
    report = verify_citations(repo, output, spec)
    failure = apply_citation_report(output, report, spec)
    results = [{"keep": "good"}, failure]
    kept = [r for r in results if r]
    assert len(kept) == 1
