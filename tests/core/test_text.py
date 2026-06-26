from __future__ import annotations

from vibe.core.utils.text import locate_edit_matches, snippet_start_line


class TestSnippetStartLine:
    def test_finds_line_number(self) -> None:
        assert snippet_start_line("a\nb\nc\nd\n", "c") == 3

    def test_first_line(self) -> None:
        assert snippet_start_line("hello\nworld", "hello") == 1

    def test_multiline_snippet(self) -> None:
        assert snippet_start_line("a\nb\nc", "\nb\n") == 2

    def test_first_occurrence_when_repeated(self) -> None:
        assert snippet_start_line("x\nx\nx", "x") == 1

    def test_leading_newline_anchors_first_content_line(self) -> None:
        assert snippet_start_line("bar\nx\nbar", "\nbar") == 3

    def test_returns_none_when_exact_snippet_absent(self) -> None:
        assert snippet_start_line("a\nb\nfoo", "foo\n") is None

    def test_not_found(self) -> None:
        assert snippet_start_line("hello\nworld", "missing") is None

    def test_blank_snippet(self) -> None:
        assert snippet_start_line("hello", "\n") is None


class TestLocateEditMatches:
    def _spans(self, content: str, old: str, **kw: object) -> list[tuple[int, int]]:
        replace_all = bool(kw.get("replace_all", False))
        return locate_edit_matches(content, old, replace_all=replace_all)

    def test_exact_single(self) -> None:
        spans = self._spans("hello world", "world")
        assert spans == [(6, 11)]

    def test_exact_multiple_returns_all(self) -> None:
        spans = self._spans("aaa bbb aaa\n", "aaa")
        assert spans == [(0, 3), (8, 11)]

    def test_exact_not_found_returns_empty(self) -> None:
        assert self._spans("hello", "missing") == []

    def test_trailing_whitespace_tolerated(self) -> None:
        # File has trailing spaces; model's old_string omits them.
        spans = self._spans("line one   \nline two\n", "line one\nline two")
        assert len(spans) == 1
        start, end = spans[0]
        # Splice on real bytes must remove the trailing spaces too.
        content = "line one   \nline two\n"
        spliced = content[:start] + "X" + content[end:]
        assert spliced == "X\n"

    def test_smart_quotes_tolerated(self) -> None:
        # File uses curly quotes; model's old_string uses straight quotes.
        spans = self._spans("say \u201chi\u201d now", 'say "hi" now')
        assert len(spans) == 1
        content = "say \u201chi\u201d now"
        start, end = spans[0]
        spliced = content[:start] + "DONE" + content[end:]
        assert spliced == "DONE"

    def test_nbsp_and_dash_tolerated(self) -> None:
        spans = self._spans("a\u00a0\u2013 b", "a - b")
        assert len(spans) == 1

    def test_fuzzy_with_replace_all_replaces_all(self) -> None:
        spans = self._spans("foo  \nbar\nfoo  \n", "foo", replace_all=True)
        assert len(spans) == 2

    def test_fuzzy_empty_old_returns_empty(self) -> None:
        assert self._spans("hello", "") == []

    def test_no_false_match(self) -> None:
        assert self._spans("completely different text", "hello") == []
