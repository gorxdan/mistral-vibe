from __future__ import annotations

from vibe.core.session.title_format import (
    MAX_TITLE_LENGTH,
    MentionSegment,
    TextSegment,
    format_session_title,
)
from vibe.core.utils.matching import name_matches
from vibe.core.utils.slug import create_slug
from vibe.core.utils.tokens import approx_token_count, truncate_middle_to_tokens

# --------------------------------------------------------------------------- #
# tokens                                                                      #
# --------------------------------------------------------------------------- #


def test_approx_token_count_ceil_division() -> None:
    assert approx_token_count("") == 0
    assert approx_token_count("abc") == 1
    assert approx_token_count("abcde") == 2  # ceil(5/4)


def test_truncate_middle_short_text_unchanged() -> None:
    assert truncate_middle_to_tokens("short", 100) == "short"


def test_truncate_middle_zero_tokens_returns_empty() -> None:
    assert truncate_middle_to_tokens("text", 0) == ""


def test_truncate_middle_long_text_inserts_marker() -> None:
    text = "x" * 200
    result = truncate_middle_to_tokens(text, 10)
    assert "[... truncated ...]" in result
    assert result.startswith("x")
    assert result.endswith("x")


# --------------------------------------------------------------------------- #
# matching                                                                    #
# --------------------------------------------------------------------------- #


def test_name_matches_glob_wildcard() -> None:
    assert name_matches("bash_tool", ["bash_*"]) is True
    assert name_matches("other", ["bash_*"]) is False


def test_name_matches_regex_prefix() -> None:
    assert name_matches("serena_tool", ["re:serena.*"]) is True
    assert name_matches("other", ["re:serena.*"]) is False


def test_name_matches_invalid_regex_skipped() -> None:
    # Invalid regex pattern is skipped, not raised
    assert name_matches("x", ["re:[invalid"]) is False


def test_name_matches_empty_and_blank_patterns_skipped() -> None:
    assert name_matches("x", ["", "  "]) is False
    assert name_matches("x", []) is False


def test_name_matches_case_insensitive() -> None:
    assert name_matches("Bash_Tool", ["bash_*"]) is True


# --------------------------------------------------------------------------- #
# slug                                                                        #
# --------------------------------------------------------------------------- #


def test_create_slug_returns_multi_word_string() -> None:
    slug = create_slug()
    parts = slug.split("-")
    assert len(parts) >= 2
    assert all(part.isalpha() for part in parts)


def test_create_slug_is_lowercase() -> None:
    slug = create_slug()
    assert slug == slug.lower()


# --------------------------------------------------------------------------- #
# title_format                                                                #
# --------------------------------------------------------------------------- #


def test_format_title_text_segments_joined() -> None:
    result = format_session_title([TextSegment("Hello"), TextSegment(" World")])
    assert result == "Hello World"


def test_format_title_collapses_whitespace() -> None:
    result = format_session_title([TextSegment("  hello   world  ")])
    assert result == "hello world"


def test_format_title_empty_returns_empty() -> None:
    assert format_session_title([TextSegment("   ")]) == ""


def test_format_title_truncates_with_ellipsis() -> None:
    long = TextSegment("x" * (MAX_TITLE_LENGTH + 20))
    result = format_session_title([long])
    assert len(result) == MAX_TITLE_LENGTH + 1  # +1 for ellipsis char
    assert result.endswith("…")


def test_format_title_mention_with_line_range() -> None:
    seg = MentionSegment(name="file.py", start_line=10, end_line=20)
    assert format_session_title([seg]) == "@file.py:10-20"


def test_format_title_mention_with_start_only() -> None:
    seg = MentionSegment(name="file.py", start_line=5)
    assert format_session_title([seg]) == "@file.py:5"


def test_format_title_mention_name_only() -> None:
    seg = MentionSegment(name="module")
    assert format_session_title([seg]) == "@module"


def test_format_title_mixed_segments() -> None:
    result = format_session_title([
        TextSegment("Fix bug in "),
        MentionSegment(name="auth.py", start_line=42),
    ])
    assert "@auth.py:42" in result
    assert "Fix bug" in result
