from __future__ import annotations

# 1:1 length-preserving typographic substitutions applied during fuzzy edit
# matching. Each key maps to exactly one replacement char so normalized offsets
# stay monotonically mappable back to source offsets.
_EDIT_SMART_MAP: dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\xa0": " ",
    "\u2013": "-",
    "\u2014": "-",
}


def first_sentence(text: str, max_chars: int) -> str:
    """Shorten text to its first sentence, falling back to a max_chars
    word-boundary cut. Keeps the lead intent the model needs to choose a tool
    while dropping the elaboration that bloats a small-window prompt.
    """
    stripped = text.strip()
    first_period = stripped.find(". ")
    if 0 <= first_period < max_chars:
        return stripped[: first_period + 1]
    if len(stripped) <= max_chars:
        return stripped
    head = stripped[:max_chars]
    space = head.rfind(" ")
    return (head[:space] if space > 0 else head).rstrip() + "…"


def line_contexts(content: str, snippet: str) -> list[tuple[int, str, str]]:
    """``(start_line, prefix, suffix)`` per match, completing it to whole lines.

    The prefix/suffix expand each match to its full source lines so a caller can
    diff whole lines instead of just the changed snippet. ``start_line`` is the
    file line of the match's first row, keeping the diff gutter offset correct.
    """
    if not snippet.strip("\n"):
        return []
    results: list[tuple[int, str, str]] = []
    pos = content.find(snippet)
    while pos != -1:
        start_line = content.count("\n", 0, pos) + 1
        line_start = content.rfind("\n", 0, pos) + 1
        prefix = content[line_start:pos]
        match_end = pos + len(snippet)
        # A match ending on a line boundary has no partial trailing line.
        if match_end > 0 and content[match_end - 1] == "\n":
            suffix = ""
        else:
            line_end = content.find("\n", match_end)
            if line_end == -1:
                line_end = len(content)
            suffix = content[match_end:line_end]
        results.append((start_line, prefix, suffix))
        pos = content.find(snippet, match_end)
    return results


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Normalize text for fuzzy matching, returning a source-position map.

    Two tolerances are folded into one pass:
    * trailing whitespace (spaces/tabs) on each line is dropped, so a model that
      omits or differs on trailing spaces still matches; and
    * typographic chars (smart quotes, NBSP, en/em-dash) are folded to their
      ASCII equivalents (1:1, length-preserving).

    Returns ``(normalized, positions)`` where ``positions[j]`` is the source
    index in ``s`` of the first original char contributing to ``normalized[j]``.
    The map lets a caller translate a match span found in normalized space back
    to an exact span in the original bytes, so the splice never corrupts.
    """
    out_chars: list[str] = []
    out_pos: list[int] = []
    i = 0
    n = len(s)
    while i < n:
        nl = s.find("\n", i)
        line_end = n if nl == -1 else nl
        line = s[i:line_end]
        stripped = line.rstrip()
        for k, c in enumerate(stripped):
            out_chars.append(_EDIT_SMART_MAP.get(c, c))
            out_pos.append(i + k)
        if nl != -1:
            out_chars.append("\n")
            out_pos.append(nl)
            i = nl + 1
        else:
            i = n
    return "".join(out_chars), out_pos


def locate_edit_matches(
    content: str, old_string: str, *, replace_all: bool
) -> list[tuple[int, int]]:
    """Find spans in ``content`` (original bytes) to replace with ``new_string``.

    Exact matching is tried first and preferred. When exact matching fails, a
    fuzzy fallback tolerates trailing-whitespace and typographic-char
    differences. Every returned ``(start, end)`` span indexes ``content`` itself
    — the splice always replaces real bytes, so normalization never silently
    alters the result.

    Returns an empty list when nothing matches. For ``replace_all=False`` the
    fuzzy fallback only applies when it yields exactly one match; an ambiguous
    fuzzy match is reported as no match so the caller surfaces a clear error.
    """
    if not old_string:
        return []
    # Pass 1: exact (current behavior, unchanged). Return every exact span so the
    # caller can detect the multi-match-without-replace_all case itself.
    if old_string in content:
        return _all_exact_spans(content, old_string)

    # Pass 2: normalized fuzzy fallback.
    norm_content, posmap_content = _normalize_with_map(content)
    norm_old, _ = _normalize_with_map(old_string)
    if not norm_old:
        return []
    count = norm_content.count(norm_old)
    if count == 0:
        return []
    if not replace_all and count > 1:
        # Ambiguous under fuzzy tolerance — refuse rather than guess.
        return []
    spans: list[tuple[int, int]] = []
    search_from = 0
    while True:
        a = norm_content.find(norm_old, search_from)
        if a == -1:
            break
        b = a + len(norm_old)
        start = posmap_content[a]
        end = posmap_content[b - 1] + 1
        spans.append((start, end))
        search_from = b
    return spans


def _all_exact_spans(content: str, old_string: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    search_from = 0
    while True:
        idx = content.find(old_string, search_from)
        if idx == -1:
            break
        spans.append((idx, idx + len(old_string)))
        search_from = idx + len(old_string)
    return spans
