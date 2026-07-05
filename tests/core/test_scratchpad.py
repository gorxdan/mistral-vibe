from __future__ import annotations

import os
import tempfile

import vibe.core.scratchpad as scratchpad
from vibe.core.scratchpad import (
    SCRATCHPAD_PREFIX,
    get_scratchpad_dir,
    init_scratchpad,
    is_foreign_scratchpad_path,
    is_scratchpad_path,
)


class TestScratchpadCleanup:
    def test_cleanup_all_removes_active_dirs(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        scratchpad._active_scratchpads.clear()
        d = init_scratchpad("cleanup-sess")
        assert d is not None and d.is_dir()
        scratchpad.cleanup_all_scratchpads()
        assert not d.exists()
        assert not scratchpad._active_scratchpads

    def test_gc_reclaims_stale_keeps_fresh_and_active(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        scratchpad._active_scratchpads.clear()
        stale = tmp_path / "vibe-scratchpad-old-x"
        stale.mkdir()
        os.utime(stale, (1, 1))  # ancient mtime
        fresh = tmp_path / "vibe-scratchpad-new-y"
        fresh.mkdir()  # current mtime
        active = init_scratchpad("live-sess")  # in tmp_path, registered active

        scratchpad.gc_stale_scratchpads(max_age_s=3600)

        assert not stale.exists()  # past cutoff -> reclaimed
        assert fresh.exists()  # recent -> kept
        assert active is not None and active.exists()  # active -> never touched
        scratchpad.cleanup_all_scratchpads()


class TestInitScratchpad:
    def test_creates_directory(self):
        result = init_scratchpad("test-session")
        assert result is not None
        assert result.is_dir()

    def test_idempotent_same_session(self):
        first = init_scratchpad("session-1")
        second = init_scratchpad("session-1")
        assert first == second

    def test_different_sessions_get_different_dirs(self):
        first = init_scratchpad("session-1")
        second = init_scratchpad("session-2")
        assert first != second

    def test_session_id_in_dir_name(self):
        result = init_scratchpad("abcdef123456")
        assert result is not None
        assert "abcdef12" in result.name

    def test_sets_module_state(self):
        init_scratchpad("test-session")
        assert get_scratchpad_dir("test-session") is not None


class TestGetScratchpadDir:
    def test_none_for_unknown_session(self):
        assert get_scratchpad_dir("nonexistent") is None

    def test_returns_path_after_init(self):
        path = init_scratchpad("test-session")
        assert get_scratchpad_dir("test-session") == path


class TestIsScratchpadPath:
    def test_false_when_not_initialized(self):
        assert not is_scratchpad_path("/tmp/anything")

    def test_true_for_file_inside(self):
        sp = init_scratchpad("test-session")
        assert sp is not None
        assert is_scratchpad_path(str(sp / "file.txt"))

    def test_true_for_nested_file(self):
        sp = init_scratchpad("test-session")
        assert sp is not None
        assert is_scratchpad_path(str(sp / "subdir" / "file.txt"))

    def test_true_for_dir_itself(self):
        sp = init_scratchpad("test-session")
        assert sp is not None
        assert is_scratchpad_path(str(sp))

    def test_true_across_sessions(self):
        sp1 = init_scratchpad("session-1")
        sp2 = init_scratchpad("session-2")
        assert sp1 is not None and sp2 is not None
        assert is_scratchpad_path(str(sp1 / "file.txt"))
        assert is_scratchpad_path(str(sp2 / "file.txt"))

    def test_false_for_outside_path(self):
        init_scratchpad("test-session")
        assert not is_scratchpad_path("/etc/passwd")

    def test_false_for_traversal_attack(self):
        sp = init_scratchpad("test-session")
        assert sp is not None
        traversal = str(sp / ".." / ".." / ".." / "etc" / "passwd")
        assert not is_scratchpad_path(traversal)

    def test_false_for_sibling_directory(self):
        sp = init_scratchpad("test-session")
        assert sp is not None
        sibling = str(sp.parent / "other-dir" / "file.txt")
        assert not is_scratchpad_path(sibling)


class TestIsForeignScratchpadPath:
    def test_true_for_other_session_scratchpad(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        scratchpad._active_scratchpads.clear()
        mine = init_scratchpad("my-session")
        foreign = tmp_path / f"{SCRATCHPAD_PREFIX}other-session-x"
        foreign.mkdir()
        assert mine is not None
        assert is_foreign_scratchpad_path(str(foreign / "bg" / "asub-1.log"))

    def test_false_for_own_scratchpad(self):
        mine = init_scratchpad("my-session")
        assert mine is not None
        assert not is_foreign_scratchpad_path(str(mine / "file.txt"))

    def test_false_for_non_scratchpad_path(self):
        assert not is_foreign_scratchpad_path("/etc/passwd")

    def test_false_for_unrelated_tmp_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        unrelated = tmp_path / "not-a-scratchpad"
        unrelated.mkdir()
        assert not is_foreign_scratchpad_path(str(unrelated / "f.txt"))
