from __future__ import annotations

from typing import Any

import pytest

from vibe.cli.textual_ui.widgets.workflow_save_app import WorkflowSaveApp, _default_name


class _FakeInput:
    def __init__(self, value: str) -> None:
        self.value = value
        self.cursor_position = 0
        self._focused = False

    def focus(self) -> None:
        self._focused = True

    def update(self, _text: str) -> None:
        # Stands in for the location label's update() during toggle tests.
        return None


class TestDefaultName:
    def test_strips_wf_prefix(self) -> None:
        assert _default_name("wf-3") == "workflow-3"

    def test_no_prefix_kept(self) -> None:
        assert _default_name("custom-run") == "workflow-custom-run"


class TestWorkflowSaveAppMessages:
    def test_save_confirmed_carries_fields(self) -> None:
        msg = WorkflowSaveApp.SaveConfirmed(
            run_id="wf-1", script_source="src", name="audit", location="user"
        )
        assert msg.run_id == "wf-1"
        assert msg.script_source == "src"
        assert msg.name == "audit"
        assert msg.location == "user"

    def test_cancelled_carries_run_id(self) -> None:
        msg = WorkflowSaveApp.Cancelled(run_id="wf-2")
        assert msg.run_id == "wf-2"


class TestWorkflowSaveAppBindings:
    def _keys(self) -> list[str]:
        keys: list[str] = []
        for b in WorkflowSaveApp.BINDINGS:
            keys.extend(b.key.split(","))
        return keys

    def test_has_enter_confirm(self) -> None:
        assert "enter" in self._keys()

    def test_has_escape_cancel(self) -> None:
        assert "escape" in self._keys()

    def test_has_tab_toggle(self) -> None:
        assert "tab" in self._keys()


class TestWorkflowSaveAppActions:
    def _make(
        self, monkeypatch: pytest.MonkeyPatch, name_value: str = "my-audit"
    ) -> tuple[WorkflowSaveApp, _FakeInput]:
        app = WorkflowSaveApp(run_id="wf-3", script_source="async def main(): return 1")
        fake_input = _FakeInput(name_value)
        # query_one returns the fake input for the name field; actions call it
        # instead of a real (app-bound) Textual Input widget.
        monkeypatch.setattr(app, "query_one", lambda *a, **k: fake_input)
        return app, fake_input

    def test_confirm_posts_save_confirmed_with_edited_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _ = self._make(monkeypatch, name_value="my-audit")
        posted: list[Any] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_confirm()

        assert len(posted) == 1
        assert isinstance(posted[0], WorkflowSaveApp.SaveConfirmed)
        assert posted[0].name == "my-audit"
        assert posted[0].location == "project"

    def test_toggle_location_switches_project_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _ = self._make(monkeypatch)
        assert app._location == "project"
        # query_one returns the fake input; toggle also queries the location
        # label to update() it — fake input's no-op update stands in.
        app.action_toggle_location()
        assert app._location == "user"
        app.action_toggle_location()
        assert app._location == "project"

    def test_confirm_uses_default_when_name_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _ = self._make(monkeypatch, name_value="   ")
        posted: list[Any] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_confirm()

        assert posted[0].name == _default_name("wf-3")

    def test_cancel_posts_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app, _ = self._make(monkeypatch)
        posted: list[Any] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_cancel()

        assert len(posted) == 1
        assert isinstance(posted[0], WorkflowSaveApp.Cancelled)
        assert posted[0].run_id == "wf-3"
