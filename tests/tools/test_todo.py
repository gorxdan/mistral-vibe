from __future__ import annotations

from vibe.core.tools.builtins.todo import (
    Todo,
    TodoConfig,
    TodoItem,
    TodoPriority,
    TodoState,
    TodoStatus,
    _should_nudge,
)


def _todo(i: int, content: str, status: TodoStatus = TodoStatus.COMPLETED) -> TodoItem:
    return TodoItem(
        id=str(i), content=content, status=status, priority=TodoPriority.MEDIUM
    )


def test_nudge_fires_on_three_completed_todos_without_verify_item() -> None:
    todos = [
        _todo(1, "Add config flag"),
        _todo(2, "Add profile"),
        _todo(3, "Wire prompt"),
    ]
    assert _should_nudge(todos, verification_enabled=True) is True


def test_nudge_suppressed_when_a_verify_item_is_present() -> None:
    todos = [
        _todo(1, "Add config flag"),
        _todo(2, "Add profile"),
        _todo(3, "Verify the implementation end-to-end"),
    ]
    assert _should_nudge(todos, verification_enabled=True) is False


def test_nudge_suppressed_below_min_size() -> None:
    todos = [_todo(1, "Add config flag"), _todo(2, "Add profile")]
    assert _should_nudge(todos, verification_enabled=True) is False


def test_nudge_suppressed_when_not_all_completed() -> None:
    todos = [
        _todo(1, "Add config flag"),
        _todo(2, "Add profile", status=TodoStatus.IN_PROGRESS),
        _todo(3, "Wire prompt"),
    ]
    assert _should_nudge(todos, verification_enabled=True) is False


def test_nudge_suppressed_when_verification_subsystem_disabled() -> None:
    todos = [
        _todo(1, "Add config flag"),
        _todo(2, "Add profile"),
        _todo(3, "Wire prompt"),
    ]
    assert _should_nudge(todos, verification_enabled=False) is False


def test_write_todos_appends_nudge_message_and_flag() -> None:
    tool = Todo(config_getter=lambda: TodoConfig(), state=TodoState())
    todos = [
        _todo(1, "Add config flag"),
        _todo(2, "Add profile"),
        _todo(3, "Wire prompt"),
    ]

    result = tool._write_todos(todos, verification_enabled=True)

    assert result.verification_nudge is True
    assert "verifier" in result.message.lower()
    assert "NOTE" in result.message


def test_write_todos_no_nudge_when_verification_disabled() -> None:
    tool = Todo(config_getter=lambda: TodoConfig(), state=TodoState())
    todos = [
        _todo(1, "Add config flag"),
        _todo(2, "Add profile"),
        _todo(3, "Wire prompt"),
    ]

    result = tool._write_todos(todos, verification_enabled=False)

    assert result.verification_nudge is False
    assert "NOTE" not in result.message
