from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.lsp import _integration as integration
from vibe.core.lsp._registry import DiagnosticRegistry
from vibe.core.lsp._server import ServerConfig


def _diagnostics(
    path: str | Path, count: int, *, prefix: str = "error", severity: int = 1
) -> dict:
    uri = path.as_uri() if isinstance(path, Path) else path
    return {
        "uri": uri,
        "diagnostics": [
            {
                "range": {
                    "start": {"line": index, "character": 0},
                    "end": {"line": index, "character": 1},
                },
                "severity": severity,
                "message": f"{prefix}-{index}",
                "source": "test-source",
            }
            for index in range(count)
        ],
    }


def _empty(path: str | Path) -> dict:
    uri = path.as_uri() if isinstance(path, Path) else path
    return {"uri": uri, "diagnostics": []}


def _diagnostic_counts(batch: dict) -> dict[str, int]:
    return {
        Path(file_entry["path"]).name: len(file_entry["diagnostics"])
        for file_entry in batch["files"]
    }


def _messages(batches: list[dict]) -> list[str]:
    return [
        diagnostic.message
        for batch in batches
        for file_entry in batch["files"]
        for diagnostic in file_entry["diagnostics"]
    ]


def test_publish_replaces_unconsumed_state_for_source(tmp_path) -> None:
    registry = DiagnosticRegistry()
    path = tmp_path / "main.py"

    registry.publish(_diagnostics(path, 1, prefix="old"), "pyright")
    registry.publish(_diagnostics(path, 1, prefix="current"), "pyright")

    assert _messages(registry.consume()) == ["current-0"]


def test_replacement_removes_absent_delivered_keys(tmp_path) -> None:
    registry = DiagnosticRegistry()
    path = tmp_path / "main.py"

    registry.publish(_diagnostics(path, 1, prefix="a"), "pyright")
    assert _messages(registry.consume()) == ["a-0"]
    registry.publish(_diagnostics(path, 1, prefix="b"), "pyright")
    assert _messages(registry.consume()) == ["b-0"]
    registry.publish(_diagnostics(path, 1, prefix="a"), "pyright")

    assert _messages(registry.consume()) == ["a-0"]


def test_empty_publish_clears_only_publishing_source(tmp_path) -> None:
    registry = DiagnosticRegistry()
    path = tmp_path / "main.py"
    pyright = _diagnostics(path, 1, prefix="pyright")
    ruff = _diagnostics(path, 1, prefix="ruff")

    registry.publish(pyright, "pyright")
    registry.publish(ruff, "ruff")
    assert set(_messages(registry.consume())) == {"pyright-0", "ruff-0"}

    registry.publish(_empty(path), "pyright")
    registry.publish(ruff, "ruff")
    assert registry.consume() == []

    registry.publish(pyright, "pyright")
    batches = registry.consume()
    assert _messages(batches) == ["pyright-0"]
    assert batches[0]["sources"] == ["pyright"]


def test_filtered_empty_publish_clears_only_publishing_source(tmp_path) -> None:
    registry = DiagnosticRegistry()
    path = tmp_path / "main.py"
    pyright = _diagnostics(path, 1, prefix="pyright")
    ruff = _diagnostics(path, 1, prefix="ruff")

    registry.publish(pyright, "pyright")
    registry.publish(ruff, "ruff")
    assert registry.consume()

    registry.publish(_diagnostics(path, 1, prefix="information", severity=3), "pyright")
    registry.publish(ruff, "ruff")
    assert registry.consume() == []

    registry.publish(pyright, "pyright")
    batches = registry.consume()
    assert _messages(batches) == ["pyright-0"]
    assert batches[0]["sources"] == ["pyright"]


def test_equivalent_relative_path_replaces_uri_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    registry = DiagnosticRegistry()
    path = tmp_path / "main.py"

    registry.publish(_diagnostics(path, 1, prefix="old"), "pyright")
    registry.publish(_diagnostics("main.py", 1, prefix="current"), "pyright")

    assert _messages(registry.consume()) == ["current-0"]


@pytest.mark.parametrize("uri", ["file:", "file://[malformed"])
def test_publish_ignores_empty_or_malformed_uri(uri: str) -> None:
    registry = DiagnosticRegistry()

    registry.publish(_diagnostics(uri, 1), "pyright")

    assert registry.consume() == []


def test_clear_normalizes_relative_and_percent_encoded_paths(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    registry = DiagnosticRegistry()
    path = tmp_path / "my file.py"
    published = _diagnostics(path, 1)

    registry.publish(published, "pyright")
    assert registry.consume()
    registry.clear_for_path("my file.py")
    registry.publish(published, "pyright")

    assert registry.consume()


@pytest.mark.parametrize(
    ("uri", "path"),
    [
        ("file:///C:/Workspace/My%20File.py", r"c:\workspace\my file.py"),
        ("file://SERVER/Share/Main.py", r"\\server\share\main.py"),
    ],
)
def test_clear_normalizes_windows_uri_and_path_forms(uri: str, path: str) -> None:
    registry = DiagnosticRegistry()
    published = _diagnostics(uri, 1)

    registry.publish(published, "pyright")
    assert registry.consume()
    registry.clear_for_path(path)
    registry.publish(published, "pyright")

    assert registry.consume()


def test_caps_retain_overflow_and_rotate_to_later_files(tmp_path) -> None:
    registry = DiagnosticRegistry()
    first = _diagnostics(tmp_path / "first.py", 30, prefix="first")
    second = _diagnostics(tmp_path / "second.py", 10, prefix="second")
    third = _diagnostics(tmp_path / "third.py", 10, prefix="third")
    fourth = _diagnostics(tmp_path / "fourth.py", 10, prefix="fourth")

    for published, source in (
        (first, "first-server"),
        (second, "second-server"),
        (third, "third-server"),
        (fourth, "fourth-server"),
    ):
        registry.publish(published, source)

    batches = registry.consume()
    assert len(batches) == 1
    assert _diagnostic_counts(batches[0]) == {
        "first.py": 10,
        "second.py": 10,
        "third.py": 10,
    }
    assert batches[0]["sources"] == ["first-server", "second-server", "third-server"]

    batches = registry.consume()
    assert len(batches) == 1
    assert _diagnostic_counts(batches[0]) == {"fourth.py": 10, "first.py": 10}
    assert batches[0]["sources"] == ["first-server", "fourth-server"]
    assert _messages(batches) == [
        *[f"fourth-{index}" for index in range(10)],
        *[f"first-{index}" for index in range(10, 20)],
    ]

    assert _messages(registry.consume()) == [
        f"first-{index}" for index in range(20, 30)
    ]
    assert registry.consume() == []


class FakeLanguageServer:
    def __init__(self) -> None:
        self.config = ServerConfig(
            name="pyright", command=["pyright-langserver"], languages={".py": "python"}
        )
        self.changes: list[tuple[str, int]] = []
        self.saves: list[tuple[str, int]] = []

    async def ensure_started(self) -> None:
        pass

    def is_open(self, path: str) -> bool:
        return True

    async def did_change(self, path: str, text: str) -> None:
        self.changes.append((path, len(text.encode("utf-8"))))

    async def did_save(self, path: str, text: str) -> None:
        self.saves.append((path, len(text.encode("utf-8"))))


class FakeLSPManager:
    def __init__(self, server: FakeLanguageServer) -> None:
        self.server = server
        self.diagnostics = DiagnosticRegistry()

    def get_server_for_file(self, path: str | Path) -> FakeLanguageServer:
        return self.server

    def clear_diagnostics_for(self, path: str | Path) -> None:
        self.diagnostics.clear_for_path(str(path))


@pytest.mark.asyncio
async def test_notify_file_changed_invalidates_all_sources_for_normalized_path(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "main.py"
    pyright = _diagnostics(path, 1, prefix="pyright")
    ruff = _diagnostics(path, 1, prefix="ruff")
    server = FakeLanguageServer()
    manager = FakeLSPManager(server)
    monkeypatch.setattr(integration, "get_lsp_manager", lambda: manager)

    manager.diagnostics.publish(pyright, "pyright")
    manager.diagnostics.publish(ruff, "ruff")
    assert manager.diagnostics.consume()

    await integration.notify_file_changed("main.py", "changed = True\n")
    manager.diagnostics.publish(pyright, "pyright")
    manager.diagnostics.publish(ruff, "ruff")

    batches = manager.diagnostics.consume()
    assert set(_messages(batches)) == {"pyright-0", "ruff-0"}
    assert batches[0]["sources"] == ["pyright", "ruff"]
    assert server.changes == [("main.py", len("changed = True\n"))]
    assert server.saves == [("main.py", len("changed = True\n"))]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["ascii", "utf8"])
async def test_oversized_edit_invalidates_before_notification_returns(
    payload: str, monkeypatch, tmp_path
) -> None:
    limit = 16
    monkeypatch.setattr(integration, "_MAX_NOTIFY_BYTES", limit)
    path = tmp_path / "main.py"
    published = _diagnostics(path, 1)
    server = FakeLanguageServer()
    manager = FakeLSPManager(server)
    monkeypatch.setattr(integration, "get_lsp_manager", lambda: manager)

    manager.diagnostics.publish(published, "pyright")
    assert manager.diagnostics.consume()

    if payload == "ascii":
        text = "x" * (limit + 1)
    else:
        unit = "😀"
        text = unit * (limit // len(unit.encode("utf-8")) + 1)
    await integration.notify_file_changed(path, text)
    manager.diagnostics.publish(published, "pyright")

    assert manager.diagnostics.consume()
    assert server.changes == []
    assert server.saves == []
