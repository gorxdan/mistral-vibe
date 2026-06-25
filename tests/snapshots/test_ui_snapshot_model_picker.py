from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from textual.pilot import Pilot

from tests.conftest import build_test_vibe_config
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp
from tests.snapshots.snap_compare import SnapCompare
from vibe.core.config._settings import ModelConfig


@contextmanager
def _no_discovery() -> Iterator[None]:
    # The picker probes live local runtimes (e.g. a dev-machine ollama) on open.
    # Patch it off so the snapshot captures only the configured models and stays
    # deterministic across machines.
    with patch(
        "vibe.core.llm.model_discovery.discover_extra_models",
        new=AsyncMock(return_value=[]),
    ):
        yield


def _model_picker_config():
    models = [
        ModelConfig(
            name="mistral-large-latest", provider="mistral", alias="mistral-large"
        ),
        ModelConfig(name="devstral-latest", provider="mistral", alias="devstral"),
        ModelConfig(name="codestral-latest", provider="mistral", alias="codestral"),
        ModelConfig(
            name="mistral-small-latest", provider="mistral", alias="mistral-small"
        ),
        ModelConfig(name="devstral", provider="llamacpp", alias="local"),
    ]
    return build_test_vibe_config(
        models=models,
        active_model="devstral",
        disable_welcome_banner_animation=True,
        displayed_workdir="/test/workdir",
    )


class ModelPickerTestApp(BaseSnapshotTestApp):
    def __init__(self):
        super().__init__(config=_model_picker_config())

    async def on_mount(self) -> None:
        await super().on_mount()
        await self._switch_to_model_picker_app()


def test_snapshot_model_picker_initial(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)

    with _no_discovery():
        assert snap_compare(
            "test_ui_snapshot_model_picker.py:ModelPickerTestApp",
            terminal_size=(100, 36),
            run_before=run_before,
        )


def test_snapshot_model_picker_navigate_down(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)
        await pilot.press("down")
        await pilot.pause(0.1)

    with _no_discovery():
        assert snap_compare(
            "test_ui_snapshot_model_picker.py:ModelPickerTestApp",
            terminal_size=(100, 36),
            run_before=run_before,
        )


def test_snapshot_model_picker_select_different_model(
    snap_compare: SnapCompare,
) -> None:
    """Select the second model and verify the picker closes back to input."""

    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.2)

    with _no_discovery(), patch("vibe.cli.textual_ui.app.VibeConfig.save_updates"):
        assert snap_compare(
            "test_ui_snapshot_model_picker.py:ModelPickerTestApp",
            terminal_size=(100, 36),
            run_before=run_before,
        )


def test_snapshot_model_picker_escape_cancels(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause(0.2)

    with _no_discovery():
        assert snap_compare(
            "test_ui_snapshot_model_picker.py:ModelPickerTestApp",
            terminal_size=(100, 36),
            run_before=run_before,
        )
