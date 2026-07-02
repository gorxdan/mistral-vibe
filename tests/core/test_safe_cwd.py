from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.paths import safe_cwd
import vibe.core.paths._safe_cwd as safe_cwd_module


@pytest.fixture(autouse=True)
def _reset_last_good(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safe_cwd_module, "_last_good_cwd", None)


def test_healthy_cwd_passthrough_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert safe_cwd() == Path.cwd()
    assert safe_cwd_module._last_good_cwd == Path.cwd()


def test_deleted_cwd_falls_back_to_last_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keeper = tmp_path / "keeper"
    keeper.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    monkeypatch.chdir(keeper)
    seeded = safe_cwd()
    monkeypatch.chdir(victim)
    victim.rmdir()
    assert safe_cwd() == seeded


def test_deleted_last_good_falls_back_to_pwd_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    monkeypatch.chdir(victim)
    safe_cwd()
    victim.rmdir()
    monkeypatch.setenv("PWD", str(anchor))
    assert safe_cwd() == anchor


def test_all_fallbacks_gone_returns_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    monkeypatch.chdir(victim)
    safe_cwd()
    victim.rmdir()
    monkeypatch.delenv("PWD", raising=False)
    assert safe_cwd() == Path.home()


def test_recovery_reseeds_after_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    keeper = tmp_path / "keeper"
    keeper.mkdir()
    monkeypatch.chdir(victim)
    safe_cwd()
    victim.rmdir()
    safe_cwd()
    monkeypatch.chdir(keeper)
    assert safe_cwd() == keeper
    assert safe_cwd_module._last_good_cwd == keeper
