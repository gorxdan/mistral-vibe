from __future__ import annotations

import subprocess

import pytest

from vibe.setup.auth import zai_protocol_handler as handler


def _pin_desktop_dirs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def test_install_zai_protocol_handler_registers_when_unclaimed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(handler.sys, "platform", "linux")
    _pin_desktop_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        handler.shutil,
        "which",
        lambda cmd: {
            "xdg-mime": "/usr/bin/xdg-mime",
            "vibe": "/usr/local/bin/vibe",
        }.get(cmd),
    )

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "query":
            return subprocess.CompletedProcess(args, 0, stdout="")
        return subprocess.CompletedProcess(args, 0, stdout="")

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    result = handler.install_zai_protocol_handler()

    assert result.status == "installed"
    assert result.handler == "vibe-zcode-handler.desktop"
    assert (
        result.path
        == tmp_path / "data" / "applications" / "vibe-zcode-handler.desktop"
    )
    desktop_path = result.path
    assert desktop_path is not None
    assert desktop_path.read_text(encoding="utf-8") == (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Mistral Vibe Z.ai Callback\n"
        "Exec=/usr/local/bin/vibe --zai-callback %u\n"
        "MimeType=x-scheme-handler/zcode;\n"
        "NoDisplay=true\n"
        "Terminal=false\n"
    )
    assert calls == [
        ["/usr/bin/xdg-mime", "query", "default", "x-scheme-handler/zcode"],
        [
            "/usr/bin/xdg-mime",
            "default",
            "vibe-zcode-handler.desktop",
            "x-scheme-handler/zcode",
        ],
    ]
    assert (tmp_path / "config" / "mimeapps.list").read_text(encoding="utf-8") == (
        "[Default Applications]\n"
        "x-scheme-handler/zcode=vibe-zcode-handler.desktop\n"
        "\n"
        "[Added Associations]\n"
        "x-scheme-handler/zcode=vibe-zcode-handler.desktop;\n"
    )


def test_install_zai_protocol_handler_does_not_replace_existing_handler(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(handler.sys, "platform", "linux")
    _pin_desktop_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        handler.shutil,
        "which",
        lambda cmd: {
            "xdg-mime": "/usr/bin/xdg-mime",
            "vibe": "/usr/local/bin/vibe",
        }.get(cmd),
    )

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="zcode.desktop\n")

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    result = handler.install_zai_protocol_handler()

    assert result.status == "existing_handler"
    assert result.handler == "zcode.desktop"
    assert calls == [
        ["/usr/bin/xdg-mime", "query", "default", "x-scheme-handler/zcode"]
    ]
    assert not (
        tmp_path / "data" / "applications" / "vibe-zcode-handler.desktop"
    ).exists()
    assert not (tmp_path / "config" / "mimeapps.list").exists()


def test_install_zai_protocol_handler_reports_already_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handler.sys, "platform", "linux")
    _pin_desktop_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        handler.shutil,
        "which",
        lambda cmd: {
            "xdg-mime": "/usr/bin/xdg-mime",
            "vibe": "/usr/local/bin/vibe",
        }.get(cmd),
    )

    def fake_run(args, **kwargs):
        if args[1] == "query":
            return subprocess.CompletedProcess(
                args, 0, stdout="vibe-zcode-handler.desktop\n"
            )
        return subprocess.CompletedProcess(args, 0, stdout="")

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    result = handler.install_zai_protocol_handler()

    assert result.status == "already_configured"
    assert (
        result.path
        == tmp_path / "data" / "applications" / "vibe-zcode-handler.desktop"
    )
    assert "x-scheme-handler/zcode=vibe-zcode-handler.desktop;" in (
        tmp_path / "config" / "mimeapps.list"
    ).read_text(encoding="utf-8")


def test_install_zai_protocol_handler_reports_missing_xdg_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handler.sys, "platform", "linux")
    monkeypatch.setattr(handler.shutil, "which", lambda _cmd: None)

    result = handler.install_zai_protocol_handler()

    assert result.status == "missing_xdg_mime"


def test_install_zai_protocol_handler_quotes_spaced_vibe_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handler.sys, "platform", "linux")
    _pin_desktop_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        handler.shutil,
        "which",
        lambda cmd: {
            "xdg-mime": "/usr/bin/xdg-mime",
            "vibe": "/home/me/tools with spaces/vibe",
        }.get(cmd),
    )
    monkeypatch.setattr(
        handler.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=""),
    )

    result = handler.install_zai_protocol_handler()

    assert result.status == "installed"
    assert 'Exec="/home/me/tools with spaces/vibe" --zai-callback %u' in (
        result.path.read_text(encoding="utf-8") if result.path else ""
    )


def test_install_zai_protocol_handler_ignores_snap_scoped_xdg_dirs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    snap_data_home = home / "snap" / "code" / "247" / ".local" / "share"
    snap_config_home = home / "snap" / "code" / "247" / ".config"
    seen_envs: list[dict[str, str]] = []
    monkeypatch.setattr(handler.Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(handler.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(snap_data_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(snap_config_home))
    monkeypatch.setattr(
        handler.shutil,
        "which",
        lambda cmd: {
            "xdg-mime": "/usr/bin/xdg-mime",
            "vibe": "/usr/local/bin/vibe",
        }.get(cmd),
    )

    def fake_run(args, **kwargs):
        seen_envs.append(dict(kwargs["env"]))
        if args[1] == "query":
            return subprocess.CompletedProcess(args, 0, stdout="")
        return subprocess.CompletedProcess(args, 0, stdout="")

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    result = handler.install_zai_protocol_handler()

    assert result.status == "installed"
    assert result.path == home / ".local" / "share" / "applications" / (
        "vibe-zcode-handler.desktop"
    )
    assert (home / ".config" / "mimeapps.list").is_file()
    assert all(
        env["XDG_DATA_HOME"] == str(home / ".local" / "share") for env in seen_envs
    )
    assert all(env["XDG_CONFIG_HOME"] == str(home / ".config") for env in seen_envs)
    assert not (
        snap_data_home / "applications" / "vibe-zcode-handler.desktop"
    ).exists()
    assert not (snap_config_home / "mimeapps.list").exists()


def test_install_zai_protocol_handler_prefers_launched_vibe(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched = tmp_path / "bin" / "vibe"
    launched.parent.mkdir()
    launched.write_text("#!/bin/sh\n", encoding="utf-8")
    _pin_desktop_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(handler.sys, "platform", "linux")
    monkeypatch.setattr(handler.sys, "argv", [str(launched), "--setup"])
    monkeypatch.setattr(
        handler.shutil,
        "which",
        lambda cmd: {
            "xdg-mime": "/usr/bin/xdg-mime",
            "vibe": "/usr/local/bin/vibe",
        }.get(cmd),
    )
    monkeypatch.setattr(
        handler.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=""),
    )

    result = handler.install_zai_protocol_handler()

    assert result.path is not None
    assert f"Exec={launched} --zai-callback %u" in result.path.read_text(
        encoding="utf-8"
    )
