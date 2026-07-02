from __future__ import annotations

import pytest

from vibe.cli import entrypoint as entrypoint_mod
from vibe.setup.auth.zai_callback import consume_zai_callback


def test_hidden_zai_callback_flag_does_not_show_in_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["vibe", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        entrypoint_mod.parse_arguments()

    assert exc_info.value.code == 0
    assert "--zai-callback" not in capsys.readouterr().out


def test_main_captures_hidden_zai_callback_before_cli_start(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    uri = "zcode://zai-auth/callback?code=abc&state=current"
    monkeypatch.setattr("sys.argv", ["vibe", "--zai-callback", uri])
    monkeypatch.setattr(
        "vibe.core.config.harness_files.init_harness_files_manager",
        lambda *_args, **_kwargs: pytest.fail("callback must not start Mistral Vibe"),
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint_mod.main()

    assert exc_info.value.code == 0
    assert "callback captured" in capsys.readouterr().out
    assert consume_zai_callback("current") == uri


def test_main_captures_bare_zcode_callback_before_argument_parse(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    uri = "zcode://zai-auth/callback?code=abc&state=current"
    monkeypatch.setattr("sys.argv", ["vibe", uri])
    monkeypatch.setattr(
        entrypoint_mod,
        "parse_arguments",
        lambda: pytest.fail("bare zcode callback must bypass argument parsing"),
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint_mod.main()

    assert exc_info.value.code == 0
    assert consume_zai_callback("current") == uri
