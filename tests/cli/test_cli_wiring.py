from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ValidationError
import pytest

from tests.conftest import build_test_vibe_config
from vibe.cli import cli as cli_mod
from vibe.core.config import SessionLoggingConfig
from vibe.core.hooks.models import HookConfigResult
from vibe.core.types import OutputFormat
from vibe.core.utils import ConversationLimitException

_EMPTY_HOOKS = HookConfigResult(hooks=[], issues=[])


def _make_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "initial_prompt": None,
        "prompt": "hello",
        "max_turns": None,
        "max_price": None,
        "max_tokens": None,
        "enabled_tools": None,
        "model": None,
        "output": "text",
        "agent": "default",
        "setup": False,
        "workdir": None,
        "add_dir": [],
        "trust": False,
        "teleport": False,
        "continue_session": False,
        "resume": None,
        "keep_alive": None,
        "worktree": False,
        "no_worktree": False,
        "auto_approve": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# _format_config_validation_error                                             #
# --------------------------------------------------------------------------- #


class _RaisingModel(BaseModel):
    required_field: int


def _validation_error() -> ValidationError:
    try:
        _RaisingModel(required_field="not-an-int")  # type: ignore[arg-type]
    except ValidationError as e:
        return e
    raise AssertionError("expected ValidationError")


def test_format_config_validation_error_counts_and_formats_loc() -> None:
    exc = _validation_error()
    text = cli_mod._format_config_validation_error(exc)
    assert text.startswith(f"Invalid configuration ({exc.error_count()} error(s)):")
    assert "required_field" in text
    lines = text.splitlines()
    assert any(line.strip().startswith("- required_field:") for line in lines)


# --------------------------------------------------------------------------- #
# load_config_or_exit                                                         #
# --------------------------------------------------------------------------- #


def test_load_config_or_exit_validation_error_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = _validation_error()
    monkeypatch.setattr(
        cli_mod.VibeConfig, "load", staticmethod(lambda: (_ for _ in ()).throw(exc))
    )
    with pytest.raises(SystemExit) as info:
        cli_mod.load_config_or_exit(interactive=False)
    assert info.value.code == 1


def test_load_config_or_exit_value_error_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_mod.VibeConfig,
        "load",
        staticmethod(lambda: (_ for _ in ()).throw(ValueError("bad value"))),
    )
    with pytest.raises(SystemExit) as info:
        cli_mod.load_config_or_exit(interactive=False)
    assert info.value.code == 1


# --------------------------------------------------------------------------- #
# warn_if_workdir_trust_is_unset early returns                                #
# --------------------------------------------------------------------------- #


def test_warn_if_workdir_trust_silent_in_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    # No output expected.
    cli_mod.warn_if_workdir_trust_is_unset()


def test_warn_if_workdir_trust_handles_filenotfound(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_a: object, **_k: object) -> Path:
        raise FileNotFoundError("cwd gone")

    monkeypatch.setattr(Path, "cwd", classmethod(boom))  # type: ignore[arg-type]
    cli_mod.warn_if_workdir_trust_is_unset()
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- #
# _run_programmatic_mode                                                      #
# --------------------------------------------------------------------------- #


def _patch_programmatic_shell(
    monkeypatch: pytest.MonkeyPatch,
    config: Any = None,
    run_programmatic_side_effect: Any = lambda **_k: "ok",
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> Any:
        captured.update(kwargs)
        if isinstance(run_programmatic_side_effect, BaseException):
            raise run_programmatic_side_effect
        return (
            run_programmatic_side_effect(**kwargs)
            if callable(run_programmatic_side_effect)
            else run_programmatic_side_effect
        )

    monkeypatch.setattr(cli_mod, "warn_if_workdir_trust_is_unset", lambda: None)
    monkeypatch.setattr(cli_mod, "run_programmatic", fake_run)
    cfg = config if config is not None else build_test_vibe_config()
    return {"captured": captured, "config": cfg}


def test_programmatic_mode_no_prompt_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shell = _patch_programmatic_shell(monkeypatch)
    args = _make_args(prompt=None)
    with pytest.raises(SystemExit) as info:
        cli_mod._run_programmatic_mode(
            args, shell["config"], "default", _EMPTY_HOOKS, None, None
        )
    assert info.value.code == 1
    assert "No prompt" in capsys.readouterr().err


def test_programmatic_mode_uses_stdin_prompt_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = _patch_programmatic_shell(monkeypatch)
    args = _make_args(prompt=None, output="json")
    with pytest.raises(SystemExit) as info:
        cli_mod._run_programmatic_mode(
            args, shell["config"], "default", _EMPTY_HOOKS, None, "from-stdin"
        )
    assert info.value.code == 0
    assert shell["captured"]["prompt"] == "from-stdin"
    assert shell["captured"]["output_format"] == OutputFormat.JSON
    assert shell["captured"]["headless"] is True
    assert shell["captured"]["allow_subagent"] is True


def test_programmatic_mode_teleport_gated_by_vibe_code_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_off = build_test_vibe_config(vibe_code_enabled=False)
    shell = _patch_programmatic_shell(monkeypatch, config=cfg_off)
    args = _make_args(teleport=True)
    with pytest.raises(SystemExit):
        cli_mod._run_programmatic_mode(
            args, cfg_off, "default", _EMPTY_HOOKS, None, None
        )
    assert shell["captured"]["teleport"] is False

    cfg_on = build_test_vibe_config(vibe_code_enabled=True)
    shell_on = _patch_programmatic_shell(monkeypatch, config=cfg_on)
    with pytest.raises(SystemExit):
        cli_mod._run_programmatic_mode(
            args, cfg_on, "default", _EMPTY_HOOKS, None, None
        )
    assert shell_on["captured"]["teleport"] is True


def test_programmatic_mode_disabled_tools_extended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_test_vibe_config()
    _patch_programmatic_shell(monkeypatch, config=cfg)
    args = _make_args()
    with pytest.raises(SystemExit):
        cli_mod._run_programmatic_mode(args, cfg, "default", _EMPTY_HOOKS, None, None)
    assert "ask_user_question" in cfg.disabled_tools
    assert "exit_plan_mode" in cfg.disabled_tools
    assert "enter_plan_mode" in cfg.disabled_tools


@pytest.mark.parametrize(
    "exc",
    [
        ConversationLimitException("limit hit"),
        cli_mod.TeleportError("tp fail"),
        RuntimeError("runtime"),
        ValueError("value"),
    ],
)
def test_programmatic_mode_translates_exceptions_to_exit_one(
    monkeypatch: pytest.MonkeyPatch,
    exc: BaseException,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shell = _patch_programmatic_shell(monkeypatch, run_programmatic_side_effect=exc)
    with pytest.raises(SystemExit) as info:
        cli_mod._run_programmatic_mode(
            _make_args(), shell["config"], "default", _EMPTY_HOOKS, None, None
        )
    assert info.value.code == 1


def test_programmatic_mode_empty_response_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = _patch_programmatic_shell(monkeypatch, run_programmatic_side_effect=None)
    with pytest.raises(SystemExit) as info:
        cli_mod._run_programmatic_mode(
            _make_args(), shell["config"], "default", _EMPTY_HOOKS, None, None
        )
    assert info.value.code == 0


# --------------------------------------------------------------------------- #
# load_session                                                                #
# --------------------------------------------------------------------------- #


def test_load_session_no_continue_no_resume_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_test_vibe_config()
    assert (
        cli_mod.load_session(_make_args(continue_session=False, resume=None), cfg)
        is None
    )


def test_load_session_disabled_logging_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_test_vibe_config(session_logging=SessionLoggingConfig(enabled=False))
    with pytest.raises(SystemExit) as info:
        cli_mod.load_session(_make_args(continue_session=True), cfg)
    assert info.value.code == 1


def test_load_session_bare_resume_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_test_vibe_config(session_logging=SessionLoggingConfig(enabled=True))
    assert cli_mod.load_session(_make_args(resume=True), cfg) is None


def test_load_session_resume_not_found_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_test_vibe_config(session_logging=SessionLoggingConfig(enabled=True))
    monkeypatch.setattr(
        cli_mod.SessionLoader,
        "find_session_by_id",
        staticmethod(lambda *_a, **_k: None),
    )
    with pytest.raises(SystemExit) as info:
        cli_mod.load_session(_make_args(resume="missing-id"), cfg)
    assert info.value.code == 1


def test_load_session_load_failure_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_test_vibe_config(session_logging=SessionLoggingConfig(enabled=True))
    fake_session = SimpleNamespace(id="x")
    monkeypatch.setattr(
        cli_mod.SessionLoader,
        "find_session_by_id",
        staticmethod(lambda *_a, **_k: fake_session),
    )
    monkeypatch.setattr(
        cli_mod.SessionLoader,
        "load_session",
        staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("corrupt"))),
    )
    with pytest.raises(SystemExit) as info:
        cli_mod.load_session(_make_args(resume="x"), cfg)
    assert info.value.code == 1


# --------------------------------------------------------------------------- #
# run_cli interactive + interrupt                                             #
# --------------------------------------------------------------------------- #


def _patch_interactive_shell(
    monkeypatch: pytest.MonkeyPatch, config: Any = None
) -> dict[str, Any]:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli_mod, "bootstrap_config_files", lambda: None)
    cfg = config if config is not None else build_test_vibe_config()
    monkeypatch.setattr(cli_mod, "load_config_or_exit", lambda interactive: cfg)
    monkeypatch.setattr(cli_mod, "_maybe_run_startup_update_prompt", lambda *_a: None)
    monkeypatch.setattr(cli_mod, "get_initial_agent_name", lambda *_a: "default")
    monkeypatch.setattr(cli_mod, "load_hooks_from_fs", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_mod, "setup_tracing", lambda *_a: None)
    monkeypatch.setattr(cli_mod, "load_session", lambda *_a: None)
    monkeypatch.setattr(cli_mod, "get_prompt_from_stdin", lambda: None)
    monkeypatch.setattr(
        "vibe.cli.textual_ui.app.run_textual_ui", lambda **kw: calls.update(kw)
    )
    return calls


def test_run_cli_setup_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "bootstrap_config_files", lambda: None)
    called: list[bool] = []
    monkeypatch.setattr(
        "vibe.setup.onboarding.run_onboarding", lambda **_k: called.append(True)
    )
    with pytest.raises(SystemExit) as info:
        cli_mod.run_cli(_make_args(setup=True, prompt=None))
    assert info.value.code == 0
    assert called == [True]


def test_run_cli_interactive_enabled_tools_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_test_vibe_config()
    calls = _patch_interactive_shell(monkeypatch, config=cfg)
    monkeypatch.setattr(cli_mod, "AgentLoop", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(
        "vibe.core.worktree.manager.worktree_enabled", lambda *a, **k: False
    )
    monkeypatch.setattr(cli_mod, "detect_terminal", lambda: None)
    cli_mod.run_cli(_make_args(prompt=None, enabled_tools=["bash*"]))
    assert cfg.enabled_tools == ["bash*"]
    assert "agent_loop" in calls


def test_run_cli_keyboard_interrupt_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "bootstrap_config_files", lambda: None)
    monkeypatch.setattr(
        cli_mod,
        "load_config_or_exit",
        lambda interactive: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(SystemExit) as info:
        cli_mod.run_cli(_make_args(prompt=None))
    assert info.value.code == 0


def test_run_cli_eof_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "bootstrap_config_files", lambda: None)
    monkeypatch.setattr(
        cli_mod,
        "load_config_or_exit",
        lambda interactive: (_ for _ in ()).throw(EOFError),
    )
    with pytest.raises(SystemExit) as info:
        cli_mod.run_cli(_make_args(prompt=None))
    assert info.value.code == 0


# --------------------------------------------------------------------------- #
# bootstrap_config_files                                                       #
# --------------------------------------------------------------------------- #


def test_bootstrap_writes_valid_toml_with_commented_openai_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tomllib

    from vibe.core.config import ProviderConfig

    cfg_file = tmp_path / "config.toml"
    monkeypatch.setattr(
        cli_mod,
        "get_harness_files_manager",
        lambda: SimpleNamespace(user_config_file=cfg_file),
    )
    monkeypatch.setattr(
        cli_mod, "HISTORY_FILE", SimpleNamespace(path=tmp_path / "history")
    )

    cli_mod.bootstrap_config_files()

    text = cfg_file.read_text("utf-8")
    # The generated defaults + appended comment block must stay valid TOML.
    parsed = tomllib.loads(text)
    assert "providers" in parsed

    # The OpenAI provider is present as one-uncomment-away TOML lines.
    provider_lines = [
        "[[providers]]",
        'name = "openai"',
        'api_base = "https://api.openai.com/v1"',
        'api_key_env_var = "OPENAI_API_KEY"',
    ]
    for line in provider_lines:
        assert f"# {line}" in text

    # Uncommenting those exact lines yields a valid ProviderConfig.
    reparsed = tomllib.loads("\n".join(provider_lines))
    ProviderConfig(**reparsed["providers"][0])
