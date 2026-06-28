from __future__ import annotations

import pytest

from vibe.setup.auth.zai_callback import consume_zai_callback, write_zai_callback
from vibe.setup.auth.zai_sign_in import ZaiSignInError


def test_write_and_consume_zai_callback_by_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    uri = "zcode://zai-auth/callback?code=abc&state=current"

    path = write_zai_callback(uri)

    assert path.is_file()
    assert consume_zai_callback("stale") is None
    assert consume_zai_callback("current") == uri
    assert not path.exists()


def test_write_zai_callback_rejects_url_without_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))

    with pytest.raises(ZaiSignInError, match="state"):
        write_zai_callback("zcode://zai-auth/callback?code=abc")
