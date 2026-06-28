from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Final
from urllib.parse import parse_qs, urlparse

from vibe.core.paths import VIBE_HOME
from vibe.core.utils.io import read_safe
from vibe.setup.auth.zai_sign_in import (
    ZaiSignInError,
    extract_zai_authorization_code,
    parse_zai_authorization_callback,
)

_CALLBACK_DIR_NAME: Final = "auth"
_CALLBACK_FILE_NAME: Final = "zai-callback.json"
_CALLBACK_POLL_INTERVAL_SECONDS: Final = 0.25


def _callback_path() -> Path:
    return VIBE_HOME.path / _CALLBACK_DIR_NAME / _CALLBACK_FILE_NAME


def write_zai_callback(uri: str) -> Path:
    callback = parse_zai_authorization_callback(uri)
    if callback.state is None:
        raise ZaiSignInError("Z.ai callback URL did not include a state value.")
    path = _callback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "uri": uri,
        "code": callback.code,
        "state": callback.state,
        "created_at": time.time(),
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def consume_zai_callback(expected_state: str) -> str | None:
    path = _callback_path()
    try:
        text = read_safe(path).text
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    uri = payload.get("uri")
    if not isinstance(uri, str):
        return None
    try:
        extract_zai_authorization_code(uri, expected_state=expected_state)
    except ZaiSignInError:
        return None
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return uri


async def wait_for_zai_callback(
    authorize_url: str, *, poll_interval: float = _CALLBACK_POLL_INTERVAL_SECONDS
) -> str:
    state = _state_from_authorize_url(authorize_url)
    while True:
        if callback := consume_zai_callback(state):
            return callback
        await asyncio.sleep(poll_interval)


def _state_from_authorize_url(authorize_url: str) -> str:
    state = (parse_qs(urlparse(authorize_url).query).get("state") or [""])[0]
    if not state:
        raise ZaiSignInError("Z.ai sign-in URL did not include a state value.")
    return state
