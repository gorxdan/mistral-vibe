from __future__ import annotations

import orjson
from pydantic import ValidationError

from vibe.core.hooks._port import HookExternalAttrs, HookHandler, _HookAction
from vibe.core.hooks.models import HookExecutionResult, HookStructuredResponse

__all__ = [
    "HookExternalAttrs",
    "HookHandler",
    "HookOutputError",
    "HookRetryState",
    "_HookAction",
    "_failure_reason",
    "_parse_structured_response",
]

_MAX_RETRIES = 3


class HookRetryState:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def reset(self) -> None:
        self._counts.clear()

    def remaining_retries(self, hook_name: str) -> int:
        return _MAX_RETRIES - self._counts.get(hook_name, 0)

    def track_retry(self, hook_name: str) -> None:
        self._counts[hook_name] = self._counts.get(hook_name, 0) + 1

    def track_no_retry(self, hook_name: str) -> None:
        self._counts.pop(hook_name, None)

    def should_retry(self, hook_name: str) -> bool:
        return self._counts.get(hook_name, 0) < _MAX_RETRIES

class HookOutputError(ValueError):
    """Hook stdout was non-empty but did not match the structured-response
    spec. The manager treats this as a hook failure (warning by default,
    deny / clear under ``strict``).
    """

def _parse_structured_response(stdout: str) -> HookStructuredResponse | None:
    """Return the parsed response, or ``None`` for an empty stdout.

    Raises :class:`HookOutputError` for any other non-conforming output.
    """
    if not stdout:
        return None
    try:
        parsed = orjson.loads(stdout)
    except orjson.JSONDecodeError as e:
        raise HookOutputError(
            f"stdout was not valid JSON: {e.msg} at line {e.lineno} col {e.colno}"
        ) from e
    if not isinstance(parsed, dict):
        raise HookOutputError(
            f"stdout was a JSON {type(parsed).__name__}, expected an object"
        )
    try:
        return HookStructuredResponse.model_validate(parsed)
    except ValidationError as e:
        raise HookOutputError(
            f"stdout JSON did not match the hook response schema: {e}"
        ) from e

def _failure_reason(result: HookExecutionResult) -> str:
    # Prefer stderr: stdout is reserved for the JSON response and is
    # likely empty / garbage when the hook crashed.
    if result.timed_out or result.exit_code is None:
        return "timed out"
    return result.stderr or result.stdout or f"exited with code {result.exit_code}"

def _append_text(base: str, addition: str) -> str:
    if not base:
        return addition
    return f"{base}\n{addition}"
