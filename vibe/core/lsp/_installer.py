"""Consent-gated bootstrap installer for language-server binaries.

Reverses ``_defaults.py``'s longstanding "never install" stance for the
common channels (npm/pip/go/rustup/brew/dotnet/gem). Mistral Vibe runs the user's
existing toolchain to install a server, never managing binaries itself.

Consent is mandatory. Callers pass a ``consent_callback`` invoked with the
human-readable install command before any subprocess runs; ``None`` declines
by default (headless). This matches ``ToolPermission.ASK``: install never
fires silently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import shutil
import subprocess

from vibe.core.logger import logger
from vibe.core.lsp._defaults import ServerPreset

# Bootstrap tool per channel. A preset's install_command[0] selects the channel.
_CHANNELS: dict[str, str] = {
    "npm": "npm",
    "pip": "pip",
    "pipx": "pipx",
    "uv": "uv",
    "go": "go",
    "cargo": "cargo",
    "rustup": "rustup",
    "brew": "brew",
    "dotnet": "dotnet",
    "gem": "gem",
}

_INSTALL_TIMEOUT = 120.0


@dataclass(frozen=True)
class InstallResult:
    """Outcome of an install attempt.

    success is True only when the install subprocess exited 0 AND the binary
    is now resolvable. ``output`` carries combined stdout/stderr for surfacing
    in the TUI; ``error`` is set on failure (timeout, non-zero exit, declined).
    """

    preset: ServerPreset
    success: bool
    output: str = ""
    error: str = ""


def channel_for_command(command0: str) -> str | None:
    """Map an install_command's first token to its bootstrap channel.

    Returns None when the command uses a tool Mistral Vibe does not bootstrap (e.g.
    a hand-rolled curl|tar pipeline). Those presets stay hint-only.
    """
    return _CHANNELS.get(command0)


def channel_available(channel: str) -> bool:
    """Whether the bootstrap tool for ``channel`` is on PATH."""
    binary = _CHANNELS.get(channel)
    if binary is None:
        return False
    return shutil.which(binary) is not None


def _validate_installable(preset: ServerPreset) -> tuple[tuple[str, ...], str] | None:
    """Return (install_command, channel) when installable, else None.

    Encapsulates the three no-spawn early-return conditions (empty command,
    unsupported channel, channel tool absent) so install_for_preset stays
    linear.
    """
    command = preset.install_command
    if not command:
        return None
    channel = channel_for_command(command[0])
    if channel is None or not channel_available(channel):
        return None
    return command, channel


def _install_error(preset: ServerPreset) -> str:
    """Explanatory error for the validation failure _validate_installable skipped."""
    if not preset.install_command:
        return f"{preset.display_name} has no install_command; install manually."
    tool = preset.install_command[0]
    if channel_for_command(tool) is None:
        return f"{preset.display_name} uses {tool} which Mistral Vibe does not bootstrap."
    return f"{tool} not on PATH; install {tool} first."


def install_for_preset(
    preset: ServerPreset,
    *,
    consent_callback: Callable[[str], bool] | None = None,
    root_path: str | None = None,
) -> InstallResult:
    """Install ``preset``'s server binary via its declared channel.

    Consent flow: ``consent_callback`` is invoked with a human-readable
    description (the install command + channel). A None callback, or one that
    returns False, declines without spawning anything. This is the only path
    that runs an install subprocess — there is no silent bootstrap.

    Returns ``InstallResult``. The preset must declare an ``install_command``
    and the matching bootstrap tool must be on PATH; otherwise the result is
    unsuccessful with an explanatory error so the caller falls back to the
    install-hint path.
    """
    validated = _validate_installable(preset)
    if validated is None:
        return InstallResult(preset=preset, success=False, error=_install_error(preset))
    command, _channel = validated
    description = f"Run `{' '.join(command)}` to install {preset.display_name}?"
    if consent_callback is None or not consent_callback(description):
        return InstallResult(preset=preset, success=False, error="declined by user")
    try:
        result = subprocess.run(
            tuple(command),
            capture_output=True,
            text=True,
            timeout=_INSTALL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("lsp install failed for %s: %s", preset.key, exc)
        return InstallResult(preset=preset, success=False, error=str(exc))
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return InstallResult(
            preset=preset,
            success=False,
            output=output,
            error=f"exit {result.returncode}",
        )
    from pathlib import Path

    from vibe.core.lsp._defaults import _resolve_binary

    root = Path(root_path) if root_path else None
    binary = preset.detection_command[0]
    if _resolve_binary(binary, root) is None:
        return InstallResult(
            preset=preset,
            success=False,
            output=output,
            error=f"install exited 0 but {binary} still not on PATH",
        )
    return InstallResult(preset=preset, success=True, output=output)


__all__ = [
    "InstallResult",
    "channel_available",
    "channel_for_command",
    "install_for_preset",
]
