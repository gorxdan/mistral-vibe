from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe.core.auth.mcp_oauth import (
        Fingerprint,
        KeyringTokenStorage,
        LoopbackCallbackHandler,
        MCPOAuthError,
        MCPOAuthHeadlessError,
        MCPOAuthInvalidGrant,
        MCPOAuthLoginFailed,
        MCPOAuthPortInUse,
        build_non_interactive_provider,
        build_oauth_provider,
        clear_stored_credentials,
        is_logged_in,
        make_non_interactive_handlers,
        perform_oauth_login,
    )

__all__ = [
    "Fingerprint",
    "KeyringTokenStorage",
    "LoopbackCallbackHandler",
    "MCPOAuthError",
    "MCPOAuthHeadlessError",
    "MCPOAuthInvalidGrant",
    "MCPOAuthLoginFailed",
    "MCPOAuthPortInUse",
    "build_non_interactive_provider",
    "build_oauth_provider",
    "clear_stored_credentials",
    "is_logged_in",
    "make_non_interactive_handlers",
    "perform_oauth_login",
]


def __getattr__(name: str) -> object:
    # Lazy re-export via PEP 562: mcp_oauth pulls the mcp SDK (~145ms), so
    # importing this package or a sibling submodule must not pay it at startup.
    if name in __all__:
        from vibe.core.auth import mcp_oauth

        return getattr(mcp_oauth, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
