from __future__ import annotations

from collections.abc import Mapping
import os
from urllib.parse import urlsplit

_INHERITED_ENV = frozenset({
    "ANDROID_HOME",
    "ANDROID_SDK_ROOT",
    "ALL_PROXY",
    "APPDATA",
    "ASDF_DATA_DIR",
    "BUN_INSTALL",
    "CARGO_HOME",
    "CI",
    "CLASSPATH",
    "COLORTERM",
    "COMPILER_PATH",
    "COMPOSER_HOME",
    "COMSPEC",
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "CONDA_PREFIX",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "DEVELOPER_DIR",
    "DOTNET_CLI_HOME",
    "DOTNET_ROOT",
    "DOTNET_ROOT_X64",
    "GEM_HOME",
    "GEM_PATH",
    "GOCACHE",
    "GO111MODULE",
    "GOENV",
    "GOEXPERIMENT",
    "GOINSECURE",
    "GOMODCACHE",
    "GOARCH",
    "GONOPROXY",
    "GONOSUMDB",
    "GOOS",
    "GOPATH",
    "GOPRIVATE",
    "GOPROXY",
    "GOROOT",
    "GOTOOLCHAIN",
    "GOWORK",
    "GRADLE_HOME",
    "GRADLE_USER_HOME",
    "HOMEDRIVE",
    "HOME",
    "HOMEPATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "JAVA_HOME",
    "JDK_HOME",
    "KOTLIN_HOME",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LIBRARY_PATH",
    "LOCALAPPDATA",
    "LOGNAME",
    "M2_HOME",
    "MAVEN_HOME",
    "MISE_DATA_DIR",
    "NODE_PATH",
    "NO_PROXY",
    "NO_COLOR",
    "NUGET_PACKAGES",
    "NVM_BIN",
    "NVM_DIR",
    "NVM_INC",
    "OBJC_INCLUDE_PATH",
    "PATH",
    "PATHEXT",
    "PHP_INI_SCAN_DIR",
    "PNPM_HOME",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "PYENV_ROOT",
    "PYTHONHOME",
    "PYTHONIOENCODING",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONUTF8",
    "RBENV_ROOT",
    "RUBYLIB",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
    "RUSTUP_TOOLCHAIN",
    "RUSTUP_HOME",
    "RUST_TOOLCHAIN",
    "SDKMAN_DIR",
    "SDKROOT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TERM_PROGRAM",
    "TMP",
    "TMPDIR",
    "TOOLCHAINS",
    "USER",
    "USERPROFILE",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "VIRTUAL_ENV",
    "WINDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_DIRS",
    "XDG_CONFIG_HOME",
    "XDG_DATA_DIRS",
    "XDG_DATA_HOME",
})


def _proxy_value_is_safe(value: str) -> bool:
    for fallback in value.split(","):
        for raw_endpoint in fallback.split("|"):
            endpoint = raw_endpoint.strip()
            if not endpoint or endpoint.lower() in {"direct", "off"}:
                continue
            if "@" in endpoint:
                return False
            try:
                parsed = urlsplit(endpoint)
            except ValueError:
                return False
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                return False
    return True


def _safe_to_inherit(key: str, value: str) -> bool:
    normalized = key.upper()
    if normalized not in _INHERITED_ENV and not normalized.startswith("LC_"):
        return False
    if normalized in {"ALL_PROXY", "GOPROXY", "HTTP_PROXY", "HTTPS_PROXY"}:
        return _proxy_value_is_safe(value)
    return True


def language_server_env(
    configured: Mapping[str, str], base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build a restricted LSP environment, then apply explicit server values.

    This removes ambient credential variables; it is defense-in-depth, not a
    filesystem or process sandbox. Isolated worktree agents therefore disable
    language-server startup at the lifecycle boundary.
    """
    source = base if base is not None else os.environ
    inherited = {
        key: value for key, value in source.items() if _safe_to_inherit(key, value)
    }
    inherited.update(configured)
    return inherited
