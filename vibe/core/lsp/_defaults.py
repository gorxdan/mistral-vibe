from __future__ import annotations

from dataclasses import dataclass
import shutil

from vibe.core.config import LSPServer


@dataclass(frozen=True)
class ServerPreset:
    """A ready-made language server definition users can opt into.

    ``install_hint`` is the command a user runs to put the binary on PATH.
    Chaton never shells out to install it — the user does.
    """

    key: str
    display_name: str
    server: LSPServer
    install_hint: str
    detection_command: tuple[str, ...]


_PYRIGHT = ServerPreset(
    key="pyright",
    display_name="Python (pyright)",
    server=LSPServer(
        name="pyright",
        command="pyright-langserver",
        languages={".py": "python"},
        args=["--stdio"],
    ),
    install_hint="npm install -g pyright  (or: pip install pyright)",
    detection_command=("pyright-langserver", "--version"),
)

_TSLANGUAGE = ServerPreset(
    key="typescript",
    display_name="TypeScript / JavaScript",
    server=LSPServer(
        name="typescript-language-server",
        command="typescript-language-server",
        languages={
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
        },
        args=["--stdio"],
    ),
    install_hint="npm install -g typescript-language-server typescript",
    detection_command=("typescript-language-server", "--version"),
)

_RUST_ANALYZER = ServerPreset(
    key="rust",
    display_name="Rust (rust-analyzer)",
    server=LSPServer(
        name="rust-analyzer", command="rust-analyzer", languages={".rs": "rust"}
    ),
    install_hint="rustup component add rust-analyzer",
    detection_command=("rust-analyzer", "--version"),
)

_GOPLS = ServerPreset(
    key="go",
    display_name="Go (gopls)",
    server=LSPServer(
        name="gopls", command="gopls", languages={".go": "go", ".gomod": "gomod"}
    ),
    install_hint="go install golang.org/x/tools/gopls@latest",
    detection_command=("gopls", "version"),
)

_CLANGD = ServerPreset(
    key="clangd",
    display_name="C / C++ (clangd)",
    server=LSPServer(
        name="clangd",
        command="clangd",
        languages={
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
        },
    ),
    install_hint="apt install clangd  (or: brew install llvm)",
    detection_command=("clangd", "--version"),
)

PRESETS: dict[str, ServerPreset] = {
    p.key: p for p in [_PYRIGHT, _TSLANGUAGE, _RUST_ANALYZER, _GOPLS, _CLANGD]
}


def preset_for_extension(ext: str) -> ServerPreset | None:
    normalized = ext.lower().lstrip(".")
    for preset in PRESETS.values():
        for key in preset.server.languages:
            if key.lower().lstrip(".") == normalized:
                return preset
    return None


def preset_binary_on_path(preset: ServerPreset) -> bool:
    return shutil.which(preset.detection_command[0]) is not None


def available_presets() -> list[ServerPreset]:
    """Presets whose server binary is installed on PATH.

    These are the languages Vibe can support without any config: the moment
    LSP is enabled (installed_components), every available preset is
    registered and lazy-started on first use of a matching file.
    """
    return [p for p in PRESETS.values() if preset_binary_on_path(p)]
