from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from vibe.core.config import LSPServer
from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME

_MANAGED_CACHE_DIR = VIBE_HOME.path / "lsp-servers"


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
        manifest_markers=(
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "Pipfile",
            "uv.lock",
        ),
    ),
    install_hint="npm install -g pyright  (or: pip install pyright)",
    detection_command=("pyright", "--version"),
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
        manifest_markers=("package.json", "tsconfig.json", "deno.json"),
    ),
    install_hint="npm install -g typescript-language-server typescript",
    detection_command=("typescript-language-server", "--version"),
)

_RUST_ANALYZER = ServerPreset(
    key="rust",
    display_name="Rust (rust-analyzer)",
    server=LSPServer(
        name="rust-analyzer",
        command="rust-analyzer",
        languages={".rs": "rust"},
        manifest_markers=("Cargo.toml",),
    ),
    install_hint="rustup component add rust-analyzer",
    detection_command=("rust-analyzer", "--version"),
)

_GOPLS = ServerPreset(
    key="go",
    display_name="Go (gopls)",
    server=LSPServer(
        name="gopls",
        command="gopls",
        languages={".go": "go", ".gomod": "gomod"},
        manifest_markers=("go.mod",),
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
        manifest_markers=("compile_commands.json", "CMakeLists.txt"),
    ),
    install_hint="apt install clangd  (or: brew install llvm)",
    detection_command=("clangd", "--version"),
)

_JDTLS = ServerPreset(
    key="java",
    display_name="Java (jdtls)",
    server=LSPServer(
        name="jdtls",
        command="jdtls",
        languages={".java": "java"},
        manifest_markers=(
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
        ),
    ),
    install_hint="brew install jdtls  (or: download from eclipse.org/jdtls)",
    detection_command=("jdtls", "--help"),
)

_OMNISHARP = ServerPreset(
    key="csharp",
    display_name="C# (OmniSharp)",
    server=LSPServer(
        name="omnisharp",
        command="OmniSharp",
        languages={".cs": "csharp"},
        manifest_markers=(
            "*.csproj",
            "*.sln",
            "Directory.Build.props",
        ),
    ),
    install_hint="dotnet tool install --global OmniSharp",
    detection_command=("OmniSharp", "--version"),
)

_INTELEPHENSE = ServerPreset(
    key="php",
    display_name="PHP (intelephense)",
    server=LSPServer(
        name="intelephense",
        command="intelephense",
        languages={".php": "php"},
        manifest_markers=("composer.json",),
        args=["--stdio"],
    ),
    install_hint="npm install -g intelephense",
    detection_command=("intelephense", "--version"),
)

_RUBY_LSP = ServerPreset(
    key="ruby",
    display_name="Ruby (ruby-lsp)",
    server=LSPServer(
        name="ruby-lsp",
        command="ruby-lsp",
        languages={".ruby": "ruby", ".rb": "ruby", ".rake": "ruby"},
        manifest_markers=("Gemfile", "Gemfile.lock", "*.gemspec", "Rakefile"),
    ),
    install_hint="gem install ruby-lsp",
    detection_command=("ruby-lsp", "--version"),
)

_SOURCEKIT_LSP = ServerPreset(
    key="swift",
    display_name="Swift (sourcekit-lsp)",
    server=LSPServer(
        name="sourcekit-lsp",
        command="sourcekit-lsp",
        languages={".swift": "swift"},
        manifest_markers=("Package.swift",),
    ),
    install_hint="brew install sourcekit-lsp  (or: xcode on macOS)",
    detection_command=("sourcekit-lsp", "--version"),
)

PRESETS: dict[str, ServerPreset] = {
    p.key: p
    for p in [
        _PYRIGHT,
        _TSLANGUAGE,
        _RUST_ANALYZER,
        _GOPLS,
        _CLANGD,
        _JDTLS,
        _OMNISHARP,
        _INTELEPHENSE,
        _RUBY_LSP,
        _SOURCEKIT_LSP,
    ]
}


def preset_for_extension(ext: str) -> ServerPreset | None:
    normalized = ext.lower().lstrip(".")
    for preset in PRESETS.values():
        for key in preset.server.languages:
            if key.lower().lstrip(".") == normalized:
                return preset
    return None


_PROBE_TIMEOUT = 3.0


@dataclass(frozen=True)
class PresetProbe:
    """Outcome of probing a preset's binary.

    status is one of:
      - "available": binary present and version probe exits 0
      - "absent"   : binary not on PATH (nothing to surface but an install hint)
      - "broken"   : binary present but probe failed; ``stderr``/``returncode``
                     explain why (e.g. a rustup proxy for an uninstalled
                     component). Broken presets are excluded from the server
                     registry but surfaced in /lsp status so the user knows the
                     language is installed-but-not-working instead of silent.
    """

    preset: ServerPreset
    status: str
    returncode: int | None = None
    stderr: str = ""


def _resolve_binary(binary_name: str, root_path: Path | None) -> str | None:
    """Resolve a language-server binary to its preferred absolute path.

    Order: project venv → managed cache → PATH. The venv preference means a
    project that pins a server (e.g. pyright) in its dev deps wins over a stray
    global install — closing the version-skew class where the LSP tool spawns a
    different binary than the project's own toolchain uses.
    """
    if root_path is not None:
        venv_bin = root_path / ".venv" / "bin" / binary_name
        if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
            return str(venv_bin)
    cache_bin = _MANAGED_CACHE_DIR / binary_name
    if cache_bin.is_file() and os.access(cache_bin, os.X_OK):
        return str(cache_bin)
    return shutil.which(binary_name)


def _probe(
    preset: ServerPreset, root_path: Path | None = None
) -> PresetProbe:
    resolved = _resolve_binary(preset.detection_command[0], root_path)
    if resolved is None:
        return PresetProbe(preset=preset, status="absent")
    try:
        result = subprocess.run(
            (resolved, *preset.detection_command[1:]),
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PresetProbe(preset=preset, status="broken", stderr=str(exc))
    if result.returncode == 0:
        return PresetProbe(preset=preset, status="available")
    return PresetProbe(
        preset=preset,
        status="broken",
        returncode=result.returncode,
        stderr=(result.stderr or result.stdout).strip(),
    )


def preset_probe_passes(preset: ServerPreset) -> bool:
    return _probe(preset).status == "available"


def available_presets(root_path: Path | None = None) -> list[ServerPreset]:
    candidates = list(PRESETS.values())
    if root_path is not None:
        candidates = [p for p in candidates if preset_matches_root(p, root_path)]

    usable: list[ServerPreset] = []
    for probe in (_probe(preset, root_path) for preset in candidates):
        if probe.status == "available":
            usable.append(probe.preset)
        elif probe.status == "broken":
            logger.warning(
                "lsp preset %s on PATH but probe %s failed; excluded",
                probe.preset.key,
                probe.preset.detection_command,
            )
    return usable


def preset_matches_root(preset: ServerPreset, root_path: Path) -> bool:
    """Whether ``preset`` is relevant to the project at ``root_path``.

    A preset is relevant when any of its ``manifest_markers`` exists at the
    project root. Markers containing glob characters (``*``, ``?``, ``[``)
    are matched with :meth:`Path.glob` so variable-name files like
    ``*.csproj`` or ``*.sln`` work. Presets without markers are always
    relevant so a marker-less server isn't silently dropped.
    """
    markers = preset.server.manifest_markers
    if not markers:
        return True
    for marker in markers:
        if any(c in marker for c in "*?["):
            if any(root_path.glob(marker)):
                return True
        elif (root_path / marker).exists():
            return True
    return False


def broken_presets() -> list[PresetProbe]:
    """Presets whose binary is on PATH but fails its probe.

    Surfaced in /lsp status so a user with a half-installed toolchain (e.g. a
    rustup proxy missing the rust-analyzer component) gets an actionable
    message instead of the language silently disappearing.
    """
    return [
        probe
        for probe in (_probe(preset) for preset in PRESETS.values())
        if probe.status == "broken"
    ]


def preset_states() -> list[PresetProbe]:
    """Probe result for every preset, in declaration order."""
    return [probe for probe in (_probe(preset) for preset in PRESETS.values())]
