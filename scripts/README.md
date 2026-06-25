# Project Management Scripts

This directory contains scripts that support project versioning and deployment workflows.

## Versioning

The version is derived from git tags via [hatch-vcs](https://github.com/ofek/hatch-vcs)
(see `dynamic = ["version"]` and `[tool.hatch.version]` in `pyproject.toml`).
Every commit past a tag produces a unique PEP 440 dev version; a tag is a release.
`__version__` resolves from the installed distribution metadata (`importlib.metadata`).

### Releasing

`release.py` computes the next semver from the latest tag, patches the Zed
`extension.toml`, scaffolds the changelog/whats-new, and creates the `vX.Y.Z` tag:

```bash
# Bump minor version (0.1.0 -> 0.2.0, tag v0.2.0)
uv run scripts/release.py minor

# Bump patch version (0.1.0 -> 0.1.1, tag v0.1.1)
uv run scripts/release.py patch

# Bump major version (0.1.0 -> 1.0.0, tag v1.0.0)
uv run scripts/release.py major

# Preview without creating the tag or editing files
uv run scripts/release.py --dry-run minor

# First-time baseline tag to seed hatch-vcs
uv run scripts/release.py --init-baseline 0.1.0
```

After cutting a tag, run `uv sync` so `importlib.metadata` picks up the new version.

## Releasing

`prepare_release.py` builds the release branch from the previous public release tag, cherry-picks commits from the matching `-private` tags, and (by default) squashes them into a single release commit.

As part of release branch creation, the script **freezes the full transitive dependency graph** into both `[project].dependencies` and `[dependency-groups].build` of `pyproject.toml` using the current `uv.lock`:

```bash
uv export --no-hashes --no-dev --no-emit-project --frozen --format requirements.txt
uv export --only-group build --no-emit-project --no-hashes --frozen --format requirements.txt
```

The pinned `[project].dependencies` is what `uv build` reads in `.github/workflows/release.yml`, so the wheel published to PyPI carries `Requires-Dist:` entries pinned to exact versions (with environment markers preserved). End users installing `mistral-vibe` from PyPI get the same dependency set the team tested against.

The pinned `[dependency-groups].build` is what `uv sync --no-dev --group build` reads in `.github/workflows/build-and-upload.yml`, so the PyInstaller binaries on each release tag are built against the exact same PyInstaller / truststore versions every time.

`main` keeps `>=` ranges, so day-to-day upgrades on `main` (`uv lock --upgrade-package …`, Renovate PRs, etc.) are unaffected. Each new release re-snapshots `uv.lock` — there is no hand-maintained pin list.
