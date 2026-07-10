from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from vibe.core.tasking.models import TaskManifestIdentity

__all__ = ["TaskManifestError", "TaskToolManifest", "resolve_task_manifest"]


class TaskManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TaskToolManifest:
    name: str
    version: str
    tools: tuple[str, ...]
    digest: str

    @property
    def identity(self) -> TaskManifestIdentity:
        return TaskManifestIdentity(
            name=self.name, version=self.version, digest=self.digest
        )


def _manifest(name: str, version: str, tools: tuple[str, ...]) -> TaskToolManifest:
    canonical_tools = tuple(sorted(set(tools)))
    payload = json.dumps(
        {"name": name, "tools": canonical_tools, "version": version},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return TaskToolManifest(
        name=name,
        version=version,
        tools=canonical_tools,
        digest=hashlib.sha256(payload).hexdigest(),
    )


_MANIFESTS = {
    (manifest.name, manifest.version): manifest
    for manifest in (
        _manifest(
            "investigate",
            "1",
            ("glob", "grep", "lsp", "read", "skill", "web_fetch", "web_search"),
        ),
        _manifest(
            "implement-verify",
            "1",
            (
                "edit",
                "glob",
                "grep",
                "lsp",
                "read",
                "task_checks",
                "todo",
                "write_file",
            ),
        ),
        _manifest(
            "verify", "1", ("glob", "grep", "lsp", "read", "skill", "task_checks")
        ),
        _manifest(
            "mechanical-edit",
            "1",
            ("edit", "glob", "grep", "lsp", "read", "todo", "write_file"),
        ),
    )
}


def resolve_task_manifest(identity: TaskManifestIdentity) -> TaskToolManifest:
    manifest = _MANIFESTS.get((identity.name, identity.version))
    if manifest is None:
        raise TaskManifestError(f"untrusted task manifest: {identity.identity}")
    if identity.digest is not None and identity.digest != manifest.digest:
        raise TaskManifestError(
            f"task manifest digest mismatch for {identity.name}@{identity.version}"
        )
    return manifest
