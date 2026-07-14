from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

import pytest

from vibe.core._immutable_store import ImmutableFileStore, ImmutableStoreError
from vibe.core._trusted_command import (
    TRUSTED_SYSTEM_PATH,
    TrustedCommandError,
    minimal_trusted_git_environment,
    resolve_trusted_system_executable,
)
from vibe.core._trusted_host_runner import (
    FrozenSourceSnapshot,
    TrustedExecutable,
    _CombinedOutput,
    _drain_pipe,
    cleanup_trusted_executable,
    resolve_environment_attestation,
    resolve_trusted_executable,
    stable_file_sha256,
    validate_environment_attestation,
    validate_trusted_executable,
)
from vibe.core._verification_receipt import VerificationReceiptError
from vibe.core._verification_runner import (
    TrustedCheck,
    _cleanup_sandbox_invocation,
    _prepare_sandbox_invocation,
)
from vibe.core.tools.sandbox import SandboxSpec
from vibe.core.utils.io import read_safe, write_safe


def test_minimal_trusted_git_environment_drops_ambient_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/injected-python")
    monkeypatch.setenv("GIT_DIR", "/tmp/injected-git")

    env = minimal_trusted_git_environment(tmp_path / "home")

    assert env["PATH"] == TRUSTED_SYSTEM_PATH
    assert env["HOME"] == str((tmp_path / "home").resolve())
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert env["GIT_ATTR_NOSYSTEM"] == "1"
    assert "LD_PRELOAD" not in env
    assert "PYTHONPATH" not in env
    assert "GIT_DIR" not in env


def test_system_executable_resolution_ignores_ambient_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injected = tmp_path / "git"
    write_safe(injected, "#!/bin/sh\nexit 99\n")
    injected.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    resolved = resolve_trusted_system_executable("git")

    assert resolved != injected
    assert resolved.name == "git"


@pytest.mark.parametrize("name", ["/tmp/git", "../git", "bin/git", ""])
def test_system_executable_resolution_rejects_non_bare_name(name: str) -> None:
    with pytest.raises(TrustedCommandError, match="bare name"):
        resolve_trusted_system_executable(name)


def test_trusted_executable_rejects_shell_hidden_behind_alias(tmp_path: Path) -> None:
    alias = tmp_path / "pytest"
    alias.symlink_to("/bin/sh")

    with pytest.raises(VerificationReceiptError, match="cannot invoke a shell"):
        resolve_trusted_executable(
            str(alias),
            forbidden_roots=(),
            expected_sha256=None,
            materialization_root=tmp_path / "materialized",
        )


def test_trusted_executable_is_materialized_in_runner_owned_storage(
    tmp_path: Path,
) -> None:
    source = resolve_trusted_system_executable("true")
    digest = stable_file_sha256(source)
    executable = resolve_trusted_executable(
        str(source),
        forbidden_roots=(),
        expected_sha256=digest,
        materialization_root=tmp_path / "materialized",
    )
    try:
        assert executable.materialized_path != source
        assert executable.materialized_path.parent == executable.materialization_root
        assert stable_file_sha256(executable.materialized_path) == digest
        assert executable.materialized_path.stat().st_mode & 0o222 == 0
        assert executable.materialization_root.stat().st_mode & 0o222 == 0
        validate_trusted_executable(executable)
    finally:
        cleanup_trusted_executable(executable)


def test_trusted_executable_rejects_shebang_wrapper(tmp_path: Path) -> None:
    wrapper = tmp_path / "pytest"
    write_safe(wrapper, "#!/usr/bin/python3\n")
    wrapper.chmod(0o755)

    with pytest.raises(VerificationReceiptError, match="shebang wrappers"):
        resolve_trusted_executable(
            str(wrapper),
            forbidden_roots=(),
            expected_sha256=hashlib.sha256(b"#!/usr/bin/python3\n").hexdigest(),
            materialization_root=tmp_path / "materialized",
        )


def test_immutable_store_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    store = ImmutableFileStore(linked)

    with pytest.raises(ImmutableStoreError, match="symlinked"):
        store.write(PurePosixPath("receipt.json"), b"payload")


def test_immutable_store_rejects_symlinked_file(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    target = tmp_path / "target"
    write_safe(target, "payload")
    (root / "receipt.json").symlink_to(target)
    store = ImmutableFileStore(root)

    with pytest.raises(ImmutableStoreError, match="symlinked"):
        store.read(PurePosixPath("receipt.json"), max_bytes=1024)


def test_immutable_store_rejects_hardlinked_file(tmp_path: Path) -> None:
    store = ImmutableFileStore(tmp_path / "store")
    relative = PurePosixPath("receipt.json")
    store.write(relative, b"payload")
    os.link(store.root / "receipt.json", tmp_path / "alias.json")

    with pytest.raises(ImmutableStoreError, match="hard link"):
        store.read(relative, max_bytes=1024)


def test_immutable_store_never_overwrites_different_content(tmp_path: Path) -> None:
    store = ImmutableFileStore(tmp_path / "store")
    relative = PurePosixPath("receipt.json")
    store.write(relative, b"first")

    with pytest.raises(ImmutableStoreError, match="different content"):
        store.write(relative, b"second")

    assert store.read(relative, max_bytes=1024) == b"first"


def test_immutable_store_detects_ancestor_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ImmutableFileStore(tmp_path / "store")
    relative = PurePosixPath("receipt.json")
    store.write(relative, b"payload")
    moved = tmp_path / "moved-store"
    original_read = os.read
    swapped = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            store.root.rename(moved)
            store.root.mkdir()
        return original_read(descriptor, size)

    monkeypatch.setattr("vibe.core._immutable_store.os.read", replacing_read)

    with pytest.raises(ImmutableStoreError, match="ancestor changed"):
        store.read(relative, max_bytes=1024)


def test_immutable_store_detects_listed_directory_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ImmutableFileStore(tmp_path / "store")
    store.write(PurePosixPath("receipts/repository/receipt.json"), b"payload")
    moved = tmp_path / "moved-receipts"
    original_listdir = os.listdir
    swapped = False

    def replacing_listdir(path) -> list[str]:
        nonlocal swapped
        entries = original_listdir(path)
        if not swapped:
            swapped = True
            (store.root / "receipts").rename(moved)
            (store.root / "receipts").mkdir()
        return entries

    monkeypatch.setattr("vibe.core._immutable_store.os.listdir", replacing_listdir)

    with pytest.raises(ImmutableStoreError, match="ancestor changed"):
        store.list_directory(PurePosixPath("receipts/repository"))


def test_environment_attestation_is_revalidated(tmp_path: Path) -> None:
    attestation = tmp_path / "environment.json"
    write_safe(attestation, '{"lock":"v1"}\n')
    digest = hashlib.sha256(read_safe(attestation).text.encode()).hexdigest()
    bound = resolve_environment_attestation(
        str(attestation), digest, forbidden_roots=()
    )
    assert bound is not None

    write_safe(attestation, '{"lock":"v2"}\n')

    with pytest.raises(VerificationReceiptError, match="changed during"):
        validate_environment_attestation(bound)


def test_environment_attestation_rejects_symlink_ancestry(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    attestation = real / "environment.json"
    write_safe(attestation, "environment")
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    digest = hashlib.sha256(b"environment").hexdigest()

    with pytest.raises(VerificationReceiptError, match="read safely"):
        resolve_environment_attestation(
            str(linked / "environment.json"), digest, forbidden_roots=()
        )


class _BrokenPipe:
    def read(self, _size: int) -> bytes:
        raise OSError("collector failed")


def test_output_collector_records_reader_failure() -> None:
    output = _CombinedOutput(1024)

    _drain_pipe(cast(BinaryIO, _BrokenPipe()), "stdout", output)

    diagnostic = output.collector_diagnostic(readers_alive=False)
    assert diagnostic is not None
    assert "collector failed" in diagnostic
    assert output.completed_readers == {"stdout"}


def test_trusted_sandbox_policy_exposes_only_exported_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    source_root.mkdir(parents=True)
    captured: list[SandboxSpec] = []
    executable = TrustedExecutable(
        lexical_path=Path("/usr/bin/true"),
        resolved_path=Path("/usr/bin/true"),
        materialization_root=run_root / "materialized",
        materialized_path=run_root / "materialized" / "executable",
        sha256="a" * 64,
        source_identity=(1, 2, 3, 4, 5),
        materialized_identity=(6, 7, 8, 9, 10),
        read_roots=(Path("/usr"),),
    )
    snapshot = FrozenSourceSnapshot(
        run_root=run_root,
        source_root=source_root,
        candidate_head="b" * 40,
        candidate_tree="c" * 40,
        content_sha256="d" * 64,
    )

    def capture(spec: SandboxSpec, _backend: str):
        captured.append(spec)
        return ["bwrap", "--"], "bwrap", None

    monkeypatch.setattr(
        "vibe.core._verification_runner.detect_backend", lambda _override: "bwrap"
    )
    monkeypatch.setattr(
        "vibe.core._verification_runner._git_common_root", lambda _root: git_common
    )
    monkeypatch.setattr(
        "vibe.core._verification_runner.resolve_trusted_executable",
        lambda *_args, **_kwargs: executable,
    )
    monkeypatch.setattr("vibe.core._verification_runner.build_sandbox_command", capture)

    invocation = _prepare_sandbox_invocation(
        TrustedCheck(name="policy", argv=("true",)), repository, snapshot, source_root
    )
    try:
        [spec] = captured
        assert source_root in spec.read_roots
        assert repository not in spec.read_roots
        assert git_common not in spec.read_roots
        assert executable.materialization_root in spec.read_roots
        assert not (source_root / ".git").exists()
        assert any(repository.is_relative_to(root) for root in spec.hidden_roots)
        assert invocation.argv[-4:] == [
            "--argv0",
            str(executable.lexical_path),
            "--",
            str(executable.materialized_path),
        ]
    finally:
        _cleanup_sandbox_invocation(invocation)
