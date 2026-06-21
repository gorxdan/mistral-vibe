from __future__ import annotations

from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from typing import Any

from git import Repo
import pytest

from vibe.core.workflows.contract import ContractFailure, ContractReport, ContractSpec
from vibe.core.workflows.runtime import WorkflowRuntime
import vibe.core.worktree.ephemeral as ephemeral

_FAKE_VIBE_WRITE = """\
import os, sys
# Simulate an isolated code agent writing its deliverable into the worktree.
with open(os.path.join(os.getcwd(), "auth.py"), "w") as f:
    f.write("JWT_TOKEN = generate()\\n")
sys.stdout.write("done\\n")
sys.exit(0)
"""

_FAKE_VIBE_NOOP = """\
import sys
sys.stdout.write("did nothing\\n")
sys.exit(0)
"""


def _setup_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    repo = Repo.init(str(tmp_path))
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "t@t.com")
    (tmp_path / "f.txt").write_text("base\n")
    repo.index.add(["f.txt"])
    repo.index.commit("init")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ephemeral, "VIBE_HOME", SimpleNamespace(path=tmp_path / "vh"))
    fake = tmp_path / "fake_vibe.py"
    fake.write_text(body)
    monkeypatch.setenv(
        "VIBE_ISOLATED_EXECUTOR_CMD",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(fake))}",
    )
    return tmp_path


def test_resolve_contract_requires_worktree_isolation() -> None:
    assert WorkflowRuntime._resolve_contract(None, None) is None
    with pytest.raises(Exception, match="isolation='worktree'"):
        WorkflowRuntime._resolve_contract({}, None)
    spec = WorkflowRuntime._resolve_contract(
        {"outputs": [{"path": "a.py"}]}, "worktree"
    )
    assert isinstance(spec, ContractSpec)
    assert spec.outputs[0].path == "a.py"


@pytest.mark.asyncio
async def test_default_executor_contract_passes_and_delivers(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_repo(tmp_path, monkeypatch, _FAKE_VIBE_WRITE)
    rt = WorkflowRuntime()
    contract = ContractSpec.model_validate({
        "outputs": [{"path": "auth.py", "must_contain": ["JWT"]}]
    })
    _output, _stats, report = await rt._default_isolated_executor(
        "impl", "auto-approve", "lbl", 40, contract=contract
    )
    assert report is not None
    assert report.passed
    assert report.delivered
    # Delivery ff-merged the worktree into the parent; the file landed.
    assert (root / "auth.py").read_text().startswith("JWT_TOKEN")


@pytest.mark.asyncio
async def test_default_executor_contract_failure_keeps_work(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_repo(tmp_path, monkeypatch, _FAKE_VIBE_NOOP)
    rt = WorkflowRuntime()
    contract = ContractSpec.model_validate({
        "outputs": [{"path": "auth.py", "must_contain": ["JWT"]}]
    })
    _output, _stats, report = await rt._default_isolated_executor(
        "impl", "auto-approve", "lbl", 40, contract=contract
    )
    assert report is not None
    assert not report.passed
    assert not report.delivered
    # Failed contract -> nothing merged into the parent.
    assert not (root / "auth.py").exists()


def test_isolated_failure_value_returns_contract_failure() -> None:
    rt = WorkflowRuntime()
    failed = ContractReport(passed=False, violations=[])
    result = rt._isolated_failure_value(failed, None, [], "boom", "raw")
    assert isinstance(result, ContractFailure)
    # The report is carried as JSON-safe data (model_dump), not the live object,
    # so the failure survives json.dumps. The passed flag round-trips.
    assert result.report == failed.model_dump(mode="json")
    assert not result.report["passed"]
    assert not result


def test_isolated_failure_value_none_when_no_recoverable_failure() -> None:
    rt = WorkflowRuntime()
    assert rt._isolated_failure_value(None, None, [], "boom", "raw") is None
