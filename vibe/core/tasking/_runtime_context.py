from __future__ import annotations

from pathlib import Path

from vibe.core.config import VibeConfig
from vibe.core.tasking._policy import BoundTaskContract, TaskContractAuthority
from vibe.core.tasking._process_context import load_task_process_context
from vibe.core.usage._process_context import load_spend_process_context
from vibe.core.usage._session import SessionSpendAdapter, SpendAdmissionBlockedError
from vibe.core.verification_state import VerificationState

__all__ = ["bind_process_runtime_context"]


def bind_process_runtime_context(
    config: VibeConfig, workspace_root: Path
) -> tuple[BoundTaskContract | None, SessionSpendAdapter | None]:
    task_context = load_task_process_context()
    spend_context = load_spend_process_context()
    contract = (
        BoundTaskContract.bind(
            task_context.brief,
            authority=TaskContractAuthority.LEAD,
            workspace_root=workspace_root,
            verification_state=VerificationState.from_recipe(
                config.trusted_verification_recipe
            ),
        )
        if task_context is not None
        else None
    )
    if task_context is not None and spend_context is None:
        raise SpendAdmissionBlockedError(
            "task process context requires a host-created spend scope"
        )
    if task_context is None:
        if spend_context is None:
            return None, None
        if spend_context.task_brief_hash is not None:
            raise SpendAdmissionBlockedError(
                "task-bound spend context requires its matching task context"
            )
        return None, SessionSpendAdapter.attach(config.spend, spend_context)
    if spend_context is None or contract is None:
        raise SpendAdmissionBlockedError("task contract could not be bound")
    return contract, SessionSpendAdapter.attach(
        config.spend,
        spend_context,
        required_task_brief_hash=task_context.brief_hash,
        required_limits=contract.spend_limits(),
    )
