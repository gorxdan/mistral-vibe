from __future__ import annotations

from vibe.core.teams._escalate import EscalationDenied
from vibe.core.teams._safety import require_shared_ask, shared_ask_enabled
from vibe.core.tools.base import ToolError, ToolPermission
from vibe.core.tools.permissions import PermissionContext


def _with_permission_detail(description: str, ctx: PermissionContext | None) -> str:
    if ctx is None:
        return description
    parts = [description]
    if ctx.reason:
        parts.append(f"Reason: {ctx.reason}")
    labels = [rp.label for rp in ctx.required_permissions]
    if labels:
        parts.append(f"Required permissions: {', '.join(labels)}")
    return "\n".join(parts)


async def enforce_shared_ask(
    tool: str,
    description: str,
    ctx: PermissionContext | None,
    default_permission: ToolPermission,
) -> None:
    if not shared_ask_enabled():
        return

    permission = ctx.permission if ctx is not None else default_permission
    if permission is ToolPermission.ALWAYS:
        return
    if permission is ToolPermission.NEVER:
        reason = ctx.reason if ctx is not None else None
        raise ToolError(reason or f"{tool} is disabled in shared-ask mode.")

    try:
        await require_shared_ask(tool, _with_permission_detail(description, ctx))
    except EscalationDenied as exc:
        raise ToolError(str(exc)) from exc
