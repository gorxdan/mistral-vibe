from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from vibe.core._agent_limits import HOST_AGENT_LANE_LIMIT

__all__ = ["AGENT_SPAWN_BATCH_DENIAL", "SpawnBatchPlan", "plan_agent_spawn_batch"]

AGENT_SPAWN_BATCH_DENIAL = (
    "The host permits at most two agent slots in one assistant tool "
    "batch. This call was not launched. Review the returned agent evidence in a "
    "later assistant turn before requesting another bounded delegation."
)


class _SpawnCall(Protocol):
    @property
    def tool_name(self) -> str: ...

    @property
    def tool_class(self) -> type: ...


@dataclass(frozen=True, slots=True)
class SpawnBatchPlan[CallT]:
    accepted: tuple[CallT, ...]
    rejected: tuple[CallT, ...]


def plan_agent_spawn_batch[CallT: _SpawnCall](
    calls: Sequence[CallT],
) -> SpawnBatchPlan[CallT]:
    accepted: list[CallT] = []
    rejected: list[CallT] = []
    used = 0
    for call in calls:
        cost = _agent_slot_cost(call)
        if cost and used + cost > HOST_AGENT_LANE_LIMIT:
            rejected.append(call)
            continue
        used += cost
        accepted.append(call)
    return SpawnBatchPlan(tuple(accepted), tuple(rejected))


def _agent_slot_cost(call: _SpawnCall) -> int:
    if call.tool_name == "launch_workflow":
        return HOST_AGENT_LANE_LIMIT
    if call.tool_name == "team_spawn" or call.tool_class.is_subagent_spawner:
        return 1
    return 0
