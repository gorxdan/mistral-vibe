from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from dataclasses import dataclass

DEFAULT_MAX_WORKSPACE_ROOTS = 8


@dataclass(frozen=True)
class WorkspaceRouteAdmission:
    evicted_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkspaceRoutePoolSnapshot:
    resident_dynamic_roots: int
    max_dynamic_roots: int
    leased_dynamic_roots: int
    resident_roots: int
    known_roots: int
    workspace_symbol_partial: bool
    revision: int


class WorkspaceRoutePool:
    def __init__(self, max_dynamic_roots: int = DEFAULT_MAX_WORKSPACE_ROOTS) -> None:
        if max_dynamic_roots < 1:
            raise ValueError("max_dynamic_roots must be at least 1")
        self._max_dynamic_roots = max_dynamic_roots
        self._dynamic: OrderedDict[str, None] = OrderedDict()
        self._protected: set[str] = set()
        self._known_dynamic: set[str] = set()
        self._leases: Counter[str] = Counter()
        self._condition = asyncio.Condition()
        self._revision = 0

    async def acquire(
        self, root_uri: str, *, protected: bool = False
    ) -> WorkspaceRouteAdmission:
        async with self._condition:
            while True:
                if protected or root_uri in self._protected:
                    self._protect(root_uri)
                    self._leases[root_uri] += 1
                    return WorkspaceRouteAdmission()
                if root_uri in self._dynamic:
                    self._dynamic.move_to_end(root_uri)
                    self._leases[root_uri] += 1
                    return WorkspaceRouteAdmission()
                if len(self._dynamic) < self._max_dynamic_roots:
                    self._admit_dynamic(root_uri)
                    self._leases[root_uri] += 1
                    return WorkspaceRouteAdmission()
                victim = next(
                    (
                        candidate
                        for candidate in self._dynamic
                        if self._leases[candidate] == 0
                    ),
                    None,
                )
                if victim is None:
                    await self._condition.wait()
                    continue
                del self._dynamic[victim]
                self._admit_dynamic(root_uri)
                self._leases[root_uri] += 1
                return WorkspaceRouteAdmission(evicted_roots=(victim,))

    async def release(self, root_uri: str) -> None:
        async with self._condition:
            if self._leases[root_uri] <= 1:
                self._leases.pop(root_uri, None)
            else:
                self._leases[root_uri] -= 1
            self._condition.notify_all()

    async def pin_resident(self, root_uris: tuple[str, ...]) -> tuple[str, ...]:
        async with self._condition:
            resident = tuple(
                root_uri
                for root_uri in dict.fromkeys(root_uris)
                if root_uri in self._protected or root_uri in self._dynamic
            )
            for root_uri in resident:
                self._leases[root_uri] += 1
            return resident

    async def release_many(self, root_uris: tuple[str, ...]) -> None:
        async with self._condition:
            for root_uri in root_uris:
                if self._leases[root_uri] <= 1:
                    self._leases.pop(root_uri, None)
                else:
                    self._leases[root_uri] -= 1
            self._condition.notify_all()

    def touch_or_admit(self, root_uri: str, *, protected: bool = False) -> bool:
        if protected or root_uri in self._protected:
            self._protect(root_uri)
            return True
        if root_uri in self._dynamic:
            self._dynamic.move_to_end(root_uri)
            return True
        if len(self._dynamic) >= self._max_dynamic_roots:
            return False
        self._admit_dynamic(root_uri)
        return True

    def is_resident(self, root_uri: str) -> bool:
        return root_uri in self._protected or root_uri in self._dynamic

    def is_leased(self, root_uri: str) -> bool:
        return self._leases[root_uri] > 0

    def reset(self, *, preserve_leases: bool = False) -> None:
        changed = bool(
            self._dynamic or self._protected or (self._leases and not preserve_leases)
        )
        self._dynamic.clear()
        self._protected.clear()
        self._known_dynamic.clear()
        if not preserve_leases:
            self._leases.clear()
        if changed:
            self._revision += 1

    def snapshot(self) -> WorkspaceRoutePoolSnapshot:
        resident = self._protected | set(self._dynamic)
        known = self._protected | self._known_dynamic
        return WorkspaceRoutePoolSnapshot(
            resident_dynamic_roots=len(self._dynamic),
            max_dynamic_roots=self._max_dynamic_roots,
            leased_dynamic_roots=sum(
                self._leases[root_uri] > 0 for root_uri in self._dynamic
            ),
            resident_roots=len(resident),
            known_roots=len(known),
            workspace_symbol_partial=not known.issubset(resident),
            revision=self._revision,
        )

    def _protect(self, root_uri: str) -> None:
        changed = root_uri not in self._protected
        if root_uri in self._dynamic:
            del self._dynamic[root_uri]
            changed = True
        self._known_dynamic.discard(root_uri)
        self._protected.add(root_uri)
        if changed:
            self._revision += 1

    def _admit_dynamic(self, root_uri: str) -> None:
        self._dynamic[root_uri] = None
        self._known_dynamic.add(root_uri)
        self._revision += 1


__all__ = [
    "DEFAULT_MAX_WORKSPACE_ROOTS",
    "WorkspaceRouteAdmission",
    "WorkspaceRoutePool",
    "WorkspaceRoutePoolSnapshot",
]
