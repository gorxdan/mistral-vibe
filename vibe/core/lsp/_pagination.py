from __future__ import annotations

import base64
import binascii
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import hashlib
import hmac
import secrets
import threading
import time
from typing import Any

import orjson

_DEFAULT_TTL_SECONDS = 120.0
_DEFAULT_MAX_SNAPSHOTS = 16
_DEFAULT_MAX_RETAINED_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_PAGE_SIZE = 1_000
_ITEM_MEMORY_OVERHEAD = 64
_TOKEN_PREFIX = "lspc1."
_TOKEN_VERSION = 1
_HANDLE_BYTES = 16
_OFFSET_BYTES = 8
_SIGNATURE_BYTES = hashlib.sha256().digest_size
_TOKEN_BODY_BYTES = 1 + _HANDLE_BYTES + _OFFSET_BYTES
_TOKEN_BYTES = _TOKEN_BODY_BYTES + _SIGNATURE_BYTES
_MAX_TOKEN_LENGTH = 256
_INVALID_TOKEN_MESSAGE = (
    "Invalid or expired LSP continuation token; rerun the original query."
)


class LspContinuationError(ValueError):
    def __init__(self) -> None:
        super().__init__(_INVALID_TOKEN_MESSAGE)


class LspContinuationReloadRequired(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "The continuation snapshot requires the original query to be reloaded."
        )


class LspContinuationSerializationError(ValueError):
    def __init__(self) -> None:
        super().__init__("LSP continuation items must be JSON-serializable.")


@dataclass(frozen=True, slots=True, kw_only=True)
class LspQueryBinding:
    operation: str
    file_path: str | None = field(repr=False)
    line: int | None
    character: int | None
    query: str | None = field(repr=False)
    session_id: str | None = field(repr=False)
    task_brief_hash: str | None = field(repr=False)
    lsp_generation: int
    workspace_root: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class LspContinuationPage:
    _encoded_items: tuple[bytes, ...] = field(repr=False)
    offset: int
    total_count: int
    continuation_token: str | None = field(default=None, repr=False)

    @property
    def items(self) -> tuple[Any, ...]:
        return tuple(orjson.loads(item) for item in self._encoded_items)

    @property
    def returned_count(self) -> int:
        return len(self._encoded_items)

    @property
    def has_more(self) -> bool:
        return self.continuation_token is not None


@dataclass(frozen=True, slots=True)
class _Snapshot:
    handle: bytes
    binding_digest: bytes
    items_digest: bytes
    total_count: int
    page_size: int
    expires_at: float
    retained_items: tuple[bytes, ...] | None = field(repr=False)
    retained_bytes: int


@dataclass(frozen=True, slots=True)
class _ScanResult:
    page_items: tuple[bytes, ...]
    retained_items: tuple[bytes, ...] | None
    retained_bytes: int
    items_digest: bytes
    total_count: int


class LspContinuationStore:
    def __init__(
        self,
        *,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_snapshots: int = _DEFAULT_MAX_SNAPSHOTS,
        max_retained_bytes: int = _DEFAULT_MAX_RETAINED_BYTES,
        max_snapshot_bytes: int = _DEFAULT_MAX_SNAPSHOT_BYTES,
        max_page_size: int = _DEFAULT_MAX_PAGE_SIZE,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be at least 1")
        if max_retained_bytes < 0:
            raise ValueError("max_retained_bytes cannot be negative")
        if max_snapshot_bytes < 0:
            raise ValueError("max_snapshot_bytes cannot be negative")
        if max_page_size < 1:
            raise ValueError("max_page_size must be at least 1")
        self._ttl_seconds = ttl_seconds
        self._max_snapshots = max_snapshots
        self._max_retained_bytes = max_retained_bytes
        self._max_snapshot_bytes = max_snapshot_bytes
        self._max_page_size = max_page_size
        self._clock = clock or time.monotonic
        self._secret = secrets.token_bytes(32)
        self._snapshots: OrderedDict[bytes, _Snapshot] = OrderedDict()
        self._retained_bytes = 0
        self._lock = threading.RLock()

    @property
    def snapshot_count(self) -> int:
        with self._lock:
            self._prune_expired_locked(self._clock())
            return len(self._snapshots)

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            self._prune_expired_locked(self._clock())
            return self._retained_bytes

    def first_page(
        self, binding: LspQueryBinding, items: Iterable[Any], *, page_size: int
    ) -> LspContinuationPage:
        if page_size < 1 or page_size > self._max_page_size:
            raise ValueError(f"page_size must be between 1 and {self._max_page_size}")
        scan = self._scan_items(
            items, offset=0, page_size=page_size, retain_limit=self._max_snapshot_bytes
        )
        now = self._clock()
        if scan.total_count <= page_size:
            with self._lock:
                self._prune_expired_locked(now)
            return LspContinuationPage(
                scan.page_items, offset=0, total_count=scan.total_count
            )

        binding_digest = self._binding_digest(binding)
        with self._lock:
            self._prune_expired_locked(now)
            retained_items = scan.retained_items
            retained_bytes = scan.retained_bytes
            if retained_bytes > self._max_retained_bytes:
                retained_items = None
                retained_bytes = 0
            while self._snapshots and (
                len(self._snapshots) >= self._max_snapshots
                or self._retained_bytes + retained_bytes > self._max_retained_bytes
            ):
                self._evict_locked(next(iter(self._snapshots)))
            handle = self._new_handle_locked()
            snapshot = _Snapshot(
                handle=handle,
                binding_digest=binding_digest,
                items_digest=scan.items_digest,
                total_count=scan.total_count,
                page_size=page_size,
                expires_at=now + self._ttl_seconds,
                retained_items=retained_items,
                retained_bytes=retained_bytes,
            )
            self._snapshots[handle] = snapshot
            self._retained_bytes += retained_bytes
            token = self._encode_token(handle, page_size)
        return LspContinuationPage(
            scan.page_items,
            offset=0,
            total_count=scan.total_count,
            continuation_token=token,
        )

    def get_page(
        self,
        token: str,
        binding: LspQueryBinding,
        *,
        reloaded_items: Iterable[Any] | None = None,
    ) -> LspContinuationPage:
        binding_digest = self._binding_digest(binding)
        with self._lock:
            snapshot, offset = self._resolve_locked(token, binding_digest)
            if snapshot.retained_items is not None:
                encoded = snapshot.retained_items[offset : offset + snapshot.page_size]
                next_token = self._next_token(snapshot, offset, len(encoded))
                return LspContinuationPage(
                    encoded,
                    offset=offset,
                    total_count=snapshot.total_count,
                    continuation_token=next_token,
                )
        if reloaded_items is None:
            raise LspContinuationReloadRequired()

        scan = self._scan_items(
            reloaded_items,
            offset=offset,
            page_size=snapshot.page_size,
            retain_limit=None,
        )
        with self._lock:
            current, current_offset = self._resolve_locked(token, binding_digest)
            if current is not snapshot or current_offset != offset:
                raise LspContinuationError()
            if scan.total_count != snapshot.total_count or not hmac.compare_digest(
                scan.items_digest, snapshot.items_digest
            ):
                self._evict_locked(snapshot.handle)
                raise LspContinuationError()
            next_token = self._next_token(snapshot, offset, len(scan.page_items))
        return LspContinuationPage(
            scan.page_items,
            offset=offset,
            total_count=snapshot.total_count,
            continuation_token=next_token,
        )

    def clear(self) -> None:
        with self._lock:
            self._snapshots.clear()
            self._retained_bytes = 0

    def _binding_digest(self, binding: LspQueryBinding) -> bytes:
        payload = orjson.dumps(
            {
                "operation": binding.operation,
                "file_path": binding.file_path,
                "line": binding.line,
                "character": binding.character,
                "query": binding.query,
                "session_id": binding.session_id,
                "task_brief_hash": binding.task_brief_hash,
                "lsp_generation": binding.lsp_generation,
                "workspace_root": binding.workspace_root,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return hmac.digest(self._secret, payload, "sha256")

    def _scan_items(
        self,
        items: Iterable[Any],
        *,
        offset: int,
        page_size: int,
        retain_limit: int | None,
    ) -> _ScanResult:
        digest = hashlib.sha256()
        page: list[bytes] = []
        retained: list[bytes] | None = [] if retain_limit is not None else None
        retained_bytes = 0
        total_count = 0
        for index, item in enumerate(items):
            try:
                encoded = orjson.dumps(item, option=orjson.OPT_SORT_KEYS)
            except TypeError as exc:
                raise LspContinuationSerializationError() from exc
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            if offset <= index < offset + page_size:
                page.append(encoded)
            if retained is not None:
                item_bytes = len(encoded) + _ITEM_MEMORY_OVERHEAD
                if retained_bytes + item_bytes > (retain_limit or 0):
                    retained = None
                    retained_bytes = 0
                else:
                    retained.append(encoded)
                    retained_bytes += item_bytes
            total_count += 1
        digest.update(total_count.to_bytes(8, "big"))
        return _ScanResult(
            page_items=tuple(page),
            retained_items=tuple(retained) if retained is not None else None,
            retained_bytes=retained_bytes,
            items_digest=digest.digest(),
            total_count=total_count,
        )

    def _resolve_locked(
        self, token: str, binding_digest: bytes
    ) -> tuple[_Snapshot, int]:
        self._prune_expired_locked(self._clock())
        handle, offset = self._decode_token(token)
        snapshot = self._snapshots.get(handle)
        if snapshot is None or not hmac.compare_digest(
            snapshot.binding_digest, binding_digest
        ):
            raise LspContinuationError()
        if (
            offset <= 0
            or offset >= snapshot.total_count
            or offset % snapshot.page_size != 0
        ):
            raise LspContinuationError()
        self._snapshots.move_to_end(handle)
        return snapshot, offset

    def _next_token(
        self, snapshot: _Snapshot, offset: int, returned_count: int
    ) -> str | None:
        next_offset = offset + returned_count
        if returned_count == 0 or next_offset >= snapshot.total_count:
            return None
        return self._encode_token(snapshot.handle, next_offset)

    def _encode_token(self, handle: bytes, offset: int) -> str:
        body = bytes([_TOKEN_VERSION]) + handle + offset.to_bytes(_OFFSET_BYTES, "big")
        signature = hmac.digest(self._secret, body, "sha256")
        encoded = base64.urlsafe_b64encode(body + signature).rstrip(b"=").decode()
        return f"{_TOKEN_PREFIX}{encoded}"

    def _decode_token(self, token: str) -> tuple[bytes, int]:
        if len(token) > _MAX_TOKEN_LENGTH or not token.startswith(_TOKEN_PREFIX):
            raise LspContinuationError()
        encoded = token.removeprefix(_TOKEN_PREFIX)
        try:
            raw = base64.b64decode(
                encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
            )
        except (binascii.Error, UnicodeEncodeError, ValueError):
            raise LspContinuationError() from None
        if len(raw) != _TOKEN_BYTES:
            raise LspContinuationError()
        body = raw[:_TOKEN_BODY_BYTES]
        signature = raw[_TOKEN_BODY_BYTES:]
        expected = hmac.digest(self._secret, body, "sha256")
        if not hmac.compare_digest(signature, expected):
            raise LspContinuationError()
        if body[0] != _TOKEN_VERSION:
            raise LspContinuationError()
        handle = body[1 : 1 + _HANDLE_BYTES]
        offset = int.from_bytes(body[1 + _HANDLE_BYTES :], "big")
        return handle, offset

    def _new_handle_locked(self) -> bytes:
        while True:
            handle = secrets.token_bytes(_HANDLE_BYTES)
            if handle not in self._snapshots:
                return handle

    def _prune_expired_locked(self, now: float) -> None:
        expired = [
            handle
            for handle, snapshot in self._snapshots.items()
            if snapshot.expires_at <= now
        ]
        for handle in expired:
            self._evict_locked(handle)

    def _evict_locked(self, handle: bytes) -> None:
        snapshot = self._snapshots.pop(handle, None)
        if snapshot is None:
            return
        self._retained_bytes -= snapshot.retained_bytes
