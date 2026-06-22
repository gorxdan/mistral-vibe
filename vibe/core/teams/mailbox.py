from __future__ import annotations

from pathlib import Path
import re
import time
from uuid import uuid4

from filelock import FileLock

from vibe.core.logger import logger
from vibe.core.teams.models import Message
from vibe.core.utils.io import read_safe

# A recipient name becomes a path component (the per-recipient inbox dir). These
# names reach the mailbox from model-controlled tool args, so they must be a
# single safe component — no separators, no "..", no absolute paths — to prevent
# reading/writing outside the mailbox directory (path traversal).
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _safe_name(name: str) -> str:
    if name == ".." or not _NAME_RE.match(name):
        raise ValueError(f"invalid team member name: {name!r}")
    return name


class Mailbox:
    def __init__(self, team_dir: Path) -> None:
        self._team_dir = team_dir
        self._mailbox_dir = team_dir / "mailbox"
        self._mailbox_dir.mkdir(parents=True, exist_ok=True)

    def _inbox(self, name: str) -> Path:
        return self._mailbox_dir / _safe_name(name)

    def send(self, from_name: str, to_name: str, content: str) -> Message:
        # Validate both names: to_name forms the inbox path; from_name is
        # recorded on the message and used as a recipient elsewhere.
        _safe_name(from_name)
        inbox = self._inbox(to_name)
        msg = Message(
            id=str(uuid4()),
            from_name=from_name,
            to_name=to_name,
            content=content,
            timestamp=time.time(),
        )
        inbox.mkdir(parents=True, exist_ok=True)
        msg_file = inbox / f"{msg.id}.json"
        lock = FileLock(str(inbox / ".lock"), timeout=5)
        with lock:
            msg_file.write_text(msg.model_dump_json(indent=2))
        return msg

    def _read_in_order(self, inbox: Path) -> list[tuple[Path, Message]]:
        """Return (file, message) pairs ordered by send time, then id.

        Filenames are random uuid4 strings, so a lexical glob sort does not
        reflect send order. Sort by the message timestamp (tiebroken by id)
        so recipients see messages in the order they were sent.
        """
        pairs: list[tuple[Path, Message]] = []
        for msg_file in inbox.glob("*.json"):
            try:
                msg = Message.model_validate_json(read_safe(msg_file).text)
            except Exception as e:
                logger.warning("Failed to read message %s: %s", msg_file, e)
                continue
            pairs.append((msg_file, msg))
        pairs.sort(key=lambda p: (p[1].timestamp, p[1].id))
        return pairs

    def read(self, recipient: str, *, mark_read: bool = True) -> list[Message]:
        inbox = self._inbox(recipient)
        if not inbox.is_dir():
            return []
        messages: list[Message] = []
        lock = FileLock(str(inbox / ".lock"), timeout=5)
        with lock:
            for msg_file, msg in self._read_in_order(inbox):
                if mark_read and not msg.read:
                    msg.read = True
                    msg_file.write_text(msg.model_dump_json(indent=2))
                messages.append(msg)
        return messages

    def get_unread(self, recipient: str) -> list[Message]:
        inbox = self._inbox(recipient)
        if not inbox.is_dir():
            return []
        messages: list[Message] = []
        lock = FileLock(str(inbox / ".lock"), timeout=5)
        with lock:
            for _msg_file, msg in self._read_in_order(inbox):
                if not msg.read:
                    messages.append(msg)
        return messages

    def clear(self, recipient: str) -> None:
        inbox = self._inbox(recipient)
        if not inbox.is_dir():
            return
        lock = FileLock(str(inbox / ".lock"), timeout=5)
        with lock:
            for msg_file in inbox.glob("*.json"):
                msg_file.unlink(missing_ok=True)
