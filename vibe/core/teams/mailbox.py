from __future__ import annotations

from pathlib import Path
import time
from uuid import uuid4

from filelock import FileLock

from vibe.core.logger import logger
from vibe.core.teams.models import Message


class Mailbox:
    def __init__(self, team_dir: Path) -> None:
        self._team_dir = team_dir
        self._mailbox_dir = team_dir / "mailbox"
        self._mailbox_dir.mkdir(parents=True, exist_ok=True)

    def send(self, from_name: str, to_name: str, content: str) -> Message:
        msg = Message(
            id=str(uuid4()),
            from_name=from_name,
            to_name=to_name,
            content=content,
            timestamp=time.time(),
        )
        inbox = self._mailbox_dir / to_name
        inbox.mkdir(parents=True, exist_ok=True)
        msg_file = inbox / f"{msg.id}.json"
        lock = FileLock(str(inbox / ".lock"), timeout=5)
        with lock:
            msg_file.write_text(msg.model_dump_json(indent=2))
        return msg

    def read(self, recipient: str, *, mark_read: bool = True) -> list[Message]:
        inbox = self._mailbox_dir / recipient
        if not inbox.is_dir():
            return []
        messages: list[Message] = []
        lock = FileLock(str(inbox / ".lock"), timeout=5)
        with lock:
            for msg_file in sorted(inbox.glob("*.json")):
                try:
                    msg = Message.model_validate_json(msg_file.read_text())
                    if mark_read and not msg.read:
                        msg.read = True
                        msg_file.write_text(msg.model_dump_json(indent=2))
                    messages.append(msg)
                except Exception as e:
                    logger.warning("Failed to read message %s: %s", msg_file, e)
        return messages

    def get_unread(self, recipient: str) -> list[Message]:
        inbox = self._mailbox_dir / recipient
        if not inbox.is_dir():
            return []
        messages: list[Message] = []
        lock = FileLock(str(inbox / ".lock"), timeout=5)
        with lock:
            for msg_file in sorted(inbox.glob("*.json")):
                try:
                    msg = Message.model_validate_json(msg_file.read_text())
                    if not msg.read:
                        messages.append(msg)
                except Exception as e:
                    logger.warning("Failed to read message %s: %s", msg_file, e)
        return messages

    def clear(self, recipient: str) -> None:
        inbox = self._mailbox_dir / recipient
        if not inbox.is_dir():
            return
        lock = FileLock(str(inbox / ".lock"), timeout=5)
        with lock:
            for msg_file in inbox.glob("*.json"):
                msg_file.unlink(missing_ok=True)
