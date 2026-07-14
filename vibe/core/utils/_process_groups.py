from __future__ import annotations

import os
import signal


def signal_owned_process_group(pid: int, sig: int | signal.Signals) -> bool:
    try:
        process_group_id = os.getpgid(pid)
        session_id = os.getsid(pid)
    except (AttributeError, OSError):
        return False
    if process_group_id != pid or session_id != pid:
        return False
    try:
        os.killpg(process_group_id, sig)
    except (AttributeError, OSError):
        return False
    return True


__all__ = ["signal_owned_process_group"]
