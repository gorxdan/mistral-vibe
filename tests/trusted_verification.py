from __future__ import annotations

from pathlib import Path
import sys

from vibe.core._trusted_host_runner import stable_file_sha256

HOST_PYTHON = Path(sys.executable).resolve()
HOST_PYTHON_SHA256 = stable_file_sha256(HOST_PYTHON)
HOST_ENVIRONMENT = HOST_PYTHON
HOST_ENVIRONMENT_SHA256 = HOST_PYTHON_SHA256

__all__ = [
    "HOST_ENVIRONMENT",
    "HOST_ENVIRONMENT_SHA256",
    "HOST_PYTHON",
    "HOST_PYTHON_SHA256",
]
