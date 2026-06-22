from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RawConfig(BaseModel):
    """Permissive default schema that preserves all fields as extras."""

    model_config = ConfigDict(extra="allow")
