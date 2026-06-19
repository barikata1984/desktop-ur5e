from __future__ import annotations

from typing import Any, Protocol


class Controller(Protocol):
    """Base controller protocol. Task-specific controllers implement this."""

    def compute_control(self, state: Any) -> Any: ...
