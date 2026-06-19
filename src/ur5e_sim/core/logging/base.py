from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Logger(Protocol):
    def record(self, **kwargs: Any) -> None: ...
    def save(self, path: Path) -> Path: ...
