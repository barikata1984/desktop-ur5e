from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class NpzLogger:
    """Accumulate named values per step and save them as a compressed .npz."""

    def __init__(self):
        self._fields: dict[str, list[Any]] = defaultdict(list)

    def record(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self._fields[key].append(value)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.array(values) for key, values in self._fields.items()}
        np.savez_compressed(path, **arrays)
        return path
