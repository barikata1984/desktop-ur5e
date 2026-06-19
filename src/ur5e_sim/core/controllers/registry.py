from __future__ import annotations

from typing import Callable

from ur5e_sim.core.controllers.base import Controller

_REGISTRY: dict[str, type] = {}


def register_controller(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls

    return deco


def make_controller(name: str, **kwargs) -> Controller:
    """Instantiate a registered controller by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown controller '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_controllers() -> list[str]:
    return sorted(_REGISTRY)
