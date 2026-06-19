"""Repo-relative path resolution for the pushing package.

Everything is resolved relative to the repository root (two parents up from this
package), so the project is location-independent and importable after
``pip install -e .``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENES_DIR = REPO_ROOT / "scenes"
ASSETS_DIR = REPO_ROOT / "assets"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_SCENE = "tasks/push.xml"


def scene_path(name: str = DEFAULT_SCENE) -> str:
    """Absolute path to a scene XML under ``scenes/`` (e.g. ``tasks/push.xml``)."""
    return str(SCENES_DIR / name)


def results_dir() -> Path:
    """The (gitignored) directory that holds timestamped trial outputs."""
    return RESULTS_DIR
