from __future__ import annotations

import subprocess
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image

from ur5e_sim.core.types import FramePose, get_body_frame, get_site_frame


class FrameRenderer:
    """Off-screen renderer that writes one PNG per captured frame."""

    def __init__(
        self,
        model: mujoco.MjModel,
        pics_dir: Path,
        width: int,
        height: int,
        camera: str,
    ):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.pics_dir = Path(pics_dir)
        self.frame_count = 0

    def capture(self, data: mujoco.MjData) -> None:
        self.renderer.update_scene(data, camera=self.cam_id)
        pixels = self.renderer.render()
        Image.fromarray(pixels).save(self.pics_dir / f"frame_{self.frame_count:06d}.png")
        self.frame_count += 1

    def close(self) -> None:
        del self.renderer


def encode_video(pics_dir: Path, output_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(Path(pics_dir) / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "fast",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[-500:]}")
    else:
        print(f"Video saved: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def _add_connector(
    scene: mujoco.MjvScene, from_point: np.ndarray, to_point: np.ndarray, rgba: np.ndarray
) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 16.0, from_point, to_point)
    scene.ngeom += 1


def _add_frame_overlay(
    scene: mujoco.MjvScene,
    frame: FramePose | None,
    axis_length: float,
    colors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    if frame is None:
        return

    origin = np.array(frame.position, dtype=np.float64)
    rotation = np.array(frame.rotation, dtype=np.float64)
    for axis_index, color in enumerate(colors):
        direction = rotation[:, axis_index]
        _add_connector(scene, origin, origin + axis_length * direction, color)


def add_base_frame_overlay(
    model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene, axis_length: float
) -> None:
    colors = (
        np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
        np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),
        np.array([0.1, 0.4, 1.0, 1.0], dtype=np.float32),
    )
    _add_frame_overlay(scene, get_body_frame(model, data, "base"), axis_length, colors)


def add_ee_frame_overlay(
    model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene, axis_length: float
) -> None:
    colors = (
        np.array([1.0, 0.4, 0.4, 1.0], dtype=np.float32),
        np.array([0.4, 1.0, 0.4, 1.0], dtype=np.float32),
        np.array([0.4, 0.7, 1.0, 1.0], dtype=np.float32),
    )
    frame = get_site_frame(model, data, "attachment_site")
    _add_frame_overlay(scene, frame, axis_length, colors)


def render_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: str | Path,
    width: int = 1280,
    height: int = 960,
    camera: str | None = None,
    show_base_frame: bool = False,
    show_ee_frame: bool = False,
    axis_length: float = 0.3,
) -> Path:
    """Render a single frame of the current state to an image file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = min(width, int(model.vis.global_.offwidth))
    height = min(height, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=height, width=width)
    if camera is None:
        renderer.update_scene(data)
    else:
        renderer.update_scene(data, camera=camera)

    if show_base_frame:
        add_base_frame_overlay(model, data, renderer.scene, axis_length)
    if show_ee_frame:
        add_ee_frame_overlay(model, data, renderer.scene, axis_length)

    image = renderer.render()
    iio.imwrite(output_path, image)
    renderer.close()
    return output_path
