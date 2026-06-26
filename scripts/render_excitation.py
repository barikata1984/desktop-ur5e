"""Render an optimized excitation trajectory as a video on MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
import tyro

from ur5e_sim.core.model_builder import build_ur5e_model
from ur5e_sim.core.renderer import add_ee_frame_overlay
from ur5e_sim.identification.io import load_optimization_result, result_to_trajectory


@dataclass
class RenderPlaybackConfig:
    model: str = ""  # kept for backwards compat; unused (build_ur5e_model used instead)
    result_json: str = "results/excitation_result.json"
    output: str = "results/excitation_playback.mp4"
    width: int = 960
    height: int = 544
    camera: str | None = None
    fps_video: int = 30
    show_ee_frame: bool = True
    axis_length: float = 0.15
    playback_speed: float = 1.0
    multi_camera: bool = False
    grid_cameras: tuple[str, ...] = ("", "view_x", "view_y", "view_z")
    save_frames: bool = False
    frames_dir: str = "results/frames"


def _render_single_view(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str | None,
    show_ee_frame: bool,
    axis_length: float,
) -> np.ndarray:
    if camera:
        renderer.update_scene(data, camera=camera)
    else:
        renderer.update_scene(data)
    if show_ee_frame:
        add_ee_frame_overlay(model, data, renderer.scene, axis_length)
    return renderer.render().copy()


def main() -> None:
    config = tyro.cli(RenderPlaybackConfig)

    print(f"Loading result from {config.result_json}")
    result = load_optimization_result(config.result_json)
    trajectory = result_to_trajectory(result)
    traj_fps = result.config.fps
    duration = result.config.duration
    n_steps = len(trajectory.time)
    print(f"Trajectory: {n_steps} steps, {duration:.1f}s, {traj_fps:.0f} fps")

    model, data = build_ur5e_model(payload_xml="scenes/objects/payload_flat.xml")

    tile_w = min(config.width, int(model.vis.global_.offwidth))
    tile_h = min(config.height, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=tile_h, width=tile_w)

    if config.multi_camera:
        cameras: list[str | None] = [(c if c else None) for c in config.grid_cameras]
        while len(cameras) < 4:
            cameras.append(None)
        cameras = cameras[:4]

    video_fps = config.fps_video
    speed = config.playback_speed
    video_duration = duration / speed
    n_video_frames = int(video_duration * video_fps)
    traj_indices = np.linspace(0, n_steps - 1, n_video_frames, dtype=int)

    print(
        f"Rendering {n_video_frames} frames at {video_fps} fps "
        f"({video_duration:.1f}s video, {speed}x speed)"
    )

    frames: list[np.ndarray] = []
    n_joints = trajectory.position.shape[1]
    frame_dirs: dict[str, Path] = {}

    if config.save_frames:
        base_dir = Path(config.frames_dir)
        if config.multi_camera:
            for cam in cameras:
                label = cam or "default"
                cam_dir = base_dir / label
                cam_dir.mkdir(parents=True, exist_ok=True)
                frame_dirs[label] = cam_dir
        else:
            label = config.camera or "default"
            cam_dir = base_dir / label
            cam_dir.mkdir(parents=True, exist_ok=True)
            frame_dirs[label] = cam_dir

    for frame_idx, traj_idx in enumerate(traj_indices):
        q = trajectory.position[traj_idx]
        data.qpos[:n_joints] = q
        mujoco.mj_forward(model, data)

        if config.multi_camera:
            tiles = []
            for cam in cameras:
                tile = _render_single_view(
                    renderer, model, data, cam, config.show_ee_frame, config.axis_length
                )
                tiles.append(tile)
                if config.save_frames:
                    label = cam or "default"
                    iio.imwrite(str(frame_dirs[label] / f"{frame_idx:04d}.png"), tile)
            top = np.concatenate([tiles[0], tiles[1]], axis=1)
            bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
            grid = np.concatenate([top, bottom], axis=0)
            frames.append(grid)
        else:
            image = _render_single_view(
                renderer, model, data, config.camera, config.show_ee_frame, config.axis_length
            )
            frames.append(image)
            if config.save_frames:
                label = config.camera or "default"
                iio.imwrite(str(frame_dirs[label] / f"{frame_idx:04d}.png"), image)

        if (frame_idx + 1) % 100 == 0 or frame_idx == n_video_frames - 1:
            t_traj = trajectory.time[traj_idx]
            print(f"  frame {frame_idx + 1}/{n_video_frames} (t={t_traj:.2f}s)")

    renderer.close()

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_array = np.stack(frames)
    iio.imwrite(str(output_path), video_array, fps=video_fps, macro_block_size=1)
    print(f"\nVideo saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
