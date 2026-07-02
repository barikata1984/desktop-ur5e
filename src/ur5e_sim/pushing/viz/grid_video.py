"""Render a trial's push trajectory as a polished 2x2 grid video.

Standard experiment-video format for this project: replays the joint/slider
trajectory stored in ``<trial>/data.npz`` and renders four synchronized panes
(overview / top / front / side) with glare suppressed (matte materials, dimmed
lights, planar reflection off) and the goal marker visible. The overview pane
carries a time / slider-y overlay.

Usage:
    pixi run python scripts/render_video.py <trial_dir>

Expects ``data.npz`` to contain ``joint_pos`` (N x 6 arm angles) and the slider
pose series ``slider_x``, ``slider_y``, ``slider_quat`` (N x 4, w x y z), as
produced by the push runner. Output: ``<trial_dir>/result_grid.mp4``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ur5e_sim.core.layout import DofLayout
from ur5e_sim.pushing.scene import build_push_model

PANE_W, PANE_H = 480, 360
FPS = 30
SLIDER_Z = 0.342  # slider rests on the work surface; z is not logged

# (label, azimuth, elevation, distance, lookat) -- tuned to the workspace.
CAMERAS = [
    ("overview", 130, -22, 1.90, (0.0, 0.60, 0.34)),
    ("top", 90, -90, 1.55, (0.0, 0.62, 0.33)),
    ("front", 270, -7, 1.50, (0.0, 0.62, 0.40)),
    ("side", 0, -7, 1.62, (0.0, 0.55, 0.40)),
]


def apply_antiglare(m: mujoco.MjModel) -> None:
    """Matte materials + dim lights (render-only; physics and XML unchanged)."""
    m.mat_reflectance[:] = 0.0
    m.mat_specular[:] = 0.0
    m.vis.headlight.diffuse[:] = 0.35
    m.vis.headlight.specular[:] = 0.0
    m.vis.headlight.ambient[:] = 0.4
    for i in range(m.nlight):
        m.light_diffuse[i] = [0.35, 0.35, 0.35]
        m.light_specular[i] = [0.0, 0.0, 0.0]


def render_grid_video(trial_dir: Path) -> Path:
    m, d = build_push_model()
    apply_antiglare(m)
    layout = DofLayout.from_model(m)

    # Slider is a freejoint (see scenes/objects/slider.xml): 3 pos + 4 quat qpos entries,
    # starting at its own jnt_qposadr (not fixed literals).
    slider_joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slider_joint")
    if slider_joint_id < 0:
        raise ValueError("Joint 'slider_joint' not found in model")
    slider_qpos_addr = int(m.jnt_qposadr[slider_joint_id])
    slider_pos_qpos = slice(slider_qpos_addr, slider_qpos_addr + 3)
    slider_quat_qpos = slice(slider_qpos_addr + 3, slider_qpos_addr + 7)

    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")
    grip_closed = m.key_qpos[key][layout.gripper_qpos].copy()

    z = np.load(trial_dir / "data.npz")
    jp, sx, sy, squat, t = (
        z["joint_pos"],
        z["slider_x"],
        z["slider_y"],
        z["slider_quat"],
        z["time"],
    )
    n = len(jp)
    print(f"replaying {n} frames from {trial_dir}")

    frames_dir = Path("/tmp/grid_video_frames")
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("f_*.png"):
        old.unlink()

    renderer = mujoco.Renderer(m, height=PANE_H, width=PANE_W)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    def pane(az: float, el: float, dist: float, lookat: tuple) -> Image.Image:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        renderer.update_scene(d, camera=cam)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return Image.fromarray(renderer.render())

    for i in range(n):
        d.qpos[layout.arm_qpos] = jp[i]
        d.qpos[layout.gripper_qpos] = grip_closed
        d.qpos[slider_pos_qpos] = [sx[i], sy[i], SLIDER_Z]
        d.qpos[slider_quat_qpos] = squat[i]
        mujoco.mj_forward(m, d)

        grid = Image.new("RGB", (2 * PANE_W, 2 * PANE_H), (20, 20, 20))
        for k, (name, az, el, dist, la) in enumerate(CAMERAS):
            img = pane(az, el, dist, la)
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, 120, 30], fill=(0, 0, 0))
            dr.text((6, 4), name, fill=(255, 255, 255), font=font)
            if name == "overview":
                dr.text(
                    (6, PANE_H - 28),
                    f"t={t[i] - t[0]:4.1f}s  y={sy[i]:.3f}",
                    fill=(255, 255, 0),
                    font=font,
                )
            grid.paste(img, ((k % 2) * PANE_W, (k // 2) * PANE_H))
        grid.save(frames_dir / f"f_{i:05d}.png")

    out = trial_dir / "result_grid.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(frames_dir / "f_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "fast",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr[-400:]}")
    print(f"saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out
