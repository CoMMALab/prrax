"""Viser checkboxes that show a solved path raw, shortcut, B-spline smoothed, or both (one MR. POP
simplify round on the GPU per combination, cached)."""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

from prrax.jax.prrtc import simplify_paths


def add_simplify_toggles(server, robot, ee_idx: int, path, collision_context, lo, hi, step_size: float,
                         slider, color=(0, 220, 60), line_width: float = 3.0) -> list:
    """Draws the path's end-effector curve at /solution_path and keeps `slider` on its waypoints.
    Returns a one-element list holding the shown path (updated when a box is toggled)."""
    raw = np.asarray(path, np.float32)
    cache = {(False, False): (raw, 0.0)}
    shown = [raw]
    shortcut = server.gui.add_checkbox("Shortcut", initial_value=False)
    spline = server.gui.add_checkbox("Spline (B-spline smoothing)", initial_value=False)
    info = server.gui.add_markdown("")

    def show(_=None) -> None:
        key = (shortcut.value, spline.value)
        if key not in cache:
            t0 = time.perf_counter()
            p = simplify_paths([raw], collision_context, shortcut=key[0], smooth=key[1], step_size=step_size,
                               min_vals=lo, max_vals=hi)[0]
            cache[key] = (p, (time.perf_counter() - t0) * 1e3)
        p, ms = cache[key]
        ee = np.asarray(robot.forward_kinematics(jnp.asarray(p))[:, ee_idx, 4:7])
        server.scene.add_spline_catmull_rom("/solution_path", positions=ee, color=color, line_width=line_width)
        shown[0] = p
        slider.max = len(p) - 1
        slider.value = min(slider.value, len(p) - 1)
        length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        info.content = (f"{len(p)} waypoints, joint length {length:.3f}"
                        + (f" ({ms:.1f} ms to simplify)" if ms else ""))
        print(f"  path (shortcut={key[0]}, spline={key[1]}): {info.content}")

    shortcut.on_update(show)
    spline.on_update(show)
    show()
    return shown
