#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open3D BEV skeleton video renderer (robust: headless-safe)

What this fixes vs the previous version:
- In some headless environments, `Visualizer.get_view_control()` returns None (no GL context).
  This file provides an **auto fallback** to OffscreenRenderer.
- You can also force backend="offscreen" or backend="visualizer".

Backends
--------
1) backend="visualizer"  : Open3D Visualizer + capture_screen_float_buffer
   - Needs a valid OpenGL context (desktop OR Xvfb/EGL)
2) backend="offscreen"   : Open3D OffscreenRenderer + render_to_image
   - No window; best for servers/HPC

Requirements
------------
    pip install open3d opencv-python numpy

Quick usage
-----------
    from pathlib import Path
    from o3d_bev_video_robust import Open3DBevVideoRenderer, COCO_EDGES

    r = Open3DBevVideoRenderer(
        out_path=Path("bev_demo.mp4"),
        width=1280, height=720, fps=30,
        edges=COCO_EDGES,
        backend="auto",    # auto -> visualizer if available else offscreen
        visible=False,
    )

    for kpts_world in all_kpts_world:  # (T,J,3)
        r.render(kpts_world)

    r.close()

Tip (if you insist on backend="visualizer" in headless):
    xvfb-run -s "-screen 0 1280x720x24" python run.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np
import cv2
import open3d as o3d


# ----------------------------- Skeleton edges (COCO-ish) -----------------------------
COCO_EDGES: List[Tuple[int, int]] = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
]


@dataclass
class BevView:
    """Camera parameters for a BEV look."""
    # For Visualizer backend (ViewControl)
    front: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    lookat: Tuple[float, float, float] = (0.0, 0.0, 10.0)
    zoom: float = 0.7

    # For Offscreen backend (scene.camera.look_at)
    eye_height: float = 25.0  # meters above lookat


Backend = Literal["auto", "visualizer", "offscreen"]


# =====================================================================================
# Base class (shared utilities)
# =====================================================================================
class _BaseRenderer:
    def __init__(
        self,
        out_path: Path,
        width: int,
        height: int,
        fps: int,
        edges: Sequence[Tuple[int, int]],
        mp4_fourcc: str,
    ) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.edges = np.asarray(list(edges), dtype=np.int32)

        self._video = cv2.VideoWriter(
            str(self.out_path),
            cv2.VideoWriter_fourcc(*mp4_fourcc),
            self.fps,
            (self.width, self.height),
        )
        if not self._video.isOpened():
            raise RuntimeError(
                f"Failed to open video writer: {self.out_path}. "
                f"Try mp4_fourcc='avc1' or ensure your OpenCV has proper codecs."
            )

    def _write_bgr(self, bgr: np.ndarray) -> None:
        if bgr.dtype != np.uint8:
            bgr = bgr.clip(0, 255).astype(np.uint8)
        if bgr.shape[:2] != (self.height, self.width):
            bgr = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self._video.write(bgr)

    def close(self) -> None:
        if getattr(self, "_video", None) is not None:
            self._video.release()
        self._video = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# =====================================================================================
# Visualizer backend
# =====================================================================================
class _VisualizerRenderer(_BaseRenderer):
    def __init__(
        self,
        out_path: Path,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        edges: Sequence[Tuple[int, int]] = COCO_EDGES,
        meters_grid: Tuple[float, float] = (20.0, 30.0),   # (x_range, z_range)
        grid_origin: Tuple[float, float, float] = (-10.0, -0.01, 0.0),
        visible: bool = False,
        view: Optional[BevView] = None,
        mp4_fourcc: str = "mp4v",
        draw_keypoints: bool = True,
        kp_radius: float = 0.08,
    ) -> None:
        super().__init__(out_path, width, height, fps, edges, mp4_fourcc)
        self.draw_keypoints = bool(draw_keypoints)
        self.kp_radius = float(kp_radius)
        self._view = view or BevView()

        self._vis = o3d.visualization.Visualizer()
        ok = self._vis.create_window(width=self.width, height=self.height, visible=visible)
        # create_window may return False in some builds
        if ok is False:
            raise RuntimeError(
                "Open3D Visualizer.create_window() failed (no OpenGL context). "
                "Use backend='offscreen' or run with Xvfb."
            )

        # Scene: ground
        x_range, z_range = meters_grid
        ground = o3d.geometry.TriangleMesh.create_box(x_range, 0.01, z_range)
        ground.translate(np.array(grid_origin, dtype=np.float64))
        ground.paint_uniform_color([0.92, 0.92, 0.92])
        ground.compute_vertex_normals()
        self._vis.add_geometry(ground)

        # Skeleton
        self._skeleton = o3d.geometry.LineSet()
        self._skeleton.lines = o3d.utility.Vector2iVector(self.edges)
        self._skeleton.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * len(self.edges))
        self._vis.add_geometry(self._skeleton)

        # Keypoint spheres
        self._kp_meshes: List[o3d.geometry.TriangleMesh] = []

        # Warm-up to ensure renderer is ready
        self._vis.poll_events()
        self._vis.update_renderer()

        # Apply view if possible (in some headless cases get_view_control is None)
        self._try_apply_view(self._view)

    def _try_apply_view(self, view: BevView) -> None:
        ctr = self._vis.get_view_control()
        if ctr is None:
            # Don't crash; caller can still render (but view may be default).
            # In many headless cases, capture will fail too, but we keep this graceful here.
            return
        ctr.set_front(view.front)
        ctr.set_up(view.up)
        ctr.set_lookat(view.lookat)
        ctr.set_zoom(view.zoom)

    def _ensure_kp_meshes(self, num_joints: int) -> None:
        if not self.draw_keypoints:
            return
        if len(self._kp_meshes) == num_joints:
            return

        for m in self._kp_meshes:
            self._vis.remove_geometry(m, reset_bounding_box=False)
        self._kp_meshes.clear()

        for _ in range(num_joints):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=self.kp_radius)
            s.compute_vertex_normals()
            s.paint_uniform_color([1.0, 0.0, 0.0])
            self._kp_meshes.append(s)
            self._vis.add_geometry(s, reset_bounding_box=False)

    def render(self, kpts_world: np.ndarray) -> np.ndarray:
        kpts_world = np.asarray(kpts_world, dtype=np.float64)
        if kpts_world.ndim != 2 or kpts_world.shape[1] != 3:
            raise ValueError(f"kpts_world must be (J,3), got {kpts_world.shape}")
        J = kpts_world.shape[0]

        self._skeleton.points = o3d.utility.Vector3dVector(kpts_world)
        self._vis.update_geometry(self._skeleton)

        self._ensure_kp_meshes(J)
        if self.draw_keypoints:
            for j, mesh in enumerate(self._kp_meshes):
                center = np.asarray(mesh.get_center(), dtype=np.float64)
                target = kpts_world[j]
                if not np.isfinite(target).all():
                    target = np.array([1e6, 1e6, 1e6], dtype=np.float64)
                mesh.translate(target - center, relative=True)
                self._vis.update_geometry(mesh)

        self._vis.poll_events()
        self._vis.update_renderer()

        img = self._vis.capture_screen_float_buffer(do_render=True)
        rgb = (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._write_bgr(bgr)
        return bgr

    def close(self) -> None:
        try:
            super().close()
        finally:
            if getattr(self, "_vis", None) is not None:
                self._vis.destroy_window()
            self._vis = None


# =====================================================================================
# Offscreen backend (recommended for servers/HPC)
# =====================================================================================
class _OffscreenRenderer(_BaseRenderer):
    def __init__(
        self,
        out_path: Path,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        edges: Sequence[Tuple[int, int]] = COCO_EDGES,
        meters_grid: Tuple[float, float] = (20.0, 30.0),
        grid_origin: Tuple[float, float, float] = (-10.0, -0.01, 0.0),
        view: Optional[BevView] = None,
        mp4_fourcc: str = "mp4v",
        line_width: float = 3.0,
    ) -> None:
        super().__init__(out_path, width, height, fps, edges, mp4_fourcc)
        self._view = view or BevView()

        self._renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self._scene = self._renderer.scene
        self._scene.set_background([1.0, 1.0, 1.0, 1.0])

        # Materials
        self._mat_ground = o3d.visualization.rendering.MaterialRecord()
        self._mat_ground.shader = "defaultLit"

        self._mat_line = o3d.visualization.rendering.MaterialRecord()
        self._mat_line.shader = "unlitLine"
        self._mat_line.line_width = float(line_width)

        # Ground
        x_range, z_range = meters_grid
        ground = o3d.geometry.TriangleMesh.create_box(x_range, 0.01, z_range)
        ground.translate(np.array(grid_origin, dtype=np.float64))
        ground.compute_vertex_normals()
        ground.paint_uniform_color([0.92, 0.92, 0.92])
        self._scene.add_geometry("ground", ground, self._mat_ground)

        # Skeleton lineset placeholder
        self._lineset = o3d.geometry.LineSet()
        self._lineset.lines = o3d.utility.Vector2iVector(self.edges)

        # Camera (BEV)
        lookat = np.array(self._view.lookat, dtype=np.float64)
        eye = lookat + np.array([0.0, self._view.eye_height, 0.0], dtype=np.float64)
        up = np.array(self._view.up, dtype=np.float64)
        self._scene.camera.look_at(lookat, eye, up)

        # Lighting (keeps it visible)
        self._scene.scene.set_sun_light([0.2, -1.0, 0.2], [1.0, 1.0, 1.0], 75000)
        self._scene.scene.enable_sun_light(True)

        self._has_skel = False

    def render(self, kpts_world: np.ndarray) -> np.ndarray:
        kpts_world = np.asarray(kpts_world, dtype=np.float64)
        if kpts_world.ndim != 2 or kpts_world.shape[1] != 3:
            raise ValueError(f"kpts_world must be (J,3), got {kpts_world.shape}")

        self._lineset.points = o3d.utility.Vector3dVector(kpts_world)
        self._lineset.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * len(self.edges))

        # Most stable update strategy: remove + add
        if self._has_skel:
            self._scene.remove_geometry("skeleton")
        self._scene.add_geometry("skeleton", self._lineset, self._mat_line)
        self._has_skel = True

        img = self._renderer.render_to_image()
        rgb = np.asarray(img)[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._write_bgr(bgr)
        return bgr

    def close(self) -> None:
        super().close()
        self._renderer = None
        self._scene = None


# =====================================================================================
# Public wrapper: auto-select backend
# =====================================================================================
class Open3DBevVideoRenderer:
    """
    Public API.

    backend:
        - "auto": try Visualizer first; if it can't get a view control / context, fallback to OffscreenRenderer
        - "visualizer": force Visualizer backend
        - "offscreen": force OffscreenRenderer backend
    """

    def __init__(
        self,
        out_path: Path,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        edges: Sequence[Tuple[int, int]] = COCO_EDGES,
        meters_grid: Tuple[float, float] = (20.0, 30.0),
        grid_origin: Tuple[float, float, float] = (-10.0, -0.01, 0.0),
        visible: bool = False,
        view: Optional[BevView] = None,
        mp4_fourcc: str = "mp4v",
        backend: Backend = "auto",
        draw_keypoints: bool = False,  # offscreen backend doesn't draw spheres; keep False by default
        kp_radius: float = 0.08,
    ) -> None:
        self.backend = backend
        self._impl = None

        if backend == "offscreen":
            self._impl = _OffscreenRenderer(
                out_path=out_path, width=width, height=height, fps=fps,
                edges=edges, meters_grid=meters_grid, grid_origin=grid_origin,
                view=view, mp4_fourcc=mp4_fourcc,
            )
            return

        if backend == "visualizer":
            self._impl = _VisualizerRenderer(
                out_path=out_path, width=width, height=height, fps=fps,
                edges=edges, meters_grid=meters_grid, grid_origin=grid_origin,
                visible=visible, view=view, mp4_fourcc=mp4_fourcc,
                draw_keypoints=draw_keypoints, kp_radius=kp_radius,
            )
            # hard-check view control availability (if missing, it's a sign Visualizer is not usable)
            if self._impl._vis.get_view_control() is None:
                raise RuntimeError(
                    "Visualizer backend has no ViewControl (likely headless without GL). "
                    "Use backend='offscreen' or run with Xvfb."
                )
            return

        # backend == "auto"
        try:
            vis_impl = _VisualizerRenderer(
                out_path=out_path, width=width, height=height, fps=fps,
                edges=edges, meters_grid=meters_grid, grid_origin=grid_origin,
                visible=visible, view=view, mp4_fourcc=mp4_fourcc,
                draw_keypoints=draw_keypoints, kp_radius=kp_radius,
            )
            # If no view control, we treat it as not reliable and fallback
            if vis_impl._vis.get_view_control() is None:
                vis_impl.close()
                raise RuntimeError("No ViewControl in Visualizer.")
            self._impl = vis_impl
        except Exception as e:
            # Fallback to offscreen
            self._impl = _OffscreenRenderer(
                out_path=out_path, width=width, height=height, fps=fps,
                edges=edges, meters_grid=meters_grid, grid_origin=grid_origin,
                view=view, mp4_fourcc=mp4_fourcc,
            )

    def render(self, kpts_world: np.ndarray) -> np.ndarray:
        return self._impl.render(kpts_world)

    def render_many(self, kpts_seq: Iterable[np.ndarray]) -> None:
        for kpts in kpts_seq:
            self.render(kpts)

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
        self._impl = None

    def __enter__(self) -> "Open3DBevVideoRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def demo_random(out_path: str = "bev_demo.mp4", T: int = 120, J: int = 17, backend: Backend = "auto") -> None:
    rng = np.random.default_rng(0)
    base = np.zeros((J, 3), dtype=np.float64)
    base[:, 0] = rng.normal(0, 0.4, size=J)      # X
    base[:, 1] = rng.normal(1.2, 0.2, size=J)    # Y
    base[:, 2] = rng.normal(10.0, 0.5, size=J)   # Z

    seq = []
    pos = np.array([0.0, 0.0, 0.0])
    for _ in range(T):
        pos += np.array([0.02, 0.0, 0.08]) + rng.normal(0, 0.005, size=3)
        seq.append(base + pos)

    with Open3DBevVideoRenderer(Path(out_path), backend=backend, visible=False) as r:
        r.render_many(seq)
    print(f"Saved demo video to: {out_path} (backend={backend})")


if __name__ == "__main__":
    demo_random()
