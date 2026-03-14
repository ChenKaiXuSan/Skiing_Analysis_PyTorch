#!/usr/bin/env python3

import shutil
from pathlib import Path
from typing import Any, Optional, Sequence

import logging

from .visualization.save_utils import merge_frame_to_video, save_figure
from .visualization.scene_visualizer import SceneVisualizer
from .visualization.skeleton_visualizer import SkeletonVisualizer
from .metadata.mhr70 import pose_info as mhr70_pose_info
from .load import load_helper, OnePersonInfo

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = [
    ("front_left", -25, 270),
    ("front_right", -25, 90),
    ("top", 85, 270),
    ("side", 0, 0),
]

SIMPLE_VIEWS = [
    ("perspective", -20, 255),
]

POSE_KEYS = (
    "pred_keypoints_3d",
    "keypoints_3d",
    "pose_3d",
    "positions_3d",
    "joints_3d",
)


def setup_visualizer():
    """Setup visualizers with default pose meta (MHR70)."""

    skeleton_visualizer = SkeletonVisualizer(line_width=2, radius=5)
    skeleton_visualizer.set_pose_meta(mhr70_pose_info)

    scene_visualizer = SceneVisualizer(line_width=2, radius=5)
    scene_visualizer.set_pose_meta(mhr70_pose_info)

    return skeleton_visualizer, scene_visualizer


def run_visualization(
    person_info: OnePersonInfo,
    out_dir: Path,
) -> None:
    """Run visualization using an argparse.Namespace of options.

    This function contains the core logic originally in `main()` and is
    safe to import and call from other scripts.
    """
    out_dir = out_dir.resolve()

    frames_dir = out_dir / "single_frame"
    video_dir = out_dir / "video"

    frames_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    (
        left_frames,
        right_frames,
        left_2d_kpt,
        right_2d_kpt,
        fused_3d_kpt,
        fused_smoothed_3d_kpt,
    ) = load_helper(person_info)

    skeleton_visualizer, scene_visualizer = setup_visualizer()

    assert left_frames.shape[0] == right_frames.shape[0], "左右视频帧数不匹配"

    for frame_idx in range(left_frames.shape[0]):
        frame_fig = process_frame(
            left_frames=left_frames,
            right_frames=right_frames,
            left_2d_kpt=left_2d_kpt,
            right_2d_kpt=right_2d_kpt,
            fused_3d_kpt=fused_3d_kpt,
            fused_smoothed_3d_kpt=fused_smoothed_3d_kpt,
            frame_idx=frame_idx,
            out_root=out_dir,
            skeleton_visualizer=skeleton_visualizer,
            scene_visualizer=scene_visualizer,
        )

    # merge frmes to video
    video_path = video_dir / "output.mp4"

    print(f"[done] 逐帧图片目录: {frames_dir}")
    print(f"[done] 视频: {video_path}")


def process_frame(
    left_frames,
    right_frames,
    left_2d_kpt,
    right_2d_kpt,
    fused_3d_kpt,
    fused_smoothed_3d_kpt,
    frame_idx,
    out_root: Path,
    skeleton_visualizer: SkeletonVisualizer,
    scene_visualizer: SceneVisualizer,
):
    left_frame = left_frames[frame_idx]
    right_frame = right_frames[frame_idx]
    left_kpt_2d = left_2d_kpt[frame_idx]["pred_keypoints_2d"]
    right_kpt_2d = right_2d_kpt[frame_idx]["pred_keypoints_2d"]
    fused_3d_kpt = fused_3d_kpt[frame_idx] if fused_3d_kpt is not None else None
    fused_smoothed_3d_kpt = (
        fused_smoothed_3d_kpt[frame_idx] if fused_smoothed_3d_kpt is not None else None
    )

    # ---------- 画骨架图 ----------
    kpts_world = (
        fused_3d_kpt[frame_idx] if fused_3d_kpt is not None else left_kpt_3d
    )  # 直接把左视角的人当作世界里的骨架

    _skeleton = skeleton_visualizer.draw_skeleton_3d(ax=None, points_3d=kpts_world)

    skeleton_visualizer.save(
        image=_skeleton,
        save_path=out_root / "fused" / f"{frame_idx}.png",
    )

    # 画左右frame + scene

    left_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=left_frame, keypoints=left_kpt_2d
    )
    right_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=right_frame, keypoints=right_kpt_2d
    )

    _frame_scene = scene_visualizer.draw_frame_with_scene(
        left_frame=left_kpt_with_frame,
        right_frame=right_kpt_with_frame,
        pose_3d=kpts_world,
        C_L_world=C_L_world,
        C_R_world=C_R_world,
        left_focal_length=left_focal_len,
        right_focal_length=right_focal_len,
    )

    scene_visualizer.save(
        image=_frame_scene,
        save_path=out_root / "frame_scene" / f"{frame_idx}.png",
    )

    return left_frame, right_frame, kpts_world
