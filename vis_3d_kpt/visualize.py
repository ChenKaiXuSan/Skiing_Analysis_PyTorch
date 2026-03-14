#!/usr/bin/env python3

import logging
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm import tqdm
import numpy as np

from .load import OnePersonInfo, load_helper
from .metadata.mhr70 import pose_info as mhr70_pose_info
from .visualization.save_utils import merge_frame_to_video, save_figure
from .visualization.scene_visualizer import SceneVisualizer
from .visualization.skeleton_visualizer import SkeletonVisualizer

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

    (
        left_frames,
        right_frames,
        left_2d_kpt,
        right_2d_kpt,
        fused_3d_kpt,
        fused_smoothed_3d_kpt,
    ) = load_helper(person_info)

    skeleton_visualizer, scene_visualizer = setup_visualizer()

    # * pro的左右视频长度不一致，按最短的来
    frame_count = min(
        left_frames.shape[0],
        right_frames.shape[0],
        left_2d_kpt.shape[0],
        right_2d_kpt.shape[0],
        fused_3d_kpt.shape[0],
        fused_smoothed_3d_kpt.shape[0],
    )

    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        process_frame(
            left_frames=left_frames[frame_idx],
            right_frames=right_frames[frame_idx],
            left_2d_kpt=left_2d_kpt[frame_idx],
            right_2d_kpt=right_2d_kpt[frame_idx],
            fused_3d_kpt=fused_3d_kpt[frame_idx],
            fused_smoothed_3d_kpt=fused_smoothed_3d_kpt[frame_idx],
            frame_idx=frame_idx,
            out_root=out_dir,
            skeleton_visualizer=skeleton_visualizer,
            scene_visualizer=scene_visualizer,
        )

    # TODO: 这里合成图片为vidoe
    # merge frmes to video
    # video_dir = out_dir / "video"
    # video_dir.mkdir(parents=True, exist_ok=True)

    # merge_frame_to_video(
    #     frame_dir=out_dir / "frame_scene",
    #     output_path=video_dir / f"{person_info.person_name}_frame_scene.mp4",
    #     fps=30,
    # )


def process_frame(
    left_frames: np.ndarray,
    right_frames: np.ndarray,
    left_2d_kpt: np.ndarray,
    right_2d_kpt: np.ndarray,
    fused_3d_kpt: np.ndarray,
    fused_smoothed_3d_kpt: np.ndarray,
    frame_idx: int,
    out_root: Path,
    skeleton_visualizer: SkeletonVisualizer,
    scene_visualizer: SceneVisualizer,
) -> None:
    # ---------- 画骨架图 ----------
    # * 时间优化前的
    _skeleton = skeleton_visualizer.draw_skeleton_3d(ax=None, points_3d=fused_3d_kpt)

    skeleton_visualizer.save(
        image=_skeleton,
        save_path=out_root / "fused" / f"{frame_idx}.png",
    )

    # * 时间优化后的
    _skeleton_smoothed = skeleton_visualizer.draw_skeleton_3d(
        ax=None, points_3d=fused_smoothed_3d_kpt
    )

    skeleton_visualizer.save(
        image=_skeleton_smoothed,
        save_path=out_root / "smoothed" / f"{frame_idx}.png",
    )

    # 画左右frame + 2D骨架

    left_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=left_frames, keypoints=left_2d_kpt
    )
    right_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=right_frames, keypoints=right_2d_kpt
    )

    _frame_scene = scene_visualizer.draw_frame_with_scene(
        left_frame=left_kpt_with_frame,
        right_frame=right_kpt_with_frame,
        pose_3d=fused_3d_kpt,
    )

    scene_visualizer.save(
        image=_frame_scene,
        save_path=out_root / "frame_scene" / f"{frame_idx}.png",
    )
