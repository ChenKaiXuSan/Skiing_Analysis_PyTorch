#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/bundle_adjustment/main.py
Project: /workspace/code/vggt/bundle_adjustment
Created Date: Tuesday November 25th 2025
Author: Kaixu Chen
-----
Comment:
Local bundle adjustment with configurable optimisation modes.
You can choose to refine only 3D pose, pose+translation, or
full pose+camera (R,t) under multi-view geometric constraints.

Have a good code time :)
-----
Last Modified: Tuesday November 25th 2025 10:30:00 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
"""

from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig

from tqdm import tqdm

from .fuse.fuse import rigid_transform_3D
from .load import (
    load_info,
    load_sam_3d_body_results,
    load_vggt_results,
    load_videopose3d_results,
)
from .loss import (
    baseline_reg_loss,
    bone_length_loss,
    camera_smooth_loss,
    pose_temporal_loss,
    reprojection_loss,
)
from .metadata.mhr70 import pose_info as mhr70_pose_info
from .visualization.scene_visualizer import SceneVisualizer
from .visualization.skeleton_visualizer import SkeletonVisualizer
from .visualization.merge import merge_frame_to_video


def setup_visualizer():
    """Set up skeleton visualizer with MHR70 pose info"""
    skeleton_visualizer = SkeletonVisualizer(line_width=2, radius=5)
    skeleton_visualizer.set_pose_meta(mhr70_pose_info)

    scene_visualizer = SceneVisualizer(line_width=2, radius=5)
    scene_visualizer.set_pose_meta(mhr70_pose_info)

    return skeleton_visualizer, scene_visualizer


def process_one_person(
    left_video_path: Path,
    left_pt_path: Path,
    left_sam3d_body_path: Path,
    right_video_path: Path,
    right_pt_path: Path,
    right_sam3d_body_path: Path,
    vggt_files: List[Path],
    videopose3d_files: List[Path],
    out_root: Path,
    cfg: DictConfig,
) -> Optional[Path]:
    """
    Process one person with multi-view bundle adjustment.

    Parameters
    ----------
    left_video_path : Path
        Path to the left video file.
    left_pt_path : Path
        Path to the left 2D keypoints file.
    right_video_path : Path
        Path to the right video file.
    right_pt_path : Path
        Path to the right 2D keypoints file.
    vggt_files : List[Path]
        List of VGGT numpy files for all views.
    videopose3d_files : List[Path]
        List of VideoPose3D numpy files for all views.
    out_root : Path
        Output root directory.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    out_dir : Optional[Path]
        Output directory if successful, None otherwise.
    """

    skeleton_visualizer, scene_visualizer = setup_visualizer()

    # left_kpt, left_kpt_score, left_bboxes_xyxy, left_bboxes_scores, left_frame = (
    #     load_info(
    #         video_file_path=left_video_path.as_posix(),
    #         pt_file_path=left_pt_path.as_posix(),
    #         assume_normalized=False,
    #     )
    # )

    # right_kpt, right_kpt_score, right_bboxes_xyxy, right_bboxes_scores, right_frame = (
    #     load_info(
    #         video_file_path=right_video_path.as_posix(),
    #         pt_file_path=right_pt_path.as_posix(),
    #         assume_normalized=False,
    #     )
    # )

    left_sam3d_body_res = load_sam_3d_body_results(left_sam3d_body_path.as_posix())
    right_sam3d_body_res = load_sam_3d_body_results(right_sam3d_body_path.as_posix())

    # videopose3d_res = load_videopose3d_results(videopose3d_files)
    # vggt_res = load_vggt_results(vggt_files)

    for frame_idx in tqdm(range(len(left_sam3d_body_res)), desc="Processing frames"):
        # if frame_idx > 60:
        #     break

        process_frame(
            left_sam3d_body_res=left_sam3d_body_res,
            right_sam3d_body_res=right_sam3d_body_res,
            frame_idx=frame_idx,
            skeleton_visualizer=skeleton_visualizer,
            scene_visualizer=scene_visualizer,
            out_root=out_root,
        )

    # merge frame to video if needed
    merge_frame_to_video(
        save_path=out_root,
        flag="fused",
    )

    merge_frame_to_video(
        save_path=out_root,
        flag="scene",
    )

    merge_frame_to_video(
        save_path=out_root,
        flag="frame_scene",
    )


def process_frame(
    left_sam3d_body_res,
    right_sam3d_body_res,
    frame_idx,
    skeleton_visualizer: SkeletonVisualizer,
    scene_visualizer: SceneVisualizer,
    out_root: Path,
):
    # 模拟 Ground Truth (Target B)

    left_frame = left_sam3d_body_res[frame_idx]["frame"]
    right_frame = right_sam3d_body_res[frame_idx]["frame"]
    left_kpt_2d = left_sam3d_body_res[frame_idx]["pred_keypoints_2d"]
    right_kpt_2d = right_sam3d_body_res[frame_idx]["pred_keypoints_2d"]
    left_kpt_3d = left_sam3d_body_res[frame_idx]["pred_keypoints_3d"]
    right_kpt_3d = right_sam3d_body_res[frame_idx]["pred_keypoints_3d"]
    C_L_world = left_sam3d_body_res[frame_idx]["pred_cam_t"]
    C_R_person = right_sam3d_body_res[frame_idx]["pred_cam_t"]

    left_focal_len = left_sam3d_body_res[frame_idx]["focal_length"]
    right_focal_len = right_sam3d_body_res[frame_idx]["focal_length"]

    # TODO: 因为相机的坐标系和kpt的坐标系需要同时移动，所以规范化的代码需要在外面
    fused, diag = rigid_transform_3D(
        target=left_kpt_3d,
        source=right_kpt_3d,
        wL=None,
        wR=None,
        return_diagnostics=True,
    )

    # 世界系 = 左人系
    # s, R_RL, t_RL: 用 Umeyama 求出的 右→左 相似变换
    s, R_RL, t_RL = (
        diag["per_frame"][0]["s"],
        diag["per_frame"][0]["R"],
        diag["per_frame"][0]["t"],
    )

    # sam 3d预测的，左、右相机中心
    # 这里的任务需要把右相机从右人系，变换到世界系（左人系）

    # * 右相机根据右人系坐标 + 右→左的相似变换，得到右世界系坐标
    # C_R_world = s * (R_RL @ C_R_person) + t_RL  # 右世界系
    # ! 直接把plt的坐标转换为opencv的坐标系，不进行坐标层面的改变，只进行渲染层面的改变
    C_L_world[1] = -C_L_world[1]
    C_R_person = -C_R_person
    C_R_world = C_R_person

    # 因为plt的z反转了，所以这里也要反转一下kpt的z轴
    fused[:, 2] = -fused[:, 2]

    # ---------- 画骨架图 ----------
    kpts_world = fused  # 直接把左视角的人当作世界里的骨架

    plt_skeleton = skeleton_visualizer.draw_skeleton_3d(ax=None, points_3d=kpts_world)

    out_skeleton = out_root / "fused"
    out_skeleton.mkdir(parents=True, exist_ok=True)

    plt_skeleton.savefig(out_skeleton / f"{frame_idx}.png", dpi=300)

    # ---------- 画场景图 ----------
    # TODO: 相机的位置还是有问题，会出现飘逸现象，需要进一步调试
    plt_scene = scene_visualizer.draw_scene(
        ax=None,
        kpts_world=kpts_world,
        C_L_world=C_L_world,
        C_R_world=C_R_world,
        left_focal_length=left_focal_len,
        right_focal_length=right_focal_len,
    )

    out_scene = out_root / "scene"
    out_scene.mkdir(parents=True, exist_ok=True)

    plt_scene.savefig(out_scene / f"{frame_idx}.png", dpi=300)

    # 画左右frame + scene

    left_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=left_frame, keypoints=left_kpt_2d
    )
    right_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=right_frame, keypoints=right_kpt_2d
    )

    plt_frame_scene = scene_visualizer.draw_frame_with_scene(
        left_frame=left_kpt_with_frame,
        right_frame=right_kpt_with_frame,
        pose_3d=kpts_world,
        C_L_world=C_L_world,
        C_R_world=C_R_world,
        left_focal_length=left_focal_len,
        right_focal_length=right_focal_len,
    )

    out_frame_scene = out_root / "frame_scene"
    out_frame_scene.mkdir(parents=True, exist_ok=True)

    plt_frame_scene.savefig(out_frame_scene / f"{frame_idx}.png", dpi=300)
