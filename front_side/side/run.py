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

import numpy as np

from .fuse.fuse import rigid_transform_3D

from .metadata.mhr70 import pose_info as mhr70_pose_info
from .reproject import reproject_and_visualize
from .visualization.merge import merge_frame_to_video
from .visualization.scene_visualizer import SceneVisualizer
from .visualization.skeleton_visualizer import SkeletonVisualizer

K = np.array(
    [
        1116.9289548941917,
        0.0,
        955.77175993563799,
        0.0,
        1117.3341496962166,
        538.91061167202145,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)


def setup_visualizer():
    """Set up skeleton visualizer with MHR70 pose info."""

    skeleton_visualizer = SkeletonVisualizer(line_width=2, radius=5)
    skeleton_visualizer.set_pose_meta(mhr70_pose_info)

    scene_visualizer = SceneVisualizer(line_width=2, radius=5)
    scene_visualizer.set_pose_meta(mhr70_pose_info)

    return skeleton_visualizer, scene_visualizer


def process_side_frame(
    left_sam3d_body_res,
    right_sam3d_body_res,
    frame_idx,
    out_root: Path,
):
    skeleton_visualizer, scene_visualizer = setup_visualizer()

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
    # UPDATE: 暂时不进行坐标系的移动了
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

    # 镜像翻转坐标系
    # TODO：如果刚体变换的rt能用的话，就用右边相机变化之后的位置
    # * 右相机根据右人系坐标 + 右→左的相似变换，得到右世界系坐标
    # C_R_world = s * (R_RL @ C_R_person) + t_RL  # 右世界系

    # ! 直接把的坐标转换为opencv的坐标系，不进行坐标层面的改变，只进行渲染层面的改变
    C_L_world = C_L_world * np.array([1, -1, -1])  # 镜像翻转
    C_R_world = C_R_person * np.array([-1, -1, -1])  # 镜像翻转

    C_R_world[0] = -C_R_world[0]  # 关于 X 轴镜像
    C_R_world[2] = -C_R_world[2]  # 关于 Z 轴镜像

    # ! 因为的左右相机的Z反转了，所以这里也要反转一下kpt的z轴
    # fused = fused * np.array([1, 1, -1])  # 镜像翻转

    # ---------- 画骨架图 ----------
    kpts_world = fused  # 直接把左视角的人当作世界里的骨架

    _skeleton = skeleton_visualizer.draw_skeleton_3d(ax=None, points_3d=kpts_world)

    skeleton_visualizer.save(
        image=_skeleton,
        save_path=out_root / "fused" / f"{frame_idx}.png",
    )

    # ---------- 画场景图 ----------
    # TODO: 相机的位置还是有问题，会出现飘逸现象，需要进一步调试
    _scene = scene_visualizer.draw_scene(
        ax=None,
        kpts_world=kpts_world,
        C_L_world=C_L_world,
        C_R_world=C_R_world,
        left_focal_length=left_focal_len,
        right_focal_length=right_focal_len,
    )

    scene_visualizer.save(
        image=_scene,
        save_path=out_root / "scene" / f"{frame_idx}.png",
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

    return left_frame, right_frame, kpts_world, R_RL, t_RL
