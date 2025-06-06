#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/main.py
Project: /workspace/code/project
Created Date: Monday May 5th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday May 5th 2025 9:08:50 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""


import os
import logging
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def triangulate_from_disparity(
    pts_l,
    pts_r,
    image_size=(1280, 720),
    focal_length=1000.0,
    baseline=0.1,
    cx=None,
    cy=None,
):
    """
    使用左右视角的2D点和视差，估算相对3D坐标（以左相机为参考）
    """
    if cx is None:
        cx = image_size[0] / 2
    if cy is None:
        cy = image_size[1] / 2

    # 计算视差
    disparity = pts_l[:, 0] - pts_r[:, 0]  # (J, )

    # 深度 = f * B / disparity
    depth = (focal_length * baseline) / (disparity + 1e-6)  # 避免除0

    # 反投影
    x = (pts_l[:, 0] - cx) * depth / focal_length
    y = (pts_l[:, 1] - cy) * depth / focal_length
    z = depth

    points_3d = torch.stack([x, y, z], dim=1)
    return points_3d  # (J, 3)


def build_3d_pose_from_2d_and_depth_torch(
    keypoints_2d, relative_depth, image_size=(1280, 720), root_index=0
):
    cx, cy = image_size[0] / 2, image_size[1] / 2
    x = keypoints_2d[:, 0] - cx
    y = keypoints_2d[:, 1] - cy
    z = relative_depth
    pose3d = torch.stack([x, y, z], dim=1)
    root = pose3d[root_index]
    pose3d_centered = pose3d - root
    scale = torch.norm(pose3d_centered, dim=1).max()
    pose3d_normalized = pose3d_centered / scale
    return pose3d_normalized


def build_3d_pose_from_2d_and_depth_absolute(
    keypoints_2d, relative_depth, image_size=(1280, 720)
):
    """
    使用图像中心为原点，构建真实尺度的3D姿态（不进行中心化和归一化）
    输入:
        keypoints_2d: (J, 2) 像素坐标 (u, v)
        relative_depth: (J,) 相对或绝对深度值
        image_size: (W, H) 图像大小
    输出:
        pose3d: (J, 3) 真实尺度下的 3D 坐标（单位：像素+深度单位）
    """
    cx, cy = image_size[0] / 2, image_size[1] / 2
    x = keypoints_2d[:, 0]  # X 轴向右
    y = keypoints_2d[:, 1]  # Y 轴向下
    z = relative_depth  # Z 轴向前（或深度方向）

    pose3d = torch.stack([x, y, z], dim=1)
    return pose3d


def get_depth_from_keypoints_torch(keypoints_2d, depth_map):
    """
    从深度图中提取关键点对应的深度值（支持batch）

    参数:
        keypoints_2d: (B, J, 2), 关键点坐标，单位为像素 (x, y)
        depth_map: (B, 1, H, W), 每像素深度值（可为 float32）
        image_size: (H, W), 图像尺寸
    返回:
        depth_values: (B, J), 每个关键点处的深度值
    """

    print("keypoints_2d:", keypoints_2d.shape)
    print("depth_map:", depth_map.shape)

    T, J, _ = keypoints_2d.shape
    T, C, H, W = depth_map.shape

    depth_values = torch.empty((T, J))

    for f in range(T):
        for j in range(J):
            if (
                keypoints_2d[f, j, 0] < 0
                or keypoints_2d[f, j, 0] >= W
                or keypoints_2d[f, j, 1] < 0
                or keypoints_2d[f, j, 1] >= H
            ):
                raise ValueError(
                    f"Keypoint {j} at frame {f} is out of bounds: "
                    f"({keypoints_2d[f, j, 0]}, {keypoints_2d[f, j, 1]}) "
                    f"for depth map size ({H}, {W})"
                )
            else:
                kx = int(keypoints_2d[f, j, 0])
                ky = int(keypoints_2d[f, j, 1])

                depth_values[f, j] = depth_map[f, 0, ky, kx]

    return depth_values


def reconstruct_3d_pose(keypoints_2d, depth, image_size):
    cx, cy = image_size[0] / 2, image_size[1] / 2
    x = keypoints_2d[:, :, 0] - cx
    y = keypoints_2d[:, :, 1] - cy
    z = depth
    pose3d = torch.stack([x, y, z], dim=-1)  # (J, 3)
    return pose3d


def plot_3d_pose(pose3d, title="3D Pose", save_path=None, frame_num: int = 0):
    """
    可视化 3D 姿态
    """

    # 转换方向（Y 轴向上）
    x = pose3d[:, 0].numpy()
    y = -pose3d[:, 1].numpy()  # 注意：Y 轴翻转
    z = -pose3d[:, 2].numpy()

    # COCO skeleton 连接关系（骨架）
    coco_skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    # 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 画点
    ax.scatter(x, y, z, c="r", s=40)
    for i in range(17):
        ax.text(x[i], y[i], z[i], str(i), fontsize=8)

    # 画连线
    for i, j in coco_skeleton:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c="black")

    ax.view_init(elev=45, azim=-45)  # elev=仰角, azim=水平角度

    # 坐标轴标签
    ax.set_title("3D COCO Skeleton (Y Axis Up)")
    ax.set_xlabel("X (Right)")
    ax.set_ylabel("Y (Up)")
    ax.set_zlabel("Z (Forward)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{save_path}/{frame_num}_test.jpg')


def process_one_person(parames: DictConfig, person: str):
    """
    Process one person's video data.
    """

    log_path = parames.log_path 

    left_pt_info = torch.load(str(person / "osmo_2.pt"))
    right_pt_info = torch.load(str(person / "osmo_1.pt"))
    logger.info(f"Start process the {person} video")

    # left video information
    left_frames = left_pt_info["frames"]
    video_name = left_pt_info["video_name"]
    video_path = left_pt_info["video_path"]
    img_shape = left_pt_info["img_shape"]
    none_index = left_pt_info["none_index"]
    left_bboxes = left_pt_info["bbox"]
    left_masks = left_pt_info["mask"]
    left_optical_flows = left_pt_info["optical_flow"]
    left_depths = left_pt_info["depth"]

    left_keypoints = left_pt_info["keypoint"]["keypoint"]
    left_keypoints_score = left_pt_info["keypoint"]["keypoint_score"]
    left_keypoints_depth = get_depth_from_keypoints_torch(left_keypoints, left_depths)

    # right video information
    right_frames = right_pt_info["frames"]
    right_bboxes = right_pt_info["bbox"]
    right_masks = right_pt_info["mask"]
    right_optical_flows = right_pt_info["optical_flow"]
    right_depths = right_pt_info["depth"]

    right_keypoints = right_pt_info["keypoint"]["keypoint"]
    right_keypoints_score = right_pt_info["keypoint"]["keypoint_score"]
    right_keypoints_depth = get_depth_from_keypoints_torch(
        right_keypoints, right_depths
    )

    pose3d_left = reconstruct_3d_pose(
        left_keypoints, left_keypoints_depth, image_size=img_shape
    )
    pose3d_right = reconstruct_3d_pose(
        right_keypoints, right_keypoints_depth, image_size=img_shape
    )

    # fuse
    pose3d_fused = (pose3d_left + pose3d_right) / 2.0
    fused_keypoints = (left_keypoints + right_keypoints) / 2.0
    fused_depth = (left_keypoints_depth + right_keypoints_depth) / 2.0

    for i in range(pose3d_fused.shape[0]):
        
        plot_3d_pose(pose3d_fused[i], title=f"Fused 3D Pose for {video_name}", frame_num=i, save_path=log_path)

@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="config.yaml",
)
def init_params(config):
    #######################
    # prepare dataset index
    #######################

    pt_path = Path(config.pt_path)

    person = [i for i in pt_path.iterdir() if i.is_dir()]

    for p in person:
        process_one_person(config, p)


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
