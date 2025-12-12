#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/bundle_adjustment/visualization/skeleton_visualizer copy.py
Project: /workspace/code/bundle_adjustment/visualization
Created Date: Sunday December 7th 2025
Author: Kaixu Chen
-----
Comment:
以人物的地面为世界中心
左相机默认的pred cam t
右相机是按照刚体对其准的

相机看向人物中心，根据focal_length计算FOV

Have a good code time :)
-----
Last Modified: Sunday December 7th 2025 2:20:39 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..fuse.fuse import TORSO_IDX
from .utils import draw_text, parse_pose_metainfo

# sam 3d body 的关键关节索引（你已经给定）
# NECK = 69
# L_HIP, R_HIP = 9, 10
# L_SHO, R_SHO = 5, 6

# # 用于估计刚体变换的“躯干点”
# TORSO_IDX = [NECK, L_HIP, R_HIP, L_SHO, R_SHO]


class SceneVisualizer:
    def __init__(
        self,
        bbox_color: Optional[Union[str, Tuple[int]]] = "green",
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = "red",
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        line_width: Union[int, float] = 1,
        radius: Union[int, float] = 3,
        alpha: float = 1.0,
        show_keypoint_weight: bool = False,
    ):
        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight

        # Pose specific meta info if available.
        self.pose_meta = {}
        self.skeleton = None

    def set_pose_meta(self, pose_meta: Dict):
        parsed_meta = parse_pose_metainfo(pose_meta)

        self.pose_meta = parsed_meta.copy()
        self.bbox_color = parsed_meta.get("bbox_color", self.bbox_color)
        self.kpt_color = parsed_meta.get("keypoint_colors", self.kpt_color)
        self.link_color = parsed_meta.get("skeleton_link_colors", self.link_color)
        self.skeleton = parsed_meta.get("skeleton_links", self.skeleton)

    # ---------- 计算相机视锥体的 4 个角点 ----------
    def compute_frustum_points(
        self, C, forward, up=np.array([0, 1, 0]), fov_deg=60, depth=1.0
    ):
        """
        C       : (3,) 相机中心（世界系）
        forward : (3,) 相机朝向
        up      : (3,) 粗略的“向上”方向向量
        fov_deg : 视野角度（大概给个 60° 即可）
        depth   : 视锥体长度（看多远）
        """
        C = np.asarray(C)
        forward = np.asarray(forward)
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)

        h = depth * np.tan(np.radians(fov_deg) / 2.0)

        p_center = C + forward * depth
        p1 = p_center + right * h + up * h
        p2 = p_center - right * h + up * h
        p3 = p_center - right * h - up * h
        p4 = p_center + right * h - up * h

        return np.stack([p1, p2, p3, p4], axis=0)  # (4,3)

    # ---------- 在 3D 里画视锥体 ----------
    def draw_frustum(self, ax, C, frustum_pts, color="r"):
        C = np.asarray(C)
        # 相机中心到四个角点
        for p in frustum_pts:
            ax.plot([C[0], p[0]], [C[1], p[1]], [C[2], p[2]], color=color)

        # 四个角点围成的四边形
        idx = [0, 1, 2, 3, 0]
        for i in range(4):
            a = frustum_pts[idx[i]]
            b = frustum_pts[idx[i + 1]]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color)

    def draw_camera_axes(self, ax, C, R, length=0.1):
        # TODO: 现在的R还不知道，只是假设的
        """
        在 3D 里画相机坐标系的三个轴
        C : (3,) 相机中心（世界系）
        R : (3,3) 相机旋转矩阵（世界系到相机系）
        length : 轴的长度
        """
        C = np.asarray(C)
        R = np.asarray(R)

        # 相机坐标系的三个轴
        x_axis = R[:, 0] * length
        y_axis = R[:, 1] * length
        z_axis = R[:, 2] * length
        # 画出三个轴
        ax.plot(
            [C[0], C[0] + x_axis[0]],
            [C[1], C[1] + x_axis[1]],
            [C[2], C[2] + x_axis[2]],
            color="r",
            lw=self.line_width,
        )
        ax.plot(
            [C[0], C[0] + y_axis[0]],
            [C[1], C[1] + y_axis[1]],
            [C[2], C[2] + y_axis[2]],
            color="g",
            lw=self.line_width,
        )
        ax.plot(
            [C[0], C[0] + z_axis[0]],
            [C[1], C[1] + z_axis[1]],
            [C[2], C[2] + z_axis[2]],
            color="b",
            lw=self.line_width,
        )

    def calculate_fov_deg(self, focal_length, image_dimension):
        """
        根据焦距和图像尺寸计算视角。

        Args:
            focal_length (float): 相机的焦距 (像素单位)
            image_dimension (int): 计算方向上的图像宽度或高度 (像素单位)

        Returns:
            float: 视角 (度数单位)
        """
        # 式: 2 * arctan((Dimension / 2) / focal_length)
        fov_rad = 2 * np.arctan(image_dimension / (2 * focal_length))

        # ラジアンを度数に変換
        fov_deg = np.rad2deg(fov_rad)

        return fov_deg

    # ---------- 主函数：画人物 + 左右相机 + 视锥体 ----------
    def draw_scene(
        self,
        ax: plt.axes,
        kpts_world,
        C_L_world,
        C_R_world,
        left_focal_length: np.ndarray,
        right_focal_length: np.ndarray,
        frustum_depth=0.5,
        elev=-30,
        azim=270,
    ):
        """
        在给定的 ax 上画 3D 场景；如果 ax 为 None，则自己新建 fig+ax。
        """
        kpts_world = np.asarray(kpts_world)
        C_L_world = np.asarray(C_L_world)
        C_R_world = np.asarray(C_R_world)

        created_fig = None
        if ax is None:
            created_fig = plt.figure(figsize=(15, 15))
            ax = created_fig.add_subplot(111, projection="3d")

        # --- 颜色处理 ---
        kpt_color = "r"
        raw_kpt_colors_mp = self.kpt_color
        if raw_kpt_colors_mp is not None and not isinstance(raw_kpt_colors_mp, str):
            kpt_color = np.array(raw_kpt_colors_mp, dtype=np.float32) / 255.0

        link_color = "b"
        raw_link_colors_mp = self.link_color
        if raw_link_colors_mp is not None and not isinstance(raw_link_colors_mp, str):
            link_color = np.array(raw_link_colors_mp, dtype=np.float32) / 255.0

        # --- 1. 人体骨架 ---
        ax.scatter(
            kpts_world[:, 0],
            kpts_world[:, 1],
            kpts_world[:, 2],
            c=kpt_color,
            marker="o",
            s=self.radius * 10,
            alpha=self.alpha,
        )

        if self.skeleton is not None:
            link_colors_mp = link_color
            for i, (p1_idx, p2_idx) in enumerate(self.skeleton):
                p1 = kpts_world[p1_idx]
                p2 = kpts_world[p2_idx]
                color = (
                    link_colors_mp[i % len(link_colors_mp)]
                    if not isinstance(link_colors_mp, str)
                    else link_colors_mp
                )
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color,
                    linewidth=self.line_width * 2,
                    alpha=self.alpha,
                )

        # 标记世界坐标系原点
        ax.scatter([0], [0], [0], s=60)
        ax.text(0, 0, 0, "world center (0,0,0)")

        # --- 2. 左相机 ---
        ax.scatter(
            C_L_world[0], C_L_world[1], C_L_world[2], marker="^", s=80, color="r"
        )
        ax.text(
            C_L_world[0], C_L_world[1], C_L_world[2], f"Cam L ({C_L_world})", color="r"
        )

        # 相机看向人物中心
        person_center = kpts_world[TORSO_IDX].mean(axis=0)

        left_forward = person_center - C_L_world
        left_forward = left_forward / (np.linalg.norm(left_forward) + 1e-8)

        fov_deg_L = self.calculate_fov_deg(
            focal_length=left_focal_length, image_dimension=1080
        )
        frustum_L = self.compute_frustum_points(
            C_L_world, forward=left_forward, fov_deg=fov_deg_L, depth=frustum_depth
        )
        self.draw_frustum(ax, C_L_world, frustum_L, color="r")
        self.draw_camera_axes(ax, C_L_world, np.eye(3), length=0.1)

        # --- 3. 右相机 ---
        ax.scatter(
            C_R_world[0], C_R_world[1], C_R_world[2], marker="s", s=80, color="b"
        )
        ax.text(
            C_R_world[0],
            C_R_world[1],
            C_R_world[2],
            f"Cam R ({C_R_world})",
            color="b",
        )

        # 相机看向人物中心

        right_forward = person_center - C_R_world
        right_forward = right_forward / (np.linalg.norm(right_forward) + 1e-8)

        fov_deg_R = self.calculate_fov_deg(
            focal_length=right_focal_length, image_dimension=1080
        )
        frustum_R = self.compute_frustum_points(
            C_R_world, forward=right_forward, fov_deg=fov_deg_R, depth=frustum_depth
        )
        self.draw_frustum(ax, C_R_world, frustum_R, color="b")
        self.draw_camera_axes(ax, C_R_world, np.eye(3), length=0.1)

        # --- 4. 视线连线（可选） ---
        ax.plot(
            [C_L_world[0], left_forward[0] + C_L_world[0]],
            [C_L_world[1], left_forward[1] + C_L_world[1]],
            [C_L_world[2], left_forward[2] + C_L_world[2]],
            linestyle="--",
            color="r",
            alpha=0.5,
        )
        ax.plot(
            [C_R_world[0], right_forward[0] + C_R_world[0]],
            [C_R_world[1], right_forward[1] + C_R_world[1]],
            [C_R_world[2], right_forward[2] + C_R_world[2]],
            linestyle="--",
            color="b",
            alpha=0.5,
        )

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-2, 0)
        ax.set_zlim3d(-5, 5)

        ax.set_box_aspect((1, 1, 1))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 翻转 Z 轴显示方向
        # zmin, zmax = ax.get_zlim()
        # ax.set_zlim(zmax, zmin)

        ax.view_init(elev=elev, azim=azim)

        return created_fig if created_fig is not None else ax

    def draw_frame_with_scene(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        pose_3d: np.ndarray,  # (J,3)
        left_focal_length: np.ndarray,
        right_focal_length: np.ndarray,
        C_L_world: np.ndarray,
        C_R_world: np.ndarray,
        frame_num: int = 0,
    ):
        """
        渲染一个 frame：左图+右图+3D pose，并返回 figure。
        """

        fig = plt.figure(figsize=(15, 15))
        fig.suptitle(f"Frame {frame_num}")
        gs = GridSpec(2, 3, figure=fig)

        # -------- 左视角 ---------- #
        axL = fig.add_subplot(gs[0, 0])
        axL.imshow(left_frame)
        axL.axis("off")
        axL.set_title("Left view")

        # -------- 右视角 ---------- #
        axR = fig.add_subplot(gs[1, 0])
        axR.imshow(right_frame)
        axR.axis("off")
        axR.set_title("Right view")

        # -------- 3D pose ---------- #
        ax_3d_left = fig.add_subplot(gs[0, 1], projection="3d")
        ax_3d_left.set_title("left side view")
        self.draw_scene(
            kpts_world=pose_3d,
            C_L_world=C_L_world,
            C_R_world=C_R_world,
            left_focal_length=left_focal_length,
            right_focal_length=right_focal_length,
            ax=ax_3d_left,
            elev=-60,
            azim=-90,
        )

        ax_3d_right = fig.add_subplot(gs[1, 1], projection="3d")
        ax_3d_right.set_title("right side view")
        self.draw_scene(
            kpts_world=pose_3d,
            C_L_world=C_L_world,
            C_R_world=C_R_world,
            left_focal_length=left_focal_length,
            right_focal_length=right_focal_length,
            ax=ax_3d_right,
            elev=130,
            azim=90,
        )

        ax_3d_top_left = fig.add_subplot(gs[0, 2], projection="3d")
        ax_3d_top_left.set_title("top left view")
        self.draw_scene(
            kpts_world=pose_3d,
            C_L_world=C_L_world,
            C_R_world=C_R_world,
            left_focal_length=left_focal_length,
            right_focal_length=right_focal_length,
            ax=ax_3d_top_left,
            elev=0,
            azim=-90,
        )

        ax_3d_top_right = fig.add_subplot(gs[1, 2], projection="3d")
        ax_3d_top_right.set_title("top right view")
        self.draw_scene(
            kpts_world=pose_3d,
            C_L_world=C_L_world,
            C_R_world=C_R_world,
            left_focal_length=left_focal_length,
            right_focal_length=right_focal_length,
            ax=ax_3d_top_right,
            elev=180,
            azim=90,
        )

        fig.tight_layout()
        return fig

    def save(
        self,
        image: plt.figure,
        save_path: Path,
    ):
        """Save the drawn image to disk.

        Args:
            image (np.ndarray): The drawn image.
            save_path (str): The path to save the image.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # image.savefig(save_path, dpi=300)
        image.savefig(
            save_path,
            dpi=300,
            facecolor="white",  # 背景白
            edgecolor="white",
        )

        plt.close(image)
