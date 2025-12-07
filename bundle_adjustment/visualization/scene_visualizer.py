#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/bundle_adjustment/visualization/skeleton_visualizer copy.py
Project: /workspace/code/bundle_adjustment/visualization
Created Date: Sunday December 7th 2025
Author: Kaixu Chen
-----
Comment:

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
# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import draw_text, parse_pose_metainfo


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
        forward : (3,) 相机朝向（需要归一化），例如朝向人物原点的方向
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

    def calculate_fov_deg(self, focal_length, image_dimension):
        """
        焦点距離 (f) と画像の次元 (W または H) から画角 (FOV) を計算する。

        Args:
            focal_length (float): 焦点距離 (ピクセル単位)
            image_dimension (int): 計算したい方向の画像の幅または高さ (ピクセル単位)

        Returns:
            float: 画角 (度数 / deg)
        """
        # 式: 2 * arctan((Dimension / 2) / focal_length)
        fov_rad = 2 * np.arctan(image_dimension / (2 * focal_length))

        # ラジアンを度数に変換
        fov_deg = np.rad2deg(fov_rad)

        return fov_deg

    def convert_opencv_matplotlib(self, opencv_point: np.ndarray, flag: str):
        # ------- OpenCV(world) -> Matplotlib(world) 线性映射 -------
        # x 保持；z(前) -> y(前)；-y(上) -> z(上)
        M = np.array(
            [
                [1.0, 0.0, 0.0],  # x -> x
                [0.0, 0.0, 1.0],  # z -> y
                [0.0, -1.0, 0.0],  # -y -> z
            ],
            dtype=float,
        )

        # 将点从opencv > matplotlib
        if flag == "cam":
            matplotlib_point = (M @ opencv_point).reshape(3)
        elif flag == "person":
            matplotlib_point = (M @ opencv_point.T).T

        return matplotlib_point

    # ---------- 主函数：画人物 + 左右相机 + 视锥体 ----------
    def draw_scene(
        self,
        kpts_world,
        C_L_world,
        C_R_world,
        focal_length: np.ndarray,
        frustum_depth=1.0,
        title="Person-centered world with two cameras",
    ):
        """
        kpts_world : (J,3)  人体 3D 关键点（以骨盆为原点）
        C_L_world  : (3,)   左相机位置（世界系）
        C_R_world  : (3,)   右相机位置（世界系）
        edges      : list[(i,j)] 骨架拓扑，没给的话只画点
        fov_deg    : 视锥 FOV
        frustum_depth : 视锥长度
        """
        kpts_world = np.asarray(kpts_world)
        C_L_world = np.asarray(C_L_world)
        C_R_world = np.asarray(C_R_world)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # --- 1. 人体骨架 ---

        # Matplotlib 颜色需要是 0.0 到 1.0 的 RGB/RGBA
        raw_kpt_colors_mp = self.kpt_color
        # 关键点颜色转换
        if raw_kpt_colors_mp is not None and not isinstance(raw_kpt_colors_mp, str):
            # 将颜色转换为 NumPy 数组并归一化到 0.0-1.0 范围
            kpt_color = np.array(raw_kpt_colors_mp, dtype=np.float32) / 255.0

        # 连线颜色转换
        raw_link_colors_mp = self.link_color
        if raw_link_colors_mp is not None and not isinstance(raw_link_colors_mp, str):
            link_color = np.array(raw_link_colors_mp, dtype=np.float32) / 255.0

        kpts_world = self.convert_opencv_matplotlib(kpts_world, flag="person")

        # 绘制 3D 关键点
        ax.scatter(
            kpts_world[:, 0],
            kpts_world[:, 1],
            kpts_world[:, 2],
            c=kpt_color,
            marker="o",
            s=self.radius * 10,  # 调整点的大小以便在 3D 中可见
            alpha=self.alpha,
        )

        # 绘制 3D 骨架连线
        if self.skeleton is not None:
            link_colors_mp = link_color

            for i, (p1_idx, p2_idx) in enumerate(self.skeleton):
                # 获取连接线的颜色，确保在 0.0-1.0 范围
                color = link_colors_mp[i % len(link_colors_mp)]  # 循环使用颜色

                # 提取两个点的坐标
                p1 = kpts_world[p1_idx]
                p2 = kpts_world[p2_idx]

                # 绘制连接线
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color,
                    linewidth=self.line_width * 2,  # 调整线宽
                    alpha=self.alpha,
                )

        # 标记骨盆为原点（假设 0 是骨盆）
        ax.scatter([0], [0], [0], s=60)
        ax.text(0, 0, 0, "pelvis(0,0,0)")

        # --- 2. 左相机 ---

        C_L_world = self.convert_opencv_matplotlib(C_L_world, flag="cam")

        ax.scatter(
            C_L_world[0], C_L_world[1], C_L_world[2], marker="^", s=80, color="r"
        )
        ax.text(C_L_world[0], C_L_world[1], C_L_world[2], "Cam L", color="r")

        # 朝向：默认看向人物原点
        dL = -C_L_world
        dL = dL / (np.linalg.norm(dL) + 1e-8)
        frustum_L = self.compute_frustum_points(
            C_L_world, dL, fov_deg=fov_deg, depth=frustum_depth
        )
        self.draw_frustum(ax, C_L_world, frustum_L, color="r")

        # --- 3. 右相机 ---

        C_R_world = self.convert_opencv_matplotlib(C_R_world, flag="cam")

        ax.scatter(
            C_R_world[0], C_R_world[1], C_R_world[2], marker="s", s=80, color="b"
        )
        ax.text(C_R_world[0], C_R_world[1], C_R_world[2], "Cam R", color="b")

        dR = -C_R_world
        dR = dR / (np.linalg.norm(dR) + 1e-8)
        frustum_R = self.compute_frustum_points(
            C_R_world, dR, fov_deg=fov_deg, depth=frustum_depth
        )
        self.draw_frustum(ax, C_R_world, frustum_R, color="b")

        # --- 4. 视线（相机到人）的连线（可选） ---
        ax.plot(
            [C_L_world[0], 0],
            [C_L_world[1], 0],
            [C_L_world[2], 0],
            linestyle="--",
            color="r",
            alpha=0.5,
        )
        ax.plot(
            [C_R_world[0], 0],
            [C_R_world[1], 0],
            [C_R_world[2], 0],
            linestyle="--",
            color="b",
            alpha=0.5,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        # 你可以在这里调一下默认视角：
        # ax.view_init(elev=20, azim=-60)

        plt.tight_layout()

        return fig
