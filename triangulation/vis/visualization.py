#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/visualitation.py
Project: /workspace/code/triangulation
Created Date: Wednesday September 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday September 3rd 2025 9:55:15 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import os
import numpy as np
import cv2
import torch
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py


# 这里以 COCO-17 骨架为例 (你也可以换成自己数据集的连接)
COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_and_save_keypoints_from_frame(
    frame,
    keypoints,
    save_path,
    color=(0, 255, 0),
    radius=4,
    thickness=-1,
    with_index=True,
    skeleton=COCO_SKELETON,  # 可以传 None 表示不画线
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转 numpy，确保是 BGR 三通道
    img = frame.numpy() if isinstance(frame, torch.Tensor) else frame.copy()
    if img.ndim == 2:  # 灰度转彩色
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 转成 numpy 数组
    keypoints = np.array(keypoints, dtype=float)

    # 画点
    for i, (x, y) in enumerate(keypoints):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        center = (int(x), int(y))
        cv2.circle(img, center, radius, color, thickness)
        if with_index:
            cv2.putText(
                img,
                str(i),
                (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    # 画连线
    if skeleton is not None:
        for i, j in skeleton:
            if (
                i < len(keypoints)
                and j < len(keypoints)
                and np.isfinite(keypoints[i][0])
                and np.isfinite(keypoints[i][1])
                and np.isfinite(keypoints[j][0])
                and np.isfinite(keypoints[j][1])
            ):
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(img, pt1, pt2, (0, 255, 255), 2)  # 黄色线

    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"[INFO] Saved image with keypoints to: {save_path}")


def draw_camera(ax, R, T, scale=0.1, label="Cam"):
    origin = T.reshape(3)
    x_axis = R @ np.array([1, 0, 0]) * scale + origin
    y_axis = R @ np.array([0, 1, 0]) * scale + origin
    z_axis = R @ np.array([0, 0, 1]) * scale + origin
    view_dir = R @ np.array([0, 0, -1]) * scale * 1.5 + origin  # 摄像头朝向（-Z轴）

    ax.plot(
        [origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], c="r"
    )
    ax.plot(
        [origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], c="g"
    )
    ax.plot(
        [origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], c="b"
    )

    # 视线方向箭头（黑色）
    ax.plot(
        [origin[0], view_dir[0]],
        [origin[1], view_dir[1]],
        [origin[2], view_dir[2]],
        c="k",
        linestyle="--",
    )

    # 相机标签
    ax.text(*origin, label, color="black")


def set_axes_equal(ax):
    """让三维坐标轴等比例显示"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_3d_joints(
    joints_3d, R, T, save_path, title="Triangulated 3D Joints", skeleton=COCO_SKELETON
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 相机位置可视化
    draw_camera(ax, np.eye(3), np.zeros(3), label="Cam1")
    draw_camera(ax, R, T, label="Cam2")

    # 画关节点
    xs, ys, zs = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    ax.scatter(xs, ys, zs, c="blue", s=30)
    for i, (x, y, z) in enumerate(joints_3d):
        ax.text(x, y, z, str(i), size=8)

    # 画骨架连线（如果有）
    if skeleton is not None:
        for i, j in skeleton:
            if i < len(joints_3d) and j < len(joints_3d):
                xline = [joints_3d[i, 0], joints_3d[j, 0]]
                yline = [joints_3d[i, 1], joints_3d[j, 1]]
                zline = [joints_3d[i, 2], joints_3d[j, 2]]
                ax.plot(xline, yline, zline, c="red", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_axes_equal(ax)  # 等比例

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


def visualize_3d_scene_interactive(joints_3d, R, T, save_path):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=joints_3d[:, 0],
            y=joints_3d[:, 1],
            z=joints_3d[:, 2],
            mode="markers+text",
            text=[str(i) for i in range(len(joints_3d))],
            marker=dict(size=4, color="blue"),
        )
    )

    def get_camera_lines(R, T, label):
        scale = 0.1
        lines = []
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        colors = ["red", "green", "blue"]
        origin = T.reshape(3)
        for axis, color in zip(axes, colors):
            end = R @ axis * scale + origin
            lines.append(
                go.Scatter3d(
                    x=[origin[0], end[0]],
                    y=[origin[1], end[1]],
                    z=[origin[2], end[2]],
                    mode="lines",
                    line=dict(color=color, width=4),
                )
            )
        view_dir = R @ np.array([0, 0, -1]) * scale * 1.5 + origin
        lines.append(
            go.Scatter3d(
                x=[origin[0], view_dir[0]],
                y=[origin[1], view_dir[1]],
                z=[origin[2], view_dir[2]],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name=f"{label}_view",
            )
        )
        return lines

    for trace in get_camera_lines(np.eye(3), np.zeros(3), "Cam1"):
        fig.add_trace(trace)
    for trace in get_camera_lines(R, T, "Cam2"):
        fig.add_trace(trace)

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="Interactive 3D Scene",
        margin=dict(l=0, r=0, b=0, t=30),
    )
    py.plot(fig, filename=save_path, auto_open=False)
    print(f"[INFO] Saved interactive HTML to: {save_path}")

