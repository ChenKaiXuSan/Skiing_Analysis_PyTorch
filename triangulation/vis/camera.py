#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/camera.py
Project: /workspace/code/triangulation/vis
Created Date: Wednesday September 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday September 3rd 2025 12:02:32 pm
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
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _camera_center(R, T, convention="opencv"):
    R = np.asarray(R).reshape(3, 3)
    T = np.asarray(T).reshape(3)
    if convention.lower() == "opencv":
        # Xc = R Xw + T → Cw = -R^T T
        return -R.T @ T
    elif convention.lower() == "cam2world":
        # Xw = R Xc + T → Cw = T
        return T
    else:
        raise ValueError("convention must be 'opencv' or 'cam2world'")


def _set_axes_equal(ax):
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xr, yr, zr = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    m = max(xr, yr, zr)
    ax.set_xlim3d(np.mean(xs) - m / 2, np.mean(xs) + m / 2)
    ax.set_ylim3d(np.mean(ys) - m / 2, np.mean(ys) + m / 2)
    ax.set_zlim3d(np.mean(zs) - m / 2, np.mean(zs) + m / 2)


def save_camera_positions_3d(
    R_list,
    T_list,
    save_path,
    labels=None,
    convention="opencv",
    draw_frustum=False,
    K=None,
    image_size=None,
    frustum_depth=0.5,
    axis_len=0.2,
    elev=20,
    azim=-60,
    title="Camera Layout (3D)",
):
    """
    把相机位置和朝向画成 3D 图片并保存。
    R_list/T_list: 长度为 N 的外参列表
    labels: 相机名（可选）
    convention: 'opencv' 或 'cam2world'
    draw_frustum: 是否画视锥（需传 K 和 image_size=(W,H)）
    """
    N = len(R_list)
    labels = labels or [f"Cam{i+1}" for i in range(N)]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    centers = []
    for i, (R, T, name) in enumerate(zip(R_list, T_list, labels)):
        R = np.asarray(R).reshape(3, 3)
        C = _camera_center(R, T, convention)
        centers.append(C)

        # 坐标轴（相机系在世界中的朝向）
        Rc2w = R.T if convention.lower() == "opencv" else R
        axes = np.eye(3) * axis_len
        axes_w = (Rc2w @ axes.T).T + C
        ax.plot(
            [C[0], axes_w[0, 0]],
            [C[1], axes_w[0, 1]],
            [C[2], axes_w[0, 2]],
            lw=2,
            c="r",
        )
        ax.plot(
            [C[0], axes_w[1, 0]],
            [C[1], axes_w[1, 1]],
            [C[2], axes_w[1, 2]],
            lw=2,
            c="g",
        )
        ax.plot(
            [C[0], axes_w[2, 0]],
            [C[1], axes_w[2, 1]],
            [C[2], axes_w[2, 2]],
            lw=2,
            c="b",
        )
        ax.scatter(C[0], C[1], C[2], s=30)
        ax.text(C[0], C[1], C[2], name)

        # 视锥（可选）
        if draw_frustum and K is not None and image_size is not None:
            K = np.asarray(K).reshape(3, 3)
            W, H = image_size
            corners = np.array(
                [[0, 0, 1], [W - 1, 0, 1], [W - 1, H - 1, 1], [0, H - 1, 1]],
                dtype=float,
            )
            rays_c = (np.linalg.inv(K) @ corners.T).T
            scale = frustum_depth / rays_c[:, 2:3]
            pts_c = rays_c * scale
            pts_w = (Rc2w @ pts_c.T).T + C
            for p in pts_w:
                ax.plot([C[0], p[0]], [C[1], p[1]], [C[2], p[2]], lw=1, c="k")
            idx = [0, 1, 2, 3, 0]
            ax.plot(pts_w[idx, 0], pts_w[idx, 1], pts_w[idx, 2], lw=1, c="k")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    _set_axes_equal(ax)
    ax.set_title(title)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return np.vstack(centers)


def save_camera_positions_topdown(
    R_list,
    T_list,
    save_path,
    labels=None,
    convention="opencv",
    arrow_len=0.2,
    title="Camera Layout (Top-Down XZ)",
):
    """
    俯视图：把相机在 X–Z 平面的位置和朝向画成 2D 图片并保存。
    朝向用相机系的 -Z 方向（光轴）在世界中的投影表示。
    """
    N = len(R_list)
    labels = labels or [f"Cam{i+1}" for i in range(N)]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    centers = []

    for R, T, name in zip(R_list, T_list, labels):
        R = np.asarray(R).reshape(3, 3)
        C = _camera_center(R, T, convention)
        centers.append(C)

        # 光轴方向（相机系 -Z）
        Rc2w = R.T if convention.lower() == "opencv" else R
        look_dir_w = Rc2w @ np.array([0, 0, -1.0])  # 方向向量
        # 只画 XZ 平面的投影
        ax.scatter(C[0], C[2], s=30)
        ax.annotate(name, (C[0], C[2]), textcoords="offset points", xytext=(5, 5))
        ax.arrow(
            C[0],
            C[2],
            look_dir_w[0] * arrow_len,
            look_dir_w[2] * arrow_len,
            head_width=arrow_len * 0.15,
            length_includes_head=True,
        )

    centers = np.vstack(centers)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    pad = max(1e-6, 0.1 * np.max(np.ptp(centers[:, [0, 2]], axis=0)))
    ax.set_xlim(centers[:, 0].min() - pad, centers[:, 0].max() + pad)
    ax.set_ylim(centers[:, 2].min() - pad, centers[:, 2].max() + pad)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(title)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return centers


def save_camera(R, T, output_path, frame_num):

    # 3d
    os.makedirs(os.path.join(output_path, "3d"), exist_ok=True)

    # topdown
    os.makedirs(os.path.join(output_path, "topdown"), exist_ok=True)

    # 列表示例
    R1 = np.eye(3)
    T1 = np.zeros(3)
    R2 = R
    T2 = T  # 传入的相机外参

    # 3D 图（可加入视锥）
    centers3d = save_camera_positions_3d(
        [R1, R2],
        [T1, T2],
        save_path=os.path.join(output_path, "3d", f"camera_{frame_num}.png"),
        labels=["Left", "Right"],
        convention="opencv",
        draw_frustum=False,  # 有 K 与 (W,H) 时可置 True
    )

    # 俯视图（X-Z）
    centers2d = save_camera_positions_topdown(
        [R1, R2],
        [T1, T2],
        save_path=os.path.join(output_path, "topdown", f"camera_{frame_num}.png"),
        labels=["Left", "Right"],
        convention="opencv",
    )
    print(centers3d)
