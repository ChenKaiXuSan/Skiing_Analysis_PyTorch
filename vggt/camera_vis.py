#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/camera_vis.py
Project: /workspace/code/vggt
Created Date: Friday November 7th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday November 7th 2025 10:36:28 am
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
import matplotlib.pyplot as plt


def plot_cameras_matplotlib(
    preds,
    out_dir=".",
    axis_len=0.1,
    title="Camera Poses",
    show_id=True,
    include_points=False,
):
    """
    可视化相机位置，并把多个视角绘制在同一张图片中（多子图）。

    Args:
        preds: dict，包含 "extrinsic" (S,3,4) 或 (S,4,4)
        out_dir: 输出文件夹路径
        axis_len: 每个相机坐标轴的长度
        title: 图标题
        show_id: 是否显示相机编号
        include_points: 是否绘制世界点云 (preds["world_points_from_depth"])
    """
    os.makedirs(out_dir, exist_ok=True)

    E = preds["extrinsic"]
    # 支持 (S,3,4) 或 (S,4,4)
    R = E[..., :3, :3]
    t = E[..., :3, 3]

    # 相机中心（世界坐标系）
    C = -np.einsum("sij,sj->si", R.transpose(0, 2, 1), t)

    # 世界点云（可选）
    if include_points and "world_points_from_depth" in preds:
        pts = preds["world_points_from_depth"].reshape(-1, 3)
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
    else:
        pts = None

    # === 视角配置 ===
    views = [
        dict(name="default", elev=30, azim=60),
        dict(name="front", elev=0, azim=90),
        dict(name="top", elev=90, azim=-90),
        dict(name="side", elev=0, azim=0),
    ]
    n_views = len(views)

    # 子图布局：2x2
    n_cols = 2
    n_rows = (n_views + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    def _plot_one(ax, elev, azim, view_name):
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=20, label="Cameras")

        # 相机坐标轴（在世界坐标系下的方向）
        Xw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([1, 0, 0]))
        Yw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([0, 1, 0]))
        Zw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([0, 0, 1]))

        for i in range(C.shape[0]):
            o = C[i]
            ax.plot(
                [o[0], o[0] + axis_len * Xw[i, 0]],
                [o[1], o[1] + axis_len * Xw[i, 1]],
                [o[2], o[2] + axis_len * Xw[i, 2]],
                "r",
                lw=1,
            )
            ax.plot(
                [o[0], o[0] + axis_len * Yw[i, 0]],
                [o[1], o[1] + axis_len * Yw[i, 1]],
                [o[2], o[2] + axis_len * Yw[i, 2]],
                "g",
                lw=1,
            )
            ax.plot(
                [o[0], o[0] + axis_len * Zw[i, 0]],
                [o[1], o[1] + axis_len * Zw[i, 1]],
                [o[2], o[2] + axis_len * Zw[i, 2]],
                "b",
                lw=1,
            )
            if show_id:
                ax.text(
                    o[0],
                    o[1],
                    o[2],
                    f"{i}",
                    color="black",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    weight="bold",
                )

        if pts is not None:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=1,
                c="gray",
                alpha=0.3,
                label="Points",
            )

        ax.set_title(f"{view_name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)

    # 绘制每个视角到不同 subplot
    for idx, view in enumerate(views):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        _plot_one(ax, view["elev"], view["azim"], view["name"])

    fig.suptitle(title)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "cameras_all_views.png")
    plt.savefig(out_path, dpi=200)
    print(f"[Saved] {out_path}")
    plt.close()


def plot_cameras_timeline(
    preds,
    out_path="./vis_cameras/cameras_timeline.png",
    axis_len=1,
    dx=0.2,  # 每一帧沿时间轴的平移步长
    timeline_axis="x",  # "x" | "y" | "z"
    stride=1,  # 下采样：每隔多少帧绘制一次
    wrap=None,  # 每行最多放多少帧；None=不换行
    show_id=True,
    elev=25,
    azim=-60,  # 视角
):
    """
    将每一帧相机按时间在指定轴方向平移（t * dx），生成“时间走廊”效果。

    Args
    ----
    preds: dict，包含 "extrinsic" (T,3,4) 或 (T,4,4)
    out_path: 输出图片路径
    axis_len: 单个相机局部坐标轴长度
    dx: 帧间平移步长（时间轴方向的间距）
    timeline_axis: 时间轴使用的世界轴向（"x"/"y"/"z"）
    stride: 帧下采样（1 表示每帧都画，2 表示隔帧等）
    wrap: 每行最多放多少帧（达到后换行到下一条“走廊”），None 表示不换行
    show_id: 是否标注帧号
    elev, azim: 3D 视角
    """
    # ---- 解析外参，得到相机中心与旋转 ----
    E = preds["extrinsic"]
    if E.shape[-2:] == (3, 4):
        R = E[..., :3, :3]  # (T,3,3)
        t = E[..., :3, 3]  # (T,3)
    else:  # (T,4,4)
        R = E[..., :3, :3]
        t = E[..., :3, 3]
    T = R.shape[0]
    if T == 0:
        raise ValueError("No camera extrinsics found.")

    # 相机中心（世界坐标）
    C = -np.einsum("tij,tj->ti", R.transpose(0, 2, 1), t)  # (T,3)

    # 选帧
    idxs = np.arange(0, T, stride)
    if len(idxs) == 0:
        idxs = np.array([0])

    # 时间轴单位向量
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    if timeline_axis not in axis_map:
        raise ValueError("timeline_axis must be one of {'x','y','z'}")
    a = axis_map[timeline_axis]

    # ---- 计算每帧平移（考虑 wrap 换行）----
    # 行/列布局：列坐标 = 帧序号 mod wrap，行坐标 = 帧序号 // wrap
    def time_offset(k):
        if wrap is None:
            col, row = k, 0
        else:
            col, row = (k % wrap), (k // wrap)
        # 横向沿时间轴平移 col*dx，纵向把每一行再下移（或侧移）一个固定距离 row*row_gap
        # 行间距用 dx*1.5（与 axis 不同方向分离）
        # 找一条与 timeline_axis 不同的正交轴作为换行方向：
        if timeline_axis == "x":
            row_axis = np.array([0.0, 0.0, 1.0])  # 用 z 方向拉开行
        elif timeline_axis == "y":
            row_axis = np.array([1.0, 0.0, 0.0])  # 用 x 方向拉开行
        else:  # "z"
            row_axis = np.array([1.0, 0.0, 0.0])  # 用 x 方向拉开行
        row_gap = dx * 1.5
        return col * dx * a + row * row_gap * row_axis

    # 为所有被画的帧准备平移后中心与基轴
    C_shift_list = []
    Xw_list, Yw_list, Zw_list = [], [], []
    for i, t_idx in enumerate(idxs):
        o = time_offset(i)  # (3,)
        C_shift_list.append(C[t_idx] + o)  # 平移后的中心
        # 将相机局部轴变换到世界：R^T * ex/ey/ez
        Rt = R[t_idx].T
        Xw_list.append(Rt @ np.array([1.0, 0.0, 0.0]))
        Yw_list.append(Rt @ np.array([0.0, 1.0, 0.0]))
        Zw_list.append(Rt @ np.array([0.0, 0.0, 1.0]))

    C_shift = np.stack(C_shift_list, axis=0)  # (N,3)
    Xw = np.stack(Xw_list, axis=0)  # (N,3)
    Yw = np.stack(Yw_list, axis=0)
    Zw = np.stack(Zw_list, axis=0)

    # ---- 坐标范围（稳定）----
    xyz_min = C_shift.min(axis=0)
    xyz_max = C_shift.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    half = (xyz_max - xyz_min).max() / 2.0
    margin = 0.2 * max(half, 1e-6)
    xyz_min = center - (half + margin)
    xyz_max = center + (half + margin)

    # ---- 绘图 ----
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(xyz_min[0], xyz_max[0])
    # ax.set_ylim(xyz_min[1], xyz_max[1])
    ax.set_zlim(xyz_min[2], xyz_max[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"Cameras laid along time on {timeline_axis.upper()} (dx={dx}, stride={stride}, wrap={wrap})"
    )

    # 画相机中心与时间轨迹（按顺序连线）
    ax.plot(
        C_shift[:, 0],
        C_shift[:, 1],
        C_shift[:, 2],
        color="royalblue",
        lw=1.5,
        label="timeline path",
    )
    ax.scatter(
        C_shift[:, 0],
        C_shift[:, 1],
        C_shift[:, 2],
        c="crimson",
        s=15,
        label="cam centers",
    )

    # 画每个相机的局部坐标轴（短线）
    for i in range(C_shift.shape[0]):
        o = C_shift[i]
        ax.plot(
            [o[0], o[0] + axis_len * Xw[i, 0]],
            [o[1], o[1] + axis_len * Xw[i, 1]],
            [o[2], o[2] + axis_len * Xw[i, 2]],
            "r",
            lw=1,
        )
        ax.plot(
            [o[0], o[0] + axis_len * Yw[i, 0]],
            [o[1], o[1] + axis_len * Yw[i, 1]],
            [o[2], o[2] + axis_len * Yw[i, 2]],
            "g",
            lw=1,
        )
        ax.plot(
            [o[0], o[0] + axis_len * Zw[i, 0]],
            [o[1], o[1] + axis_len * Zw[i, 1]],
            [o[2], o[2] + axis_len * Zw[i, 2]],
            "b",
            lw=1,
        )
        if show_id:
            ax.text(
                o[0],
                o[1],
                o[2],
                f"{idxs[i]}",
                color="k",
                fontsize=8,
                ha="center",
                va="bottom",
                weight="bold",
            )

    ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=0, azim=90)  # 默认视角

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {out_path}")
