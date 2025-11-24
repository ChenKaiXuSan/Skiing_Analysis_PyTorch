#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# --------------------------- 常量 ---------------------------

# COCO-17 骨架（左/右臂、腿、躯干、头部）
COCO_SKELETON: List[Tuple[int, int]] = [
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

__all__ = [
    "COCO_SKELETON",
    "draw_camera",
    "visualize_3d_joints",
    "compute_bone_lengths",
    "compute_bone_stats",
]

# ---- 新增：骨长计算 ----


def compute_bone_lengths(
    pts: np.ndarray,
    skeleton: Iterable[Tuple[int, int]],
    *,
    ignore_nan: bool = True,
) -> np.ndarray:
    """
    计算一帧 3D 关键点在给定骨架下的骨长。
    pts: (K,3)
    返回: (E,) 对应 skeleton 中每条边的长度；无效边为 np.nan
    """
    P = np.asarray(pts, dtype=float)
    L: List[float] = []
    for i, j in skeleton:
        if i >= len(P) or j >= len(P):
            L.append(np.nan)
            continue
        a, b = P[i], P[j]
        if ignore_nan and (not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
            L.append(np.nan)
            continue
        L.append(float(np.linalg.norm(a - b)))
    return np.asarray(L, dtype=float)


def compute_bone_stats(lengths: np.ndarray) -> Dict[str, float]:
    """
    对骨长（含 nan）做统计，返回 mean/median/std/min/max/valid_count。
    """
    x = np.asarray(lengths, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return dict(
            mean=np.nan,
            median=np.nan,
            std=np.nan,
            min=np.nan,
            max=np.nan,
            valid_count=0,
        )
    xv = x[valid]
    return dict(
        mean=float(np.nanmean(xv)),
        median=float(np.nanmedian(xv)),
        std=float(np.nanstd(xv)),
        min=float(np.nanmin(xv)),
        max=float(np.nanmax(xv)),
        valid_count=int(valid.sum()),
    )


# --------------------------- 3D 可视化 ---------------------------


def draw_camera(
    ax: plt.Axes,
    R: np.ndarray,  # (3,3)
    T: np.ndarray,  # (3,) or (3,1)
    K: np.ndarray,  # (3,3)
    image_size: Tuple[int, int],  # (W, H) in px
    axis_len: float = 1,
    frustum_depth: float = 1,
    colors: Tuple[str, str, str] = ("r", "g", "b"),
    label: Optional[str] = None,
    ray_scale_mode: Literal[
        "depth", "focal"
    ] = "depth",  # "depth": 固定 z_cam；"focal": 以焦距近似尺度
    linewidths: Dict[str, float] = None,  # {"axis": 2.0, "frustum": 1.0}
    frustum_alpha: float = 1.0,
) -> np.ndarray:
    """
    在 Matplotlib 3D 轴上绘制 OpenCV 相机 (坐标轴 & 视锥)，并对齐到 Matplotlib 3D 坐标系：
      OpenCV:     x→右, y→下, z→前
      Matplotlib: x→右, y→前, z→上

    参数
    ----
    R, T : 相机外参
      - 如果 convention="world2cam"（默认），满足 Xc = R @ Xw + T
      - 如果 convention="cam2world"，满足 Xw = R @ Xc + T
    K : 内参矩阵 (3x3)
    image_size : (W, H) 像素
    axis_len : 相机坐标轴长度（世界单位）
    frustum_depth : 视锥体深度（世界单位），当 ray_scale_mode="depth" 时生效
    colors : 坐标轴颜色 (X, Y, Z)
    label : 相机中心标注
    convention : 外参定义方式
    ray_scale_mode :
        - "depth": 将像素反投影射线缩放到 z_cam=frustum_depth
        - "focal": 以焦距近似尺度（更像真实FOV），深度感基于 fx/fy
    linewidths : 线宽配置，默认为 {"axis": 2.0, "frustum": 1.0}
    frustum_alpha : 视锥边透明度

    返回
    ----
    C_plt : (3,) 相机中心（已映射到 Matplotlib 世界系）
    """
    # ------- 参数与形状 -------
    if linewidths is None:
        linewidths = {"axis": 2.0, "frustum": 1.0}

    R = np.asarray(R, dtype=float).reshape(3, 3)
    T = np.asarray(T, dtype=float).reshape(3, 1)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    W, H = [float(x) for x in image_size]

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

    # ------- 相机中心 (OpenCV 世界系) -------
    # Xc = R Xw + T  => 0 = R C + T  => C = -R^T T
    C_cv = -R.T @ T
    R_wc = R.T  # world <- cam
    t_wc = C_cv.reshape(3)  # same as camera center

    # ------- 相机中心映射到 Matplotlib 世界系 -------
    C_plt = (M @ C_cv).reshape(3)

    # ------- 画相机坐标轴（使用 R_wc 的行向量作为相机轴在世界系中的表示）-------
    # rows of R_wc: x_cam,y_cam,z_cam expressed in world(OpenCV) coords
    axes_cv = R_wc.copy()
    lw_axis = float(linewidths.get("axis", 2.0))
    for axis_cv, color in zip(axes_cv, colors):
        end_vec_cv = axis_cv * axis_len  # (3,)
        end_vec_plt = M @ end_vec_cv
        ax.plot(
            [C_plt[0], C_plt[0] + end_vec_plt[0]],
            [C_plt[1], C_plt[1] + end_vec_plt[1]],
            [C_plt[2], C_plt[2] + end_vec_plt[2]],
            c=color,
            lw=lw_axis,
        )

    # ------- 构造视锥：像素四角反投影 -> 相机系 -> 世界系(OpenCV) -> Matplotlib -------
    # 像素四角齐次坐标
    corners_px = np.array(
        [
            [0.0, 0.0, 1.0],
            [W - 1, 0.0, 1.0],
            [W - 1, H - 1, 1.0],
            [0.0, H - 1, 1.0],
        ],
        dtype=float,
    )

    Kinv = np.linalg.inv(K)
    # 射线方向（相机系）
    rays_cam = (Kinv @ corners_px.T).T  # (4,3)

    if ray_scale_mode == "depth":
        # 令 z_cam = frustum_depth
        scale = frustum_depth / np.clip(rays_cam[:, 2:3], 1e-12, None)
        rays_cam = rays_cam * scale
    else:
        # 按焦距尺度：以 fx/fy 的均值近似一个“单位深度”量级
        fx, fy = K[0, 0], K[1, 1]
        s = float((fx + fy) * 0.5)
        if s <= 1e-12:
            s = 1.0
        rays_cam = rays_cam / s * max(axis_len, frustum_depth)

    # cam -> world(OpenCV)
    corners_w_cv = (R_wc @ rays_cam.T).T + t_wc.reshape(1, 3)  # (4,3)
    # world(OpenCV) -> world(Matplotlib)
    corners_w_plt = (M @ corners_w_cv.T).T

    # ------- 画视锥边 -------
    lw_frustum = float(linewidths.get("frustum", 1.0))
    for p in corners_w_plt:
        ax.plot(
            [C_plt[0], p[0]],
            [C_plt[1], p[1]],
            [C_plt[2], p[2]],
            c="k",
            lw=lw_frustum,
            alpha=frustum_alpha,
        )
    loop = [0, 1, 2, 3, 0]
    ax.plot(
        corners_w_plt[loop, 0],
        corners_w_plt[loop, 1],
        corners_w_plt[loop, 2],
        c="k",
        lw=lw_frustum,
        alpha=frustum_alpha,
    )

    # ------- 标签 -------
    if label:
        ax.text(C_plt[0], C_plt[1], C_plt[2], label, fontsize=9, color="black")

    return C_plt


def _ensure_joints3d_arr(joints_3d: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    arr = torch.as_tensor(joints_3d).detach().cpu().numpy()
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr.T
    raise ValueError(f"joints_3d must be (K,3) or (3,K), got {arr.shape}")


def visualize_3d_joints(
    joints_3d: Union[np.ndarray, torch.Tensor],
    R: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int],
    save_path: Union[str, Path],
    title: str = "Triangulated 3D Joints",
    dpi: int = 300,
    show_stats: bool = True,  # 是否叠加骨长统计文本
) -> Optional[Dict[str, float]]:
    """
    保存 3D 关键点+相机示意图，并可：
      - 统一等比例 & 统一刻度（axis_limits / tick_step）
      - 计算并显示骨长统计（均值/中位数/方差等）
      - 可选在每条骨中点标注长度
    返回: 若 show_stats=True 则返回统计字典；否则 None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pts = _ensure_joints3d_arr(joints_3d).astype(float)
    R = np.asarray(R, dtype=float).reshape(2, 3, 3)
    T = np.asarray(T, dtype=float).reshape(2, 3)

    # 初始化图形
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 相机
    draw_camera(ax=ax, R=R[0], T=T[0], K=K[0], image_size=image_size, label="Cam1")
    draw_camera(ax=ax, R=R[1], T=T[1], K=K[1], image_size=image_size, label="Cam2")

    # 可视化坐标置换（Y 朝上）
    # 把pts从cv2世界系映射到matplotlib世界系
    M = np.array(
        [
            [1.0, 0.0, 0.0],  # x -> x
            [0.0, 0.0, 1.0],  # z -> y
            [0.0, -1.0, 0.0],  # -y -> z
        ],
        dtype=float,
    )
    pts_plot = (M @ pts.T).T
    xlab, ylab, zlab = "X", "Z", "Y (up)"

    # 点与索引
    ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], c="blue", s=30)
    for i, (x, y, z) in enumerate(pts_plot):
        ax.text(x, y, z, str(i), size=8)

    # 骨架与长度
    skeleton = COCO_SKELETON
    if skeleton is not None:
        lengths = compute_bone_lengths(
            pts, skeleton
        )  # 用原坐标算长度（与可视化变换无关）

        # 画线（在绘图坐标系）
        for i, j in skeleton:
            if i < len(pts_plot) and j < len(pts_plot):
                if not (
                    np.all(np.isfinite(pts_plot[i]))
                    and np.all(np.isfinite(pts_plot[j]))
                ):
                    continue
                ax.plot(
                    [pts_plot[i, 0], pts_plot[j, 0]],
                    [pts_plot[i, 1], pts_plot[j, 1]],
                    [pts_plot[i, 2], pts_plot[j, 2]],
                    c="red",
                    linewidth=2,
                )
        stats = compute_bone_stats(lengths)
    else:
        stats = None

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)

    # 叠加统计文本
    if show_stats and stats is not None:
        txt = (
            f"bones: {stats['valid_count']}\n"
            f"mean:  {stats['mean']:.3f}\n"
            f"median:{stats['median']:.3f}\n"
            f"std:   {stats['std']:.3f}\n"
            f"min:   {stats['min']:.3f}\n"
            f"max:   {stats['max']:.3f}"
        )
        ax.text2D(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    plt.tight_layout()

    # ax.set_zlim(ax.get_zlim()[::-1])
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 25)
    # ax.set_zlim(-10, 30)

    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")

    return stats if show_stats else None
