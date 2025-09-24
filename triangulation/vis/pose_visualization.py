#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objs as go  # noqa: E402
import plotly.offline as py  # noqa: E402

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
    "set_axes_equal",
    "visualize_3d_joints",
    "visualize_3d_scene_interactive",
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
    ax: plt.Axes, R: np.ndarray, T: np.ndarray, scale: float = 0.1, label: str = "Cam"
) -> None:
    """
    在 3D 轴上画一个相机坐标系（右手，Z 朝前）。
    R: (3,3) 旋转矩阵；T: (3,) 或 (3,1) 平移。
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    T = np.asarray(T, dtype=float).reshape(3)
    origin = T

    axes = np.eye(3)
    colors = ("r", "g", "b")
    for axis, color in zip(axes, colors):
        end = (R @ axis) * scale + origin
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], c=color)

    # 视线方向（-Z）
    view_dir = (R @ np.array([0, 0, -1.0])) * scale * 1.5 + origin
    ax.plot(
        [origin[0], view_dir[0]],
        [origin[1], view_dir[1]],
        [origin[2], view_dir[2]],
        c="k",
        linestyle="--",
    )
    ax.text(*origin, label, color="black")


# ---- 可选：更通用的等比例工具，支持指定范围/半径/中心/刻度 ----
def set_axes_equal(
    ax: plt.Axes,
    *,
    limits: Optional[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = None,
    center: Optional[Tuple[float, float, float]] = None,
    radius: Optional[float] = None,
    tick_step: Optional[float] = None,
) -> None:
    """
    等比例三维坐标：
      - 若提供 limits=(xlim,ylim,zlim) 则直接采用；
      - 否则以 center+radius 设置立方体范围；
      - 若都未提供，回退到基于当前数据自适应的等比例。
      - tick_step: 指定三个轴统一的刻度间隔（可选）。
    """
    if limits is not None:
        (x0, x1), (y0, y1), (z0, z1) = limits
    else:
        if radius is None or center is None:
            # 回退：基于当前 axis 数据范围
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            x_range = abs(x_limits[1] - x_limits[0])
            y_range = abs(y_limits[1] - y_limits[0])
            z_range = abs(z_limits[1] - z_limits[0])
            x_mid = np.mean(x_limits)
            y_mid = np.mean(y_limits)
            z_mid = np.mean(z_limits)
            rad = 0.5 * float(max(x_range, y_range, z_range))
            x0, x1 = x_mid - rad, x_mid + rad
            y0, y1 = y_mid - rad, y_mid + rad
            z0, z1 = z_mid - rad, z_mid + rad
        else:
            cx, cy, cz = center
            rad = float(radius)
            x0, x1 = cx - rad, cx + rad
            y0, y1 = cy - rad, cy + rad
            z0, z1 = cz - rad, cz + rad

    ax.set_xlim3d([x0, x1])
    ax.set_ylim3d([y0, y1])
    ax.set_zlim3d([z0, z1])

    # 统一刻度
    if tick_step is not None and tick_step > 0:

        def _ticks(a, b):
            # 包含端点的等间隔
            start = np.floor(a / tick_step) * tick_step
            end = np.ceil(b / tick_step) * tick_step
            return np.arange(start, end + 1e-6, tick_step)

        ax.set_xticks(_ticks(x0, x1))
        ax.set_yticks(_ticks(y0, y1))
        ax.set_zticks(_ticks(z0, z1))


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
    save_path: Union[str, Path],
    *,
    title: str = "Triangulated 3D Joints",
    skeleton: Optional[Iterable[Tuple[int, int]]] = COCO_SKELETON,
    dpi: int = 300,
    y_up: bool = False,
    axis_limits: Optional[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = None,
    tick_step: Optional[float] = None,
    margin: float = 0.15,  # 自动范围时的边界余量
    show_stats: bool = True,  # 是否叠加骨长统计文本
    show_lengths: bool = False,  # 是否在每条骨的中点标注长度
    length_fmt: str = "{:.2f}",
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
    R = np.asarray(R, dtype=float).reshape(3, 3)
    T = np.asarray(T, dtype=float).reshape(3)

    # 可视化坐标置换（Y 朝上）
    if y_up:
        P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)
        pts_plot = (P @ pts.T).T
        R_plot = P @ R
        T_plot = P @ T
        xlab, ylab, zlab = "X", "Z", "Y (up)"
    else:
        pts_plot = pts
        R_plot, T_plot = R, T
        xlab, ylab, zlab = "X", "Y", "Z"

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 相机
    draw_camera(ax, np.eye(3), np.zeros(3), label="Cam1")
    draw_camera(ax, R_plot, T_plot, label="Cam2")

    # 点与索引
    ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], c="blue", s=30)
    for i, (x, y, z) in enumerate(pts_plot):
        ax.text(x, y, z, str(i), size=8)

    # 骨架与长度
    if skeleton is not None:
        lengths = compute_bone_lengths(
            pts, skeleton
        )  # 用原坐标算长度（与可视化变换无关）
        if show_lengths:
            for (i, j), L in zip(skeleton, lengths):
                if not np.isfinite(L) or i >= len(pts_plot) or j >= len(pts_plot):
                    continue
                p, q = pts_plot[i], pts_plot[j]
                mid = (p + q) / 2.0
                ax.text(
                    mid[0], mid[1], mid[2], length_fmt.format(L), color="purple", size=7
                )

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

    ax.set_title(title + (" (Y up)" if y_up else ""))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)

    # ---- 统一等比例 + 刻度 ----
    if axis_limits is None:
        # 基于数据自适应：考虑关键点与两台相机位置
        all_xyz = np.vstack([pts_plot, np.zeros((1, 3)), T_plot.reshape(1, 3)])
        mins = np.nanmin(all_xyz, axis=0)
        maxs = np.nanmax(all_xyz, axis=0)
        center = (mins + maxs) / 2.0
        rad = 0.5 * float(np.nanmax(maxs - mins)) * (1.0 + float(margin))
        set_axes_equal(ax, center=tuple(center), radius=rad, tick_step=tick_step)
    else:
        set_axes_equal(ax, limits=axis_limits, tick_step=tick_step)

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
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")

    return stats if show_stats else None


def visualize_3d_scene_interactive(
    joints_3d: Union[np.ndarray, torch.Tensor],
    R: np.ndarray,
    T: np.ndarray,
    save_path: Union[str, Path],
    *,
    title: str = "Interactive 3D Scene",
) -> None:
    """
    用 Plotly 生成可交互 HTML 的 3D 场景。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pts = _ensure_joints3d_arr(joints_3d)

    fig = go.Figure()

    # 点云
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers+text",
            text=[str(i) for i in range(len(pts))],
            marker=dict(size=4, color="blue"),
            name="joints",
        )
    )

    # 相机线段
    def camera_traces(Rm: np.ndarray, Tm: np.ndarray, label: str) -> List[go.Scatter3d]:
        Rm = np.asarray(Rm, dtype=float).reshape(3, 3)
        Tm = np.asarray(Tm, dtype=float).reshape(3)
        scale = 0.1
        traces: List[go.Scatter3d] = []

        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        colors = ["red", "green", "blue"]
        for axis, color in zip(axes, colors):
            end = Rm @ axis * scale + Tm
            traces.append(
                go.Scatter3d(
                    x=[Tm[0], end[0]],
                    y=[Tm[1], end[1]],
                    z=[Tm[2], end[2]],
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=f"{label}_{color}",
                    showlegend=False,
                )
            )
        view_dir = Rm @ np.array([0, 0, -1]) * scale * 1.5 + Tm
        traces.append(
            go.Scatter3d(
                x=[Tm[0], view_dir[0]],
                y=[Tm[1], view_dir[1]],
                z=[Tm[2], view_dir[2]],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name=f"{label}_view",
                showlegend=False,
            )
        )
        return traces

    for tr in camera_traces(np.eye(3), np.zeros(3), "Cam1"):
        fig.add_trace(tr)
    for tr in camera_traces(np.asarray(R), np.asarray(T), "Cam2"):
        fig.add_trace(tr)

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title=title,
        margin=dict(l=0, r=0, b=0, t=30),
    )
    py.plot(fig, filename=str(save_path), auto_open=False)
    print(f"[INFO] Saved interactive HTML to: {save_path}")
