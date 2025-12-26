#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/front_side/run.py
Project: /workspace/code/front_side
Created Date: Thursday December 25th 2025
Author: Kaixu Chen
-----
Comment:
侧面用左右视角进行刚体变换
前面用鸟览图进行融合

最后合成一个完整的运动轨迹视频
Have a good code time :)
-----
Last Modified: Thursday December 25th 2025 4:20:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import gc
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .front.run import process_front_frame
from .load import load_sam3_results, load_sam_3d_body_results
from .side.run import process_side_frame
from .utils.merge import merge_frame_to_video


def process_one_person(
    left_sam3d_body_path: Path,
    right_sam3d_body_path: Path,
    front_sam3_results: Path,
    output_dir: Path,
) -> None:
    """
    Process one person with multi-view bundle adjustment.
    """

    left_sam3d_body_res = load_sam_3d_body_results(left_sam3d_body_path.as_posix())
    right_sam3d_body_res = load_sam_3d_body_results(right_sam3d_body_path.as_posix())

    front_sam3_res = load_sam3_results(front_sam3_results.as_posix())

    for frame_idx in tqdm(range(len(left_sam3d_body_res)), desc="Processing frames"):
        # if frame_idx > 30:
        #     break

        # process side view
        left_frame, right_frame, kpts_world, R_RL, t_RL = process_side_frame(
            left_sam3d_body_res=left_sam3d_body_res,
            right_sam3d_body_res=right_sam3d_body_res,
            frame_idx=frame_idx,
            out_root=output_dir / "side",
        )

        # process front view
        image = front_sam3_res[frame_idx]["frame"]  # np.ndarray, (H,W,3)
        bbox_xyxy = front_sam3_res[frame_idx]["out_boxes_xywh"]  # np.ndarray, (4,) xyxy
        obj_ids = front_sam3_res[frame_idx].get("out_obj_ids", None)
        probs = front_sam3_res[frame_idx].get("out_probs", None)
        binary_masks = front_sam3_res[frame_idx].get("out_binary_masks")
        # foot point in bev pixels
        foot_xy_px, raw_img, bev_img = process_front_frame(
            image, bbox_xyxy, output_dir=output_dir / "front", frame_idx=frame_idx
        )

        # merge side and front results
        merge(
            kpts_world=kpts_world,
            foot_xy_px=foot_xy_px,
            raw_img=raw_img,
            bev_img=bev_img,
            output_dir=output_dir / "merge",
            frame_idx=frame_idx,
        )

    # merge side video
    merge_frame_to_video(save_path=output_dir / "side", flag="frame_scene", fps=30)
    merge_frame_to_video(save_path=output_dir / "side", flag="fused", fps=30)
    merge_frame_to_video(save_path=output_dir / "side", flag="scene", fps=30)

    # merge front video
    merge_frame_to_video(save_path=output_dir / "front", flag="raw_vis", fps=30)
    merge_frame_to_video(save_path=output_dir / "front", flag="bev_vis", fps=30)
    # merge merge video
    merge_frame_to_video(save_path=output_dir / "merge", flag="bev_vis", fps=30)
    # 清空内存
    gc.collect()


COCO_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
]


def project_world_to_bev_centered(
    kpts_world: np.ndarray,
    center_world: np.ndarray,
    center_px: tuple[int, int],
    meters_per_pixel: float,
    use_axes: tuple[int, int] = (0, 2),
    rot90_left: bool = False,  # True: rotate (x,z) left by 90° around center
):
    """需要把kpt world的坐标和bev的坐标对起来

    Args:
        kpts_world (np.ndarray): _description_
        center_world (np.ndarray): _description_
        center_px (tuple[int, int]): _description_
        meters_per_pixel (float): _description_
        use_axes (tuple[int, int], optional): _description_. Defaults to (0, 2).
        rot90_left (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    x_idx, z_idx = use_axes
    cx, cz = center_world[x_idx], center_world[z_idx]
    u0, v0 = center_px

    pts_uv = []
    for j in range(kpts_world.shape[0]):
        if not np.isfinite(kpts_world[j]).all():
            pts_uv.append(None)
            continue

        x, z = kpts_world[j, x_idx], kpts_world[j, z_idx]
        dx, dz = x - cx, z - cz

        # ---- rotate in XZ plane (around center) ----
        if rot90_left:
            dx, dz = dz, dx

        du = dx / meters_per_pixel
        dv = -dz / meters_per_pixel

        u = int(round(u0 + du))
        v = int(round(v0 + dv))
        pts_uv.append((u, v))
    return pts_uv


def draw_skeleton(bev_img: np.ndarray, pts_uv, edges=COCO_EDGES):
    h, w = bev_img.shape[:2]

    # lines
    for a, b in edges:
        if a < len(pts_uv) and b < len(pts_uv):
            pa, pb = pts_uv[a], pts_uv[b]
            if pa is None or pb is None:
                continue
            if 0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h:
                cv2.line(bev_img, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)

    # points
    for p in pts_uv:
        if p is None:
            continue
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(bev_img, p, 3, (0, 0, 255), -1, cv2.LINE_AA)

    return bev_img


def draw_bev_axes(
    img: np.ndarray,
    origin_px: tuple[int, int],
    meters_per_pixel: float,
    axis_len_m: float = 1.0,  # 轴长度（米）
    color_x=(0, 0, 255),  # X轴：红
    color_z=(0, 255, 0),  # Z轴：绿
    thickness: int = 2,
):
    """
    在 BEV 图上画局部坐标轴：
      +X → 右
      +Z → 下
    """
    # TODO: 这里需要修改一下

    u0, v0 = origin_px
    axis_len_px = int(round(axis_len_m / meters_per_pixel))

    # +X 轴（右）
    end_x = (u0 + axis_len_px, v0)
    cv2.arrowedLine(
        img,
        (u0, v0),
        end_x,
        color=color_x,
        thickness=thickness,
        tipLength=0.2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "+X",
        (end_x[0] + 5, end_x[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_x,
        2,
        cv2.LINE_AA,
    )

    # +Z 轴（下）
    end_z = (u0, v0 + axis_len_px)
    cv2.arrowedLine(
        img,
        (u0, v0),
        end_z,
        color=color_z,
        thickness=thickness,
        tipLength=0.2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "+Z",
        (end_z[0] + 5, end_z[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_z,
        2,
        cv2.LINE_AA,
    )

    return img


def merge(
    kpts_world: np.ndarray,
    foot_xy_px: dict,  # e.g. {"x": 320, "y": 240} or {"u":..., "v":...}
    raw_img: np.ndarray,
    bev_img: np.ndarray,
    frame_idx: int,
    output_dir: Path,
) -> None:
    bev_vis_path = output_dir / f"bev_vis"
    bev_vis_path.mkdir(parents=True, exist_ok=True)

    # 1) BEV中心像素（你说 foot_xy_px 就是中心）
    # TODO: 这里需要判断哪个是化学运动员
    u0 = int(round(foot_xy_px[2][0]))
    v0 = int(round(foot_xy_px[2][1]))
    center_px = (u0, v0)

    # 2) world中心：你要保证它和 foot_xy_px 对应的是同一个“中心定义”
    #    最稳：直接用关节点的均值；更好：用双脚/骨盆等稳定点
    center_world = np.nanmean(kpts_world, axis=0)  # (3,)

    # 3) m/px：你必须给一个尺度（根据你的BEV底图是什么尺度）
    meters_per_pixel = 0.02  # 2cm/px 举例，按你的图改

    bev_vis = bev_img.copy()

    pts_uv = project_world_to_bev_centered(
        kpts_world,
        center_world,
        center_px,
        meters_per_pixel,
        rot90_left=True,
    )

    # 1️⃣ 画坐标轴（先画，保证在底层）
    bev_vis = draw_bev_axes(
        bev_vis,
        origin_px=center_px,
        meters_per_pixel=meters_per_pixel,
        axis_len_m=1.0,  # 1 米坐标轴
    )

    # TODO:把运动员的运动方向画到图片上面去

    # 2️⃣ 再画中心点
    cv2.circle(bev_vis, center_px, 5, (255, 0, 0), -1, cv2.LINE_AA)

    # 3️⃣ 再画骨架
    bev_vis = draw_skeleton(bev_vis, pts_uv)

    # 保存结果
    out_path = bev_vis_path / f"{frame_idx:06d}.png"
    cv2.imwrite(str(out_path), bev_vis)
