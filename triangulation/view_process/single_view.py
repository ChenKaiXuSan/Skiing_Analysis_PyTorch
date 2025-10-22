#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/two_view.py
Project: /workspace/code/triangulation
Created Date: Thursday September 25th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday September 25th 2025 3:59:12 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import os
import csv
import numpy as np
import logging

logger = logging.getLogger(__name__)

from triangulation.vis.camera import save_camera

from triangulation.camera_position.camera_position import (
    estimate_camera_pose_from_kpt,
    estimate_camera_pose_from_ORB,
    estimate_camera_pose_from_SIFT,
    estimate_pose_from_bbox_region,
)
from triangulation.camera_position.camera_position_kpt_bbox import (
    estimate_pose_from_bbox_and_kpt,
)


# ---------- 小工具 ----------
def cam_center_from_RT(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    外参满足 x_c = R x_w + t （world->cam）
    相机中心(世界系) C = -R^T t
    """
    return -R.T @ t


def _stack_or_empty(lst, shape):
    if len(lst) == 0:
        return np.zeros(shape, dtype=float)
    return np.stack(lst, axis=0)


class PoseLogger:
    """
    记录多种方法的R/T，并输出 npz + csv
    """

    def __init__(self):
        # methods -> {'R': [3x3...], 't': [3...], 'meta': [dict...]}
        self.data = {}

    def add(self, method: str, frame_idx: int, R: np.ndarray, t: np.ndarray, **meta):
        d = self.data.setdefault(method, {"R": [], "t": [], "frame": [], "meta": []})
        d["R"].append(np.asarray(R, float))
        d["t"].append(np.asarray(t, float).reshape(3))
        d["frame"].append(int(frame_idx))
        d["meta"].append(meta or {})

    def save_npz(self, path_npz: str):
        save_dict = {}
        for m, d in self.data.items():
            R = _stack_or_empty(d["R"], (0, 3, 3))
            t = _stack_or_empty(d["t"], (0, 3))
            frames = np.array(d["frame"], dtype=int)
            save_dict[f"{m}_R"] = R
            save_dict[f"{m}_t"] = t
            save_dict[f"{m}_frame"] = frames
        np.savez(path_npz, **save_dict)

    def save_csv(self, path_csv: str):
        """
        写一份汇总：方法、帧号、||t||、相机中心C（世界系）的x,y,z
        （注意：若你的R,t是相对位姿或不是 world->cam，需要在add时传入已经对齐到同一参考系的R,t）
        """
        rows = []
        for m, d in self.data.items():
            for f, R, t, meta in zip(d["frame"], d["R"], d["t"], d["meta"]):
                tnorm = float(np.linalg.norm(t))
                C = cam_center_from_RT(R, t)  # 世界系相机中心
                rows.append(
                    {
                        "method": m,
                        "frame": f,
                        "t_norm": tnorm,
                        "C_x": C[0],
                        "C_y": C[1],
                        "C_z": C[2],
                        **{f"meta_{k}": v for k, v in meta.items()},
                    }
                )
        # 排序更好看
        rows.sort(key=lambda r: (r["frame"], r["method"]))
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
        with open(path_csv, "w", newline="", encoding="utf-8") as fp:
            fieldnames = sorted(
                {k for r in rows for k in r.keys()},
                key=lambda x: (
                    ["method", "frame", "t_norm", "C_x", "C_y", "C_z"].index(x)
                    if x in ["method", "frame", "t_norm", "C_x", "C_y", "C_z"]
                    else 1000
                ),
            )
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ---------- 你的流程，嵌入 PoseLogger ----------
def process_single_video(
    K,
    single_kpts,
    single_vframes,
    single_bbox,
    output_path,
    baseline_m,
):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "camera"), exist_ok=True)

    pose_logger = PoseLogger()  # ← 新增

    for i in range(0, single_kpts.shape[0]-1):
        l_kpt, r_kpt = single_kpts[i], single_kpts[i+1]

         # drop the 0 value keypoints
        assert (
            l_kpt.shape == r_kpt.shape
        ), f"Keypoints shape mismatch: {l_kpt.shape} vs {r_kpt.shape}"

        l_frame = single_vframes[i] if single_vframes is not None else None
        r_frame = single_vframes[i+1] if single_vframes is not None else None

        l_bbox = single_bbox[i] if single_bbox is not None else None
        r_bbox = single_bbox[i+1] if single_bbox is not None else None

        # --- SIFT ---
        R, t, *_ = estimate_camera_pose_from_SIFT(l_frame, r_frame, K, baseline_m)
        save_camera(
            K, R, t, os.path.join(output_path, "camera/SIFT"), frame_num=i
        )
        pose_logger.add("SIFT", i, R, t, baseline=baseline_m)

        # --- KPT（注意：你原代码 baseline_m=20 是硬编码，这里保留但也记录到meta） ---
        Rk, tk, mask_pose = estimate_camera_pose_from_kpt(
            l_kpt, r_kpt, K, baseline_m=20
        )
        save_camera(
            K, Rk, tk, os.path.join(output_path, "camera/kpt"), frame_num=i
        )
        pose_logger.add(
            "KPT",
            i,
            Rk,
            tk,
            inliers=int(mask_pose.sum()) if mask_pose is not None else None,
            baseline=20,
        )

        # --- ORB ---
        Ro, to, *_ = estimate_camera_pose_from_ORB(l_frame, r_frame, K, baseline_m)
        save_camera(
            K, Ro, to, os.path.join(output_path, "camera/ORB"), frame_num=i
        )
        pose_logger.add("ORB", i, Ro, to, baseline=baseline_m)

        # --- BBOX区域特征 ---
        Rb, tb, bbox_meta = estimate_pose_from_bbox_region(
            l_frame, r_frame, l_bbox, r_bbox, K, baseline_m
        )
        save_camera(
            K, Rb, tb, os.path.join(output_path, "camera/bbox"), frame_num=i
        )
        pose_logger.add("BBOX", i, Rb, tb, baseline=baseline_m)

        # --- KPT + BBOX 结合 ---
        R_combined, t_combined, _, _ = estimate_pose_from_bbox_and_kpt(
            l_frame, r_frame, l_bbox, r_bbox, K, baseline_m, kptsL=l_kpt, kptsR=r_kpt
        )
        save_camera(
            K,
            R_combined,
            t_combined,
            os.path.join(output_path, "camera/combined"),
            f"camera_{i:04d}.png",
        )
        pose_logger.add("COMBINED", i, R_combined, t_combined, baseline=baseline_m)

        # --- FIXED（演示：把右相机放到指定位置与朝向） ---
        def Ry(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

        C2 = np.array([0.0, 0.0, 20.0])  # 世界坐标下的相机中心
        R2 = Ry(np.deg2rad(180.0))  # 对视（绕y转180°）
        t2 = -R2 @ C2  # t = -R C
        save_camera(
            K, R2, t2, os.path.join(output_path, "camera/fixed"), frame_num=i
        )
        pose_logger.add(
            "FIXED", i, R2, t2, note="demo_fixed_pose", z=20.0, yaw_deg=180.0
        )

    # --- 统一保存 ---
    npz_path = os.path.join(output_path, "camera_position_all_methods.npz")
    csv_path = os.path.join(output_path, "camera_position_summary.csv")
    pose_logger.save_npz(npz_path)
    pose_logger.save_csv(csv_path)

    logger.info(f"Saved poses:\n- {npz_path}\n- {csv_path}")
    return pose_logger.data
