#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/estimate_camera_position.py
Project: /workspace/code/triangulation
Created Date: Wednesday September 24th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday September 24th 2025 9:30:28 pm
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
from triangulation.vis.camera import save_camera
from pathlib import Path

from triangulation.camera_position.camera_position import (
    estimate_camera_pose_from_kpt,
    estimate_camera_pose_from_ORB,
    estimate_camera_pose_from_SIFT,
)


def save_RT_matrices(R_list, T_list, save_path: str):
    """
    保存一系列相机外参 (R, T) 到文件。

    参数:
        R_list: list of (3,3) ndarray
        T_list: list of (3,) or (3,1) ndarray
        save_path: str, 输出路径 (.npz 或 .npy)
    """
    R_array = np.stack(R_list, axis=0)
    T_array = np.stack([np.ravel(T) for T in T_list], axis=0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 自动选择格式
    if save_path.suffix == ".npz":
        np.savez_compressed(save_path, R=R_array, T=T_array)
    elif save_path.suffix == ".npy":
        np.save(save_path, {"R": R_array, "T": T_array})
    else:
        # 默认保存 npz
        np.savez_compressed(str(save_path) + ".npz", R=R_array, T=T_array)

    print(f"[INFO] Saved R,T to {save_path}")


# ---------- 主处理函数 ----------
def process_two_video(
    K, left_kpts, left_vframes, right_kpts, right_vframes, output_path, baseline_m
):

    r_list, t_list = [], []

    os.makedirs(output_path, exist_ok=True)

    # relative pose from left anr right kpts
    if left_kpts.shape[0] != right_kpts.shape[0]:
        raise ValueError(
            f"Frame mismatch: {left_kpts.shape[0]} vs {right_kpts.shape[0]}"
        )

    for i in range(left_kpts.shape[0]):
        l_kpt, r_kpt = left_kpts[i], right_kpts[i]

        # drop the 0 value keypoints
        assert (
            l_kpt.shape == r_kpt.shape
        ), f"Keypoints shape mismatch: {l_kpt.shape} vs {r_kpt.shape}"

        l_frame = left_vframes[i] if left_vframes is not None else None
        r_frame = right_vframes[i] if right_vframes is not None else None

        # * estimate camera position from SIFT
        R, T, *_ = estimate_camera_pose_from_SIFT(
            l_frame,
            r_frame,
            K,
            baseline_m,
        )

        save_camera(
            K, R, T, os.path.join(output_path, "camera/SIFT"), f"camera_{i:04d}.png"
        )

        # * estimate camera position from kpts
        R, T, mask_pose = estimate_camera_pose_from_kpt(l_kpt, r_kpt, K, baseline_m=20)

        save_camera(
            K, R, T, os.path.join(output_path, "camera/kpt"), f"camera_{i:04d}.png"
        )

        R, T, *_ = estimate_camera_pose_from_ORB(l_frame, r_frame, K, baseline_m)
        save_camera(
            K, R, T, os.path.join(output_path, "camera/ORB"), f"camera_{i:04d}.png"
        )

        # * 固定相机位置
        # Left camera = world
        def Ry(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

        # Right camera: at z=20m and yaw 180° (对视)
        C2 = np.array([0, 0, 20.0])
        R2 = Ry(np.deg2rad(180))
        T2 = -R2 @ C2

        save_camera(
            K, R2, T2, os.path.join(output_path, "camera/fixed"), f"camera_{i:04d}.png"
        )

        # * COLMAP
        # visualize_SIFT_matches(
        #     to_gray_cv_image(l_frame),
        #     to_gray_cv_image(r_frame),
        #     pts1,
        #     pts2,
        #     os.path.join(output_path, "SIFT"),
        #     i,
        # )

        r_list.append(R2)
        t_list.append(T2)

    save_RT_matrices(r_list, t_list, os.path.join(output_path, "RT_matrices.npz"))

    return r_list, t_list


def process_one_video(K, kpts, vframes, baseline_m, out_dir: str):
    r_list, t_list = [], []
    os.makedirs(out_dir, exist_ok=True)

    # * estimate camera position from frames
    for i in range(len(vframes) - 1):
        print(f"\n[INFO] Processing frame pair {i} and {i+1}")

        R, T, _ = estimate_camera_pose_from_ORB(
            vframes[i], vframes[i + 1], K, baseline_m=baseline_m
        )
        print("Rotation:\n", R)
        print("Translation direction:", T)

        save_camera(K, R, T, os.path.join(out_dir, "camera/ORB"), f"camera_{i:04d}.png")

        # * estimate camera position from SIFT
        R, T, *_ = estimate_camera_pose_from_SIFT(
            vframes[i], vframes[i + 1], K, baseline_m=baseline_m
        )
        print("Rotation:\n", R)
        print("Translation direction:", T)

        save_camera(
            K, R, T, os.path.join(out_dir, "camera/SIFT"), f"camera_{i:04d}.png"
        )

    # * estimate camera position from kpts
    for kpt in range(kpts.shape[0] - 1):
        R, T, mask_pose = estimate_camera_pose_from_kpt(
            kpts[kpt], kpts[kpt + 1], K, baseline_m=baseline_m
        )
        save_camera(
            K, R, T, os.path.join(out_dir, "camera/kpt"), f"camera_kpt_{kpt:04d}.png"
        )

        print("Rotation:\n", R)
        print("Translation direction:", T)

    r_list.append(R)
    t_list.append(T)

    # save RT
    save_RT_matrices(r_list, t_list, os.path.join(out_dir, "RT_matrices.npz"))

    return r_list, t_list
