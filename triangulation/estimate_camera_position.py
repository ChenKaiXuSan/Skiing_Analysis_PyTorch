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

from triangulation.camera_position.camera_position import (
    estimate_camera_pose_from_kpt,
    estimate_camera_pose_from_ORB,
    estimate_camera_pose_from_SIFT,
)


from triangulation.vis.camera import save_camera


# ---------- 主处理函数 ----------
def process_two_video(
    K, left_kpts, left_vframes, right_kpts, right_vframes, output_path
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
        )

        save_camera(
            K, R, T, os.path.join(output_path, "camera/SIFT"), f"camera_{i:04d}.png"
        )

        # * estimate camera position from kpts
        R, T, mask_pose = estimate_camera_pose_from_kpt(
            l_kpt,
            r_kpt,
            K,
        )

        save_camera(
            K, R, T, os.path.join(output_path, "camera/kpt"), f"camera_{i:04d}.png"
        )

        R, T, *_ = estimate_camera_pose_from_ORB(
            l_frame,
            r_frame,
            K,
        )
        save_camera(
            K, R, T, os.path.join(output_path, "camera/ORB"), f"camera_{i:04d}.png"
        )

        # * 固定相机位置
        # Left camera = world
        def Ry(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

        def pose_from_center_R(C, R):
            C = np.asarray(C, dtype=float).reshape(3, 1)
            return R, -R @ C

        # Right camera: at z=20m and yaw 180° (对视)
        C2_world = np.array([0, 0, 20.0])
        R2 = Ry(np.deg2rad(180))
        R2, T2 = pose_from_center_R(C2_world, R2)

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

        r_list.append(R)
        t_list.append(T)

    return r_list, t_list


def process_one_video(K, kpts, vframes, out_dir: str):
    r_list, t_list = [], []
    os.makedirs(out_dir, exist_ok=True)

    # * estimate camera position from frames
    for i in range(len(vframes) - 1):
        print(f"\n[INFO] Processing frame pair {i} and {i+1}")
        R, T, (yaw, pitch, roll) = estimate_camera_pose_from_ORB(
            vframes[i],
            vframes[i + 1],
            K,
        )
        print("Rotation:\n", R)
        print("Translation direction:", T)
        print(
            "Euler angles (deg): Yaw=%.2f, Pitch=%.2f, Roll=%.2f" % (yaw, pitch, roll)
        )

        save_camera(K, R, T, os.path.join(out_dir, "camera/ORB"), f"camera_{i:04d}.png")

        # * estimate camera position from SIFT
        R, T, *_ = estimate_camera_pose_from_SIFT(
            vframes[i],
            vframes[i + 1],
            K,
        )
        print("Rotation:\n", R)
        print("Translation direction:", T)
        print(
            "Euler angles (deg): Yaw=%.2f, Pitch=%.2f, Roll=%.2f" % (yaw, pitch, roll)
        )

        save_camera(
            K, R, T, os.path.join(out_dir, "camera/SIFT"), f"camera_{i:04d}.png"
        )

    # * estimate camera position from kpts
    for kpt in range(kpts.shape[0] - 1):
        R, T, mask_pose = estimate_camera_pose_from_kpt(
            kpts[kpt],
            kpts[kpt + 1],
            K,
        )
        save_camera(
            K, R, T, os.path.join(out_dir, "camera/kpt"), f"camera_kpt_{kpt:04d}.png"
        )

        print("Rotation:\n", R)
        print("Translation direction:", T)

    r_list.append(R)
    t_list.append(T)
    return r_list, t_list
