#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Triangulate 3D joints from two-view 2D keypoints using either video frames or pre-extracted keypoints.
Supports modular triangulation and pose estimation pipeline.

Author: Kaixu Chen
Last Modified: August 4th, 2025
"""

import os
import numpy as np
import cv2
import torch
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- 相机参数 ----------
K = np.array(
    [[1675.1430, 0.0, 880.9680], [0.0, 1286.3486, 1025.9397], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

R_DEFAULT = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
T_DEFAULT = np.array([1, 0.0, 0.0], dtype=np.float32)


# ---------- 核心模块 ----------
def triangulate_joints(keypoints1, keypoints2, K, R, T):
    if keypoints1.shape != keypoints2.shape or keypoints1.shape[1] != 2:
        raise ValueError(
            f"Keypoints shape mismatch: {keypoints1.shape} vs {keypoints2.shape}"
        )
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T.reshape(3, 1)))
    pts_4d = cv2.triangulatePoints(P1, P2, keypoints1.T, keypoints2.T)
    return (pts_4d[:3, :] / pts_4d[3, :]).T


def visualize_3d_joints(joints_3d, save_path, title="Triangulated 3D Joints"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c="blue", s=30)
    for i, (x, y, z) in enumerate(joints_3d):
        ax.text(x, y, z, str(i), size=8)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


# ---------- 相机姿态估计模块 ----------
def estimate_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask_pose


# ---------- 从.pt文件重建 ----------
def load_keypoints_from_pt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    print(f"[INFO] Loading: {file_path}")
    data = torch.load(file_path, map_location="cpu")
    keypoints = np.array(data["keypoint"]["keypoint"]).squeeze(0)
    if keypoints.ndim != 3 or keypoints.shape[2] != 2:
        raise ValueError(f"Invalid shape: {keypoints.shape}")
    return keypoints


def process_one_video(left_path, right_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    kpt1 = load_keypoints_from_pt(left_path)
    kpt2 = load_keypoints_from_pt(right_path)

    if kpt1.shape[0] != kpt2.shape[0]:
        raise ValueError(f"Frame mismatch: {kpt1.shape[0]} vs {kpt2.shape[0]}")

    for i in range(kpt1.shape[0]):
        pts1, pts2 = kpt1[i], kpt2[i]

        # 姿态估计
        R, T, mask = estimate_pose(pts1, pts2, K)
        if R is None or T is None:
            print(f"[WARN] Frame {i}: pose estimation failed")
            continue

        # 应用姿态进行三角测量
        joints_3d = triangulate_joints(pts1, pts2, K, R, T)
        img_path = os.path.join(output_path, f"frame_{i:04d}.png")
        visualize_3d_joints(joints_3d, save_path=img_path)


# ---------- 多人批量处理入口 ----------
def main_pt(input_root, output_root):
    subjects = sorted(glob.glob(f"{input_root}/*/"))
    if not subjects:
        raise FileNotFoundError(f"No folders found in: {input_root}")
    print(f"[INFO] Found {len(subjects)} subjects in {input_root}")
    for person_dir in subjects:
        person_name = os.path.basename(person_dir.rstrip("/"))
        print(f"\n[INFO] Processing: {person_name}")
        left = os.path.join(person_dir, "osmo_1.pt")
        right = os.path.join(person_dir, "osmo_2.pt")
        out_dir = os.path.join(output_root, person_name)
        try:
            process_one_video(left, right, out_dir)
        except Exception as e:
            print(f"[ERROR] Failed: {person_name} – {e}")


if __name__ == "__main__":
    input_path = "/workspace/data/pt"
    output_path = "/workspace/code/logs/triangulated_3d_joints"
    main_pt(input_path, output_path)

    # Or use this for direct video input:
    # main_video("left.mp4", "right.mp4")
