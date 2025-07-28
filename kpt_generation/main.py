#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Triangulate 3D joints from two-view 2D keypoints using camera intrinsics and extrinsics.

Author: Kaixu Chen
Last Modified: July 28th, 2025
"""

import os
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")  # ✅ 防止在无GUI环境下崩溃
import matplotlib.pyplot as plt
import torch


def triangulate_joints(keypoints1, keypoints2, K, R, T):
    """
    使用两个视角的2D关节点和相机参数进行三角测量，恢复3D坐标。
    """
    if keypoints1.shape != keypoints2.shape or keypoints1.shape[1] != 2:
        raise ValueError(
            f"Keypoints shape mismatch or not (N, 2): {keypoints1.shape} vs {keypoints2.shape}"
        )

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # 投影矩阵1
    P2 = K @ np.hstack((R, T.reshape(3, 1)))  # 投影矩阵2

    pts1 = keypoints1.T.astype(np.float32)  # (2, N)
    pts2 = keypoints2.T.astype(np.float32)

    pts_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)  # (4, N)
    joints_3d = (pts_4d[:3, :] / pts_4d[3, :]).T  # (N, 3)

    return joints_3d


def visualize_3d_joints(
    joints_3d, save_path="triangulated_3d_joints.png", title="Triangulated 3D Joints"
):
    """
    可视化三维关键点
    """
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
    print(f"[INFO] 3D joints plot saved to: {save_path}")


def load_keypoints_from_pt(file_path):
    """
    加载.pt文件中的2D关键点
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[INFO] Loading keypoints from {file_path}")
    data = torch.load(file_path, map_location="cpu")

    try:
        keypoints = np.array(data["keypoint"]["keypoint"])
    except Exception as e:
        raise KeyError(f"Missing keypoint data in {file_path}: {e}")

    return keypoints


if __name__ == "__main__":
    # === 文件路径 ===
    left_path = "/workspace/data/pt/run_1/osmo_1.pt"
    right_path = "/workspace/data/pt/run_1/osmo_2.pt"

    # === 加载关键点 ===
    keypoints1 = load_keypoints_from_pt(left_path).squeeze(0)  # (T, N, 2)
    keypoints2 = load_keypoints_from_pt(right_path).squeeze(0)  # (T, N, 2)

    # === 相机内参矩阵 ===
    K = np.array(
        [
            [1.67514300e03, 0.00000000e00, 8.80968039e02],
            [0.00000000e00, 1.28634860e03, 1.02593966e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        dtype=np.float32,
    )

    # === 外参：视角2相对于视角1的平移和旋转 ===
    R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)

    T = np.array([0.1, 0.0, 0.0], dtype=np.float32)  # Camera2 相对于 Camera1 右移10cm

    assert (
        keypoints1.shape[0] == keypoints2.shape[0]
    ), "Keypoints must have the same number of frames"

    # === 三角测量 ===
    for k in range(keypoints1.shape[0]):

        joints_3d = triangulate_joints(keypoints1[k], keypoints2[k], K, R, T)
        print("[INFO] Triangulated 3D joints:\n", joints_3d)

    # === 可视化 ===
    visualize_3d_joints(joints_3d)
