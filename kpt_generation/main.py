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
import glob
import matplotlib

matplotlib.use("Agg")  # ✅ 禁用图形界面，使用非交互后端
import matplotlib.pyplot as plt

import torch


def triangulate_joints(keypoints1, keypoints2, K, R, T):
    if keypoints1.shape != keypoints2.shape or keypoints1.shape[1] != 2:
        raise ValueError(
            f"Keypoints shape mismatch: {keypoints1.shape} vs {keypoints2.shape}"
        )
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T.reshape(3, 1)))
    pts1 = keypoints1.T.astype(np.float32)
    pts2 = keypoints2.T.astype(np.float32)
    pts_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
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


def load_keypoints_from_pt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    print(f"[INFO] Loading: {file_path}")
    try:
        data = torch.load(file_path, map_location="cpu")
        keypoints = np.array(data["keypoint"]["keypoint"]).squeeze(0)  # (T, N, 2)
        if keypoints.ndim != 3 or keypoints.shape[2] != 2:
            raise ValueError(f"Invalid shape: {keypoints.shape}")
        return keypoints
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}: {e}")


def process_one_video(left_path, right_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    kpt1 = load_keypoints_from_pt(left_path)
    kpt2 = load_keypoints_from_pt(right_path)

    if kpt1.shape[0] != kpt2.shape[0]:
        raise ValueError(f"Frame mismatch: {kpt1.shape[0]} vs {kpt2.shape[0]}")

    # 相机内参
    K = np.array(
        [
            [1675.1430, 0.0000, 880.9680],
            [0.0000, 1286.3486, 1025.9397],
            [0.0000, 0.0000, 1.0000],
        ],
        dtype=np.float32,
    )

    # 外参：180度绕Y轴 + 右移10cm
    R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    T = np.array([1, 0.0, 0.0], dtype=np.float32)

    for i in range(kpt1.shape[0]):
        joints_3d = triangulate_joints(kpt1[i], kpt2[i], K, R, T)
        img_path = os.path.join(output_path, f"frame_{i:04d}.png")
        visualize_3d_joints(joints_3d, save_path=img_path)


def main(input_root, output_root):
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
    main(input_path, output_path)
