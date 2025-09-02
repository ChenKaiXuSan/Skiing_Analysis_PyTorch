#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Triangulate 3D joints from two-view 2D keypoints using either video frames or pre-extracted keypoints.
Supports modular triangulation, pose estimation, and interactive 3D visualization.

Author: Kaixu Chen
Last Modified: August 4th, 2025
"""

import os
import numpy as np
import cv2
import torch
from torchvision.io import read_video
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py

from triangulation.camera_position import (
    estimate_camera_pose_from_sift_imgs,
    estimate_camera_pose_from_kpt,
    to_gray_cv_image,
    visualize_SIFT_matches,
)

from triangulation.postprocess import (
    post_triage_single,
    post_triage_sequence,
    plot_skeleton_3d,
)

from triangulation.reproject import reproject_and_visualize


# ---------- 可视化工具 ----------
def draw_and_save_keypoints_from_frame(
    frame,
    keypoints,
    save_path,
    color=(0, 255, 0),
    radius=4,
    thickness=-1,
    with_index=True,
):
    img = frame.numpy() if isinstance(frame, torch.Tensor) else frame.copy()
    for i, (x, y) in enumerate(keypoints):
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        if with_index:
            cv2.putText(
                img,
                str(i),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"[INFO] Saved image with keypoints to: {save_path}")


def draw_camera(ax, R, T, scale=0.1, label="Cam"):
    origin = T.reshape(3)
    x_axis = R @ np.array([1, 0, 0]) * scale + origin
    y_axis = R @ np.array([0, 1, 0]) * scale + origin
    z_axis = R @ np.array([0, 0, 1]) * scale + origin
    view_dir = R @ np.array([0, 0, -1]) * scale * 1.5 + origin  # 摄像头朝向（-Z轴）

    ax.plot(
        [origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], c="r"
    )
    ax.plot(
        [origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], c="g"
    )
    ax.plot(
        [origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], c="b"
    )

    # 视线方向箭头（黑色）
    ax.plot(
        [origin[0], view_dir[0]],
        [origin[1], view_dir[1]],
        [origin[2], view_dir[2]],
        c="k",
        linestyle="--",
    )

    # 相机标签
    ax.text(*origin, label, color="black")


# ---------- 相机参数 ----------
# K = np.array(
#     [[1675.1430, 0.0, 880.9680], [0.0, 1286.3486, 1025.9397], [0.0, 0.0, 1.0]],
#     dtype=np.float32,
# )

# * 这个是用录得视频推测的相机内参
K = np.array(
    [
        [1.10308405e03, 0.00000000e00, 9.47946068e02],
        [0.00000000e00, 1.10601861e03, 5.31242592e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

K_dist = np.array([0.17697328, -0.45675065, -0.0026601, -0.00330938, 0.35538705])


# ---------- 三角测量 ----------
def triangulate_joints(keypoints1, keypoints2, K, R, T):
    if keypoints1.shape != keypoints2.shape or keypoints1.shape[1] != 2:
        raise ValueError(
            f"Keypoints shape mismatch: {keypoints1.shape} vs {keypoints2.shape}"
        )
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T.reshape(3, 1)))
    pts_4d = cv2.triangulatePoints(P1, P2, keypoints1.T, keypoints2.T)
    return (pts_4d[:3, :] / pts_4d[3, :]).T


def visualize_3d_joints(joints_3d, R, T, save_path, title="Triangulated 3D Joints"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_camera(ax, np.eye(3), np.zeros(3), label="Cam1")
    draw_camera(ax, R, T, label="Cam2")

    ax.scatter(joints_3d[:, 0], joints_3d[:, 2], joints_3d[:, 1], c="blue", s=30)
    for i, (x, y, z) in enumerate(joints_3d):
        ax.text(x, z, y, str(i), size=8)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


def visualize_3d_scene_interactive(joints_3d, R, T, save_path):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=joints_3d[:, 0],
            y=joints_3d[:, 1],
            z=joints_3d[:, 2],
            mode="markers+text",
            text=[str(i) for i in range(len(joints_3d))],
            marker=dict(size=4, color="blue"),
        )
    )

    def get_camera_lines(R, T, label):
        scale = 0.1
        lines = []
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        colors = ["red", "green", "blue"]
        origin = T.reshape(3)
        for axis, color in zip(axes, colors):
            end = R @ axis * scale + origin
            lines.append(
                go.Scatter3d(
                    x=[origin[0], end[0]],
                    y=[origin[1], end[1]],
                    z=[origin[2], end[2]],
                    mode="lines",
                    line=dict(color=color, width=4),
                )
            )
        view_dir = R @ np.array([0, 0, -1]) * scale * 1.5 + origin
        lines.append(
            go.Scatter3d(
                x=[origin[0], view_dir[0]],
                y=[origin[1], view_dir[1]],
                z=[origin[2], view_dir[2]],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name=f"{label}_view",
            )
        )
        return lines

    for trace in get_camera_lines(np.eye(3), np.zeros(3), "Cam1"):
        fig.add_trace(trace)
    for trace in get_camera_lines(R, T, "Cam2"):
        fig.add_trace(trace)

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="Interactive 3D Scene",
        margin=dict(l=0, r=0, b=0, t=30),
    )
    py.plot(fig, filename=save_path, auto_open=False)
    print(f"[INFO] Saved interactive HTML to: {save_path}")


# ---------- 加载关键点 ----------
def load_keypoints_from_pt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    print(f"[INFO] Loading: {file_path}")
    data = torch.load(file_path, map_location="cpu")
    video_path = data.get("video_path", None)
    vframes = (
        read_video(video_path, pts_unit="sec", output_format="THWC")[0]
        if video_path
        else None
    )
    keypoints = np.array(data["keypoint"]["keypoint"]).squeeze(0)
    keypoints_score = np.array(data["keypoint"]["keypoint_score"]).squeeze(0)
    if keypoints.ndim != 3 or keypoints.shape[2] != 2:
        raise ValueError(f"Invalid shape: {keypoints.shape}")
    if vframes is not None:
        keypoints[:, :, 0] *= vframes.shape[2]
        keypoints[:, :, 1] *= vframes.shape[1]
    return keypoints, keypoints_score, vframes


# ---------- 主处理函数 ----------
def process_one_video(left_path, right_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    left_kpts, left_kpts_score, left_vframes = load_keypoints_from_pt(left_path)
    right_kpts, right_kpts_score, right_vframes = load_keypoints_from_pt(right_path)

    if left_kpts.shape[0] != right_kpts.shape[0]:
        raise ValueError(
            f"Frame mismatch: {left_kpts.shape[0]} vs {right_kpts.shape[0]}"
        )

    for i in range(min(6, left_kpts.shape[0])):
        l_kpt, r_kpt = left_kpts[i], right_kpts[i]

        # drop the 0 value keypoints
        assert (
            l_kpt.shape == r_kpt.shape
        ), f"Keypoints shape mismatch: {l_kpt.shape} vs {r_kpt.shape}"

        l_frame = left_vframes[i] if left_vframes is not None else None
        r_frame = right_vframes[i] if right_vframes is not None else None

        if l_frame is not None and r_frame is not None:
            draw_and_save_keypoints_from_frame(
                l_frame,
                l_kpt,
                os.path.join(output_path, f"left_frame_{i:04d}.png"),
                color=(0, 255, 0),
            )
            draw_and_save_keypoints_from_frame(
                r_frame,
                r_kpt,
                os.path.join(output_path, f"right_frame_{i:04d}.png"),
                color=(0, 0, 255),
            )

        # estimate camera position from imgs
        R, T, pts1, pts2, mask_pose = estimate_camera_pose_from_sift_imgs(
            to_gray_cv_image(l_frame),
            to_gray_cv_image(r_frame),
            K,
        )

        # estimate camera position from kpts
        R, T, mask_pose = estimate_camera_pose_from_kpt(
            l_kpt,
            r_kpt,
            K,
        )

        T=-T
        T = T / np.linalg.norm(T) * 10  # baseline_m 是两相机真实物理距离


        visualize_SIFT_matches(
            to_gray_cv_image(l_frame),
            to_gray_cv_image(r_frame),
            pts1,
            pts2,
            os.path.join(output_path, f"sift_matches_{i:04d}.png"),
        )

        # * 这里是没有过滤的3d pose
        joints_3d = triangulate_joints(l_kpt, r_kpt, K, R, T)
        visualize_3d_joints(
            joints_3d,
            R,
            T,
            os.path.join(output_path, f"frame_{i:04d}.png"),
            title=f"Frame {i} - 3D Joints",
        )
        # 保存交互式3D场景
        html_path = os.path.join(output_path, f"scene_{i:04d}.html")
        visualize_3d_scene_interactive(joints_3d, R, T, html_path)

        # 已有：X3d  (T,J,3)  ← 你的三角测量输出
        #       kptL,kptR (T,J,2)
        #       K_left,K_right, dist_left, dist_right, R, T

        # * 这里是过滤了的3d pose
        # FIXME: 这里过滤之后的kpt是有问题的，需要修复
        X_clean, rep = post_triage_single(
            joints_3d,
            l_kpt,
            r_kpt,
            K,
            K,
            R,
            T,
            dist1=K_dist,
            dist2=K_dist,
            confL=left_kpts_score[i],
            confR=right_kpts_score[i],
            conf_thr=0.3,
            err_thresh_px=2.0,
        )
        print(rep)  # 看 rmse、pos_depth_ratio、kept_ratio

        visualize_3d_joints(
            X_clean,
            R,
            T,
            os.path.join(output_path, f"frame_{i:04d}_filtered.png"),
            title=f"Frame {i} - Filtered 3D",
        )

        res = reproject_and_visualize(
            img1=left_vframes[i],
            img2=right_vframes[i],
            X3=joints_3d[i],
            kptL=l_kpt,  # (J,2)
            kptR=r_kpt,  # (J,2)
            K1=K,
            dist1=K_dist,
            K2=K,
            dist2=K_dist,
            R=R,
            T=T,  # 注意：T要用“有尺度”的（基线已缩放）
            joint_names=None,  # 或者传 COCO 的关节名列表
            out_path=os.path.join(output_path, f"reproj_{i:04d}.jpg"),
        )
    print(res["mean_err_L"], res["mean_err_R"])
    print("Saved to:", res["out_path"])


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

        process_one_video(left, right, out_dir)


if __name__ == "__main__":
    input_path = "/workspace/data/pt"
    output_path = "/workspace/code/logs/triangulated_3d_joints"
    main_pt(input_path, output_path)
