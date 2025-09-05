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
import glob
import hydra

from triangulation.camera_position.camera_position import (
    estimate_camera_pose_from_sift_imgs,
    estimate_camera_pose_from_kpt,
    to_gray_cv_image,
)

from triangulation.camera_position.SIFT_kpt import estimate_camera_pose_hybrid

from triangulation.load import load_keypoints_from_d2_pt, load_keypoints_from_yolo_pt

from triangulation.reproject import reproject_and_visualize

from triangulation.postprocess import (
    post_triage_single,
    post_triage_sequence,
)

# vis
from triangulation.vis.visualization import (
    draw_and_save_keypoints_from_frame,
    visualize_3d_joints,
    visualize_3d_scene_interactive,
)
from triangulation.vis.SIFT import (
    visualize_SIFT_matches,
)
from triangulation.vis.camera import save_camera

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


# ---------- 主处理函数 ----------
def process_one_video(left_path, right_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # YOLO 关键点加载
    # left_kpts, left_kpts_score, left_vframes = load_keypoints_from_yolo_pt(left_path)
    # right_kpts, right_kpts_score, right_vframes = load_keypoints_from_yolo_pt(
    #     right_path
    # )
    # D2 关键点加载
    left_kpts, left_kpts_score, left_vframes = load_keypoints_from_d2_pt(left_path)
    right_kpts, right_kpts_score, right_vframes = load_keypoints_from_d2_pt(right_path)

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

        if l_frame is not None and r_frame is not None:
            draw_and_save_keypoints_from_frame(
                l_frame,
                l_kpt,
                os.path.join(output_path, "left_frame", f"{i:04d}.png"),
                color=(0, 255, 0),
            )
            draw_and_save_keypoints_from_frame(
                r_frame,
                r_kpt,
                os.path.join(output_path, "right_frame", f"{i:04d}.png"),
                color=(0, 0, 255),
            )

        # estimate camera position from imgs
        # R, T, pts1, pts2, mask_pose = estimate_camera_pose_from_sift_imgs(
        #     to_gray_cv_image(l_frame),
        #     to_gray_cv_image(r_frame),
        #     K,
        # )

        # estimate camera position from kpts
        # R, T, mask_pose = estimate_camera_pose_from_kpt(
        #     l_kpt,
        #     r_kpt,
        #     K,
        # )

        # K 同一台内参；若两台不同，请自行改成 K1/K2 并分别归一化
        R, T, info = estimate_camera_pose_hybrid(
            l_frame,
            r_frame,
            K,
            kpt1=l_kpt,
            kpt2=r_kpt,
            kpt_score1=left_kpts_score[i],
            kpt_score2=right_kpts_score[i],
            score_thresh=0.4,
            sift_ratio=0.75,
            magsac_thresh=1e-3,
            kpt_boost=3,
            refine=True,
        )
        print(info["n_sift"], info["n_kpt"], info["n_inlier"])

        # * COLMAP 
        # visualize_SIFT_matches(
        #     to_gray_cv_image(l_frame),
        #     to_gray_cv_image(r_frame),
        #     pts1,
        #     pts2,
        #     os.path.join(output_path, "SIFT"),
        #     i,
        # )

        # * 这里是没有过滤的3d pose
        joints_3d = triangulate_joints(l_kpt, r_kpt, K, R, T)

        def cam_to_world(R, T, Xc):
            R = R.T
            T = -R @ T.reshape(3)
            return (R @ Xc.T).T + T

        X_world = cam_to_world(R, T, joints_3d)

        # * 可视化3d关节
        visualize_3d_joints(
            X_world,
            R,
            T,
            os.path.join(output_path, "3d", f"frame_{i:04d}.png"),
            title=f"Frame {i} - 3D Joints",
            # y_up=True,
        )

        # * 可视化相机的位置
        save_camera(R, T, os.path.join(output_path, "camera"), f"camera_{i:04d}.png")

        # 保存交互式3D场景
        # html_path = os.path.join(output_path, f"scene_{i:04d}.html")
        # visualize_3d_scene_interactive(joints_3d, R, T, html_path)

        # 已有：X3d  (T,J,3)  ← 你的三角测量输出
        #       kptL,kptR (T,J,2)
        #       K_left,K_right, dist_left, dist_right, R, T

        # * 这里是过滤了的3d pose
        # FIXME: 这里过滤之后的kpt是有问题的，需要修复
        # X_clean, rep = post_triage_single(
        #     joints_3d,
        #     l_kpt,
        #     r_kpt,
        #     K,
        #     K,
        #     R,
        #     T,
        #     dist1=K_dist,
        #     dist2=K_dist,
        #     confL=left_kpts_score[i],
        #     confR=right_kpts_score[i],
        #     conf_thr=0.3,
        #     err_thresh_px=2.0,
        # )
        # print(rep)  # 看 rmse、pos_depth_ratio、kept_ratio

        # visualize_3d_joints(
        #     X_clean,
        #     R,
        #     T,
        #     os.path.join(output_path, f"frame_{i:04d}_filtered.png"),
        #     title=f"Frame {i} - Filtered 3D",
        # )

        res = reproject_and_visualize(
            img1=left_vframes[i],
            img2=right_vframes[i],
            X3=joints_3d,
            kptL=l_kpt,  # (J,2)
            kptR=r_kpt,  # (J,2)
            K1=K,
            dist1=K_dist,
            K2=K,
            dist2=K_dist,
            R=R,
            T=T,  # 注意：T要用“有尺度”的（基线已缩放）
            joint_names=None,  # 或者传 COCO 的关节名列表
            out_path=os.path.join(output_path, "reproj", f"{i:04d}.jpg"),
        )
    print(res["mean_err_L"], res["mean_err_R"])
    print("Saved to:", res["out_path"])


# ---------- 多人批量处理入口 ----------
@hydra.main(config_path="../configs", config_name="triangulation")
def main_pt(config):

    input_root = config.paths.input
    output_root = config.paths.output

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

    main_pt()
