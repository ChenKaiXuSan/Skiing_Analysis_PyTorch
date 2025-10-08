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

# from triangulation.estimate_camera_position import (
#     process_two_video,
#     process_one_video,
# )

from triangulation.two_view import process_two_video

from triangulation.load import (
    load_kpt_and_bbox_from_d2_pt,
    load_keypoints_from_yolo_pt,
)

from triangulation.reproject import reproject_and_visualize

from triangulation.postprocess import (
    post_triage_single,
    post_triage_sequence,
)

# vis
from triangulation.vis.pose_visualization import (
    visualize_3d_joints,
)

from triangulation.vis.frame_visualization import (
    draw_and_save_keypoints_from_frame,
)

# ---------- 相机参数 ----------
# K = np.array(
#     [[1675.1430, 0.0, 880.9680], [0.0, 1286.3486, 1025.9397], [0.0, 0.0, 1.0]],
#     dtype=np.float32,
# )

# * 这个是用录得视频推测的相机内参
# K = np.array(
#     [
#         [1.10308405e03, 0.00000000e00, 9.47946068e02],
#         [0.00000000e00, 1.10601861e03, 5.31242592e02],
#         [0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )

K = np.array(
    [
        1116.9289548941917,
        0.0,
        955.77175993563799,
        0.0,
        1117.3341496962166,
        538.91061167202145,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)

# K_dist = np.array([0.17697328, -0.45675065, -0.0026601, -0.00330938, 0.35538705])
K_dist = np.array(
    [
        -1.1940477842823853,
        -15.440461757486913,
        0.00013163161053023783,
        0.00019082529328353381,
        98.843073622415901,
        -1.3588290520381034,
        -14.555841222727574,
        96.219667412855202,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


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


def process_triangulate(
    left_kpts, right_kpts, left_vframes, right_vframes, K, R, T, output_path
):

    for l_kpt, r_kpt, l_frame, r_frame, r, t, i in zip(
        left_kpts, right_kpts, left_vframes, right_vframes, R, T, range(len(left_kpts))
    ):
        W, H = l_frame.shape[1], l_frame.shape[0]

        # * 这里是没有过滤的3d pose
        joints_3d = triangulate_joints(l_kpt, r_kpt, K, r, t)

        # * 可视化3d关节
        visualize_3d_joints(
            joints_3d=joints_3d,
            R=r,
            T=t,
            K=K,
            image_size=(W, H),
            save_path=os.path.join(output_path, f"frame_{i:04d}.png"),
            title=f"Frame {i} - 3D Joints",
            y_up=True,
        )

        res = reproject_and_visualize(
            img1=l_frame,
            img2=r_frame,
            X3=joints_3d,
            kptL=l_kpt,  # (J,2)
            kptR=r_kpt,  # (J,2)
            K1=K,
            dist1=K_dist,
            K2=K,
            dist2=K_dist,
            R=r,
            T=t,  # 注意：T要用“有尺度”的（基线已缩放）
            joint_names=None,  # 或者传 COCO 的关节名列表
            out_path=os.path.join(output_path, "reproj", f"{i:04d}.jpg"),
        )
        print(res["mean_err_L"], res["mean_err_R"])
        print("Saved to:", res["out_path"])


def process(left_path, right_path, out_dir, baseline_m):
    # YOLO 关键点加载
    # left_kpts, left_kpts_score, left_vframes = load_keypoints_from_yolo_pt(left_path)
    # right_kpts, right_kpts_score, right_vframes = load_keypoints_from_yolo_pt(
    #     right_path
    # )
    # D2 关键点加载
    left_kpts, left_kpts_score, left_bboxes_xyxy, left_bboxes_scores, left_vframes = (
        load_kpt_and_bbox_from_d2_pt(left_path)
    )

    (
        right_kpts,
        right_kpts_score,
        right_bboxes_xyxy,
        right_bboxes_scores,
        right_vframes,
    ) = load_kpt_and_bbox_from_d2_pt(right_path)

    # ! 为了测试截断
    # num = 30
    # left_kpts = left_kpts[:num]
    # left_vframes = left_vframes[:num]
    # right_kpts = right_kpts[:num]
    # right_vframes = right_vframes[:num]

    # * draw keypoints on frames and save
    for i in range(left_kpts.shape[0]):
        l_kpt, r_kpt = left_kpts[i], right_kpts[i]

        # drop the 0 value keypoints
        assert (
            l_kpt.shape == r_kpt.shape
        ), f"Keypoints shape mismatch: {l_kpt.shape} vs {r_kpt.shape}"

        l_frame = left_vframes[i] if left_vframes is not None else None
        r_frame = right_vframes[i] if right_vframes is not None else None

        draw_and_save_keypoints_from_frame(
            l_frame,
            l_kpt,
            os.path.join(out_dir, "keypoint_vis/left_frame", f"{i:04d}.png"),
            color=(0, 255, 0),
        )
        draw_and_save_keypoints_from_frame(
            r_frame,
            r_kpt,
            os.path.join(out_dir, "keypoint_vis/right_frame", f"{i:04d}.png"),
            color=(0, 0, 255),
        )

    # * process single view post-triage
    # TODO: 如果单个视点可行的话，需要从单个视点来计算两个相机的R，T
    # process_one_video(
    #     K, left_kpts, left_vframes, os.path.join(out_dir, "single_view/left"), baseline_m=baseline_m,
    # )

    # process_one_video(
    #     K, right_kpts, right_vframes, os.path.join(out_dir, "single_view/right"), baseline_m=baseline_m,
    # )

    # * 尝试使用bbox区域来提取特征


    # * process two view triangulation
    data = process_two_video(
        K=K,
        left_kpts=left_kpts,
        left_vframes=left_vframes,
        left_bbox=left_bboxes_xyxy,
        right_kpts=right_kpts,
        right_vframes=right_vframes,
        right_bbox=right_bboxes_xyxy,
        output_path=os.path.join(out_dir, "two_view"),
        baseline_m=baseline_m,
    )

    # * process two view post-triage
    for method, v in data.items():
        _out_dir = os.path.join(out_dir, "3d", method)
        r_list = v["R"]
        t_list = v["t"]
        frame_num = v["frame"]

        process_triangulate(
            left_kpts=left_kpts,
            right_kpts=right_kpts,
            left_vframes=left_vframes,
            right_vframes=right_vframes,
            K=K,
            R=r_list,
            T=t_list,
            output_path=_out_dir,
        )


# ---------- 多人批量处理入口 ----------
@hydra.main(config_path="../configs", config_name="triangulation")
def main_pt(config):

    input_root = config.paths.input
    output_root = config.paths.output
    baseline_m = config.baseline_m

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

        process(left, right, out_dir, baseline_m=baseline_m)


if __name__ == "__main__":

    main_pt()
