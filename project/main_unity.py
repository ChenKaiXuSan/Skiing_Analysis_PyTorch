#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/main.py
Project: /workspace/code/project
Created Date: Friday January 30th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 30th 2026 3:58:30 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
from pathlib import Path
import numpy as np

from project.load_unity import load
from fuse import fuse_frame_3d, temporal_smooth_ema
from confidence import (
    weakpersp_reproj_confidence,
    crossview_consistency_confidence,
)
from unity_data_compare import calculate_mpjpe
from save import save_smoothed_results

# --- 設定 ---
UNITY_MHR70_MAPPING = {
    1: "Bone_Eye_L",
    2: "Bone_Eye_R",
    5: "Upperarm_L",
    6: "Upperarm_R",
    7: "lowerarm_l",
    8: "lowerarm_r",
    9: "Thigh_L",
    10: "Thigh_R",
    11: "calf_l",
    12: "calf_r",
    13: "Foot_L",
    14: "Foot_R",
    41: "Hand_R",
    62: "Hand_L",
    69: "neck_01",
}
TARGET_IDS = list(UNITY_MHR70_MAPPING.keys())


# --- メイン実行 ---


def main():
    # パス (適宜書き換えてください)
    paths = {
        "sam_l": "/workspace/data/sam3d_body_results/unity/male/left_sam_3d_body_outputs.npz",
        "sam_r": "/workspace/data/sam3d_body_results/unity/male/right_sam_3d_body_outputs.npz",
        "gt_2d_l": "/workspace/data/unity_data/RecordingsPose/cam_left camera/male_kpt2d_left camera_trimmed.jsonl",
        "gt_2d_r": "/workspace/data/unity_data/RecordingsPose/cam_right camera/male_kpt2d_right camera_trimmed.jsonl",
        "gt_3d": "/workspace/data/unity_data/RecordingsPose/male_pose3d_trimmed.jsonl",
    }

    all_frame_results = load(paths)

    # 通过时间轴融合多帧
    fused_seq = []
    all_frame_3d_kpt_errors = []

    for frame_idx, frame_data in all_frame_results.items():
        print(f"Frame {frame_idx}:")

        # ---- GT dicts (2D) ----
        g2d_l = frame_data["L_2D"]["gt"]
        g2d_r = frame_data["R_2D"]["gt"]
        g3d_l = frame_data["L_3D"]["gt"]
        g3d_r = frame_data["R_3D"]["gt"]

        # ---- Sam preds ----
        p2d_l_raw = frame_data["L_2D"]["pred"]
        p3d_l_raw = frame_data["L_3D"]["pred"]
        p2d_r_raw = frame_data["R_2D"]["pred"]
        p3d_r_raw = frame_data["R_3D"]["pred"]

        # 计算置信度
        p3d_l_conf1, err_px, uhat, params = weakpersp_reproj_confidence(
            p3d_l_raw, p2d_l_raw, sigma_px=12.0  # (70,3)  # (70,2)
        )
        print(p3d_l_conf1.shape, p3d_l_conf1.min(), p3d_l_conf1.max())

        p3d_r_conf1, err_px, uhat, params = weakpersp_reproj_confidence(
            p3d_r_raw, p2d_r_raw, sigma_px=12.0  # (70,3)  # (70,2)
        )
        print(p3d_r_conf1.shape, p3d_r_conf1.min(), p3d_r_conf1.max())

        # 你需要把下面这些 index 换成你 MHR70 / sam3d 的真实 index
        IDX_PELVIS = 14
        IDX_LHIP = 11
        IDX_RHIP = 12
        IDX_LSHO = 5
        IDX_RSHO = 6

        conf2, dist, Xlc, Xrc, info = crossview_consistency_confidence(
            p3d_l_raw,
            p3d_r_raw,
            root_idx=IDX_PELVIS,
            left_hip_idx=IDX_LHIP,
            right_hip_idx=IDX_RHIP,
            left_shoulder_idx=IDX_LSHO,
            right_shoulder_idx=IDX_RSHO,
            sigma_3d=0.08,
            scale_mode="hip",
        )
        print(conf2.shape, conf2.min(), conf2.max())

        q_l_data = np.sqrt(p3d_l_conf1 * conf2)  # 简单又稳
        q_r_data = np.sqrt(p3d_r_conf1 * conf2)

        # ---- frame-wise fuse 3D ----
        q_l = q_l_data
        q_r = q_r_data

        fused_3d = fuse_frame_3d(p3d_l_raw, p3d_r_raw, q_l, q_r, TARGET_IDS)
        print(f"  Fused 3D Keypoints: {fused_3d}")
        fused_seq.append(fused_3d)

        # 计算误差
        mpjpe_3d = calculate_mpjpe(
            fused_3d, g3d_l
        )  # 简单起见，直接用平均2D作为3D的GT投影
        print(f"  Fused 3D MPJPE (w.r.t. avg 3D GT): {mpjpe_3d:.2f} px")
        all_frame_3d_kpt_errors.append(mpjpe_3d)

    # 输出temporal smoothing之前的平均3d误差
    mean_mpjpe_3d = np.mean(all_frame_3d_kpt_errors)
    print(f"Average Fused 3D MPJPE after fuse two view: {mean_mpjpe_3d:.2f} px")

    # ---- temporal smoothing ----
    smooth_seq = temporal_smooth_ema(fused_seq, TARGET_IDS, alpha=0.7)

    # 计算总误差
    all_frame_3d_kpt_errors = {}
    for i, fused_3d in enumerate(smooth_seq):

        mpjpe_3d = calculate_mpjpe(fused_3d, g3d_l)
        all_frame_3d_kpt_errors[i] = {"mpjpe": mpjpe_3d}

    # 输出temporal smoothing之后的平均3d误差
    mean_mpjpe_3d = np.mean([v["mpjpe"] for v in all_frame_3d_kpt_errors.values()])
    print(f"Average Fused 3D MPJPE after smoothing: {mean_mpjpe_3d:.2f} px")

    # save smoothed results
    output_path = Path("/workspace/data/fused_smoothed_results") / "unity_smoothed_3d.npy"
    saved_path = save_smoothed_results(smooth_seq, TARGET_IDS, output_path)
    print(f"Smoothed results saved to: {saved_path}")

if __name__ == "__main__":
    main()
