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
from confidence import (
    crossview_consistency_confidence,
    weakpersp_reproj_confidence,
)
from fuse import fuse_frame_3d, temporal_smooth_ema
from load_raw import load_raw
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


def main():
    paths = Path("/workspace/data/sam3d_body_results/person")

    for person in paths.iterdir():
        person_name = person.name
        print(f"Processing person: {person_name}")

        # Determine data format based on directory name
        if person_name.startswith("pro"):
            # pro format: left and right subdirectories
            sam_l_path = person / "left"
            sam_r_path = person / "right"
        elif person_name.startswith("run"):
            # run format: osmo_1 and osmo_2 files
            sam_l_path = person / "osmo_1_sam_3d_body_outputs.npz"
            sam_r_path = person / "osmo_2_sam_3d_body_outputs.npz"
        else:
            print(f"Unknown person format: {person_name}, skipping...")
            continue

        paths = {
            "sam_l": str(sam_l_path),
            "sam_r": str(sam_r_path),
        }

        all_frame_results = load_raw(paths)

        # 通过时间轴融合多帧
        fused_seq = []

        for frame_idx, frame_data in all_frame_results.items():
            print(f"Frame {frame_idx}:")

            # ---- Sam preds ----
            p2d_l_raw = frame_data["L_2D"]["pred"]
            p3d_l_raw = frame_data["L_3D"]["pred"]
            p2d_r_raw = frame_data["R_2D"]["pred"]
            p3d_r_raw = frame_data["R_3D"]["pred"]

            # 计算置信度
            p3d_l_conf1, err_px, uhat, params = weakpersp_reproj_confidence(
                p3d_l_raw,
                p2d_l_raw,
                sigma_px=12.0,  # (70,3)  # (70,2)
            )
            print(p3d_l_conf1.shape, p3d_l_conf1.min(), p3d_l_conf1.max())

            p3d_r_conf1, err_px, uhat, params = weakpersp_reproj_confidence(
                p3d_r_raw,
                p2d_r_raw,
                sigma_px=12.0,  # (70,3)  # (70,2)
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

        # ---- temporal smoothing ----
        smooth_seq = temporal_smooth_ema(fused_seq, TARGET_IDS, alpha=0.7)

        # save smoothed results
        output_path = (
            Path("/workspace/data/fused_smoothed_results") / f"{person_name}.npy"
        )
        saved_path = save_smoothed_results(smooth_seq, TARGET_IDS, output_path)
        print(f"Smoothed results saved to: {saved_path}")


if __name__ == "__main__":
    main()
