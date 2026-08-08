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
import glob
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

from triangulation.view_process.two_view import process_two_video
from triangulation.view_process.single_view import process_single_video

from triangulation.load import (
    load_kpt_and_bbox_from_d2_pt,
    load_keypoints_from_yolo_pt,
)

from triangulation.vis.frame_visualization import (
    draw_and_save_keypoints_from_frame,
)

from triangulation.save import save_3d_joints
from triangulation.triangulate import process_triangulate

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


def process(
    left_path: Path,
    right_path: Path,
    out_dir: Path,
    baseline_m: int,
    vis_options: DictConfig,
):
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
    num = 10
    # left_kpts = left_kpts[:num]
    # left_vframes = left_vframes[:num]
    # right_kpts = right_kpts[:num]
    # right_vframes = right_vframes[:num]

    # * draw keypoints on frames and save
    if vis_options.keypoint_vis:

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
    left_data = process_single_video(
        K=K,
        single_kpts=left_kpts,
        single_vframes=left_vframes,
        single_bbox=left_bboxes_xyxy,
        output_path=os.path.join(out_dir, "single_view/left"),
        baseline_m=baseline_m,
    )

    right_data = process_single_video(
        K=K,
        single_kpts=right_kpts,
        single_vframes=right_vframes,
        single_bbox=right_bboxes_xyxy,
        output_path=os.path.join(out_dir, "single_view/right"),
        baseline_m=baseline_m,
    )

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

        joints_3d_al = process_triangulate(
            left_kpts=left_kpts,
            right_kpts=right_kpts,
            left_vframes=left_vframes,
            right_vframes=right_vframes,
            K=K,
            R=r_list,
            T=t_list,
            output_path=_out_dir,
        )

        # * save 3d joints
        _joint_3d_data_out_dir = os.path.join(out_dir, "joints_3d", method)
        for i, joints_3d, r, t in zip(frame_num, joints_3d_al, r_list, t_list):

            save_3d_joints(
                joints_3d,
                save_dir=os.path.join(_joint_3d_data_out_dir),
                frame_idx=i,
                r=r,
                t=t,
                video_path={"left": left_path, "right": right_path},
                fmt="json",
            )


# ---------- 多人批量处理入口 ----------
@hydra.main(config_path="../configs", config_name="triangulation")
def main_pt(config):

    input_root = Path(config.paths.pt_path)
    output_root = Path(config.paths.log_path)
    baseline_m = config.baseline_m

    subjects = sorted(input_root.glob("*/"))
    if not subjects:
        raise FileNotFoundError(f"No folders found in: {input_root}")
    logger.info(f"Found {len(subjects)} subjects in {input_root}")

    for person_dir in subjects:
        person_name = person_dir.name
        logger.info(f"Processing: {person_name}")

        left_path = person_dir / "osmo_1.pt"
        right_path = person_dir / "osmo_2.pt"

        out_dir = output_root / person_name

        process(
            left_path,
            right_path,
            out_dir,
            baseline_m=baseline_m,
            vis_options=config.visualize,
        )


if __name__ == "__main__":

    main_pt()
