#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Triangulate 3D joints from two-view 2D keypoints using either video frames or pre-extracted keypoints.
Supports modular triangulation, pose estimation, and interactive 3D visualization.

Author: Kaixu Chen
Last Modified: August 4th, 2025
"""

import logging
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


# videopose 3d
from .common.arguments import parse_args

# fuse two view 3d poses
from .fuse.fuse import fuse_pose_no_extrinsics_h36m
from .fuse.fuse_eval import eval_fused_pose
from .run import run_video_pose_3d
from .save import save_3d_joints
from .visualization import save_coco3d_gif_multi_view

logger = logging.getLogger(__name__)


def process_video_3d(
    config: DictConfig,
    left_path: Path,
    right_path: Path,
    out_dir: Path,
    npy_dir: Path,
):
    # FIXME: 这里只是为了合并代码，改好了之后就删除掉
    args = parse_args()
    print(args)

    # * run videopose3d for left and right view
    left_kpt_3d, left_depth = run_video_pose_3d(
        args=args,
        config=config,
        pt_path=left_path,
        out_dir=out_dir / "videopose3d" / "left",
    )
    logger.info(f"Saved VideoPose3D results to: {out_dir / 'videopose3d' / 'left'}")

    right_kpt_3d, right_depth = run_video_pose_3d(
        args=args,
        config=config,
        pt_path=right_path,
        out_dir=out_dir / "videopose3d" / "right",
    )
    logger.info(f"Saved VideoPose3D results to: {out_dir / 'videopose3d' / 'right'}")

    # * fuse two view 3d poses
    all_fused = []
    for f in range(left_kpt_3d.shape[0]):
        left_kpt_3d_f = left_kpt_3d[f]
        right_kpt_3d_f = right_kpt_3d[f]

        fused_3d, diag = fuse_pose_no_extrinsics_h36m(
            left_3d=left_kpt_3d_f,
            right_3d=right_kpt_3d_f,
            tau=0.06,
            allow_scale=False,
            mirror_right_x=False,
        )
        all_fused.append(fused_3d)

        if diag is not None:
            print(
                "mean gain:", diag["mean_gain"], "bad frames:", diag["bad_frames"][:10]
            )

    save_coco3d_gif_multi_view(all_fused, out_dir / "fused", fps=30, swap_yz=False)
    logger.info(f"Saved fused GIF to: {out_dir / 'fused_pose.gif'}")

    # * save fused 3d joints as npz
    save_3d_joints(
        fused_joints_3d=np.array(all_fused),
        left_joints_3d=left_kpt_3d,
        right_joints_3d=right_kpt_3d,
        save_dir=Path(str(npy_dir) + "_fused_keypoints.npy"),
    )

    # * evaluate fused pose
    metrics = eval_fused_pose(left_kpt_3d, right_kpt_3d, np.array(all_fused))
    logger.info("Fused Pose Evaluation Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k:25s}: {v:.4f}")

    # * write metrics to a text file
    with open(out_dir / "fused_metrics.txt", "w") as f:
        f.write("Fused Pose Evaluation Metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k:25s}: {v:.4f}\n")


# ---------- 多人批量处理入口 ----------
@hydra.main(config_path="../configs", config_name="videopose3d")
def main_pt(config):
    input_root = Path(config.paths.pt_path)
    output_root = Path(config.paths.log_path)
    npy_root = Path(config.paths.npy_path)

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
        npy_dir = npy_root / person_name

        process_video_3d(
            config=config,
            left_path=left_path,
            right_path=right_path,
            out_dir=out_dir,
            npy_dir=npy_dir,
        )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main_pt()
