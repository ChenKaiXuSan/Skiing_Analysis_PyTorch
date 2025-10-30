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
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
# videopose 3d
from VideoPose3D.run import run_video_pose_3d
from VideoPose3D.common.arguments import parse_args


from triangulation.vis.frame_visualization import (
    draw_and_save_keypoints_from_frame,
)

from triangulation.save import save_3d_joints
from triangulation.triangulate import process_triangulate


def process_video_3d(
    config: DictConfig,
    left_path: Path,
    right_path: Path,
    out_dir: Path,
    vis_options: DictConfig,
):
    # FIXME: 这里只是为了合并代码，改好了之后就删除掉
    args = parse_args()
    print(args)

    # process single video
    run_video_pose_3d(
        args=args,
        config=config,
        pt_path=left_path,
        out_dir=out_dir / "videopose3d" / "left",
    )
    logger.info(f"Saved VideoPose3D results to: {out_dir / 'videopose3d' / 'left'}")

    run_video_pose_3d(
        args=args,
        config=config,
        pt_path=right_path,
        out_dir=out_dir / "videopose3d" / "right",
    )
    logger.info(f"Saved VideoPose3D results to: {out_dir / 'videopose3d' / 'right'}")

# ---------- 多人批量处理入口 ----------
@hydra.main(config_path="../configs", config_name="videopose3d")
def main_pt(config):

    input_root = Path(config.paths.pt_path)
    output_root = Path(config.paths.log_path)

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

        process_video_3d(
            config=config,
            left_path=left_path,
            right_path=right_path,
            out_dir=out_dir,
            vis_options=config.visualize,
        )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main_pt()
