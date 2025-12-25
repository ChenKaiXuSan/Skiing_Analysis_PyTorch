#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/front_side/run.py
Project: /workspace/code/front_side
Created Date: Thursday December 25th 2025
Author: Kaixu Chen
-----
Comment:
侧面用左右视角进行刚体变换
前面用鸟览图进行融合

最后合成一个完整的运动轨迹视频
Have a good code time :)
-----
Last Modified: Thursday December 25th 2025 4:20:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import gc
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .front.run import process_front_frame
from .side.run import process_side_frame
from .load import load_sam3_results, load_sam_3d_body_results


def process_one_person(
    left_sam3d_body_path: Path,
    right_sam3d_body_path: Path,
    front_sam3_results: Path,
    output_dir: Path,
) -> None:
    """
    Process one person with multi-view bundle adjustment.
    """

    left_sam3d_body_res = load_sam_3d_body_results(left_sam3d_body_path.as_posix())
    right_sam3d_body_res = load_sam_3d_body_results(right_sam3d_body_path.as_posix())

    front_sam3_res = load_sam3_results(front_sam3_results.as_posix())

    for frame_idx in tqdm(range(len(left_sam3d_body_res)), desc="Processing frames"):
        # if frame_idx > 60:
        #     break

        left_frame, right_frame, kpts_world, R_RL, t_RL = process_side_frame(
            left_sam3d_body_res=left_sam3d_body_res,
            right_sam3d_body_res=right_sam3d_body_res,
            frame_idx=frame_idx,
            out_root=output_dir / "side",
        )

        # process front view
        image = front_sam3_res[frame_idx]["frame"]  # np.ndarray, (H,W,3)
        bbox_xyxy = front_sam3_res[frame_idx]["out_boxes_xywh"]  # np.ndarray, (4,) xyxy
        obj_ids = front_sam3_res[frame_idx].get("out_obj_ids", None)
        probs = front_sam3_res[frame_idx].get("out_probs", None)
        binary_masks = front_sam3_res[frame_idx].get("out_binary_masks")
        # foot point in bev pixels
        foot_xy_px = process_front_frame(
            image, bbox_xyxy, output_dir=output_dir / "front", frame_idx=frame_idx
        )

        # merge side and front results here
        merge(
            kpts_world=kpts_world,
            foot_xy_px=foot_xy_px,
            output_dir=output_dir,
            frame_idx=frame_idx,
        )
    # 清空内存
    gc.collect()


def merge(
    kpts_world: np.ndarray, foot_xy_px: np.ndarray, output_dir: Path, frame_idx: int
) -> None:
    # merge side and front results

    pass
