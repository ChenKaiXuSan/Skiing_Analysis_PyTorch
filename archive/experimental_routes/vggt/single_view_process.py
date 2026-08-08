#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/multi_view_process copy.py
Project: /workspace/code/vggt
Created Date: Thursday December 11th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday December 11th 2025 5:40:57 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vggt_video_infer.py
从单个视频抽帧并执行 VGGT 推理，可作为函数调用。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from vggt.load import load_info
from vggt.reproject import reproject_and_visualize
from vggt.save import save_camera_info
from vggt.triangulate import triangulate_one_frame
from vggt.vggt.infer import CameraHead
from vggt.vis.pose_visualization import save_stereo_pose_frame, visualize_3d_joints

from .vis.skeleton_visualizer import SkeletonVisualizer

logger = logging.getLogger(__name__)


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


# --------------------------------------------------------------------------- #
# Processing functions
# --------------------------------------------------------------------------- #
def process_single_view_video(
    video_path: Path,
    pt_path: Path,
    out_root: Path,
    inference_output_path: Path,
    cfg: DictConfig,
) -> Optional[Path]:
    """
    处理双目视频。返回输出目录；失败返回 None。

    目前示例代码仍然只对 left_video_path 进行 VGGT 推理，
    主要提供一个“成对管理 + 输出目录区分”的框架。
    后续如果需要真正 multi-view 融合，可以在这里扩展。
    """
    subject = video_path.parent.name or "default"

    out_dir = out_root / "single_view" / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Run-SV] {video_path} → {out_dir} | ")

    # * load info from pt and video
    kpts, left_kpt_scores, left_bboxes, left_bboxes_scores, frames = load_info(
        video_file_path=video_path.as_posix(),
        pt_file_path=pt_path.as_posix(),
        assume_normalized=False,
    )

    all_frame_camera_extrinsics = []
    all_frame_camera_intrinsics = []
    all_frame_R = []
    all_frame_t = []
    all_frame_C = []

    camera_head = CameraHead(cfg, out_dir / "vggt_infer")
    visualizer = SkeletonVisualizer()

    inference_idx = [] 
    inference_imgs = []

    for idx in tqdm(range(0, len(frames), 30), desc="Processing frames"):
        # if idx > 10:
        #     break

        inference_idx.append(idx)
        inference_imgs.append(frames[idx])

        # save image
        img_dir = out_dir / "raw_frames" / f"frame_{idx:04d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        draw_kpt = visualizer.draw_skeleton_2d(
            image=frames[idx].numpy(),
            keypoints=kpts[idx],
        )

        cv2.imwrite(
            (img_dir / f"frame_{idx:04d}_kpt.png").as_posix(),
            draw_kpt,
        )

    # infer vggt
    (
        camera_extrinsics,
        camera_intrinsics_resized,
        R,
        t,
        C,
        world_points_from_depth,
    ) = camera_head.reconstruct_from_frames(
        imgs=inference_imgs,
        frame_id=0,
    )

    all_frame_camera_extrinsics.append(camera_extrinsics)
    all_frame_camera_intrinsics.append(camera_intrinsics_resized)
    all_frame_R.append(R)
    all_frame_t.append(t)
    all_frame_C.append(C)

    # save 3d info into npz
    save_camera_info(
        out_pt_path=inference_output_path / f"{subject}_multi_view_3d_info.npz",
        all_frame_camera_intrinsics=all_frame_camera_intrinsics,
        all_frame_R=all_frame_R,
        all_frame_t=all_frame_t,
        all_frame_C=all_frame_C,
    )
