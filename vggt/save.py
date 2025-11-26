#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/save.py
Project: /workspace/code/vggt
Created Date: Monday November 24th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday November 24th 2025 4:37:41 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
import numpy as np
import logging
from vggt.visual_util import predictions_to_glb

logger = logging.getLogger(__name__)


def save_inference_results(
    preds: dict,
    outdir: Path,
    conf_thres: float = 0.5,
    prediction_mode: str = "All",
) -> dict:
    """
    保存 VGGT 推理结果：
    1. 保存 npz
    2. 导出 glb

    Args:
        preds: VGGT 推理结果
        outdir: 输出目录
        imgs: 输入图像列表
        conf_thres: 导出 glb 时的置信度阈值
        prediction_mode: 导出 glb 时的预测模式

    Returns:
        dict 包含路径等信息
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 保存 npz
    npz_path = outdir / "predictions.npz"
    np.savez(npz_path, **preds)

    # 导出 glb
    glb_path = (
        outdir / f"scene_conf{conf_thres}_mode{prediction_mode.replace(' ', '_')}.glb"
    )
    glb = predictions_to_glb(
        preds,
        conf_thres=conf_thres,
        filter_by_frames="All",
        show_cam=True,
        mask_black_bg=False,
        mask_white_bg=False,
        mask_sky=False,
        target_dir=outdir,
        prediction_mode=prediction_mode,
    )
    glb.export(file_obj=glb_path)
    logger.info(f"Saved GLB → {glb_path}")

    return dict(
        npz_path=npz_path,
        glb_path=glb_path,
        preds=preds,
    )


# update 3d information to pt file
def save_camera_info(
    out_pt_path: Path,
    all_frame_x3d: list[np.ndarray],
    all_frame_camera_intrinsics: list[np.ndarray],
    all_frame_R: list[np.ndarray],
    all_frame_t: list[np.ndarray],
    all_frame_C: list[np.ndarray],
):
    """
    更新 pt 文件，添加 3D 信息

    Args:
        out_pt_path: 输出 pt 文件路径
        reprojet_err: 重投影误差字典
    """

    data = {
        "x3d_pose": np.stack(all_frame_x3d, axis=0),  # (N, J, 3)
        "camera_intrinsics": np.stack(
            all_frame_camera_intrinsics, axis=0
        ),  # (N, C, 3, 3)
        "R": np.stack(all_frame_R, axis=0),  # (N, C, 3, 3)
        "t": np.stack(all_frame_t, axis=0),  # (N, C, 3)
        "C": np.stack(all_frame_C, axis=0),  # (N, C, 3)
    }

    # 保存更新后的 pt 文件
    np.savez_compressed(out_pt_path.with_suffix(".npz"), **data)
    logger.info(f"Updated PT with 3D info → {out_pt_path}")
