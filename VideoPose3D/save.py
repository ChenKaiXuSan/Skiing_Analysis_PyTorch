#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/save.py
Project: /workspace/code/triangulation
Created Date: Friday October 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday October 10th 2025 10:58:39 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import os
import numpy as np
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def save_3d_joints(
    fused_joints_3d: np.ndarray,
    left_joints_3d: np.ndarray,
    right_joints_3d: np.ndarray,
    save_dir: Path,
    fmt: str = "npy",
):
    """
    保存3D关节坐标到文件（支持 npy / csv / json）

    Args:
        joints_3d (np.ndarray): (J,3) 关节坐标，单位可为m或任意世界单位
        save_dir (str): 输出文件夹路径
        frame_idx (int): 当前帧编号
        fmt (str): 保存格式，可选 ['npy', 'csv', 'json']
    """

    save_dir.parent.mkdir(parents=True, exist_ok=True)

    _dict = {
        "fused_joints_3d": fused_joints_3d.tolist(),
        "left_joints_3d": left_joints_3d.tolist(),
        "right_joints_3d": right_joints_3d.tolist(),
    }

    if fmt == "npy":
        np.save(save_dir, _dict)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    logger.info(f"save_3d_joints: {save_dir}")