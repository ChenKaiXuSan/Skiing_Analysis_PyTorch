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
import json
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def save_3d_joints(
    joints_3d: np.ndarray,
    save_dir: str,
    frame_idx: int,
    r: np.ndarray,
    t: np.ndarray,
    video_path: dict[str, str],
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
    os.makedirs(save_dir, exist_ok=True)
    fname_base = f"frame_{frame_idx:04d}_3dpose"

    if fmt == "npy":
        np.save(os.path.join(save_dir, f"{fname_base}.npy"), joints_3d)

    elif fmt == "csv":
        df = pd.DataFrame(joints_3d, columns=["X", "Y", "Z"])
        df.index.name = "joint_id"
        df.to_csv(os.path.join(save_dir, f"{fname_base}.csv"), float_format="%.6f")

    elif fmt == "json":
        data = {
            "frame": frame_idx,
            "num_joints": len(joints_3d),
            "joints_3d": joints_3d.tolist(),
            "R": r.tolist(),
            "T": t.tolist(),
            "video_path": {k: str(v) for k, v in video_path.items()},
        }
        with open(
            os.path.join(save_dir, f"{fname_base}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    logger.info(f"3D joints saved ({fmt}) → {os.path.join(save_dir, fname_base + '.' + fmt)}")
    